"""Frozen MapAnything spatial features of one exterior view, aligned to pi0.5's SigLIP token grid.

Runs in the MapAnything venv (PyTorch), not openpi's: /data/haotian/mapanything/venv.

Input per sample: the exterior camera's render (H, W, 3) uint8, its metric z-depth (H, W) in metres
(0 = invalid) and its 3x3 intrinsics for (H, W) in the MolmoSpaces convention (pixel i spans [i, i+1),
principal point W/2 for a centred camera). No extrinsics.

1. MapAnything input: the official preprocessing (mapanything.utils.image.preprocess_inputs,
   "fixed_mapping" 518 set: 640x368 -> 518x294, a slight centre crop) of image + depth_z + intrinsics,
   one view with the samples as its batch, so no information is shared across samples or views.
   Geometric inputs are always used (the infer() configuration: ray directions from K, depth with
   is_metric_scale=True, no poses).
2. Features: the selected layer of the info-sharing transformer, (B, 1536, 21, 37) for 518x294
   (one 1536-d token per 14x14 patch). With a single view the alternating-attention blocks reduce to
   self-attention within the view.
     "final"  - the transformer's last block (16), what the DPT / pose / scale heads decode (default)
     "inter7", "inter11" - its intermediate outputs (DPT inputs 2 and 3)
     "fused"  - its input: DINOv2-g patch features + ray-direction and depth encodings, LayerNorm'd
3. Alignment: pi0.5's exterior image is the 320x180 frame (the render resized) letterboxed into 224x224,
   so SigLIP token (r, c) covers a 14x14 square of that canvas. Each token's feature is the mean of the
   MapAnything feature map bilinearly sampled at 7x7 points of its square that fall inside the image
   (padding rows have none). Two extra channels: the fraction of those points inside the image and the
   fraction with valid depth.

Output: (B, 256, 1536 + 2) float32, row-major tokens like SigLIP's.
"""

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F

LAYERS = ("final", "inter7", "inter11", "fused")
GRID = 16  # SigLIP So400m/14 on 224 x 224
CANVAS = 224
SUB = 7  # sample points per token side
SIGLIP_IMAGE_HW = (180, 320)  # the frame pi0.5 sees (LeRobot conversion of the render), letterboxed to 224


class _Stop(Exception):
    pass


@dataclasses.dataclass
class _Layout:
    """Sample points of every token, in the MapAnything input's pixels, for one render size."""

    grid: torch.Tensor  # (1, GRID*SUB, GRID*SUB, 2) grid_sample coords on the feature map
    inside: torch.Tensor  # (GRID*SUB, GRID*SUB) bool: the point lies in the (cropped) image
    raw_ij: torch.Tensor  # (GRID*SUB, GRID*SUB, 2) long: (row, col) of the point in the render


class MapAnythingFeatures:
    def __init__(self, layer: str = "final", device: str = "cuda", model_id: str = "facebook/map-anything"):
        from mapanything.models import MapAnything

        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
        self.layer = layer
        self.device = torch.device(device)
        self.model = MapAnything.from_pretrained(model_id).to(self.device).eval()
        self.model.requires_grad_(False)
        self.dim = self.model.info_sharing.dim if layer != "fused" else self.model.encoder.enc_embed_dim
        self.patch = self.model.encoder.patch_size
        self._layouts = {}

    # -- MapAnything --------------------------------------------------------------------------------
    def _views(self, rgb, depth, K):
        """One MapAnything view batching the samples: preprocess_inputs per sample, then stacked."""
        from mapanything.utils.image import preprocess_inputs

        out = []
        for i in range(len(rgb)):
            k = np.asarray(K[i], np.float64).copy()
            k[0, 2] -= 0.5  # MolmoSpaces: pixel i spans [i, i+1); MapAnything (OpenCV): pixel i centred at i
            k[1, 2] -= 0.5
            (v,) = preprocess_inputs(
                [{"img": np.asarray(rgb[i], np.uint8), "depth_z": np.asarray(depth[i], np.float32),
                  "intrinsics": k.astype(np.float32), "is_metric_scale": torch.tensor([True])}],
                resize_mode="fixed_mapping", resolution_set=518,
            )
            v["_K_in"] = torch.from_numpy(k.astype(np.float32))[None]
            out.append(v)
        view = {key: torch.cat([v[key] for v in out]) for key in ("img", "depth_z", "intrinsics", "is_metric_scale", "_K_in")}
        view["data_norm_type"] = out[0]["data_norm_type"]
        return view

    @torch.inference_mode()
    def _features(self, view):
        """(B, C, h, w) features of the selected layer, without running the prediction heads."""
        from mapanything.utils.inference import preprocess_input_views_for_inference, validate_input_views_for_inference

        m = self.model
        views = validate_input_views_for_inference([{k: v for k, v in view.items() if k != "_K_in"}])
        for v in views:
            for k in list(v):
                if hasattr(v[k], "to") and k not in ("data_norm_type",):
                    v[k] = v[k].to(self.device)
        views = preprocess_input_views_for_inference(views)
        captured = {}

        def pre_hook(_mod, args):
            if self.layer == "fused":
                captured["x"] = args[0].features[0]
                raise _Stop

        def hook(_mod, _args, output):
            final, inter = output
            captured["x"] = {"final": final.features[0], "inter7": inter[0].features[0], "inter11": inter[1].features[0]}[self.layer]
            raise _Stop

        handles = [m.info_sharing.register_forward_pre_hook(pre_hook), m.info_sharing.register_forward_hook(hook)]
        m._configure_geometric_input_config(use_calibration=True, use_depth=True, use_pose=False, use_depth_scale=True, use_pose_scale=False)
        try:
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                m.forward(views)
        except _Stop:
            pass
        finally:
            for h in handles:
                h.remove()
            m._restore_original_geometric_input_config()
        return captured["x"].float()

    # -- alignment to the SigLIP grid ----------------------------------------------------------------
    def _layout(self, raw_hw, K_in, K_ma, ma_hw) -> _Layout:
        key = (raw_hw, ma_hw, tuple(np.round(K_in.numpy().ravel(), 3)), tuple(np.round(K_ma.numpy().ravel(), 3)))
        if key in self._layouts:
            return self._layouts[key]
        H, W = raw_hw
        h_s, w_s = SIGLIP_IMAGE_HW
        n = GRID * SUB
        c = (torch.arange(n, dtype=torch.float64) + 0.5) * CANVAS / n  # canvas sample points (continuous)
        y, x = torch.meshgrid(c, c, indexing="ij")
        scale = CANVAS / max(h_s, w_s)
        top, left = (CANVAS - h_s * scale) / 2, (CANVAS - w_s * scale) / 2
        u = (x - left) / (w_s * scale) * W  # render pixels, continuous ([0, W))
        v = (y - top) / (h_s * scale) * H
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        # render (OpenCV coords) -> MapAnything input pixels: the crop + resize is the affine K_ma K_in^-1
        A = K_ma.double() @ torch.linalg.inv(K_in.double())
        pts = torch.stack([u - 0.5, v - 0.5, torch.ones_like(u)], -1) @ A.T
        hm, wm = ma_hw
        inside &= (pts[..., 0] >= -0.5) & (pts[..., 0] < wm - 0.5) & (pts[..., 1] >= -0.5) & (pts[..., 1] < hm - 0.5)
        # feature (i, j) sits at the centre of its patch: pixel (j * p + (p - 1) / 2) in OpenCV coords
        p = self.patch
        gx = (pts[..., 0] - (p - 1) / 2) / p
        gy = (pts[..., 1] - (p - 1) / 2) / p
        fw, fh = wm // p, hm // p
        grid = torch.stack([gx / (fw - 1) * 2 - 1, gy / (fh - 1) * 2 - 1], -1)[None].float()  # align_corners=True
        raw_ij = torch.stack([v.floor().clamp(0, H - 1), u.floor().clamp(0, W - 1)], -1).long()
        out = _Layout(grid.to(self.device), inside.to(self.device), raw_ij.to(self.device))
        self._layouts[key] = out
        return out

    def __call__(self, rgb, depth, K) -> np.ndarray:
        """rgb (B, H, W, 3) uint8, depth (B, H, W) metres (0 = invalid), K (B, 3, 3) -> (B, 256, C + 2)."""
        rgb, depth, K = np.asarray(rgb), np.asarray(depth, np.float32), np.asarray(K, np.float32)
        if rgb.ndim == 3:
            rgb, depth, K = rgb[None], depth[None], K[None]
        if not (rgb.shape[:3] == depth.shape and K.shape == (len(rgb), 3, 3)):
            raise ValueError(f"rgb {rgb.shape}, depth {depth.shape}, K {K.shape}: need (B,H,W,3), (B,H,W), (B,3,3)")
        if not np.any(depth > 0):
            raise ValueError("no valid depth: MapAnything features need the exterior depth (not substituted)")
        view = self._views(rgb, depth, K)
        feat = self._features(view)  # (B, C, h, w)
        b = len(rgb)
        ma_hw = tuple(view["img"].shape[-2:])
        outs = []
        for i in range(b):  # layouts are shared by samples with the same intrinsics (all of them, in practice)
            lay = self._layout(rgb.shape[1:3], view["_K_in"][i], view["intrinsics"][i], ma_hw)
            s = F.grid_sample(feat[i : i + 1], lay.grid, mode="bilinear", align_corners=True)[0]  # (C, n, n)
            ins = lay.inside.float()
            d = torch.as_tensor(depth[i], device=self.device)[lay.raw_ij[..., 0], lay.raw_ij[..., 1]]
            valid = ins * (d > 0).float()
            C = s.shape[0]
            s = (s * ins).reshape(C, GRID, SUB, GRID, SUB).sum((2, 4))
            cnt = ins.reshape(GRID, SUB, GRID, SUB).sum((1, 3))
            tok = (s / cnt.clamp(min=1)).reshape(C, GRID * GRID).T  # (256, C)
            frac_in = (cnt / SUB**2).reshape(-1, 1)
            frac_valid = (valid.reshape(GRID, SUB, GRID, SUB).sum((1, 3)) / SUB**2).reshape(-1, 1)
            outs.append(torch.cat([tok, frac_in, frac_valid], -1))
        return torch.stack(outs).cpu().numpy().astype(np.float32)
