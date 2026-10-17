"""Geometry-conditioned residual adapter on image tokens (Pi0Config.geo_adapter).

Sits between SigLIP and the VLM on the exterior camera's tokens:

    z_i = [LN(F_i), g_i]
    F'_i = F_i + sigmoid(W_gate z_i) * W_up GELU(W_down z_i)        (W_up zero-initialised: F' = F at start)

g_i is a per-patch geometry embedding (a small MLP of patch_geometry's features) from the exterior
camera's metric depth map and the intrinsics of its grid, in the camera frame (no extrinsics). The
depth map is the image's square-padded layout (molmospaces cond_inputs.depth_map_for_model, 112 x 112),
so each of SigLIP's 16 x 16 patches covers a 7 x 7 depth block.

Feature sets (patch_geometry):
- "rgb": none (the adapter sees LN(F) alone; the no-geometry control);
- "depth": the block's raw depths / MAX_DEPTH and validity (98);
- "xyz": the block's camera-frame centroid (metres) and valid fraction (4);
- "relative": valid fraction, log(patch depth / scene depth), the block's points minus their centroid
  divided by the centroid depth (scale-free local shape, 147), the plane normal (camera frame, facing
  the camera), |normal . viewing ray| and the plane-fit residual / depth (154).
- "mapanything": precomputed per-token features (Observation.geo_features, Pi0Config.geo_feature_dim):
  frozen MapAnything features aligned to the token grid plus the fractions of the token inside the image
  and with valid depth (mapanything_features/extractor.py). The features are LayerNorm'd (zeroed where
  the token has no image) before the same geometry MLP: the trainable spatial projection.

ReprHead and repr_losses: the cross-view representation objective of Pi0Config.repr_loss.

Params live under Pi0.lora_geo_adapter: "lora" in the path makes the weight loader initialise them
(the base checkpoint has none) and the LoRA freeze filter keep them trainable.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at

GRID = 16  # SigLIP So400m/14 on 224 x 224
MAX_DEPTH = 4.0  # metres; the recorded depth range
FEATURE_DIMS = {"rgb": 0, "depth": 98, "xyz": 4, "relative": 154, "mapanything": None}  # None: geo_feature_dim


def patch_points(depth: at.Float[at.Array, "b s s"], intrinsics: at.Float[at.Array, "b 3 3"], grid: int = GRID):
    """Camera-frame points (b, grid^2, k, 3) of each patch's depth block (k = (s / grid)^2 pixels, row-major
    patches like SigLIP's tokens) and their validity (b, grid^2, k)."""
    b, s, _ = depth.shape
    blk = s // grid
    assert blk * grid == s, (s, grid)
    z = depth.astype(jnp.float32)
    k = intrinsics.astype(jnp.float32)
    uv = jnp.arange(s, dtype=jnp.float32) + 0.5  # cell centres: pixel i spans [i, i + 1)
    x = (uv[None, None, :] - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None] * z
    y = (uv[None, :, None] - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None] * z
    pts = jnp.stack([x, y, z], axis=-1)  # (b, s, s, 3)
    pts = pts.reshape(b, grid, blk, grid, blk, 3).transpose(0, 1, 3, 2, 4, 5).reshape(b, grid * grid, blk * blk, 3)
    valid = (pts[..., 2] > 0).astype(jnp.float32)
    return pts, valid


def patch_geometry(depth, intrinsics, kind: str, grid: int = GRID) -> at.Float[at.Array, "b p f"]:
    """Per-patch geometry features (b, grid^2, FEATURE_DIMS[kind]); see the module docstring."""
    pts, valid = patch_points(depth, intrinsics, grid)
    n = valid.sum(-1)  # (b, p)
    frac = n / valid.shape[-1]
    if kind == "depth":
        return jnp.concatenate([pts[..., 2] / MAX_DEPTH, valid], axis=-1)
    centroid = (pts * valid[..., None]).sum(-2) / jnp.maximum(n, 1.0)[..., None]  # (b, p, 3)
    if kind == "xyz":
        return jnp.concatenate([centroid, frac[..., None]], axis=-1)
    assert kind == "relative", kind
    ok = (n >= 3).astype(jnp.float32)  # enough points for a plane
    zc = jnp.maximum(centroid[..., 2], 1e-3)
    scene = (zc * ok).sum(-1, keepdims=True) / jnp.maximum(ok.sum(-1, keepdims=True), 1.0)
    log_rel = jnp.log(zc / jnp.maximum(scene, 1e-3)) * ok
    local = (pts - centroid[..., None, :]) * valid[..., None] / zc[..., None, None]  # scale-free
    cov = jnp.einsum("bpki,bpkj->bpij", local, local) / jnp.maximum(n, 1.0)[..., None, None]
    cov = cov + 1e-8 * jnp.eye(3)
    evals, evecs = jnp.linalg.eigh(cov)  # ascending
    normal = evecs[..., :, 0]
    ray = centroid / jnp.maximum(jnp.linalg.norm(centroid, axis=-1, keepdims=True), 1e-6)
    facing = jnp.sum(normal * ray, axis=-1, keepdims=True)
    normal = jnp.where(facing > 0, -normal, normal)  # point towards the camera
    rough = jnp.sqrt(jnp.maximum(evals[..., 0], 0.0))
    feats = jnp.concatenate(
        [
            frac[..., None],
            log_rel[..., None],
            normal * ok[..., None],
            jnp.abs(facing) * ok[..., None],
            rough[..., None] * ok[..., None],
            local.reshape(*local.shape[:2], -1) * ok[..., None],
        ],
        axis=-1,
    )
    return jax.lax.stop_gradient(feats)


class GeoAdapter(nnx.Module):
    """(b, p, width) image tokens (+ depth, intrinsics) -> corrected tokens, same shape and dtype."""

    def __init__(
        self, width: int, kind: str, rngs: nnx.Rngs, bottleneck: int = 64, geo_dim: int = 128, feature_dim: int = 0
    ):
        if kind not in FEATURE_DIMS:
            raise ValueError(f"geo_adapter must be one of {sorted(FEATURE_DIMS)}, got {kind!r}")
        self.kind = kind
        self.norm = nnx.LayerNorm(width, rngs=rngs)
        in_dim = width
        if kind == "mapanything":
            # feature_dim = MapAnything channels + (fraction in image, fraction with valid depth)
            self.feat_norm = nnx.LayerNorm(feature_dim - 2, rngs=rngs)
        if kind != "rgb":
            n_in = feature_dim if kind == "mapanything" else FEATURE_DIMS[kind]
            self.geo_in = nnx.Linear(n_in, geo_dim, rngs=rngs)
            self.geo_out = nnx.Linear(geo_dim, geo_dim, rngs=rngs)
            self.geo_norm = nnx.LayerNorm(geo_dim, rngs=rngs)
            in_dim += geo_dim
        self.down = nnx.Linear(in_dim, bottleneck, rngs=rngs)
        self.up = nnx.Linear(bottleneck, width, kernel_init=nnx.initializers.zeros, rngs=rngs)
        self.gate = nnx.Linear(in_dim, 1, rngs=rngs)

    def __call__(self, tokens, depth=None, intrinsics=None, geo_features=None):
        f = tokens.astype(jnp.float32)
        z = [self.norm(f)]
        if self.kind == "mapanything":
            if geo_features is None:
                raise ValueError("geo_adapter='mapanything' needs Observation.geo_features (cached or online)")
            g = geo_features.astype(jnp.float32)
            feat, frac = g[..., :-2], g[..., -2:]
            feat = self.feat_norm(feat) * (frac[..., :1] > 0)  # tokens without image: no feature
            g = jnp.concatenate([feat, frac], axis=-1)
            z.append(self.geo_norm(self.geo_out(nnx.gelu(self.geo_in(g)))))
        elif self.kind != "rgb":
            if depth is None or intrinsics is None:
                raise ValueError(f"geo_adapter={self.kind!r} needs Observation.depth and Observation.intrinsics")
            grid = int(round(tokens.shape[1] ** 0.5))
            g = patch_geometry(depth, intrinsics, self.kind, grid)
            z.append(self.geo_norm(self.geo_out(nnx.gelu(self.geo_in(g)))))
        z = jnp.concatenate(z, axis=-1)
        out = f + nnx.sigmoid(self.gate(z)) * self.up(nnx.gelu(self.down(z)))
        return out.astype(tokens.dtype)


class ReprHead(nnx.Module):
    """Z = h(TaskPool(F', prompt)): attention pooling of the adapted exterior tokens F' with a query from
    the mean prompt embedding (pi0.5's prompt holds the task and the discretised state), then an MLP.
    Training only (Pi0Config.repr_loss); the policy never uses it."""

    def __init__(self, width: int, rngs: nnx.Rngs, out_dim: int = 128, dim: int = 256, heads: int = 4, hidden: int = 512):
        self.heads = heads
        self.tok_norm = nnx.LayerNorm(width, rngs=rngs)
        self.lang_norm = nnx.LayerNorm(width, rngs=rngs)
        self.q = nnx.Linear(width, dim, rngs=rngs)
        self.k = nnx.Linear(width, dim, rngs=rngs)
        self.v = nnx.Linear(width, dim, rngs=rngs)
        self.h_in = nnx.Linear(dim, hidden, rngs=rngs)
        self.h_out = nnx.Linear(hidden, out_dim, rngs=rngs)

    def __call__(self, tokens, prompt_emb, prompt_mask):
        """tokens (b, p, width) adapted exterior tokens; prompt_emb (b, l, width), prompt_mask (b, l) -> (b, out_dim)."""
        t = self.tok_norm(tokens.astype(jnp.float32))
        m = prompt_mask.astype(jnp.float32)[..., None]
        lang = (prompt_emb.astype(jnp.float32) * m).sum(1) / jnp.maximum(m.sum(1), 1.0)
        b, p, _ = t.shape
        q = self.q(self.lang_norm(lang)).reshape(b, self.heads, -1)  # (b, h, d)
        k = self.k(t).reshape(b, p, self.heads, -1)
        v = self.v(t).reshape(b, p, self.heads, -1)
        att = jax.nn.softmax(jnp.einsum("bhd,bphd->bhp", q, k) / jnp.sqrt(q.shape[-1]), axis=-1)
        pooled = jnp.einsum("bhp,bphd->bhd", att, v).reshape(b, -1)
        return self.h_out(nnx.gelu(self.h_in(pooled)))


def repr_losses(z: at.Float[at.Array, "b d"], kind: str, var_gamma: float = 1.0) -> dict[str, at.Array]:
    """Cross-view terms for paired rows (2i: view A, 2i+1: view B of the same timestep).

    inv: mean_i ||Z_A^i - Z_B^i||^2 (summed over dimensions). vicreg adds, per branch and averaged over
    the two: var = mean_d relu(gamma - sqrt(Var_batch(Z_d) + 1e-4)); cov = sum of squared off-diagonal
    batch covariances / d (VICReg, Bardes et al. 2022)."""
    za, zb = z[0::2].astype(jnp.float32), z[1::2].astype(jnp.float32)
    out = {"inv": jnp.mean(jnp.sum(jnp.square(za - zb), axis=-1))}
    if kind == "vicreg":
        def var_cov(x):
            n, d = x.shape
            x = x - x.mean(0)
            std = jnp.sqrt(x.var(0) + 1e-4)
            cov = x.T @ x / (n - 1)
            off = cov - jnp.diag(jnp.diag(cov))
            return jnp.mean(jax.nn.relu(var_gamma - std)), jnp.sum(jnp.square(off)) / d
        (va, ca), (vb, cb) = var_cov(za), var_cov(zb)
        out["var"] = 0.5 * (va + vb)
        out["cov"] = 0.5 * (ca + cb)
    return out
