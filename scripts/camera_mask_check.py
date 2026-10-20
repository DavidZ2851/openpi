"""Checks for the camera-dependency experiments (configs pi05_droid_jointpos_molmospaces_lora_{wristonly,extonly,nocam}).

1. Existing configs unchanged: the base LoRA config builds the same transforms as before (no MaskCameras), and each
   masked config equals the base config except for its name and data.masked_cameras.
2. End to end through the served policy (same path as the closed-loop eval), with the full-camera two-view LoRA
   checkpoint and fixed noise: replacing a MASKED camera's image with a different real frame or random noise leaves the
   action chunk bitwise identical (its information is unreachable), while changing a kept camera changes the actions.
3. Training batches of the masked configs carry image_mask False and zero images for the masked cameras.

  CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=/data/haotian/lerobot_home .venv/bin/python scripts/camera_mask_check.py
"""

import dataclasses
import sys

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.policies import droid_policy
from openpi.policies import molmospaces_policy
from openpi.policies import policy_config

CKPT = "/data/haotian/openpi_store/checkpoints/pi05_droid_jointpos_molmospaces_lora/pick_place_potato_onehouse_two_random_lora/19999"
REPO = "local/pick_place_potato_onehouse_two_random"
BASE = "pi05_droid_jointpos_molmospaces_lora"
MASKED = {"wristonly": ("base_0_rgb",), "extonly": ("left_wrist_0_rgb",), "nocam": ("base_0_rgb", "left_wrist_0_rgb")}
EXT, WRIST = "observation/exterior_image_1_left", "observation/wrist_image_left"


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main():
    base = _config.get_config(BASE)
    base_tf = base.data.create(base.assets_dirs, base.model).data_transforms.inputs
    check(not any(isinstance(t, droid_policy.MaskCameras) for t in base_tf) and base.data.masked_cameras == (),
          f"{BASE}: no camera masking (transforms {[type(t).__name__ for t in base_tf]})")
    for k, cams in MASKED.items():
        c = _config.get_config(f"{BASE}_{k}")
        check(dataclasses.replace(c, name=BASE, data=dataclasses.replace(c.data, masked_cameras=())) == base
              and c.data.masked_cameras == cams, f"{c.name} == {BASE} except masked_cameras={cams}")

    # real frames (DROID-format dicts, as the eval client sends them)
    cfg = dataclasses.replace(base, data=dataclasses.replace(base.data, repo_id=REPO))
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = _data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    to_droid = molmospaces_policy.MolmoSpacesToDroid()
    frames = [to_droid(dict(raw[i])) for i in (100, 3000)]  # two different episodes
    obs = {k: v for k, v in frames[0].items() if k.startswith("observation/") or k == "prompt"}
    obs = {k: (np.asarray(v) if k != "prompt" else v) for k, v in obs.items()}
    other = {EXT: np.asarray(frames[1][EXT]), WRIST: np.asarray(frames[1][WRIST])}
    rng = np.random.default_rng(0)
    noise_img = {k: (rng.random(np.asarray(obs[k]).shape) * (255 if np.asarray(obs[k]).dtype == np.uint8 else 1)).astype(np.asarray(obs[k]).dtype) for k in (EXT, WRIST)}
    noise = np.random.default_rng(1).standard_normal((cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)

    for name, cams in [("full", ()), *MASKED.items()]:
        cname = BASE if name == "full" else f"{BASE}_{name}"
        policy = policy_config.create_trained_policy(_config.get_config(cname), CKPT)
        ref = policy.infer(obs, noise=noise)["actions"]
        check(np.array_equal(ref, policy.infer(obs, noise=noise)["actions"]), f"[{name}] deterministic with fixed noise")
        masked_keys = {EXT: "base_0_rgb" in cams, WRIST: "left_wrist_0_rgb" in cams}
        for key, is_masked in masked_keys.items():
            for desc, img in [("another episode's frame", other[key]), ("random noise", noise_img[key])]:
                out = policy.infer({**obs, key: img}, noise=noise)["actions"]
                diff = float(np.abs(out - ref).max())
                if is_masked:
                    check(diff == 0.0, f"[{name}] {key.split('/')[-1]} masked: {desc} -> actions identical (max |diff| {diff:.1e})")
                else:
                    check(diff > 1e-4, f"[{name}] {key.split('/')[-1]} kept: {desc} -> actions change (max |diff| {diff:.3f})")
        del policy

    for name, cams in MASKED.items():
        c = _config.get_config(f"{BASE}_{name}")
        c = dataclasses.replace(c, data=dataclasses.replace(c.data, repo_id=REPO), batch_size=4)
        loader = _data_loader.create_data_loader(c, shuffle=True, num_batches=1, skip_norm_stats=False)
        o, _ = next(iter(loader))
        for k in ("base_0_rgb", "left_wrist_0_rgb"):
            m, im = np.asarray(o.image_masks[k]), np.asarray(o.images[k])
            if k in cams:
                check(not m.any() and np.all(im == im.flat[0]), f"[{name}] training batch: {k} mask False, constant image")
            else:
                check(m.all() and im.std() > 0, f"[{name}] training batch: {k} mask True, real image")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
