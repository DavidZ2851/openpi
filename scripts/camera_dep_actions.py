"""Action-prediction difference for the camera-dependency experiments: the served policy (same transforms as the
closed-loop eval) on fixed frames of the potato two-random-view set, matched noise per frame across conditions.
Reports, per (config, checkpoint): joint error to the demonstration's action chunk and change vs the reference
(full-camera M1) chunk, in radians (7 arm joints, absolute targets).

  CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=/data/haotian/lerobot_home .venv/bin/python scripts/camera_dep_actions.py \
      --out /data/haotian/camera_dep/actions.json label=config=checkpoint ...
"""

import argparse
import dataclasses
import json
import pathlib

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.policies import molmospaces_policy
from openpi.policies import policy_config

REPO = "local/pick_place_potato_onehouse_two_random"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=config=checkpoint; the first is the reference")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--frames", type=int, default=128)
    args = ap.parse_args()
    base = _config.get_config("pi05_droid_jointpos_molmospaces_lora")
    cfg = dataclasses.replace(base, data=dataclasses.replace(base.data, repo_id=REPO))
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = _data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    to_droid = molmospaces_policy.MolmoSpacesToDroid()
    idx = np.linspace(0, len(raw) - 1, args.frames).astype(int)
    obs, gt = [], []
    for i in idx:
        d = to_droid(dict(raw[int(i)]))
        obs.append({k: (v if k == "prompt" else np.asarray(v)) for k, v in d.items() if k.startswith("observation/") or k == "prompt"})
        gt.append(np.asarray(d["actions"])[:, :7])  # absolute joint targets, (15, 7)
    gt = np.stack(gt)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((len(idx), cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    preds, ref = {}, None
    for run in args.runs:
        label, cname, ckpt = run.split("=")
        policy = policy_config.create_trained_policy(_config.get_config(cname), ckpt)
        p = np.stack([policy.infer(o, noise=n)["actions"][:, :7] for o, n in zip(obs, noise, strict=True)])
        preds[label] = p
        ref = p if ref is None else ref
        err = np.sqrt(np.square(p - gt).mean((1, 2)))
        chg = np.sqrt(np.square(p - ref).mean((1, 2)))
        results[label] = {"config": cname, "checkpoint": ckpt, "rmse_to_demo_rad": float(err.mean()),
                          "rmse_vs_reference_rad": float(chg.mean()), "reference": args.runs[0].split("=")[0],
                          "frames": len(idx)}
        print(f"{label:28s} RMSE to demo {err.mean():.4f} rad | change vs {args.runs[0].split('=')[0]} {chg.mean():.4f} rad", flush=True)
        del policy
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
