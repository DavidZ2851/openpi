"""Cache frozen MapAnything features for a LeRobot dataset (row = the frame's global index).

Inputs: the per-episode raw exterior render, depth and intrinsics exported in the dataset's episode order
(/data/haotian/mapanything/tools/export_inputs.py, checked against the dataset's states). Output: a
(frames, 256, 1538) float16 .npy (molmospaces_policy.AddCachedGeoFeatures) and a .json with its layer,
model and episode lengths.

The features depend only on the frame, so caching them is exact as long as training does not
augment the exterior image spatially (Pi0Config.exterior_spatial_aug=False). Colour jitter only reaches
SigLIP, never MapAnything, and needs no cache change.

  CUDA_VISIBLE_DEVICES=1 /data/haotian/mapanything/venv/bin/python -m mapanything_features.cache \
      <inputs dir> <out.npy> [--layer final] [--batch 16]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

from mapanything_features.extractor import MapAnythingFeatures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layer", default="final")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    files = sorted(args.inputs.glob("ep_*.npz"))
    if [f.name for f in files] != [f"ep_{i:05d}.npz" for i in range(len(files))]:
        raise ValueError("episode files must be ep_00000.npz .. contiguous")
    lengths = [int(np.load(f, mmap_mode="r")["state"].shape[0]) for f in files]
    ex = MapAnythingFeatures(layer=args.layer)
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16, shape=(sum(lengths), 256, ex.dim + 2))
    row = 0
    for f, n in zip(tqdm.tqdm(files, desc="episodes"), lengths, strict=True):
        z = np.load(f)
        for s in range(0, n, args.batch):
            e = min(n, s + args.batch)
            out[row + s : row + e] = ex(z["rgb"][s:e], z["depth"][s:e].astype(np.float32), z["K"][s:e]).astype(np.float16)
        row += n
    out.flush()
    meta = {
        "layer": args.layer,
        "model": "facebook/map-anything",
        "inputs": str(args.inputs),
        "episode_lengths": lengths,
        "shape": list(out.shape),
        "channels": f"{ex.dim} MapAnything features + fraction in image + fraction with valid depth",
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("wrote", args.out, out.shape)


if __name__ == "__main__":
    main()
