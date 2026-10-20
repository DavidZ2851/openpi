"""FoveaVLA (arXiv 2610.20064, Fig. 5) look-alike of attention_probe_episode_plot.py: letterbox padding removed.

Shares count content tokens only (the 128 padding tokens per view are attention sinks, ~70% of each view's mass);
the dashed line is the wrist share of layers 12-13 (all tokens; per-layer maps are not stored). Maps: content tokens,
bilinear-upsampled, cropped to the 16:9 content, jet colored to each model's own 99th percentile.

  python scripts/attention_probe_paper_plot.py <dir> <stride>   # dir from attention_probe.py --episode e --no-ablate
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT, STRIDE, FPS = pathlib.Path(sys.argv[1]), int(sys.argv[2]), 15
EXT_C, WR_C = "#1f6fb4", "#b5345b"
R = json.loads((OUT / "results.json").read_text())
ext, wr = np.load(OUT / "exterior_images.npy"), np.load(OUT / "wrist_images.npy")


def content(im):  # (frames, 256) token is >50% non-padding
    pix = np.abs(im.astype(int) - im[:, :1, :1].astype(int)).max(-1) > 3
    return pix.reshape(len(im), 16, 14, 16, 14).mean((2, 4)).reshape(len(im), 256) > 0.5


ce, cw = content(ext), content(wr)
rows = np.where((np.abs(ext[0].astype(int) - ext[0, :1, :1].astype(int)).max(-1) > 3).any(1))[0]
top, bot = rows.min(), rows.max() + 1
keys = [int(f * (len(ext) - 1)) for f in (0.15, 0.5, 0.85)]
models = R["models"]
fig = plt.figure(figsize=(13, 3.3 * len(models)))
outer = fig.add_gridspec(len(models), 2, width_ratios=[1.35, 1], wspace=0.12, hspace=0.42)
for r, (m, res) in enumerate(models.items()):
    maps = np.load(OUT / f"maps_{m.replace(' ', '_')}.npy")
    me, mw = maps[:, :256] * ce, maps[:, 256:] * cw
    e, w = me.sum(1), mw.sum(1)
    w_share, e_share = w / (e + w), e / (e + w)
    layer = np.array(res["share_per_frame_layer"])[:, 12:14].mean(1)
    t = np.arange(len(e)) * STRIDE / FPS
    ax = fig.add_subplot(outer[r, 0])
    ax.plot(t, e_share, color=EXT_C, lw=2, label="External")
    ax.plot(t, w_share, color=WR_C, lw=2, label="Wrist")
    ax.plot(t, layer[:, 1] / (layer[:, 0] + layer[:, 1]), color=WR_C, lw=1, ls="--", label="Wrist, layers 12-13")
    for i, k in enumerate(keys):
        ax.axvline(t[k], color="k", ls=(0, (3, 2)), lw=1.2)
        ax.text(t[k], 0.5, str(i + 1), ha="center", va="center", color="w", fontsize=9, fontweight="bold",
                bbox={"boxstyle": "circle,pad=0.25", "fc": "k"})
    ax.set(ylim=(0, 1), xlim=(0, t[-1]), ylabel="Attention share", yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(alpha=0.3)
    title = f"{m}" + (f" ({res['success']:.1f}% success)" if res.get("success") else "")
    ax.set_title(f"{title}: wrist {w_share.mean():.0%}", loc="left", fontsize=11, fontweight="bold")
    if r == 0:
        ax.legend(fontsize=8, loc="lower center", ncol=3)
    if r == len(models) - 1:
        ax.set_xlabel("Episode time (s)")
    vmax = np.percentile(np.concatenate([me[keys][ce[keys]], mw[keys][cw[keys]]]), 99)
    inner = outer[r, 1].subgridspec(3, 2, wspace=0.04, hspace=0.06)
    for i, k in enumerate(keys):
        for j, (im, mp, name, c) in enumerate([(ext, me, "External", EXT_C), (wr, mw, "Wrist", WR_C)]):
            a = fig.add_subplot(inner[i, j])
            a.imshow(im[k], extent=(0, 224, 224, 0))
            a.imshow(mp[k].reshape(16, 16), extent=(0, 224, 224, 0), cmap="jet", vmin=0, vmax=vmax, alpha=0.5,
                     interpolation="bilinear")
            a.set(xlim=(0, 224), ylim=(bot, top), xticks=[], yticks=[])
            if i == 0:
                a.set_title(name, color=c, fontsize=10)
            if j == 0:
                a.set_ylabel(str(i + 1), rotation=0, labelpad=10, fontsize=10, fontweight="bold", va="center")
fig.savefig(OUT / "paper_style.png", dpi=130, bbox_inches="tight")
print(OUT / "paper_style.png")
