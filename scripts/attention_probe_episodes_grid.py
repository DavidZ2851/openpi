"""FoveaVLA-style (arXiv 2610.20064, Fig. 5) External vs Wrist image-attention share over time, several episodes:
one row per episode; pretrained and fine-tuned pi0.5 timelines, plus the fine-tuned model's attention maps at three
keyframes. Reads the per-episode outputs of scripts/attention_probe.py --episode <e> --stride <s> --no-ablate.

  python scripts/attention_probe_episodes_grid.py <out.png> <stride> <episode dir> ...
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT, STRIDE, FPS = pathlib.Path(sys.argv[1]), int(sys.argv[2]), 15
DIRS = [pathlib.Path(d) for d in sys.argv[3:]]
MODELS = [("pretrained pi05", "pretrained π0.5"), ("two-view LoRA", "fine-tuned π0.5")]
MAP_MODEL = "two-view LoRA"

fig = plt.figure(figsize=(19, 2.5 * len(DIRS)))
gs = fig.add_gridspec(len(DIRS), 8, width_ratios=[2.6, 2.6, 1, 1, 1, 1, 1, 1], wspace=0.12, hspace=0.5)
for r, d in enumerate(DIRS):
    R = json.loads((d / "results.json").read_text())
    ep = R["episodes"][0]
    ext, wr = np.load(d / "exterior_images.npy"), np.load(d / "wrist_images.npy")
    keys = [int(f * (len(ext) - 1)) for f in (0.15, 0.5, 0.85)]
    for c, (m, label) in enumerate(MODELS):
        share = np.array(R["models"][m]["share_per_frame"])  # (frames, groups); 0/1 = exterior/wrist
        e = share[:, 0] / (share[:, 0] + share[:, 1])
        t = np.arange(len(e)) * STRIDE / FPS
        ax = fig.add_subplot(gs[r, c])
        ax.stackplot(t, e, 1 - e, colors=["#d62728", "#1f77b4"], labels=["External", "Wrist"], alpha=0.85)
        for k in keys:
            ax.axvline(t[k], color="k", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlim(t[0], t[-1])
        ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_title(f"ep {ep} · {label}: Ext {100 * e.mean():.0f}% / Wrist {100 * (1 - e.mean()):.0f}%",
                     fontsize=8.5, loc="left")
        if c == 0:
            ax.set_ylabel("image attention")
        else:
            ax.set_yticklabels([])
        if r == len(DIRS) - 1:
            ax.set_xlabel("episode time (s)")
        if r == 0 and c == 0:
            ax.legend(fontsize=7, loc="lower left")
    maps = np.load(d / f"maps_{MAP_MODEL.replace(' ', '_')}.npy")
    vmax = np.percentile(maps[keys], 99)
    for j, k in enumerate(keys):
        for v, (im, sl, name) in enumerate([(ext, slice(0, 256), "Ext"), (wr, slice(256, 512), "Wrist")]):
            a = fig.add_subplot(gs[r, 2 + 2 * j + v])
            a.imshow(im[k])
            a.imshow(np.kron(maps[k, sl].reshape(16, 16), np.ones((14, 14))), cmap="inferno", alpha=0.6, vmin=0, vmax=vmax)
            a.set_title(f"t={k * STRIDE / FPS:.1f}s {name} {100 * maps[k, sl].sum() / maps[k, :512].sum():.0f}%", fontsize=7)
            a.axis("off")
fig.suptitle("Image attention of the action tokens, External vs Wrist (External + Wrist = 100%), over each episode; "
             "maps: fine-tuned π0.5", fontsize=11, y=1.0)
fig.subplots_adjust(top=0.95)
fig.savefig(OUT, dpi=120, bbox_inches="tight")
print("saved", OUT)
