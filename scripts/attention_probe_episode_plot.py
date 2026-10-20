"""FoveaVLA-style (arXiv 2610.20064, Fig. 5) image-attention figure for one episode from
scripts/attention_probe.py --episode <e> --stride <s> --no-ablate --out <dir>:
left, each view's share of the image attention over time (External + Wrist = 100%); right, attention maps at
three keyframes, colored relative to each model's own 99th percentile.

  python scripts/attention_probe_episode_plot.py <dir> <stride>
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT, STRIDE, FPS = pathlib.Path(sys.argv[1]), int(sys.argv[2]), 15
R = json.loads((OUT / "results.json").read_text())
models = R["models"]
ext, wr = np.load(OUT / "exterior_images.npy"), np.load(OUT / "wrist_images.npy")
n = len(dict.fromkeys(range(len(R["episodes"]))))
keys = [int(f * (len(ext) - 1)) for f in (0.15, 0.5, 0.85)]
summary = {}
fig = plt.figure(figsize=(17, 2.6 * len(models)))
gs = fig.add_gridspec(len(models), 7, width_ratios=[3.2, 1, 1, 1, 1, 1, 1], wspace=0.08, hspace=0.45)
for r, (m, res) in enumerate(models.items()):
    share = np.array(res["share_per_frame"])  # (frames, groups); groups 0/1 = exterior/wrist
    img = share[:, 0] + share[:, 1]
    e, w = share[:, 0] / img, share[:, 1] / img
    t = np.arange(len(e)) * STRIDE / FPS
    maps = np.load(OUT / f"maps_{m.replace(' ', '_')}.npy")
    content = np.abs(ext.astype(int) - ext[:, :1, :1].astype(int)).max(-1) > 3
    ce = content.reshape(len(ext), 16, 14, 16, 14).mean((2, 4)).reshape(len(ext), 256) > 0.5
    cw = (np.abs(wr.astype(int) - wr[:, :1, :1].astype(int)).max(-1) > 3).reshape(len(wr), 16, 14, 16, 14).mean((2, 4)).reshape(len(wr), 256) > 0.5
    ec, wc = (maps[:, :256] * ce).sum(1), (maps[:, 256:] * cw).sum(1)
    layer = np.array(res["share_per_frame_layer"])  # (frames, layers, groups)
    le = layer[..., 0] / (layer[..., 0] + layer[..., 1])
    summary[m] = {"wrist": float(w.mean()), "wrist_content_only": float((wc / (ec + wc)).mean()),
                  "wrist_by_layer": (1 - le.mean(0)).round(3).tolist(), "success": res["success"]}
    ax = fig.add_subplot(gs[r, 0])
    ax.stackplot(t, e, w, colors=["#d62728", "#1f77b4"], labels=["External", "Wrist"], alpha=0.85)
    ax.plot(t, ec / (ec + wc), "k--", lw=1, label="External, content tokens only")
    for k in keys:
        ax.axvline(t[k], color="k", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlim(t[0], t[-1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("image attention share")
    sr = "" if res["success"] is None else f" ({res['success']:.1f}% success)"
    ax.set_title(f"{m}{sr}: wrist {100 * w.mean():.0f}% of image attention", fontsize=9, loc="left")
    if r == len(models) - 1:
        ax.set_xlabel("episode time (s)")
    if r == 0:
        ax.legend(fontsize=7, loc="lower left")
    vmax = np.percentile(maps[keys], 99)  # each model relative to its own 99th percentile
    for j, k in enumerate(keys):
        for v, (im, sl, name) in enumerate([(ext, slice(0, 256), "External"), (wr, slice(256, 512), "Wrist")]):
            a = fig.add_subplot(gs[r, 1 + 2 * j + v])
            a.imshow(im[k])
            a.imshow(np.kron(maps[k, sl].reshape(16, 16), np.ones((14, 14))), cmap="inferno", alpha=0.6, vmin=0, vmax=vmax)
            a.set_title(f"t={t[k]:.1f}s {name} {100 * maps[k, sl].sum() / (maps[k, :512].sum()):.0f}%", fontsize=7)
            a.axis("off")
fig.savefig(OUT / "episode_attention.png", dpi=120, bbox_inches="tight")
(OUT / "episode_summary.json").write_text(json.dumps(summary, indent=1))
for m, s in summary.items():
    print(f"{m:34s} wrist {100 * s['wrist']:.0f}% of image attention (content tokens only {100 * s['wrist_content_only']:.0f}%); "
          f"by layer min {100 * min(s['wrist_by_layer']):.0f}% max {100 * max(s['wrist_by_layer']):.0f}%")
