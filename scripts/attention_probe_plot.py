"""Figures for scripts/attention_probe.py: python scripts/attention_probe_plot.py /data/haotian/attention_probe"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(sys.argv[1])
R = json.loads((OUT / "results.json").read_text())
names, models = R["groups"], R["models"]
keep = [i for i, n in enumerate(names) if n != "wrist2 (masked)"]
colors = ["#d62728", "#1f77b4", "#ccc", "#7f7f7f", "#2ca02c", "#ff7f0e", "#bcbd22", "#9467bd"]
labels = [f"{m}" + ("" if r["success"] is None else f"  ({r['success']:.1f}%)") for m, r in models.items()]

# 1. attention share
fig, ax = plt.subplots(figsize=(10, 0.55 * len(models) + 1.5))
left = np.zeros(len(models))
for i in keep:
    v = np.array([r["share"][i] for r in models.values()])
    ax.barh(labels, v, left=left, color=colors[i], label=names[i])
    for y, (l, x) in enumerate(zip(left, v)):
        if x > 0.04:
            ax.text(l + x / 2, y, f"{100 * x:.0f}", ha="center", va="center", fontsize=8, color="w")
    left += v
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("share of the action tokens' attention (all layers, heads, denoising steps)")
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25))
fig.tight_layout()
fig.savefig(OUT / "attention_share.png", dpi=130)

# 2. exterior / wrist share by layer
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for (m, r), c in zip(models.items(), plt.cm.tab10.colors):
    L = np.array(r["share_by_layer"])
    ax[0].plot(L[:, 0], color=c, label=m)
    ax[1].plot(L[:, 1], color=c, label=m)
for a, t in zip(ax, ["exterior", "wrist"]):
    a.set_title(f"{t} share by layer")
    a.set_xlabel("layer")
ax[0].set_ylabel("attention share")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "attention_by_layer.png", dpi=130)

# 3. ablations
conds = sorted({k for r in models.values() for k in r["mse_vs_full"]}, key=lambda k: list(next(iter(models.values()))["mse_vs_full"]).index(k) if k in next(iter(models.values()))["mse_vs_full"] else 99)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
w = 0.8 / len(models)
for j, ((m, r), c) in enumerate(zip(models.items(), plt.cm.tab10.colors)):
    x = np.arange(len(conds)) + j * w
    ax[0].bar(x, [r["mse_vs_full"].get(k, np.nan) for k in conds], w, color=c, label=m)
    ax[1].bar(x, [r["mse_gt"].get(k, np.nan) / r["mse_gt"]["full"] for k in conds], w, color=c, label=m)
for a in ax:
    a.set_xticks(np.arange(len(conds)) + 0.4 - w / 2, conds, rotation=15, fontsize=8)
ax[0].set_title("action change vs the intact input (MSE, normalized joints)")
ax[1].set_title("error to ground truth, relative to the intact input (1 = no effect)")
ax[1].axhline(1, color="k", lw=0.8)
ax[1].set_yscale("log")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "ablations.png", dpi=130)

# 4. where in the image: mean attention per image token, for 4 frames
ext, wr = np.load(OUT / "exterior_images.npy"), np.load(OUT / "wrist_images.npy")
show = [m for m in ["pretrained pi05", "frozenvis LoRA", "E2 MapAnything + frozenvis LoRA"] if m in models]
frames = np.linspace(0, len(ext) - 1, 4).astype(int)
fig, ax = plt.subplots(len(show), 8, figsize=(16, 2.2 * len(show) + 0.4))
ax = np.atleast_2d(ax)
for r_, m in enumerate(show):
    maps = np.load(OUT / f"maps_{m.replace(' ', '_')}.npy")
    vmax = np.percentile(maps, 99.5)
    for c_, f in enumerate(frames):
        for k, (img, sl) in enumerate([(ext, slice(0, 256)), (wr, slice(256, 512))]):
            a = ax[r_, 2 * c_ + k]
            a.imshow(img[f])
            a.imshow(np.kron(maps[f, sl].reshape(16, 16), np.ones((14, 14))), cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
            a.set_title(f"{'ext' if k == 0 else 'wrist'} {100 * maps[f, sl].sum():.1f}%", fontsize=8)
            a.axis("off")
    ax[r_, 0].text(-20, 112, m, rotation=90, va="center", ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "attention_maps.png", dpi=110)
print("wrote", *sorted(p.name for p in OUT.glob("*.png")))

# 5. image content vs letterbox padding (padding tokens often act as attention sinks)
def content_tokens(img):
    c = np.abs(img.astype(np.int16) - img[:, :1, :1].astype(np.int16)).max(-1) > 3
    return c.reshape(len(img), 16, 14, 16, 14).mean((2, 4)).reshape(len(img), 256) > 0.5

ce, cw = content_tokens(ext), content_tokens(wr)
rows = []
for m, r in models.items():
    maps = np.load(OUT / f"maps_{m.replace(' ', '_')}.npy")
    e, w_ = maps[:, :256], maps[:, 256:512]
    rows.append({
        "model": m, "success": r["success"],
        "ext_content": float((e * ce).sum(1).mean()), "ext_padding": float((e * ~ce).sum(1).mean()),
        "wrist_content": float((w_ * cw).sum(1).mean()), "wrist_padding": float((w_ * ~cw).sum(1).mean()),
        "ext_per_content_token": float(((e * ce).sum(1) / ce.sum(1)).mean()),
        "wrist_per_content_token": float(((w_ * cw).sum(1) / cw.sum(1)).mean()),
    })
(OUT / "content_split.json").write_text(json.dumps(rows, indent=1))
print(f"content tokens: exterior {ce.sum(1).mean():.0f}/256, wrist {cw.sum(1).mean():.0f}/256")
for r in rows:
    print(f"{r['model']:34s} ext content {100*r['ext_content']:5.1f}%  pad {100*r['ext_padding']:5.1f}% | wrist content {100*r['wrist_content']:5.1f}%  pad {100*r['wrist_padding']:5.1f}% | per content token ext/wrist {r['ext_per_content_token']/r['wrist_per_content_token']:.2f}")
