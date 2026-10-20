"""Results table + bar chart for the camera-dependency experiments (data/haotian/camera_dep/run_camera_dep.sh).

  python scripts/camera_dep_summary.py /data/haotian/camera_dep
"""

import json
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(sys.argv[1])
EV = pathlib.Path("/data/haotian/pi05_eval/onehouse")
SEEDS = (1111, 2222, 3333)
ROWS = [  # (camera configuration, inference-masking tag, fine-tuned tag)
    ("Wrist + External", "camdep_e1_full", "camdep_e1_full"),  # M1 = the full-camera checkpoint itself
    ("Wrist Only", "camdep_e1_wristonly", "camdep_m2_wristonly"),
    ("External Only", "camdep_e1_extonly", "camdep_m3_extonly"),
    ("No Cameras", "camdep_e1_nocam", None),
]


def sr(tag):
    per = []
    for s in SEEDS:
        f = EV / f"eval_{tag}_s{s}.log"
        m = re.findall(r"Success count: (\d+), Total count: (\d+)", f.read_text(errors="ignore")) if f.exists() else []
        if m:
            per.append((int(m[-1][0]), int(m[-1][1])))
    if not per:
        return None
    k, n = sum(a for a, _ in per), sum(b for _, b in per)
    rates = [100 * a / b for a, b in per]
    return {"success": k, "total": n, "sr": 100 * k / n, "per_seed": [f"{a}/{b}" for a, b in per],
            "std_over_seeds": float(np.std(rates, ddof=1)) if len(rates) > 1 else None}


res = {name: {"masking": sr(t1), "finetuned": sr(t2) if t2 else None} for name, t1, t2 in ROWS}
fmt = lambda r: "TBD" if r is None else f"{r['sr']:.1f}% ({r['success']}/{r['total']}; {', '.join(r['per_seed'])})"
lines = ["| Camera configuration | Inference masking SR (M1 ckpt) | Fine-tuned SR |", "|---|---|---|"]
for name, _, t2 in ROWS:
    lines.append(f"| {name} | {fmt(res[name]['masking'])} | {fmt(res[name]['finetuned']) if t2 else 'Not required'} |")
full = res["Wrist + External"]["masking"]
for key, lab in [("masking", "inference masking"), ("finetuned", "fine-tuned")]:
    w, e = res["Wrist Only"][key], res["External Only"][key]
    if full and w:
        lines.append(f"\nΔ_ext ({lab}) = SR(full) − SR(wrist only) = {full['sr'] - w['sr']:+.1f} pts")
    if full and e:
        lines.append(f"Δ_wrist ({lab}) = SR(full) − SR(external only) = {full['sr'] - e['sr']:+.1f} pts")
acts = OUT / "actions.json"
if acts.exists():
    lines += ["", "| Condition | Action RMSE to demo (rad) | Change vs full M1 (rad) |", "|---|---|---|"]
    for k, v in json.loads(acts.read_text()).items():
        lines.append(f"| {k} | {v['rmse_to_demo_rad']:.4f} | {v['rmse_vs_reference_rad']:.4f} |")
text = "\n".join(lines)
(OUT / "summary.md").write_text(text + "\n")
(OUT / "summary.json").write_text(json.dumps(res, indent=1))
print(text)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(ROWS))
for j, (key, lab, c) in enumerate([("masking", "inference masking (M1 checkpoint)", "#7f7f7f"), ("finetuned", "fine-tuned with that configuration", "#1f77b4")]):
    v = [res[n][key]["sr"] if res[n][key] else np.nan for n, _, _ in ROWS]
    e = [res[n][key]["std_over_seeds"] if res[n][key] and res[n][key]["std_over_seeds"] else 0 for n, _, _ in ROWS]
    b = ax.bar(x + (j - 0.5) * 0.38, v, 0.38, yerr=e, capsize=3, color=c, label=lab)
    for xi, vi in zip(x + (j - 0.5) * 0.38, v):
        if not np.isnan(vi):
            ax.text(xi, vi + 2, f"{vi:.0f}", ha="center", fontsize=8)
ax.set_xticks(x, [n for n, _, _ in ROWS])
ax.set_ylabel("success rate (%)  potato2v, 3 seeds x 50")
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
ax.set_title("Does pi0.5 need the external camera?")
fig.tight_layout()
fig.savefig(OUT / "success_rates.png", dpi=130)
