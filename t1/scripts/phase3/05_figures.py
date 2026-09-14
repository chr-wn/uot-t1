"""Phase-3 figures: natural transfer per site; detector AUROCs (both labelings)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
root = Path(__file__).resolve().parents[2]; out = root / "figures/phase3"; out.mkdir(parents=True, exist_ok=True)
nt = json.load(open(root / "runs/E3.1/natural_transfer_qwen3-4b-base.json"))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
sites = list(nt); x = np.arange(len(sites)); w = 0.2
for j, (k, lab) in enumerate((("transfer_categorical", "synthetic→natural, categorical"), ("transfer_linear_nearest", "synthetic→natural, linear"), ("within_natural_loo_categorical", "within-natural LOO-lexeme"), ("control", "control task"))):
    axes[0].bar(x + (j - 1.5) * w, [nt[s][k] for s in sites], w, label=lab)
axes[0].axhline(nt[sites[0]]["majority"], ls="--", c="k", lw=0.7); axes[0].set_xticks(x); axes[0].set_xticklabels(sites, fontsize=7); axes[0].set_ylabel("12-way accuracy"); axes[0].legend(fontsize=6); axes[0].set_title("E3.1 naturalistic mentions (Qwen3-4B)", fontsize=9)
det = {}
for f in root.glob("runs/E3.2/detector_*.json"): det = json.load(open(f))
if det:
    conds = list(det); keys = [k for k in ("A_probe", "A_selfcheck", "A_pint", "B_probe", "B_selfcheck") if all(k in det[c] for c in conds)]
    for j, k in enumerate(keys):
        axes[1].bar(np.arange(len(conds)) + (j - (len(keys) - 1) / 2) * 0.15, [det[c][k] for c in conds], 0.15, label=k.replace("A_", "clean-vs-corrupted: ").replace("B_", "parser truth: "))
    axes[1].set_xticks(range(len(conds))); axes[1].set_xticklabels(conds); axes[1].set_ylim(0.4, 1.02); axes[1].axhline(0.5, c="0.6", lw=0.7); axes[1].set_ylabel("AUROC"); axes[1].legend(fontsize=6); axes[1].set_title("E3.2 dimension-inconsistency detector", fontsize=9)
fig.tight_layout(); fig.savefig(out / "figP3.png", dpi=160); fig.savefig(out / "figP3.pdf"); print("wrote", out)
