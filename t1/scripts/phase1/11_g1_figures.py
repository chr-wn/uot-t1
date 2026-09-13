"""G1 figures: (a) LOO-lexeme decodability per model/site (linear vs categorical vs control);
(b) in-distribution vs held-out-lattice nearest accuracy per model (semantic set P1b)."""
import glob, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
root = Path(__file__).resolve().parents[2]; out = root / "figures/phase1"; out.mkdir(parents=True, exist_ok=True)
models = ["qwen3-4b-base", "qwen3-8b-base", "qwen3-14b-base", "olmo3-7b", "gemma2-9b"]
sites = ["unit:28", "mention_end:12", "anaphor:12", "last:-1"]
loo = {}
for m in models:
    for s in (0, 1):
        f = root / f"runs/E1.2c/named_loo_{m}_P1b_s{s}.json"
        if f.exists(): loo[(m, s)] = json.load(open(f))
fig, axes = plt.subplots(1, len(sites), figsize=(3.2 * len(sites), 3.2), sharey=True)
w = 0.25
for ax, site in zip(axes, sites):
    for i, m in enumerate(models):
        vals = [loo[(m, s)][site] for s in (0, 1) if (m, s) in loo]
        if not vals: continue
        lin = np.mean([v["linear_nearest"] for v in vals]); cat = np.mean([v["categorical"] for v in vals]); ctl = np.mean([v["control_nearest"] for v in vals])
        ax.bar(i - w, lin, w, color="C0", label="linear (exponent) probe" if i == 0 else None)
        ax.bar(i, cat, w, color="C1", label="categorical probe" if i == 0 else None)
        ax.bar(i + w, ctl, w, color="0.6", label="control task" if i == 0 else None)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([m.replace("-base", "") for m in models], rotation=30, fontsize=7)
    ax.set_title(f"position {site.split(':')[0]} (layer {site.split(':')[1]})", fontsize=8); ax.axhline(1 / 12, ls="--", c="k", lw=0.7)
axes[0].set_ylabel("leave-one-lexeme-out nearest-point accuracy (12 points)"); axes[0].legend(fontsize=6)
fig.suptitle("H1: dimension of a mentioned quantity decodes across unseen lexemes and at the referent", fontsize=9)
fig.tight_layout(); fig.savefig(out / "figG1a_loo_lexeme.png", dpi=160); fig.savefig(out / "figG1a_loo_lexeme.pdf")
# (b) in-distribution vs held-out lattice
fig, ax = plt.subplots(figsize=(6, 3.2))
for i, m in enumerate(models):
    vals = []
    for s in (0, 1):
        f = root / f"runs/E1.3/geometry_{m}_P1b_s{s}_semb.json"
        if f.exists(): vals.append(json.load(open(f)))
    if not vals: continue
    for j, site in enumerate(("unit:16", "anaphor:12")):
        ind = np.mean([v[site]["linear"]["nearest_acc"] for v in vals]); cat = np.mean([v[site]["categorical"]["exact_acc"] for v in vals]); lh = np.mean([v[site]["lattice_holdout"]["linear"]["nearest_acc"] for v in vals])
        x = i + (j - 0.5) * 0.4
        ax.bar(x - 0.12, ind, 0.12, color="C0", label="in-distribution, linear" if (i, j) == (0, 0) else None)
        ax.bar(x, cat, 0.12, color="C1", label="in-distribution, categorical" if (i, j) == (0, 0) else None)
        ax.bar(x + 0.12, lh, 0.12, color="C3", label="held-out lattice points, linear" if (i, j) == (0, 0) else None)
ax.set_xticks(range(len(models))); ax.set_xticklabels([m.replace("-base", "") for m in models], fontsize=8); ax.axhline(1 / 12, ls="--", c="k", lw=0.7)
ax.set_ylabel("nearest-point accuracy"); ax.set_title("H2: no extrapolation to held-out lattice points (left bars: unit token; right: anaphor)", fontsize=8); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(out / "figG1b_lattice_holdout.png", dpi=160); fig.savefig(out / "figG1b_lattice_holdout.pdf")
print("wrote", out)
