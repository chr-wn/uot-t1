"""Phase-1 probe figures: layer × position heatmaps of task / control / selectivity for each split."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", nargs="+", required=True)
ap.add_argument("--out", default="figures/phase1")
args = ap.parse_args()
root = Path(__file__).resolve().parents[2]
out = root / args.out; out.mkdir(parents=True, exist_ok=True)
for c in args.csv:
    df = pd.read_csv(c)
    tag = Path(c).stem
    positions = list(dict.fromkeys(df.position)); layers = sorted(df.layer.unique())
    metrics = [("random/task/nearest_acc", "random split: task"), ("cross_lexeme/task/nearest_acc", "cross-lexeme: task"),
               ("cross_lexeme/control/nearest_acc", "cross-lexeme: control"), ("cross_lexeme/selectivity_nearest", "cross-lexeme: selectivity"),
               ("lattice_holdout/task/nearest_acc", "lattice holdout: nearest"), ("lattice_holdout/task/r2_mean", "lattice holdout: R² (mean over axes)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.4))
    for ax, (m, title) in zip(axes, metrics):
        M = np.array([[df[(df.position == p) & (df.layer == l)][m].mean() if len(df[(df.position == p) & (df.layer == l)]) else np.nan for l in layers] for p in positions])
        im = ax.imshow(M, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(len(positions))); ax.set_yticklabels(positions, fontsize=7)
        ax.set_xticks(range(0, len(layers), max(1, len(layers) // 6))); ax.set_xticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers) // 6))], fontsize=7)
        ax.set_title(title, fontsize=8); ax.set_xlabel("layer", fontsize=7)
        for i in range(M.shape[0]):
            j = int(np.nanargmax(M[i])) if not np.all(np.isnan(M[i])) else 0
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.suptitle(tag, fontsize=9)
    fig.savefig(out / f"probes_{tag}.png", dpi=150, bbox_inches="tight"); fig.savefig(out / f"probes_{tag}.pdf", bbox_inches="tight")
    best = df.loc[df["lattice_holdout/task/nearest_acc"].idxmax()]
    print(tag, "best lattice-holdout:", best.position, "L", best.layer, f"nearest={best['lattice_holdout/task/nearest_acc']:.3f}",
          f"r2={best['lattice_holdout/task/r2_mean']:.3f}", "| best cross-lexeme selectivity:",
          df.loc[df["cross_lexeme/selectivity_nearest"].idxmax()][["position", "layer", "cross_lexeme/selectivity_nearest"]].tolist())
print("wrote", out)
