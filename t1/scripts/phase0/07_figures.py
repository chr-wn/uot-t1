"""E0.5 figures: accuracy by condition per model (T1..T5), order-swap, prior controls, accuracy vs
lattice distance and vs Dolma-1.7 log-frequency. Reads runs/E0.3/analysis/*.csv and E0.3b results.
Writes figures/phase0/*.pdf|png and a markdown snippet with the tables."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs/E0.3")
ap.add_argument("--controls", default="runs/E0.3b")
ap.add_argument("--out", default="figures/phase0")
args = ap.parse_args()
root = Path(__file__).resolve().parents[2]
runs, ctrl, out = root / args.runs, root / args.controls, root / args.out
out.mkdir(parents=True, exist_ok=True)
tab = pd.read_csv(runs / "analysis" / "accuracy_by_condition.csv")
MODEL_ORDER = ["qwen3-4b-base", "qwen3-8b-base", "qwen3-14b-base", "olmo3-7b", "olmo3-32b", "gemma2-9b", "qwen3-4b", "qwen3-8b"]
models = [m for m in MODEL_ORDER if m in set(tab.model)]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# --- Fig 1: T1 accuracy by condition, one bar group per model
t1 = tab[tab.task == "T1"]
conds = ["FAM-NAMED", "FAM-UNNAMED", "INV-LEX", "INV-LEX-TWIN", "INV-BASE", "INV-BASE-TWIN"]
fig, ax = plt.subplots(figsize=(9, 3.2))
w = 0.8 / max(len(models), 1)
for i, m in enumerate(models):
    sub = t1[t1.model == m].set_index("condition").reindex(conds)
    x = np.arange(len(conds)) + i * w
    ax.bar(x, sub["acc"], w, yerr=[sub["acc"] - sub["ci_lo"], sub["ci_hi"] - sub["acc"]], label=m, capsize=2)
ax.axhline(0.25, ls="--", c="k", lw=0.8)
ax.set_xticks(np.arange(len(conds)) + w * (len(models) - 1) / 2); ax.set_xticklabels(conds, rotation=20)
ax.set_ylabel("T1 unit-cloze accuracy (candidate readout)"); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, ncol=2)
ax.set_title("Compositional unit-of-answer: familiar vs invented units (chance 0.25)")
fig.tight_layout(); fig.savefig(out / "fig1_T1_accuracy_by_condition.pdf"); fig.savefig(out / "fig1_T1_accuracy_by_condition.png", dpi=160)

# --- Fig 2: in-order vs order-swapped
fig, ax = plt.subplots(figsize=(6, 3))
for i, m in enumerate(models):
    sub = t1[t1.model == m].set_index("condition").reindex(conds)
    ax.scatter(sub["acc_inorder"], sub["acc_swapped"], label=m, s=22)
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("accuracy, in-order items"); ax.set_ylabel("accuracy, order-swapped items")
ax.set_xlim(0.5, 1.02); ax.set_ylim(0.5, 1.02); ax.legend(fontsize=7); ax.set_title("No template-copy penalty")
fig.tight_layout(); fig.savefig(out / "fig2_order_swap.pdf"); fig.savefig(out / "fig2_order_swap.png", dpi=160)

# --- Fig 3: prior controls (full vs no-units vs no-context) per model
rows = []
for mdir in sorted(ctrl.glob("*")):
    if not mdir.is_dir():
        continue
    for rf in mdir.glob("T1_s*.*.results.jsonl"):
        variant = rf.name.split(".")[1]
        for line in open(rf):
            r = json.loads(line)
            rows.append(dict(model=mdir.name, variant=variant, condition=r["condition"], correct=r["correct_mean"]))
if rows:
    cdf = pd.DataFrame(rows).groupby(["model", "variant", "condition"])["correct"].mean().reset_index()
    full = t1[["model", "condition", "acc"]].rename(columns={"acc": "correct"}); full["variant"] = "full"
    cdf = pd.concat([cdf, full[["model", "variant", "condition", "correct"]]])
    cdf.to_csv(out / "prior_controls.csv", index=False)
    cm = [m for m in models if m in set(cdf[cdf.variant != "full"].model)]
    fig, axes = plt.subplots(1, len(cm), figsize=(2.6 * len(cm), 3), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], cm):
        sub = cdf[cdf.model == m].pivot(index="condition", columns="variant", values="correct").reindex(conds)
        sub = sub.reindex(columns=["noctx", "nounits", "full"])
        sub.plot.bar(ax=ax, width=0.8, legend=(ax is axes[0][0]))
        ax.axhline(0.25, ls="--", c="k", lw=0.8); ax.set_title(m); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=60, labelsize=7)
    axes[0][0].set_ylabel("T1 accuracy")
    fig.suptitle("Prior-only controls: no context / relation without units / full prompt", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig3_prior_controls.pdf"); fig.savefig(out / "fig3_prior_controls.png", dpi=160)

# --- Fig 4: accuracy vs lattice distance and vs log-frequency (T1)
bd = pd.read_csv(runs / "analysis" / "acc_by_distance.csv")
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for m in models:
    for cond, ls in (("FAM-NAMED", "-"), ("FAM-UNNAMED", "--"), ("INV-LEX", ":")):
        sub = bd[(bd.model == m) & (bd.condition == cond)]
        if len(sub):
            axes[0].plot(sub["lattice_distance"], sub["mean"], ls, marker="o", ms=3, label=f"{m} {cond}")
axes[0].set_xlabel("L1 lattice distance of answer dimension"); axes[0].set_ylabel("T1 accuracy"); axes[0].set_ylim(0, 1.02)
axes[0].legend(fontsize=5, ncol=2)
# frequency bins from raw results
fr = []
for mdir in runs.glob("*"):
    if not mdir.is_dir() or mdir.name == "analysis":
        continue
    for rf in mdir.glob("T1_s*.results.jsonl"):
        for line in open(rf):
            r = json.loads(line)
            if r["condition"].startswith("FAM"):
                fr.append(dict(model=mdir.name, item_id=r["item_id"], correct=r["correct_mean"]))
try:
    from uot.frequency import FrequencyCache
except ImportError:
    sys.path.insert(0, str(root / "src")); from uot.frequency import FrequencyCache
from uot.tasks import items_from_jsonl
fc = FrequencyCache(root / "data" / "frequency" / "counts.json")
ustr = {}
for sp in (root / "data" / "phase0").glob("T1_s*.jsonl"):
    for it in items_from_jsonl(str(sp)):
        ustr[it.item_id] = it.unit_strings
if fr:
    fdf = pd.DataFrame(fr)
    fdf["logf"] = fdf["item_id"].map(lambda i: fc.log_freq(ustr.get(i, [""])) if ustr.get(i) else np.nan)
    fdf["bin"] = pd.cut(fdf["logf"], [-0.1, 0.5, 1.5, 2.5, 3.5, 4.5, 9])
    g = fdf.groupby(["model", "bin"], observed=True)["correct"].agg(["mean", "count"]).reset_index()
    g.to_csv(out / "acc_by_logfreq_bin.csv", index=False)
    for m in models:
        sub = g[g.model == m]
        axes[1].plot(range(len(sub)), sub["mean"], marker="o", ms=3, label=m)
        axes[1].set_xticks(range(len(sub))); axes[1].set_xticklabels([str(b) for b in sub["bin"]], rotation=30, fontsize=6)
    axes[1].set_xlabel("log10(1 + Dolma-1.7 count of answer unit string), FAM items"); axes[1].set_ylim(0, 1.02); axes[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(out / "fig4_distance_frequency.pdf"); fig.savefig(out / "fig4_distance_frequency.png", dpi=160)

# --- Fig 5: T2-T5 accuracy by condition
other = tab[tab.task != "T1"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, task in zip(axes, ["T2", "T3", "T4", "T5"]):
    sub = other[other.task == task]
    cs = list(dict.fromkeys(sub.condition))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        s = sub[sub.model == m].set_index("condition").reindex(cs)
        x = np.arange(len(cs)) + i * w
        ax.bar(x, s["acc"], w, yerr=[s["acc"] - s["ci_lo"], s["ci_hi"] - s["acc"]], label=m, capsize=1.5)
    ax.axhline(sub["chance"].iloc[0] if len(sub) else 0.5, ls="--", c="k", lw=0.8)
    ax.set_xticks(np.arange(len(cs)) + w * (len(models) - 1) / 2); ax.set_xticklabels(cs, rotation=45, fontsize=6)
    ax.set_title({"T2": "T2 consistency (yes/no)", "T3": "T3 formula selection", "T4": "T4 conversion (yes/no)", "T5": "T5 error spotting"}[task])
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("accuracy"); axes[-1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(out / "fig5_T2_T5.pdf"); fig.savefig(out / "fig5_T2_T5.png", dpi=160)

# --- markdown tables
md = ["## Accuracy by condition (candidate readout; bootstrap 95% CI over templates × items)\n",
      tab[["model", "task", "condition", "n", "acc", "ci_lo", "ci_hi", "chance", "acc_gen_dim|parsed", "acc_inorder", "acc_swapped"]].round(3).to_markdown(index=False)]
open(out / "tables.md", "w").write("\n".join(md))
print("wrote", out, list(p.name for p in out.glob("*.png")))
