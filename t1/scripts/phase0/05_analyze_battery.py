"""E0.5 — aggregate battery results: accuracy tables with bootstrap CIs, order-swap penalty,
accuracy vs lattice distance and vs corpus frequency; figures under figures/phase0/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.frequency import FrequencyCache  # noqa: E402
from uot.tasks import items_from_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="runs/E0.3")
ap.add_argument("--stimuli-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "phase0"))
ap.add_argument("--freq-cache", default=str(Path(__file__).resolve().parents[2] / "data" / "frequency" / "counts.json"))
ap.add_argument("--out", default=None)
ap.add_argument("--metric", default="correct_mean")
args = ap.parse_args()
runs = Path(args.runs)
out = Path(args.out or runs / "analysis")
out.mkdir(parents=True, exist_ok=True)

rows = []
for mdir in sorted(runs.iterdir()):
    if not mdir.is_dir() or mdir.name == "analysis":
        continue
    for rf in sorted(mdir.glob("*.results.jsonl")):
        for line in open(rf):
            r = json.loads(line)
            r["model"] = mdir.name
            r["stimfile"] = rf.name.replace(".results.jsonl", "")
            rows.append(r)
df = pd.DataFrame(rows)
if df.empty:
    sys.exit("no results")
df["order_swapped"] = df["meta"].apply(lambda m: m.get("order_swapped", None))
df["style"] = df["meta"].apply(lambda m: m.get("style"))
df["invented"] = df["meta"].apply(lambda m: m.get("invented"))
df["formula_given"] = df["meta"].apply(lambda m: m.get("formula_given"))
df["relation"] = df["meta"].apply(lambda m: m.get("relation"))
df["l1"] = df["meta"].apply(lambda m: m.get("l1"))

# frequency of the answer unit string (T1 only) — use stimulus files for unit_strings
fc = FrequencyCache(args.freq_cache) if Path(args.freq_cache).exists() else None
ustr = {}
for sp in Path(args.stimuli_dir).glob("*.jsonl"):
    for it in items_from_jsonl(str(sp)):
        ustr[it.item_id] = it.unit_strings
# recompute generation correctness from stored generations with the current matcher
from uot.lm.scoring import generation_matches  # noqa: E402
from uot.parse_units import generation_dimension_correct  # noqa: E402
prompts = {}
for sp in Path(args.stimuli_dir).glob("*.jsonl"):
    for it in items_from_jsonl(str(sp)):
        prompts[it.item_id] = it.prompt
if "generation" in df:
    df["correct_gen"] = [generation_matches(g, ustr.get(i, [])) if isinstance(g, str) else np.nan
                         for g, i in zip(df["generation"], df["item_id"])]
    # dimension-level generation correctness: parsed dimension == answer dimension (None -> False)
    df["correct_gen_dim"] = [(generation_dimension_correct(g, prompts.get(i, ""), ad) is True) if isinstance(g, str) else np.nan
                             for g, i, ad in zip(df["generation"], df["item_id"], df["answer_dimension"])]
    df["gen_parsed"] = [(generation_dimension_correct(g, prompts.get(i, ""), ad) is not None) if isinstance(g, str) else np.nan
                        for g, i, ad in zip(df["generation"], df["item_id"], df["answer_dimension"])]
if fc is not None:
    def lf(iid):
        ss = [s for s in ustr.get(iid, []) if s.strip() in fc.counts]
        return math.log10(1 + max(fc.counts[s.strip()] for s in ss)) if ss else np.nan
    df["log_freq"] = df["item_id"].apply(lf)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def cluster_boot_ci(sub, key, col, n=2000, seed=0):
    """Bootstrap over clusters (templates) then items within."""
    groups = [g[col].to_numpy(dtype=float) for _, g in sub.groupby(key)]
    if not groups:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(n):
        gs = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        ms.append(np.concatenate([rng.choice(g, len(g)) for g in gs]).mean())
    return (float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)))


metric = args.metric


def margin_yes(r):
    """log-prob margin of candidate 0 minus candidate 1 (two-way tasks)."""
    s = r["scores"]
    return s[0]["mean_logprob"] - s[1]["mean_logprob"]


def calibrated_metrics(g):
    """For two-way tasks: AUROC of the margin vs label and accuracy after subtracting the median margin
    (removes a constant yes/no bias; Zhao et al. 2021 / Feng & Steinhardt 2023)."""
    from sklearn.metrics import roc_auc_score
    m = np.array([margin_yes(r) for _, r in g.iterrows()])
    y = (g["answer_index"].to_numpy() == 0).astype(int)
    out = {}
    if len(set(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, m))
        thr = np.median(m)
        out["acc_calibrated"] = float(((m > thr).astype(int) == y).mean())
        out["yes_rate"] = float((m > 0).mean())
    return out


tab = []
for (model, task, cond), g in df.groupby(["model", "task", "condition"]):
    lo, hi = cluster_boot_ci(g, "template_id", metric)
    chance = 1 / len(g.iloc[0]["candidate_roles"])
    rec = dict(model=model, task=task, condition=cond, n=len(g), acc=g[metric].mean(), ci_lo=lo, ci_hi=hi, chance=chance)
    if len(g.iloc[0]["candidate_roles"]) == 2:
        rec.update(calibrated_metrics(g))
    if "correct_gen" in g and g["correct_gen"].notna().any():
        rec["acc_gen"] = g["correct_gen"].mean()
        rec["acc_gen_dim"] = g["correct_gen_dim"].mean()
        rec["gen_parsed"] = g["gen_parsed"].mean()
        gp = g[g["gen_parsed"] == True]
        rec["acc_gen_dim|parsed"] = gp["correct_gen_dim"].mean() if len(gp) else np.nan
    if task == "T1":
        a = g[g.order_swapped == False][metric].mean() if (g.order_swapped == False).any() else np.nan
        b = g[g.order_swapped == True][metric].mean() if (g.order_swapped == True).any() else np.nan
        rec.update(acc_inorder=a, acc_swapped=b, swap_penalty=a - b)
        sw = g[g.order_swapped == True]
        if len(sw):
            rec["swapped_ci_lo"], rec["swapped_ci_hi"] = cluster_boot_ci(sw, "template_id", metric)
    tab.append(rec)
tab = pd.DataFrame(tab)
tab.to_csv(out / "accuracy_by_condition.csv", index=False)
pd.set_option("display.width", 200)
print(tab.round(3).to_string())

# per-template spread (prompt brittleness)
tpl = df.groupby(["model", "task", "condition", "template_id"])[metric].mean().reset_index()
spread = tpl.groupby(["model", "task", "condition"])[metric].agg(["min", "max"]).reset_index()
spread["range"] = spread["max"] - spread["min"]
spread.to_csv(out / "template_spread.csv", index=False)

# distance / frequency (T1)
t1 = df[df.task == "T1"].copy()
if len(t1):
    from scipy.stats import spearmanr
    recs = []
    for (model, cond), g in t1.groupby(["model", "condition"]):
        r = dict(model=model, condition=cond)
        if g["lattice_distance"].nunique() > 1:
            rho, p = spearmanr(g["lattice_distance"], g[metric])
            r.update(rho_distance=rho, p_distance=p)
        if "log_freq" in g and g["log_freq"].notna().sum() > 10 and g["log_freq"].nunique() > 1:
            rho, p = spearmanr(g["log_freq"], g[metric], nan_policy="omit")
            r.update(rho_logfreq=rho, p_logfreq=p)
        recs.append(r)
    pd.DataFrame(recs).to_csv(out / "distance_frequency_correlations.csv", index=False)
    print(pd.DataFrame(recs).round(3).to_string())
    t1.groupby(["model", "condition", "lattice_distance"])[metric].agg(["mean", "count"]).to_csv(out / "acc_by_distance.csv")
    t1.groupby(["model", "condition", "relation"])[metric].agg(["mean", "count"]).to_csv(out / "acc_by_relation.csv")
    t1.groupby(["model", "condition", "style"])[metric].agg(["mean", "count"]).to_csv(out / "acc_by_style.csv")
    # candidate-role confusion: which distractor wins when wrong
    def chosen_role(r):
        return r["candidate_roles"][r["pred_mean"]]
    t1["chosen_role"] = t1.apply(chosen_role, axis=1)
    t1.groupby(["model", "condition", "chosen_role"]).size().unstack(fill_value=0).to_csv(out / "chosen_roles.csv")
    print(t1.groupby(["model", "condition", "chosen_role"]).size().unstack(fill_value=0))
print("wrote", out)
