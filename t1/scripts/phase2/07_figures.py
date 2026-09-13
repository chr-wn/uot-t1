"""Phase-2 figures: IIA vs rank at u1 (alg/heur/random/probe/ceiling), per-model bars, controls."""
import glob, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
root = Path(__file__).resolve().parents[2]; out = root / "figures/phase2"; out.mkdir(parents=True, exist_ok=True)
rows = []
for f in glob.glob(str(root / "runs/E2.1/*/das_*.json")):
    d = json.load(open(f)); tag = Path(f).stem
    rows.append(dict(model=d["model"], layer=d["layer"], pos=d["position"], var=d["variable"], mode=d["mode"], rank=d["rank"], seed=int(tag[-1]) if tag[-3:-1] == "_s" else 0,
                     all=d.get("iia_all", {}).get("balanced_iia", np.nan), disagree=d.get("iia_disagree", {}).get("balanced_iia", np.nan),
                     heldout=d.get("iia_heldout_lex", {}).get("balanced_iia", np.nan), invented=d.get("iia_invented_alg", {}).get("balanced_iia", np.nan)))
df = pd.DataFrame(rows)
# Fig A: IIA vs rank at u1 L8 and L4 for Qwen3-4B
fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
for ax, L in zip(axes, (4, 8)):
    q = df[(df.model == "qwen3-4b-base") & (df.pos == "u1") & (df.layer == L)]
    for var, c in (("alg", "C0"), ("heur", "C1")):
        g = q[(q["var"] == var) & (q["mode"] == "learned")].groupby("rank")["all"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["rank"], g["mean"], yerr=g["std"].fillna(0), marker="o", color=c, label=f"learned subspace, M_{var} labels")
        pr = q[(q["var"] == var) & (q["mode"] == "probe")]
        if len(pr): ax.scatter(pr["rank"], pr["all"], marker="s", color=c, facecolors="none", label=f"probe-derived, M_{var}")
        full = q[(q["var"] == var) & (q["mode"] == "full")]["all"]
        if len(full): ax.axhline(float(full.iloc[0]), color=c, ls=":", lw=1, label=f"full-residual ceiling, M_{var}")
    r = q[q["mode"] == "random"].groupby("rank")["all"].mean().reset_index(); ax.plot(r["rank"], r["all"], "k--", marker="x", label="matched-rank random")
    ax.set_xscale("log", base=2); ax.set_xlabel("subspace rank"); ax.set_title(f"unit-1 token, layer {L}"); ax.set_ylim(0.45, 0.9); ax.axhline(0.5, c="0.7", lw=0.5)
axes[0].set_ylabel("balanced IIA"); axes[1].legend(fontsize=6, loc="upper left")
fig.suptitle("Interchange interventions on quantity 1's unit representation (Qwen3-4B-Base)", fontsize=9); fig.tight_layout()
fig.savefig(out / "figP2a_iia_vs_rank.png", dpi=160); fig.savefig(out / "figP2a_iia_vs_rank.pdf")
# Fig B: per-model at u1 L8 rank 64 (+ceilings)
fig, ax = plt.subplots(figsize=(6.5, 3.3)); models = [m for m in ["qwen3-4b-base", "olmo3-7b", "gemma2-9b"] if m in set(df.model)]
w = 0.13
for i, m in enumerate(models):
    q = df[(df.model == m) & (df.pos == "u1") & (df.layer == 8)]
    vals = [q[(q["var"] == "alg") & (q["mode"] == "learned") & (q["rank"] == 64)]["all"].mean(), q[(q["var"] == "heur") & (q["mode"] == "learned") & (q["rank"] == 64)]["all"].mean(),
            q[(q["mode"] == "random") & (q["rank"] == 64)]["all"].mean(), q[(q["var"] == "alg") & (q["mode"] == "full")]["all"].mean(), q[(q["var"] == "heur") & (q["mode"] == "full")]["all"].mean(),
            q[(q["var"] == "alg") & (q["mode"] == "learned") & (q["rank"] == 64)]["invented"].mean()]
    labels = ["M_alg rank 64", "M_heur rank 64", "random rank 64", "ceiling (alg labels)", "ceiling (heur labels)", "invented units (alg)"]
    for j, (v, lab) in enumerate(zip(vals, labels)): ax.bar(i + (j - 2.5) * w, v, w, color=f"C{j}", label=lab if i == 0 else None)
ax.set_xticks(range(len(models))); ax.set_xticklabels(models); ax.axhline(0.5, c="0.6", lw=0.7); ax.set_ylim(0.4, 0.95); ax.set_ylabel("balanced IIA (u1, layer 8)"); ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(out / "figP2b_models.png", dpi=160); fig.savefig(out / "figP2b_models.pdf")
df.to_csv(out / "das_all.csv", index=False); print("wrote", out)
