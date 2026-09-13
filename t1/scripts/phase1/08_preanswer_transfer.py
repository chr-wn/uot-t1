"""E1.4 — does a *composed* dimension exist before the answer? Train the exponent probe on mention
caches (REAL, neutral+revealing) at a given layer, apply (no retraining) to T1v2 prompts at the
pre-answer token (`last`) against the *answer* dimension, and at the input-unit tokens against the
*input* dimensions. Also fit a probe directly on T1 items (cross-relation folds) for the ceiling."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import from_jsonl
from uot.tasks import items_from_jsonl
from uot.cache import load_cache
from uot.probes import regression_metrics, group_folds, ALPHAS
from uot.dims import Dimension

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--layers", default="4,8,12,16,20,24,28,36")
ap.add_argument("--t1", default="T1v2_s0"); ap.add_argument("--out", default="runs/E1.4"); ap.add_argument("--pca", type=int, default=512)
args = ap.parse_args()
root = Path(__file__).resolve().parents[2]
Cm = load_cache(root / "cache" / args.model / "P1_s0.npz"); mitems = from_jsonl(root / "data/phase1/P1_s0.jsonl")
Ct = load_cache(root / "cache" / args.model / f"{args.t1}.npz"); titems = {it.item_id: it for it in items_from_jsonl(str(root / "data/phase0" / f"{args.t1}.jsonl"))}
tids = Ct["meta"]["item_ids"]; T = [titems[i] for i in tids]
Vm = np.array([it.vector for it in mitems]); inlat = (np.abs(Vm[:, 3:]).sum(1) == 0)
msel = np.array([it.condition in ("REAL-BASE", "REAL-NAMED", "REAL-LATTICE") and it.family in ("neutral", "revealing") for it in mitems]) & inlat
Ym = Vm[msel][:, :3]
# T1 labels: answer dimension at `last`; slot dims at unit positions; only real-lattice (no X) items
def vec(ds): d = Dimension.parse(ds); return np.array(d.vector(("L", "M", "T")))
tsel = np.array([("X" not in it.answer_dimension) and all("X" not in sd for sd in it.meta["slot_dims"]) for it in T])
Yans = np.array([vec(it.answer_dimension) for it in T]); Yu0 = np.array([vec(it.meta["slot_dims"][0]) for it in T])
Yu1 = np.array([vec(it.meta["slot_dims"][1] if len(it.meta["slot_dims"]) > 1 else it.meta["slot_dims"][0]) for it in T])
conds = np.array([it.condition for it in T]); rels = [it.meta["relation"] for it in T]
out = Path(root / args.out); out.mkdir(parents=True, exist_ok=True); results = {}
for l in [int(x) for x in args.layers.split(",")]:
    lj_m, lj_t = Cm["layers"].index(l), Ct["layers"].index(l)
    res = {}
    for mpos in ("unit", "mention_end", "last"):
        Xm = Cm["resid"][msel, Cm["position_names"].index(mpos), lj_m].astype(np.float32)
        mu = Xm.mean(0); pca = PCA(n_components=args.pca, random_state=0).fit(Xm - mu)
        sc = StandardScaler().fit(pca.transform(Xm - mu)); probe = RidgeCV(alphas=ALPHAS).fit(sc.transform(pca.transform(Xm - mu)), Ym)
        pts = np.unique(Ym, axis=0)
        for tpos, Yt in (("last", Yans), ("u0_unit", Yu0), ("u1_unit", Yu1)):
            Xt = Ct["resid"][:, Ct["position_names"].index(tpos), lj_t].astype(np.float32)
            p = probe.predict(sc.transform(pca.transform(Xt - mu)))
            r = regression_metrics(p[tsel], Yt[tsel], pts)
            # per condition
            byc = {c: round(regression_metrics(p[tsel & (conds == c)], Yt[tsel & (conds == c)], pts)["axis_acc"], 3) for c in sorted(set(conds)) if (tsel & (conds == c)).sum() > 20}
            res[f"mention:{mpos}->T1:{tpos}"] = dict(r2=round(r["r2_mean"], 3), axis=round(r["axis_acc"], 3), nearest=round(r["nearest_acc"], 3), exact=round(r["exact_acc"], 3), by_condition=byc)
            print(f"L{l:2d} mention:{mpos:11s} -> T1:{tpos:8s} r2={r['r2_mean']:.2f} axis={r['axis_acc']:.2f} nearest={r['nearest_acc']:.2f} | {byc}", flush=True)
    # direct probe on T1 `last` with cross-relation folds (ceiling for "composed dimension is there")
    Xt = Ct["resid"][:, Ct["position_names"].index("last"), lj_t].astype(np.float32)[tsel]
    Yt = Yans[tsel]; grp = [r for r, s in zip(rels, tsel) if s]
    mu = Xt.mean(0); pca = PCA(n_components=args.pca, random_state=0).fit(Xt - mu); X = pca.transform(Xt - mu)
    preds = np.zeros_like(Yt, dtype=float)
    for tr, te in group_folds(grp, 5, 0):
        sc = StandardScaler().fit(X[tr]); preds[te] = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), Yt[tr]).predict(sc.transform(X[te]))
    r = regression_metrics(preds, Yt, np.unique(Yt, axis=0))
    res["T1:last cross-relation"] = dict(r2=round(r["r2_mean"], 3), axis=round(r["axis_acc"], 3), nearest=round(r["nearest_acc"], 3), exact=round(r["exact_acc"], 3))
    print(f"L{l:2d} T1:last direct cross-relation probe r2={r['r2_mean']:.2f} axis={r['axis_acc']:.2f} nearest={r['nearest_acc']:.2f}", flush=True)
    results[l] = res
json.dump(results, open(out / f"preanswer_{args.model}_{args.t1}.json", "w"), indent=1)
print("wrote", out)
