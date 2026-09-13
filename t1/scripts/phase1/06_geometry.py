"""E1.3 — parameterisation comparison, additivity, dimensionless-at-origin, named-vs-unnamed
extrapolation, at chosen (position, layer) sites of one model's mention cache.

Parameterisations on a random 80/20 split (and cross-expression folds):
  linear   : ridge -> exponent vector (L,M,T)
  signmag  : ridge -> [sign(e), |e|] per axis, recombined
  categorical : multinomial logistic over lattice points (in-distribution only)
  mlp      : 1-hidden-layer (256) regressor -> exponent vector
Additivity: centroids c(d) per lattice point (PCA-512 space); residual
  ‖c(d1·d2) − c(d1) − c(d2) + c(1)‖ / ‖c(d1·d2) − c(1)‖ over all (d1,d2) with d1·d2 in the set,
  compared with a random-triple baseline (random d1', d2' with the same target).
Dimensionless: DIMLESS items projected by the linear probe -> mean |predicted exponent| vs base points.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import from_jsonl
from uot.cache import load_cache
from uot.probes import group_folds, lattice_holdout_folds, regression_metrics, ALPHAS

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--stimuli", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--sites", default="unit:16,mention_end:12,anaphor:12,last:36")
ap.add_argument("--pca", type=int, default=512); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
C = load_cache(args.cache); items = from_jsonl(args.stimuli)
V = np.array([it.vector for it in items])
inlat = (np.abs(V[:, 3:]).sum(1) == 0)
main = np.array([it.condition in ("REAL-BASE", "REAL-NAMED", "REAL-LATTICE") and it.family in ("neutral", "revealing") for it in items]) & inlat
dimless = np.array([it.condition == "DIMLESS" and it.family in ("neutral", "revealing") for it in items])
Y = V[main][:, :3]; units = [it.unit_id for it, s in zip(items, main) if s]
named = np.array([it.named_point is not None for it, s in zip(items, main) if s])
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
results = {}
for site in args.sites.split(","):
    pname, l = site.split(":"); l = int(l)
    pj, lj = C["position_names"].index(pname), C["layers"].index(l)
    Xall = C["resid"][:, pj, lj].astype(np.float32)
    pca = PCA(n_components=args.pca, random_state=0).fit(Xall[main] - Xall[main].mean(0))
    mu = Xall[main].mean(0)
    X = pca.transform(Xall[main] - mu); Xd = pca.transform(Xall[dimless] - mu)
    n = len(X); perm = rng.permutation(n); tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    res = {}
    # linear
    lin = RidgeCV(alphas=ALPHAS).fit(Xtr, Y[tr]); res["linear"] = regression_metrics(lin.predict(Xte), Y[te], np.unique(Y, axis=0))
    # sign/magnitude
    T = np.concatenate([np.sign(Y), np.abs(Y)], 1)
    sm = RidgeCV(alphas=ALPHAS).fit(Xtr, T[tr]); P = sm.predict(Xte)
    pred = np.sign(np.rint(P[:, :3])) * np.rint(np.clip(P[:, 3:], 0, 3))
    res["signmag"] = regression_metrics(pred, Y[te], np.unique(Y, axis=0))
    # categorical
    labels = {tuple(y): i for i, y in enumerate(np.unique(Y, axis=0))}; yc = np.array([labels[tuple(y)] for y in Y])
    cat = LogisticRegression(C=0.5, max_iter=3000).fit(Xtr, yc[tr]); res["categorical"] = dict(exact_acc=float((cat.predict(Xte) == yc[te]).mean()))
    # mlp
    mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=600, random_state=0, early_stopping=True).fit(Xtr, Y[tr])
    res["mlp"] = regression_metrics(mlp.predict(Xte), Y[te], np.unique(Y, axis=0))
    # lattice holdout: linear vs mlp vs signmag; named vs unnamed held-out points
    lh = {"linear": [], "mlp": [], "signmag": []}; named_acc = {"named": [], "unnamed": []}; r2_named = {"named": [], "unnamed": []}
    for trf, tef in lattice_holdout_folds(Y, 5, args.seed):
        sc2 = StandardScaler().fit(X[trf]); a, b = sc2.transform(X[trf]), sc2.transform(X[tef])
        p_lin = RidgeCV(alphas=ALPHAS).fit(a, Y[trf]).predict(b); lh["linear"].append(regression_metrics(p_lin, Y[tef], np.unique(Y, axis=0)))
        p_mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=600, random_state=0, early_stopping=True).fit(a, Y[trf]).predict(b)
        lh["mlp"].append(regression_metrics(p_mlp, Y[tef], np.unique(Y, axis=0)))
        Pm = RidgeCV(alphas=ALPHAS).fit(a, T[trf]).predict(b); p_sm = np.sign(np.rint(Pm[:, :3])) * np.rint(np.clip(Pm[:, 3:], 0, 3))
        lh["signmag"].append(regression_metrics(p_sm, Y[tef], np.unique(Y, axis=0)))
        for key, mask in (("named", named[tef]), ("unnamed", ~named[tef])):
            if mask.sum() > 5:
                named_acc[key].append(float((np.rint(p_lin[mask]) == Y[tef][mask]).all(1).mean()))
                r2_named[key].append(regression_metrics(p_lin[mask], Y[tef][mask])["r2_mean"])
    res["lattice_holdout"] = {k: {m: float(np.mean([d[m] for d in v])) for m in ("exact_acc", "axis_acc", "r2_mean", "sign_acc", "nearest_acc")} for k, v in lh.items()}
    res["lattice_holdout_named_vs_unnamed"] = {k: dict(exact_acc=float(np.mean(v)) if v else None, r2=float(np.mean(r2_named[k])) if r2_named[k] else None) for k, v in named_acc.items()}
    # additivity of centroids
    pts = {tuple(y): X[(Y == y).all(1)].mean(0) for y in np.unique(Y, axis=0)}
    origin = Xd.mean(0) if len(Xd) else np.zeros(X.shape[1])
    ratios, base = [], []
    keys = list(pts)
    for d1 in keys:
        for d2 in keys:
            d3 = tuple(np.array(d1) + np.array(d2))
            if d3 in pts and d1 != d3 and d2 != d3:
                num = np.linalg.norm(pts[d3] - pts[d1] - pts[d2] + origin); den = np.linalg.norm(pts[d3] - origin) + 1e-9
                ratios.append(num / den)
                r1, r2 = keys[rng.integers(len(keys))], keys[rng.integers(len(keys))]
                base.append(np.linalg.norm(pts[d3] - pts[r1] - pts[r2] + origin) / den)
    res["additivity"] = dict(residual_ratio=float(np.mean(ratios)), random_triple_ratio=float(np.mean(base)), n_triples=len(ratios))
    # dimensionless at origin
    if len(Xd):
        pd_ = lin.predict(sc.transform(Xd)); res["dimless"] = dict(mean_abs_pred=float(np.abs(pd_).mean()), frac_nearest_origin=float((np.abs(pd_).sum(1) < 0.5).mean()),
                                                                   base_mean_abs=float(np.abs(lin.predict(Xte)[np.abs(Y[te]).sum(1) == 1]).mean()))
    results[site] = res
    print(site, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'r2_per_axis'}) for k, v in res.items()}, default=str)[:900], flush=True)
json.dump(results, open(out / f"geometry_{Path(args.cache).stem}.json", "w"), indent=1, default=str)
print("wrote", out)
