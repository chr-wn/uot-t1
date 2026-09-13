"""E1.2c — clean cross-lexeme test on the semantic set: leave-one-lexeme-out over base + named units,
evaluated only on lexemes whose lattice point has >= 2 lexemes (so the point stays in training via
other lexemes). Reports per-point accuracy and overall nearest-point accuracy + control task."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import from_jsonl
from uot.cache import load_cache
from uot.probes import ALPHAS, nearest_point, control_labels

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--stimuli", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--sites", default="unit:16,unit:28,mention_end:12,anaphor:12,last:36"); ap.add_argument("--pca", type=int, default=256)
ap.add_argument("--families", default="neutral,revealing")
args = ap.parse_args()
C = load_cache(args.cache); items = from_jsonl(args.stimuli)
V = np.array([it.vector for it in items]); inlat = (np.abs(V[:, 3:]).sum(1) == 0)
fams = set(args.families.split(","))
sel = np.array([it.condition in ("REAL-BASE", "REAL-NAMED") and it.family in fams for it in items]) & inlat
Y = V[sel][:, :3]; lex = np.array([it.unit_id for it, s in zip(items, sel) if s]); pts = np.unique(Y, axis=0)
key = [tuple(y) for y in Y]; lex_by_pt = defaultdict(set)
for k, l in zip(key, lex): lex_by_pt[k].add(l)
evaluable = np.array([len(lex_by_pt[k]) >= 2 for k in key])
print("items", sel.sum(), "points", len(pts), "lexemes", len(set(lex)), "evaluable items (point has >=2 lexemes)", evaluable.sum(), "points with >=2 lexemes", sum(1 for k, v in lex_by_pt.items() if len(v) >= 2))
out = Path(args.out); out.mkdir(parents=True, exist_ok=True); results = {}
Yc = control_labels(list(lex), Y, 0)
for site in args.sites.split(","):
    pname, l = site.split(":"); l = int(l); pj, lj = C["position_names"].index(pname), C["layers"].index(l)
    Xall = C["resid"][sel, pj, lj].astype(np.float32); mu = Xall.mean(0)
    X = PCA(n_components=args.pca, random_state=0).fit_transform(Xall - mu)
    pred = np.zeros_like(Y, dtype=float); predc = np.zeros_like(Y, dtype=float); cat_pred = np.zeros(len(Y), dtype=int)
    labels = {tuple(y): i for i, y in enumerate(pts)}; yc = np.array([labels[tuple(y)] for y in Y])
    for lx in sorted(set(lex)):
        te = lex == lx
        if not evaluable[te].any():
            continue
        tr = ~te; sc = StandardScaler().fit(X[tr])
        pred[te] = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), Y[tr]).predict(sc.transform(X[te]))
        predc[te] = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), Yc[tr]).predict(sc.transform(X[te]))
        cat_pred[te] = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[tr]), yc[tr]).predict(sc.transform(X[te]))
    e = evaluable
    near = (pts[nearest_point(pred[e], pts)] == Y[e]).all(1); nearc = (pts[nearest_point(predc[e], pts)] == Yc[e]).all(1)
    cat_acc = (cat_pred[e] == yc[e])
    per_point = {}
    for k in sorted(set(key[i] for i in np.where(e)[0])):
        m = np.array([key[i] == k for i in np.where(e)[0]])
        per_point[str(k)] = dict(n=int(m.sum()), linear_nearest=round(float(near[m].mean()), 2), categorical=round(float(cat_acc[m].mean()), 2))
    res = dict(linear_nearest=round(float(near.mean()), 3), control_nearest=round(float(nearc.mean()), 3), categorical=round(float(cat_acc.mean()), 3),
               axis_acc=round(float((np.rint(pred[e]) == Y[e]).mean()), 3), per_point=per_point)
    results[site] = res
    print(f"{site:16s} LOO-lexeme: linear nearest={res['linear_nearest']:.2f} control={res['control_nearest']:.2f} categorical={res['categorical']:.2f} axis={res['axis_acc']:.2f}", flush=True)
json.dump(results, open(out / f"named_loo_{Path(args.cache).parent.name}_{Path(args.cache).stem}.json", "w"), indent=1)
worst = sorted(results[args.sites.split(",")[0]]["per_point"].items(), key=lambda kv: kv[1]["linear_nearest"])[:6]
print("hardest points (first site):", worst)
