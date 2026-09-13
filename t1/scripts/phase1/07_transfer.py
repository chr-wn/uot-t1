"""E1.2b — cross-condition / cross-family probe transfer at chosen sites.
Train a ridge exponent probe on items of condition/family A, test on B (no retraining)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import from_jsonl
from uot.cache import load_cache
from uot.probes import regression_metrics, ALPHAS

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--stimuli", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--sites", default="unit:16,mention_end:12,anaphor:12,last:36"); ap.add_argument("--pca", type=int, default=512)
args = ap.parse_args()
C = load_cache(args.cache); items = from_jsonl(args.stimuli)
V = np.array([it.vector for it in items]); inlat = (np.abs(V[:, 3:]).sum(1) == 0)
cond = np.array([it.condition for it in items]); fam = np.array([it.family for it in items])
def m(conds, fams): return np.isin(cond, conds) & np.isin(fam, fams) & inlat
PAIRS = {
    "LATTICE->NAMED": (m(["REAL-LATTICE"], ["neutral", "revealing"]), m(["REAL-NAMED"], ["neutral", "revealing"])),
    "NAMED+BASE->LATTICE": (m(["REAL-NAMED", "REAL-BASE"], ["neutral", "revealing"]), m(["REAL-LATTICE"], ["neutral", "revealing"])),
    "REAL->INV-LEX": (m(["REAL-BASE", "REAL-NAMED", "REAL-LATTICE"], ["neutral", "revealing"]), m(["INV-LEX"], ["neutral", "revealing"])),
    "REAL->XLING": (m(["REAL-BASE", "REAL-NAMED", "REAL-LATTICE"], ["neutral", "revealing"]), m(["XLING"], ["neutral", "revealing"])),
    "neutral->revealing": (m(["REAL-BASE", "REAL-NAMED", "REAL-LATTICE"], ["neutral"]), m(["REAL-BASE", "REAL-NAMED", "REAL-LATTICE"], ["revealing"])),
    "revealing->neutral": (m(["REAL-BASE", "REAL-NAMED", "REAL-LATTICE"], ["revealing"]), m(["REAL-BASE", "REAL-NAMED", "REAL-LATTICE"], ["neutral"])),
    "with-unit->noun_only": (m(["REAL-BASE", "REAL-NAMED", "REAL-LATTICE"], ["neutral", "revealing"]), m(["REAL-BASE", "REAL-NAMED", "REAL-LATTICE"], ["noun_only"])),
    "BASE->BASE(random)": (None, None),
}
out = Path(args.out); out.mkdir(parents=True, exist_ok=True); results = {}
for site in args.sites.split(","):
    pname, l = site.split(":"); l = int(l); pj, lj = C["position_names"].index(pname), C["layers"].index(l)
    Xall = C["resid"][:, pj, lj].astype(np.float32); mu = Xall[inlat].mean(0)
    pca = PCA(n_components=args.pca, random_state=0).fit(Xall[inlat] - mu); X = pca.transform(Xall - mu)
    res = {}
    for name, (tr, te) in PAIRS.items():
        if tr is None:
            continue
        sc = StandardScaler().fit(X[tr]); Ytr, Yte = V[tr][:, :3], V[te][:, :3]
        p = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), Ytr).predict(sc.transform(X[te]))
        pts = np.unique(Ytr, axis=0)
        r = regression_metrics(p, Yte, pts); r["n_train"], r["n_test"] = int(tr.sum()), int(te.sum())
        # base-dimension 3-way accuracy for test items that are base points
        base = (np.abs(Yte).sum(1) == 1)
        if base.any():
            r["base_3way_acc"] = float((np.rint(p[base]).argmax(1) == Yte[base].argmax(1)).mean()) if (Yte[base].max(1) == 1).all() else None
        res[name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "r2_per_axis"}
        print(f"{site:16s} {name:22s} n={r['n_train']}/{r['n_test']} r2={r['r2_mean']:.2f} axis={r['axis_acc']:.2f} nearest={r['nearest_acc']:.2f} exact={r['exact_acc']:.2f}", flush=True)
    results[site] = res
json.dump(results, open(out / f"transfer_{Path(args.cache).stem}.json", "w"), indent=1)
print("wrote", out)
