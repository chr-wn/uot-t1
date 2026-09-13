"""E3.1b — synthetic→natural probe transfer: probes trained on P1b (REAL-BASE+NAMED, neutral+revealing)
at a (position, layer) applied without retraining to the naturalistic mentions cache; plus within-natural
LOO-lexeme probes and the control task."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import from_jsonl
from uot.cache import load_cache
from uot.probes import ALPHAS, nearest_point, control_labels
from uot.dims import Dimension

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3-4b-base"); ap.add_argument("--sites", default="unit:16,unit:28,mention_end:12,last:36")
args = ap.parse_args(); root = Path(__file__).resolve().parents[2]
Cs = load_cache(root / "cache" / args.model / "P1b_s0.npz"); S = from_jsonl(root / "data/phase1/P1b_s0.jsonl")
Cn = load_cache(root / "cache" / args.model / "natural_s0.npz"); N = from_jsonl(root / "data/phase3/natural_s0.jsonl")
Vs = np.array([it.vector for it in S]); Vn = np.array([it.vector for it in N])
ssel = np.array([it.condition in ("REAL-BASE", "REAL-NAMED") and it.family in ("neutral", "revealing") for it in S]) & (np.abs(Vs[:, 3:]).sum(1) == 0)
Ys = Vs[ssel][:, :3]; pts = np.unique(Ys, axis=0); lab = {tuple(p): i for i, p in enumerate(pts)}
nsel = np.array([tuple(v[:3]) in lab and np.abs(v[3:]).sum() == 0 for v in Vn]); Yn = Vn[nsel][:, :3]; lexn = np.array([it.unit_id for it, s in zip(N, nsel) if s])
print("synthetic train", ssel.sum(), "| natural items in the 12-point set", nsel.sum(), "of", len(N), "| natural lexemes", len(set(lexn)))
res = {}
for site in args.sites.split(","):
    pname, l = site.split(":"); l = int(l)
    Xs = Cs["resid"][ssel, Cs["position_names"].index(pname), Cs["layers"].index(l)].astype(np.float32); Xn = Cn["resid"][nsel, Cn["position_names"].index(pname), Cn["layers"].index(l)].astype(np.float32)
    mu = Xs.mean(0); pca = PCA(n_components=256, random_state=0).fit(Xs - mu); sc = StandardScaler().fit(pca.transform(Xs - mu))
    Zs, Zn = sc.transform(pca.transform(Xs - mu)), sc.transform(pca.transform(Xn - mu))
    ys = np.array([lab[tuple(y)] for y in Ys]); yn = np.array([lab[tuple(y)] for y in Yn])
    cat = LogisticRegression(C=0.5, max_iter=3000).fit(Zs, ys); cat_acc = float((cat.predict(Zn) == yn).mean())
    lin = RidgeCV(alphas=ALPHAS).fit(Zs, Ys); lin_acc = float((pts[nearest_point(lin.predict(Zn), pts)] == Yn).all(1).mean())
    # within-natural LOO-lexeme (categorical) and control
    mun = Xn.mean(0); pcan = PCA(n_components=min(256, len(Xn) - 1), random_state=0).fit(Xn - mun); Zn2 = pcan.transform(Xn - mun)
    Yc = control_labels(list(lexn), Yn, 0); yc = np.array([lab.get(tuple(y), -1) for y in Yc])
    pred = np.zeros_like(yn); predc = np.zeros_like(yn)
    for lx in sorted(set(lexn)):
        te = lexn == lx; tr = ~te
        if len(set(yn[tr])) < 2: continue
        s2 = StandardScaler().fit(Zn2[tr]); pred[te] = LogisticRegression(C=0.5, max_iter=2000).fit(s2.transform(Zn2[tr]), yn[tr]).predict(s2.transform(Zn2[te]))
        if len(set(yc[tr])) >= 2: predc[te] = LogisticRegression(C=0.5, max_iter=2000).fit(s2.transform(Zn2[tr]), yc[tr]).predict(s2.transform(Zn2[te]))
    within = float((pred == yn).mean()); ctrl = float((predc == yc).mean()); maj = float(max(np.mean(yn == c) for c in set(yn)))
    res[site] = dict(transfer_categorical=cat_acc, transfer_linear_nearest=lin_acc, within_natural_loo_categorical=within, control=ctrl, majority=maj, n=int(nsel.sum()))
    print(f"{site:16s} synthetic→natural: categorical={cat_acc:.3f} linear-nearest={lin_acc:.3f} | within-natural LOO-lexeme categorical={within:.3f} control={ctrl:.3f} majority={maj:.3f}", flush=True)
out = root / "runs/E3.1"; out.mkdir(parents=True, exist_ok=True); json.dump(res, open(out / f"natural_transfer_{args.model}.json", "w"), indent=1)
