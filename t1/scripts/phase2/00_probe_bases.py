"""Probe-derived subspaces for the DAS 'probe' control: at (layer, position) fit (a) a ridge exponent
probe (3 directions) and (b) a multinomial logistic probe over lattice points (K directions) on the
P1b semantic set; save orthonormalised row bases as npy for ranks 3 and K (and top-k truncations)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import from_jsonl
from uot.cache import load_cache
from uot.probes import ALPHAS

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--layers", default="12,16,20"); ap.add_argument("--position", default="unit")
args = ap.parse_args(); root = Path(__file__).resolve().parents[2]
C = load_cache(root / "cache" / args.model / "P1b_s0.npz"); items = from_jsonl(root / "data/phase1/P1b_s0.jsonl")
V = np.array([it.vector for it in items]); inlat = (np.abs(V[:, 3:]).sum(1) == 0)
sel = np.array([it.condition in ("REAL-BASE", "REAL-NAMED") and it.family in ("neutral", "revealing") for it in items]) & inlat
Y = V[sel][:, :3]; pts = np.unique(Y, axis=0); lab = {tuple(p): i for i, p in enumerate(pts)}; y = np.array([lab[tuple(v)] for v in Y])
out = root / "runs/E2.0" / args.model / "probe_bases"; out.mkdir(parents=True, exist_ok=True)
for l in [int(x) for x in args.layers.split(",")]:
    X = C["resid"][sel, C["position_names"].index(args.position), C["layers"].index(l)].astype(np.float32)
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    W_r = RidgeCV(alphas=ALPHAS).fit(Xs, Y).coef_ / sc.scale_            # 3 × d in raw space
    W_c = LogisticRegression(C=0.5, max_iter=3000).fit(Xs, y).coef_ / sc.scale_  # K × d
    for name, W in (("ridge3", W_r), (f"logistic{len(pts)}", W_c)):
        q, _ = np.linalg.qr(W.T); B = q.T[: W.shape[0]]
        np.save(out / f"{name}_L{l}_{args.position}.npy", B.astype(np.float32))
    print(f"L{l}: saved ridge3 and logistic{len(pts)} bases (d={X.shape[1]})", flush=True)
