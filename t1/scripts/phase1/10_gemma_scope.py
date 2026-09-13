"""E1.6 — Gemma Scope reconnaissance: which residual-SAE latents (layer 20, width 16k) track dimension?
Encodes cached Gemma-2-9B residuals at a position with the JumpReLU SAE, then for each lattice point
finds latents with high mean activation and high selectivity (mean on point / mean elsewhere), and
tests whether the top-k latents' activations linearly decode the exponent vector (cross-lexeme LOO)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import from_jsonl
from uot.cache import load_cache

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache/gemma2-9b/P1b_s0.npz"); ap.add_argument("--stimuli", default="data/phase1/P1b_s0.jsonl")
ap.add_argument("--layer", type=int, default=20); ap.add_argument("--width", default="16k"); ap.add_argument("--position", default="unit")
ap.add_argument("--out", default="runs/E1.6")
args = ap.parse_args()
root = Path(__file__).resolve().parents[2]
# pick the canonical l0 folder for this layer/width
from huggingface_hub import list_repo_files
files = [f for f in list_repo_files("google/gemma-scope-9b-pt-res") if f.startswith(f"layer_{args.layer}/width_{args.width}/") and f.endswith("params.npz")]
l0s = sorted(files, key=lambda f: int(f.split("average_l0_")[1].split("/")[0]))
chosen = l0s[len(l0s) // 2]  # median sparsity
print("SAE:", chosen)
pth = hf_hub_download("google/gemma-scope-9b-pt-res", chosen)
P = np.load(pth); W_enc, b_enc, thr = P["W_enc"], P["b_enc"], P["threshold"]
C = load_cache(root / args.cache); items = from_jsonl(root / args.stimuli)
lj = C["layers"].index(args.layer); pj = C["position_names"].index(args.position)
V = np.array([it.vector for it in items]); inlat = (np.abs(V[:, 3:]).sum(1) == 0)
sel = np.array([it.condition in ("REAL-BASE", "REAL-NAMED") and it.family in ("neutral", "revealing") for it in items]) & inlat
X = C["resid"][sel, pj, lj].astype(np.float32); Y = V[sel][:, :3]; lex = np.array([it.unit_id for it, s in zip(items, sel) if s])
pre = X @ W_enc + b_enc; A = np.where(pre > thr, pre, 0.0)  # JumpReLU
print("items", len(X), "active latents/item (mean)", float((A > 0).sum(1).mean()))
pts = np.unique(Y, axis=0); report = {}
for p in pts:
    m = (Y == p).all(1)
    if m.sum() < 20: continue
    mean_in, mean_out = A[m].mean(0), A[~m].mean(0) + 1e-6
    score = mean_in / mean_out; frac_in = (A[m] > 0).mean(0)
    top = np.argsort(-(score * (frac_in > 0.3)))[:5]
    report[str(tuple(int(x) for x in p))] = [dict(latent=int(t), frac_active_in=round(float(frac_in[t]), 2), frac_active_out=round(float((A[~m] > 0).mean(0)[t]), 3), ratio=round(float(score[t]), 1)) for t in top]
    print(tuple(int(x) for x in p), "n", int(m.sum()), report[str(tuple(int(x) for x in p))][:3])
# linear decode of exponents from the union of top latents (LOO lexeme)
from sklearn.linear_model import RidgeCV
from uot.probes import ALPHAS, nearest_point
lat = sorted({d["latent"] for v in report.values() for d in v})
Z = A[:, lat]; pred = np.zeros_like(Y, dtype=float)
for lx in sorted(set(lex)):
    te = lex == lx
    pred[te] = RidgeCV(alphas=ALPHAS).fit(Z[~te], Y[~te]).predict(Z[te])
acc = float((pts[nearest_point(pred, pts)] == Y).all(1).mean())
print(f"LOO-lexeme nearest-point accuracy from {len(lat)} selected latents: {acc:.3f}")
out = root / args.out; out.mkdir(parents=True, exist_ok=True)
json.dump(dict(sae=chosen, n_latents_selected=len(lat), loo_nearest=acc, per_point=report), open(out / f"gemma_scope_L{args.layer}_{args.position}.json", "w"), indent=1)
