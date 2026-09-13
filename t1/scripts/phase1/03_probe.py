"""Run the probe suite (task/control × random/cross-lexeme/lattice-holdout) per layer × position.

Usage: 03_probe.py --cache cache/qwen3-4b-base/P1_s0.npz --stimuli data/phase1/P1_s0.jsonl --out runs/E1.1
Filters: --conditions REAL-BASE,REAL-NAMED,REAL-LATTICE  --families neutral,revealing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import from_jsonl  # noqa: E402
from uot.cache import load_cache  # noqa: E402
from uot.probes import run_probe_suite  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--stimuli", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--conditions", default="REAL-BASE,REAL-NAMED,REAL-LATTICE")
ap.add_argument("--families", default="neutral,revealing")
ap.add_argument("--positions", default=None)
ap.add_argument("--layers", default=None)
ap.add_argument("--alpha", type=float, default=None, help="ridge alpha; default = CV-selected")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="main")
ap.add_argument("--axes", default="0,1,2", help="indices into the exponent vector to predict (L,M,T,...)")
ap.add_argument("--pca", type=int, default=512, help="project residuals to this many PCs (fit on the selected items) before probing; 0 = none")
args = ap.parse_args()

C = load_cache(args.cache)
items = from_jsonl(args.stimuli)
assert len(items) == C["resid"].shape[0]
conds = set(args.conditions.split(","))
fams = set(args.families.split(","))
axes = [int(a) for a in args.axes.split(",")]
V = np.array([it.vector for it in items])
other = [i for i in range(V.shape[1]) if i not in axes]
# keep only items whose dimension lies in the sub-lattice spanned by the selected axes
sel = np.array([it.condition in conds and it.family in fams for it in items]) & (np.abs(V[:, other]).sum(1) == 0)
Y = V[sel][:, axes]
units = [it.unit_id or "none" for it in items]
units = [u for u, s in zip(units, sel) if s]
positions = args.positions.split(",") if args.positions else C["position_names"]
layers = [int(x) for x in args.layers.split(",")] if args.layers else C["layers"]
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rows = []
for pname in positions:
    pj = C["position_names"].index(pname)
    for l in layers:
        lj = C["layers"].index(l)
        X = C["resid"][sel, pj, lj].astype(np.float32)
        if args.pca and args.pca < X.shape[1]:
            from sklearn.decomposition import PCA
            X = PCA(n_components=args.pca, random_state=0).fit_transform(X - X.mean(0))
        r = run_probe_suite(X, Y, units, seed=args.seed, alpha=args.alpha)
        row = dict(position=pname, layer=l, n=int(sel.sum()), tag=args.tag)
        for k, v in r.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk != "r2_per_axis":
                        row[f"{k}/{kk}"] = vv
                row[f"{k}/r2_axes"] = json.dumps(v.get("r2_per_axis"))
            else:
                row[k] = v
        rows.append(row)
        print(f"{pname:12s} L{l:3d} lex: nearest={r['cross_lexeme/task']['nearest_acc']:.3f} axis={r['cross_lexeme/task']['axis_acc']:.3f} r2={r['cross_lexeme/task']['r2_mean']:.3f} ctrl_axis={r['cross_lexeme/control']['axis_acc']:.3f} | "
              f"lattice-holdout: nearest={r['lattice_holdout/task']['nearest_acc']:.3f} axis={r['lattice_holdout/task']['axis_acc']:.3f} r2={r['lattice_holdout/task']['r2_mean']:.3f}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(out / f"probes_{Path(args.cache).stem}_{args.tag}.csv", index=False)
print("wrote", out)
