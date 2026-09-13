"""E2.0a — the model's own pairwise compatibility table: yes-minus-no log-prob margin for every ordered
pair of lexemes in the Phase-2 lexeme set, for each operation (2 templates each), long style.
Output: runs/E2.0/<model>/pair_table.json  {op: {u1: {u2: mean margin}}} plus the calibrated median."""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p2_interchange import LEXEMES, OPS, _render
from uot.units import get_registry
from uot.lm.models import load_model
from uot.lm.scoring import score_candidates

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", default="runs/E2.0"); ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()
root = Path(__file__).resolve().parents[2]; reg = get_registry()
lex = [u for us in LEXEMES.values() for u in us]
spec, tok, model = load_model(args.model)
prompts, keys = [], []
for op, temps in OPS.items():
    for ti, text in enumerate(temps):
        for a, b in itertools.product(lex, lex):
            ua, ub = reg[a], reg[b]
            q1, q2 = _render(ua, 25, "long"), _render(ub, 40, "long")
            prompts.append(text.format(q1=q1, q2=q2, u2=ub.long_pl) + " Answer yes or no.\nAnswer:"); keys.append((op, ti, a, b))
print(len(prompts), "prompts", flush=True)
sc = score_candidates(model, tok, prompts, [[" yes", " no"]] * len(prompts), batch_size=args.batch)
table = {}
for (op, ti, a, b), s in zip(keys, sc):
    table.setdefault(op, {}).setdefault(a, {}).setdefault(b, []).append(s[0]["mean_logprob"] - s[1]["mean_logprob"])
out = root / args.out / args.model; out.mkdir(parents=True, exist_ok=True)
res = {op: {a: {b: float(np.mean(v)) for b, v in d.items()} for a, d in t.items()} for op, t in table.items()}
allm = [m for t in res.values() for d in t.values() for m in d.values()]
json.dump(dict(table=res, median=float(np.median(allm)), lexemes=lex), open(out / "pair_table.json", "w"), indent=0)
# quick summary: agreement of thresholded table with dimension equality
same = np.array([reg[a].dim == reg[b].dim for op in res for a in res[op] for b in res[op][a]]); marg = np.array([res[op][a][b] for op in res for a in res[op] for b in res[op][a]])
thr = np.median(marg); print("median margin", thr, "| agreement(thresholded table, same-dimension):", float(((marg > thr) == same).mean()), "| AUROC-ish: mean margin same", marg[same].mean(), "diff", marg[~same].mean())
