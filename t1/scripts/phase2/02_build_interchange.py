"""E2.0b — build interchange examples for DAS.

Each example = (base prompt, source prompt) where the intervened variable is quantity 1's dimension
(M_alg) or quantity 1's unit lexeme (M_heur).  Labels:
  y_alg  = 1[D1(src) == D2(base)]
  y_heur = 1[compat(U1(src), U2(base)) > median]   (model's own pair table for the base's operation)
Subsets flagged per example: 'agree' (labels equal), 'disagree', 'same_dim_swap' (D1(src)==D1(base),
U1(src)!=U1(base)), 'heldout_lex' (U1(src) in a held-out lexeme list), plus invented-unit twins.
Writes data/phase2/<model>_interchange_s<seed>.jsonl (train/eval split by base item)."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p2_interchange import generate, to_jsonl, LEXEMES

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-base", type=int, default=1200); ap.add_argument("--n-inv", type=int, default=400)
ap.add_argument("--heldout-lex", default="mile,ounce,week,dyne,erg,horsepower,torr,becquerel,barrel,kph")
args = ap.parse_args()
root = Path(__file__).resolve().parents[2]
pt = json.load(open(root / "runs/E2.0" / args.model / "pair_table.json")); table, med = pt["table"], pt["median"]
rng = random.Random(args.seed)
real = generate(args.n_base, args.seed, styles=("long",)); inv = generate(args.n_inv, args.seed + 1, invented=True)
heldout = set(args.heldout_lex.split(","))
out = root / "data/phase2"; out.mkdir(parents=True, exist_ok=True)
to_jsonl(real, out / f"P2_real_s{args.seed}.jsonl"); to_jsonl(inv, out / f"P2_inv_s{args.seed}.jsonl")


def pair_examples(items, invented):
    ex = []
    by_op = {}
    for it in items: by_op.setdefault(it.op, []).append(it)
    for b in items:
        pool = [s for s in by_op[b.op] if s.item_id != b.item_id]
        rng.shuffle(pool)
        # choose sources to balance label-changing / preserving under M_alg
        want_yes = [s for s in pool if (s.d1 == b.d2)]; want_no = [s for s in pool if (s.d1 != b.d2)]
        picks = (want_yes[:2] + want_no[:2]) if (want_yes and want_no) else pool[:4]
        for s in picks:
            y_alg = int(s.d1 == b.d2)
            if invented:
                y_heur = None  # no lexical table entry for invented lexemes
            else:
                y_heur = int(table[b.op][s.u1][b.u2] > med)
            ex.append(dict(base=b.item_id, src=s.item_id, y_alg=y_alg, y_heur=y_heur, y_base=int(b.same_dim),
                           same_dim_swap=int(s.d1 == b.d1 and s.u1 != b.u1), heldout_lex=int(s.u1 in heldout or b.u1 in heldout),
                           agree=(None if y_heur is None else int(y_alg == y_heur)), op=b.op))
    return ex


ex_real = pair_examples(real, False); ex_inv = pair_examples(inv, True)
bases = sorted({e["base"] for e in ex_real}); rng.shuffle(bases); n_tr = int(0.8 * len(bases)); tr_b = set(bases[:n_tr])
for e in ex_real: e["split"] = "train" if e["base"] in tr_b else "eval"
for e in ex_inv: e["split"] = "eval"
json.dump(dict(real=ex_real, inv=ex_inv), open(out / f"{args.model}_interchange_s{args.seed}.json", "w"))
import collections
c = collections.Counter((e["split"], e["y_alg"], e.get("agree")) for e in ex_real)
print("real examples", len(ex_real), dict(c)); print("invented examples", len(ex_inv), "| same_dim_swap", sum(e["same_dim_swap"] for e in ex_real), "| heldout_lex", sum(e["heldout_lex"] for e in ex_real), "| disagree", sum(1 for e in ex_real if e["agree"] == 0))
