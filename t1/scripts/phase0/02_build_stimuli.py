"""Build the Phase-0 stimulus sets (all tasks) to data/phase0/<set>.jsonl, deterministic by seed."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks import GENERATORS, items_to_jsonl  # noqa: E402
from uot.tasks.t1_unit_cloze import make_familiar_twins  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=240, help="items per (task, condition)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
ap.add_argument("--tasks", default="T1,T2,T3,T4,T5")
ap.add_argument("--suffix", default="", help="filename suffix, e.g. v2 -> T1v2_s0.jsonl")
ap.add_argument("--conditions", default=None, help="comma list (T1 only)")
ap.add_argument("--integer-output", action="store_true")
args = ap.parse_args()

out = Path(args.out or (Path(__file__).resolve().parents[2] / "data" / "phase0"))
out.mkdir(parents=True, exist_ok=True)
for t in args.tasks.split(","):
    kw = {"conditions": tuple(args.conditions.split(","))} if (args.conditions and t == "T1") else {}
    if t == "T1" and args.integer_output:
        kw["integer_output"] = True
    items = GENERATORS[t](args.n, seed=args.seed, **kw)
    if t == "T1":
        items = items + make_familiar_twins(items, seed=args.seed)
    p = out / f"{t}{args.suffix}_s{args.seed}.jsonl"
    items_to_jsonl(items, str(p))
    c = Counter(it.condition for it in items)
    print(t, len(items), dict(c), "->", p)
