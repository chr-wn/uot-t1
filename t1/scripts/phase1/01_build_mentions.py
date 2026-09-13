"""Build Phase-1 probing stimuli (single-quantity mentions with spans) to data/phase1/P1_s<seed>.jsonl."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import generate, to_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=400, help="items per condition (before noun_only skips)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()
out = Path(args.out or (Path(__file__).resolve().parents[2] / "data" / "phase1"))
out.mkdir(parents=True, exist_ok=True)
items = generate(args.n, seed=args.seed)
p = out / f"P1_s{args.seed}.jsonl"
to_jsonl(items, str(p))
print(len(items), dict(Counter((it.condition, it.family) for it in items)), "->", p)
print("distinct lattice points:", len({it.dimension for it in items}))
