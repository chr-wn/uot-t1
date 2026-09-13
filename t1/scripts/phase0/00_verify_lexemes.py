"""Build the corpus-verified invented-lexeme pool: generate candidates (wordlist-filtered), query
Dolma-1.7 counts for ' lexeme' and ' lexemes', keep those with count <= --max-count.
Writes data/lexemes/verified.json (+ counts for every candidate)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.units import generate_lexemes, pluralize  # noqa: E402
from uot.frequency import FrequencyCache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--max-count", type=int, default=50)
ap.add_argument("--seed", type=int, default=12345)
args = ap.parse_args()
root = Path(__file__).resolve().parents[2]
fc = FrequencyCache(root / "data" / "lexemes" / "lexeme_counts.json")
cands = generate_lexemes(args.n, random.Random(args.seed))
keep = []
for i, w in enumerate(cands):
    c1, c2 = fc.count(w), fc.count(pluralize(w))
    if c1 is not None and c2 is not None and max(c1, c2) <= args.max_count:
        keep.append(w)
    if i % 100 == 0:
        fc.save(); print(i, w, c1, c2, "kept", len(keep), flush=True)
fc.save()
json.dump(dict(max_count=args.max_count, index=fc.index, n_candidates=len(cands), lexemes=keep),
          open(root / "data" / "lexemes" / "verified.json", "w"), indent=0)
print("verified lexemes:", len(keep), "of", len(cands))
