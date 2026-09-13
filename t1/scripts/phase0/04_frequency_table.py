"""E0.4 — query infini-gram (Dolma 1.7) for every unit string used in the Phase-0 stimuli."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks import items_from_jsonl  # noqa: E402
from uot.frequency import FrequencyCache  # noqa: E402
from uot.units import get_registry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stimuli", nargs="+", required=True)
ap.add_argument("--cache", default=str(Path(__file__).resolve().parents[2] / "data" / "frequency" / "counts.json"))
args = ap.parse_args()

fc = FrequencyCache(args.cache)
strings: list[str] = []
for u in get_registry().units(invented=False):
    strings += [u.symbol, u.long_sg, u.long_pl]
for sp in args.stimuli:
    for it in items_from_jsonl(sp):
        strings += it.unit_strings
strings = list(dict.fromkeys(s.strip() for s in strings if s.strip()))
print(len(strings), "unique strings")
for i, s in enumerate(strings):
    fc.count(s)
    if i % 50 == 0:
        fc.save(); print(i, s, fc.counts[s], flush=True)
fc.save()
inv = [s for s in strings if s.startswith("inv:")]
print("done; cached", len(fc.counts))
