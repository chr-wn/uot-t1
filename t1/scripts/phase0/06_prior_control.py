"""E0.3b — prior-only control: score the same candidate sets with the scenario removed.

Variants written to data/phase0/controls/<stim>.<variant>.jsonl:
  nocontext : definitions prefix (if any) + "The unit is" — no relation, no numbers.
  nounits   : the original prompt with every input quantity's unit deleted (numbers kept) —
              the relation is stated but the lexemes are absent, so composition is impossible;
              measures how much the answer can be guessed from the candidate strings + relation.
Then run 03_run_battery.py on them like any stimulus file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks import items_from_jsonl, items_to_jsonl  # noqa: E402
from uot.tasks.t1_unit_cloze import RELATIONS, INV_BASE_RELATIONS  # noqa: E402
from uot.quantity import fmt_value  # noqa: E402
REL_BY_NAME = {r.name: r for r in RELATIONS + INV_BASE_RELATIONS}

ap = argparse.ArgumentParser()
ap.add_argument("--stimuli", nargs="+", required=True)
args = ap.parse_args()
for sp in args.stimuli:
    items = items_from_jsonl(sp)
    out = Path(sp).parent / "controls"
    out.mkdir(exist_ok=True)
    noctx, nounits = [], []
    for it in items:
        m = re.match(r"^((?:[^.]*? is a unit of [^.]*\.\s*)+|(?:[^.]*? is a basic physical quantity[^.]*\.\s*))", it.prompt)
        prefix = m.group(1).strip() + " " if m else ""
        a = json.loads(json.dumps(it.__dict__))
        a["prompt"] = (prefix + "The unit is").strip()
        a["item_id"] = it.item_id + "#noctx"
        noctx.append(a)
        b = json.loads(json.dumps(it.__dict__))
        rel = REL_BY_NAME[it.meta["relation"]]
        text = next(t[1] for t in rel.templates if t[0] == it.template_id)
        rendered = {f"q{k}": fmt_value(v) for k, v in enumerate(it.meta["values"])}
        body = text.format(v=fmt_value(it.meta["out_value"]), name=it.meta.get("x_quantity") or "mass", **rendered)
        b["prompt"] = (prefix + body).strip()
        b["item_id"] = it.item_id + "#nounits"
        nounits.append(b)
    from uot.tasks.base import Item
    items_to_jsonl([Item(**a) for a in noctx], str(out / (Path(sp).stem + ".noctx.jsonl")))
    items_to_jsonl([Item(**b) for b in nounits], str(out / (Path(sp).stem + ".nounits.jsonl")))
    print(sp, "->", out, "| example nounits:", nounits[3]["prompt"][:160])
