"""E0.0 — tokenization audit: how do the suite tokenizers split unit strings and invented lexemes?

Writes runs/E0.0/tokenization_audit.json and prints a summary table.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.units import get_registry, generate_lexemes, UnitExpr  # noqa: E402
from uot.lm.models import MODEL_SPECS  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "runs" / "E0.0"
OUT.mkdir(parents=True, exist_ok=True)

reg = get_registry()
rng = random.Random(0)
lexemes = generate_lexemes(300, rng)
strings = {}
for u in reg.units(invented=False):
    strings[f"real:{u.id}:short"] = " " + u.symbol
    strings[f"real:{u.id}:long_pl"] = " " + u.long_pl
m, s, kg = reg["meter"], reg["second"], reg["kilogram"]
for st in ("symbol", "ascii", "slash", "per", "long"):
    strings[f"compound:{st}"] = " " + UnitExpr.of((kg, 1), (m, 1), (s, -2)).render(st)
for w in lexemes[:100]:
    strings[f"inv:{w}"] = " " + w
    strings[f"inv:{w}:pl"] = " " + w + "s"

report = {}
keys = [k for k in sys.argv[1:]] or ["qwen3-4b-base", "olmo3-7b", "gemma2-9b"]
for key in keys:
    spec = MODEL_SPECS[key]
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    res = {}
    for name, sv in strings.items():
        ids = tok(sv, add_special_tokens=False)["input_ids"]
        res[name] = dict(n=len(ids), toks=tok.convert_ids_to_tokens(ids))
    report[key] = res
    inv = [v["n"] for k, v in res.items() if k.startswith("inv:") and not k.endswith(":pl")]
    short = [v["n"] for k, v in res.items() if k.endswith(":short")]
    longp = [v["n"] for k, v in res.items() if k.endswith(":long_pl")]
    print(f"{key}: invented lexeme tokens {Counter(inv)}; real short {Counter(short)}; real long-pl {Counter(longp)}")
    for st in ("symbol", "ascii", "slash", "per", "long"):
        print(f"   compound[{st}] -> {res['compound:' + st]['toks']}")
json.dump(report, open(OUT / "tokenization_audit.json", "w"), ensure_ascii=False, indent=1)
print("wrote", OUT / "tokenization_audit.json")
