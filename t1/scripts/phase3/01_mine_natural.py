"""E3.1a — mine naturalistic quantity mentions from Wikipedia (streaming sample): sentences containing
'<number> <unit lexeme>' where the lexeme is a single-word registry unit (long singular/plural or symbol
in a whitelist), label the dimension, record spans; write data/phase3/natural_s0.jsonl in the mention
schema (positions: value, unit, mention_end, last)."""
from __future__ import annotations
import json, re, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.units import get_registry
from uot.tasks.base import VEC_BASIS
root = Path(__file__).resolve().parents[2]; out = root / "data/phase3"; out.mkdir(parents=True, exist_ok=True)
reg = get_registry()
forms = {}
for u in reg.units(invented=False):
    if u.dim.distance() == 0: continue
    for f in (u.long_sg, u.long_pl):
        if " " not in f and f.isalpha(): forms[f] = u
    for sym in u.short:
        if sym in ("km", "kg", "cm", "mm", "mg", "ml", "mL", "kW", "MW", "GHz", "MHz", "kHz", "Hz", "kPa", "MPa", "hPa", "mph", "kph", "psi", "lb", "oz", "ft", "mi", "ha", "kWh", "kJ", "MJ", "GW"): forms[sym] = u
pat = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s+(" + "|".join(sorted(map(re.escape, forms), key=len, reverse=True)) + r")(?=[\s.,;:)!?]|$)")
from datasets import load_dataset
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
rng = random.Random(0); items = []; seen_dims = {}
for k, ex in enumerate(ds):
    if k > 60000 or len(items) >= 3000: break
    text = ex["text"]
    for m in pat.finditer(text):
        u = forms[m.group(2)]
        # sentence window: from previous sentence end to the next sentence end, plus one more sentence
        s0 = max(text.rfind(". ", 0, m.start()) + 2, max(text.rfind("\n", 0, m.start()) + 1, 0))
        e1 = text.find(". ", m.end()); e1 = len(text) if e1 < 0 else e1 + 1
        e2 = text.find(". ", e1 + 1); e2 = min(len(text), e1 + 200) if e2 < 0 else min(e2 + 1, e1 + 300)
        prompt = text[s0:e2].replace("\n", " ").strip()
        if not (40 <= len(prompt) <= 400) or prompt.count("\n") > 0: continue
        off = m.start() - s0; v0, v1 = off, off + len(m.group(1)); u0, u1 = off + m.end(2) - m.start() - len(m.group(2)), off + (m.end() - m.start())
        if prompt[u0:u1] != m.group(2): continue
        d = u.dim; key = str(d)
        if seen_dims.get(key, 0) >= 300: continue
        seen_dims[key] = seen_dims.get(key, 0) + 1
        items.append(dict(item_id=f"NAT-{len(items):05d}", condition="NATURAL", family="natural", template_id="wiki", prompt=prompt,
                          spans=dict(value=(v0, v1), unit=(u0, u1), mention_end=(u1, min(u1 + 1, len(prompt))), anaphor=(u1, min(u1 + 1, len(prompt))), last=(len(prompt) - 1, len(prompt))),
                          unit_id=u.id, unit_string=m.group(2), dimension=key, vector=list(d.vector(VEC_BASIS)), lattice_distance=d.distance(), named_point=d.name(),
                          style="natural", lang="en", value=float(m.group(1).replace(",", "")), system=u.system, invented=False, meta=dict(article=ex.get("title", ""))))
with open(out / "natural_s0.jsonl", "w") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(items), "items;", "dims:", seen_dims)
