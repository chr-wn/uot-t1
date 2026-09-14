"""E3.2a — model-generated step-by-step solutions for unit-cloze problems (T1int, FAM-NAMED and INV-LEX),
then a corrupted twin per solution where the final stated unit is replaced by a single-token unit of a
different dimension.  Writes data/phase3/cot_<model>.jsonl with the final-unit span for each version."""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks import items_from_jsonl
from uot.lm.models import load_model
from uot.lm.scoring import generate_greedy
from uot.parse_units import parse_unit_expression, invented_units_from_prompt
from uot.dims import Dimension
from uot.units import get_registry

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3-4b-base"); ap.add_argument("--n", type=int, default=240); ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args(); root = Path(__file__).resolve().parents[2]; reg = get_registry(); rng = random.Random(0)
items = [it for it in items_from_jsonl(root / "data/phase0/T1int_s0.jsonl") if it.condition in ("FAM-NAMED", "INV-LEX")]
rng.shuffle(items); items = items[: args.n]
spec, tok, model = load_model(args.model)
# strip the trailing output number so the model has to derive value and unit; ask for reasoning
prompts = []
for it in items:
    p = it.prompt; p = p[: p.rfind(" ")] if p.split()[-1].replace(".", "").isdigit() else p
    prompts.append(p + " Let's work it out step by step, and state the final answer with its unit.\n")
gens = generate_greedy(model, tok, prompts, max_new_tokens=80, batch_size=args.batch)
WRONG = {"L": ["kilograms", "seconds", "watts"], "M": ["meters", "hours", "joules"], "T": ["meters", "grams", "pascals"]}
single = [u for u in ("meters", "kilograms", "seconds", "hours", "grams", "watts", "joules", "pascals", "liters", "newtons") if len(tok(" " + u, add_special_tokens=False)["input_ids"]) == 1]
out = root / "data/phase3"; out.mkdir(parents=True, exist_ok=True); recs = []
for it, p, g in zip(items, prompts, gens):
    # final answer unit: last '<number> <unit-expr>' in the generation
    ms = list(re.finditer(r"(\d+(?:\.\d+)?)\s+([A-Za-z][^\n.,;]{0,40}?)(?=[\n.,;]|$)", g))
    if not ms: continue
    # prefer the first '<number> <unit>' after an "answer"/"=" cue in the last line that has one; else the last match
    cue = [m_ for m_ in ms if re.search(r"(answer|therefore|so|=)", g[max(0, m_.start() - 40): m_.start()], re.I)]
    m = cue[0] if cue else ms[-1]; ustr = m.group(2).strip(); extra = invented_units_from_prompt(it.prompt)
    d = parse_unit_expression(ustr, extra)
    expected = Dimension.parse(it.answer_dimension)
    full = p + g; s0 = len(p) + m.start(2); s1 = s0 + len(ustr)
    # corrupted twin: replace the unit expression by a wrong-dimension single-token unit
    wrong = rng.choice([w for w in single if reg[[u.id for u in reg.units(invented=False) if u.long_pl == w][0]].dim != expected] if True else single)
    corrupted = full[:s0] + wrong + full[s1:]
    recs.append(dict(item_id=it.item_id, condition=it.condition, expected_dim=str(expected), gen_unit=ustr, gen_dim=(str(d) if d else None),
                     gen_correct=(d == expected) if d else None,
                     clean=dict(text=full, unit_span=(s0, s1)), corrupted=dict(text=corrupted, unit_span=(s0, s0 + len(wrong)), unit=wrong)))
with open(out / f"cot_{args.model}.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
n = len(recs); ok = sum(1 for r in recs if r["gen_correct"]); none = sum(1 for r in recs if r["gen_dim"] is None)
print(f"{n} solutions with a final unit ({len(items)} prompts); dimension-correct {ok}, unparsed {none}; wrote {out}")
