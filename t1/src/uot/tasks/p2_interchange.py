"""Phase-2 stimuli: two-quantity consistency judgments with single-token answers, built over a fixed
lexeme set so that base/source interchange pairs and the two rival counterfactual label sets can be
constructed exactly.

A prompt is  "<defs> Does it make sense to <op> <v1> <u1> and <v2> <u2>? Answer yes or no.\nAnswer:"
with op ∈ {add, compare, equate, convert}.  Character spans of u1/u2 are recorded.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field, asdict

from ..dims import Dimension
from ..units import Unit, UnitExpr, get_registry
from ..quantity import fmt_value
from ..lexicon import LexemePool
from .base import VEC_BASIS

OPS = {
    "add": ["Does it make sense to add {q1} and {q2}?", "Can {q1} and {q2} be added together?"],
    "compare": ["Does it make sense to ask whether {q1} is larger than {q2}?", "Is it meaningful to compare {q1} with {q2}?"],
    "equate": ["Could {q1} and {q2} be the same amount of the same physical quantity?"],
    "convert": ["Can {q1} be converted into a value in {u2}?"],
}
# fixed lexeme set: ≥ 3 lexemes per lattice point, single-word real units
LEXEMES = {
    "L": ["meter", "kilometer", "centimeter", "mile", "foot", "inch", "yard"],
    "M": ["kilogram", "gram", "pound", "tonne", "ounce", "milligram"],
    "T": ["second", "minute", "hour", "day", "week", "year"],
    "L M/T2": ["newton", "kilonewton", "pound_force", "dyne", "kip"],
    "L2 M/T2": ["joule", "kilojoule", "calorie", "kilowatt_hour", "british_thermal_unit", "erg"],
    "L2 M/T3": ["watt", "kilowatt", "horsepower", "megawatt"],
    "M/(L T2)": ["pascal", "kilopascal", "psi", "bar", "torr", "atmosphere"],
    "1/T": ["hertz", "kilohertz", "becquerel", "rpm"],
    "L3": ["liter", "milliliter", "gallon", "pint", "barrel"],
    "L/T": ["knot", "mph", "kph"],
}


@dataclass
class PItem:
    item_id: str
    prompt: str
    spans: dict          # u1, u2, last
    op: str
    template_id: str
    u1: str; u2: str      # unit ids
    d1: str; d2: str      # dimension strings
    same_dim: bool
    style: str
    invented: bool
    meta: dict = field(default_factory=dict)


def _render(u: Unit, v: int, style: str) -> str:
    return f"{v} {u.symbol}" if style == "symbol" else f"{v} {u.long_pl if v != 1 else u.long_sg}"


def generate(n: int, seed: int, *, styles=("long",), invented: bool = False, ops=("add", "compare", "equate", "convert")) -> list[PItem]:
    rng = random.Random(seed); reg = get_registry(); pool = LexemePool(seed + 21)
    dims = list(LEXEMES)
    items: list[PItem] = []
    for i in range(n):
        op = ops[i % len(ops)]; tid = rng.randrange(len(OPS[op])); text = OPS[op][tid]
        same = (i // len(ops)) % 2 == 0
        d1 = rng.choice(dims); d2 = d1 if same else rng.choice([d for d in dims if d != d1])
        if invented:
            bd = [Dimension.parse(d1), Dimension.parse(d2)]
            u1, u2 = pool.invented_units(bd)
            defs = u1.definition + " " + u2.definition + " "
            style = "long"
        else:
            u1 = reg[rng.choice(LEXEMES[d1])]; u2 = reg[rng.choice(LEXEMES[d2])]
            defs = ""; style = rng.choice(styles)
        v1, v2 = rng.randint(2, 99), rng.randint(2, 99)
        q1, q2 = _render(u1, v1, style), _render(u2, v2, style)
        u2s = u2.long_pl if style == "long" else u2.symbol
        body = text.format(q1=q1, q2=q2, u2=u2s) + " Answer yes or no.\nAnswer:"
        prompt = defs + body
        s1 = prompt.find(q1) + len(str(v1)) + 1; e1 = s1 + len(q1) - len(str(v1)) - 1
        if op == "convert":
            s2 = prompt.find(u2s, s1); e2 = s2 + len(u2s)
        else:
            s2 = prompt.find(q2, e1) + len(str(v2)) + 1; e2 = s2 + len(q2) - len(str(v2)) - 1
        items.append(PItem(item_id=f"P2-{'INV' if invented else 'REAL'}-{i:05d}", prompt=prompt,
                           spans=dict(u1=(s1, e1), u2=(s2, e2), last=(len(prompt) - 1, len(prompt))), op=op, template_id=f"{op}{tid}",
                           u1=u1.id, u2=u2.id, d1=str(u1.dim), d2=str(u2.dim), same_dim=(u1.dim == u2.dim), style=style, invented=invented,
                           meta=dict(v1=v1, v2=v2, u1_str=prompt[s1:e1], u2_str=prompt[s2:e2])))
    return items


def to_jsonl(items, path):
    import json
    with open(path, "w") as f:
        for it in items: f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def from_jsonl(path):
    import json
    return [PItem(**json.loads(l)) for l in open(path) if l.strip()]
