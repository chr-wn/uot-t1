"""Phase-1 probing stimuli: single-quantity mentions with character spans for position classes.

Each item is a short passage in which one quantity is mentioned once and then referred to
anaphorically.  Spans: 'value' (numeral), 'unit' (unit string), 'mention_end' (token after the
unit), 'anaphor' (the referring noun), 'last' (final token).  Labels: exponent vector.

Template families:
  neutral   — the noun carries no dimension information ("the reading was 12 meters");
  revealing — the noun names the quantity ("the length was 12 meters");
  noun_only — dimension-revealing noun, no unit ("the length was 12"): control for whether the
              dimension feature is tied to the unit lexeme or to the quantity semantics.
Conditions cover: real units at many lattice points (base, named-derived, composed compound
units incl. unnamed points), dimensionless mentions, invented lexemes (defined in context),
cross-lingual long forms (es/de/fr).
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field, asdict

from ..dims import Dimension
from ..units import Unit, UnitExpr, get_registry
from ..quantity import fmt_value
from ..lexicon import LexemePool
from .base import CORE_UNITS, VEC_BASIS

NEUTRAL_NOUNS = ["reading", "value", "figure", "measurement", "result", "number", "entry", "amount recorded"]
OBJECTS = ["the sensor", "the gauge", "the report", "the logbook", "the display", "the survey", "the instrument",
           "the technician", "the lab notebook", "the spreadsheet"]
REVEALING = {  # dimension -> nouns
    "L": ["length", "distance", "height", "width"], "M": ["mass", "weight"], "T": ["duration", "time taken"],
    "L/T": ["speed", "velocity"], "L/T2": ["acceleration"], "L2": ["area"], "L3": ["volume"], "M L/T2": ["force"],
    "M L2/T2": ["energy"], "M L2/T3": ["power"], "M/(L T2)": ["pressure"], "1/T": ["frequency"], "M/L3": ["density"],
    "I": ["current"], "Th": ["temperature"], "I T": ["charge"], "M L2/(T3 I)": ["voltage"], "M L2/(T3 I2)": ["resistance"],
    "M L/T": ["momentum"], "L3/T": ["flow rate"], "1": ["ratio"],
}
ANAPHOR_CONT = [
    " That {noun} was later confirmed by a second check.",
    " The {noun} surprised nobody on the team.",
    " Nobody questioned the {noun} at the time.",
    " That {noun} appears twice in the final summary.",
]
INTRO = [
    "According to {obj}, the {noun} was {q}.",
    "The {noun} shown by {obj} came to {q}.",
    "{Obj} listed the {noun} as {q}.",
    "In the morning, {obj} recorded a {noun} of {q}.",
    "The {noun} from {obj} turned out to be {q}.",
]


@dataclass
class MentionItem:
    item_id: str
    condition: str
    family: str          # neutral | revealing | noun_only
    template_id: str
    prompt: str
    spans: dict          # name -> (start, end)
    unit_id: str | None
    unit_string: str
    dimension: str
    vector: list
    lattice_distance: int
    named_point: str | None
    style: str
    lang: str
    value: float
    system: str | None
    invented: bool
    meta: dict = field(default_factory=dict)


def _lattice_exprs(rng: random.Random, max_abs: int = 2, variants: int = 4) -> list[tuple[Dimension, UnitExpr]]:
    """Compound expressions over (L, M, T) natural units for every lattice point with |e|<=max_abs (no
    origin); `variants` distinct unit choices per point so that cross-lexeme folds exist at every point."""
    reg = get_registry()
    out = []
    for a, b, c in itertools.product(range(-max_abs, max_abs + 1), repeat=3):
        if (a, b, c) == (0, 0, 0):
            continue
        d = Dimension(L=a, M=b, T=c)
        seen = set()
        tries = 0
        while len(seen) < variants and tries < 50:
            tries += 1
            fs = []
            for sym, e in (("M", b), ("L", a), ("T", c)):
                if e:
                    fs.append((reg[rng.choice(CORE_UNITS[sym])], e))
            fs = [f for f in fs if f[1] > 0] + [f for f in fs if f[1] < 0]
            ex = UnitExpr.of(*fs)
            key = tuple(u.id for u, _ in ex.factors)
            if key in seen:
                continue
            seen.add(key)
            out.append((d, ex))
    return out


def _render_q(v: float, ue: UnitExpr | None, style: str, lang: str) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Returns text, value span, unit span (relative to text)."""
    vs = fmt_value(v)
    if ue is None:
        return vs, (0, len(vs)), (len(vs), len(vs))
    us = ue.render(style, lang=lang, plural=(v != 1))
    text = f"{vs} {us}"
    return text, (0, len(vs)), (len(vs) + 1, len(text))


def build(prompt_parts: list[tuple[str, str | None]]) -> tuple[str, dict]:
    """Concatenate (text, span_name) parts; return prompt and spans of named parts."""
    s, spans = "", {}
    for text, name in prompt_parts:
        if name:
            spans[name] = (len(s), len(s) + len(text))
        s += text
    return s, spans


def generate(n_per_condition: int, seed: int, *, conditions=("REAL-BASE", "REAL-NAMED", "REAL-LATTICE", "DIMLESS",
                                                            "INV-LEX", "XLING"), families=("neutral", "revealing", "noun_only"),
             n_override: dict | None = None) -> list[MentionItem]:
    """n_override: per-condition item counts, e.g. {"REAL-LATTICE": 3000}."""
    rng = random.Random(seed)
    reg = get_registry()
    pool = LexemePool(seed + 11)
    items: list[MentionItem] = []
    c = 0
    base_dims = [Dimension.parse(s) for s in ("L", "M", "T", "I", "Th", "N", "J")]
    named_units = reg.units(named_derived=True)
    lattice = _lattice_exprs(rng)
    for cond in conditions:
        n_cond = (n_override or {}).get(cond, n_per_condition)
        for i in range(n_cond):
            fam = families[i % len(families)]
            lang = "en"
            style = rng.choice(("symbol", "long", "slash"))
            ue: UnitExpr | None
            if cond == "REAL-BASE":
                d = base_dims[i % len(base_dims)]
                us = reg.units(d, invented=False)
                ue = UnitExpr.of(rng.choice(us))
            elif cond == "REAL-NAMED":
                u = named_units[i % len(named_units)]
                ue = UnitExpr.of(u)
            elif cond == "REAL-LATTICE":
                d, ue = lattice[i % len(lattice)]
                style = rng.choice(("symbol", "long", "slash"))
            elif cond == "DIMLESS":
                ue = None
                style = "none"
            elif cond == "INV-LEX":
                d = rng.choice(base_dims[:3])
                (u,) = pool.invented_units([d])
                ue = UnitExpr.of(u)
                style = "long"
            elif cond == "XLING":
                lang = rng.choice(("es", "de", "fr"))
                cands = [u for u in reg.units(invented=False) if lang in u.translations]
                ue = UnitExpr.of(rng.choice(cands))
                style = "long"
            else:
                raise ValueError(cond)
            d = ue.dim if ue is not None else Dimension()
            # nouns
            if fam == "neutral":
                noun = rng.choice(NEUTRAL_NOUNS)
            else:
                opts = REVEALING.get(str(d)) or REVEALING.get(d.name() or "") or ["quantity"]
                noun = rng.choice(opts)
            if fam == "noun_only":
                if not REVEALING.get(str(d)):
                    continue  # noun_only needs a revealing noun
                ue_render = None
            else:
                ue_render = ue
            v = float(rng.randint(2, 99))
            qtext, vspan, uspan = _render_q(v, ue_render, style, lang)
            ti = rng.randrange(len(INTRO))
            intro = INTRO[ti]
            obj = rng.choice(OBJECTS)
            head, tail = intro.split("{q}")
            head = head.format(obj=obj, Obj=obj[0].upper() + obj[1:], noun=noun)
            parts = [(head, None), (qtext[vspan[0]:vspan[1]], "value")]
            if ue_render is not None:
                parts += [(qtext[vspan[1]:uspan[0]], None), (qtext[uspan[0]:uspan[1]], "unit")]
            parts += [(tail, "mention_end")]  # the period right after the mention
            ai = rng.randrange(len(ANAPHOR_CONT))
            cont = ANAPHOR_CONT[ai]
            a_head, a_tail = cont.split("{noun}")
            parts += [(a_head, None), (noun, "anaphor"), (a_tail, None)]
            prompt, spans = build(parts)
            spans["last"] = (len(prompt) - 1, len(prompt))
            if ue_render is None:
                spans["unit"] = spans["value"]  # no unit token: fall back to the numeral
            defs = " ".join(u.definition for u in (ue.units if ue else ()) if u.invented)
            if defs:
                prompt2 = defs + " " + prompt
                off = len(defs) + 1
                spans = {k: (a + off, b + off) for k, (a, b) in spans.items()}
                prompt = prompt2
            u0 = ue.units[0] if ue is not None else None
            items.append(MentionItem(
                item_id=f"P1-{cond}-{fam}-{c:05d}", condition=cond, family=fam, template_id=f"i{ti}_a{ai}",
                prompt=prompt, spans=spans, unit_id=(u0.id if (u0 and ue.is_single()) else (ue.render('symbol') if ue else None)),
                unit_string=(ue.render(style, lang=lang) if (ue is not None and fam != "noun_only") else ""),
                dimension=str(d), vector=list(d.vector(VEC_BASIS)), lattice_distance=d.distance(), named_point=d.name(),
                style=style, lang=lang, value=v, system=(u0.system if u0 else None), invented=bool(u0 and u0.invented),
                meta=dict(noun=noun, n_units=len(ue.units) if ue else 0, units=[u.id for u in ue.units] if ue else []),
            ))
            c += 1
    return items


def to_jsonl(items, path):
    import json
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def from_jsonl(path):
    import json
    return [MentionItem(**json.loads(l)) for l in open(path) if l.strip()]
