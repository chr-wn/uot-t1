"""T2 — dimensional-consistency judgment (yes/no).

Operations: add / compare / equate / convert.  Label = yes iff the two quantities share a dimension.
"""
from __future__ import annotations

import random

from ..dims import Dimension
from ..units import Unit, UnitExpr, get_registry
from ..quantity import Quantity, sample_style
from ..lexicon import LexemePool
from .base import Item, dim_fields, pick_unit, definitions_prefix, CORE_UNITS

OPS = {
    "add": ("add_a", "Does it make sense to add {q0} and {q1}? Answer yes or no.\nAnswer:"),
    "add2": ("add_b", "Question: Can {q0} and {q1} be added together to give a meaningful total? Answer yes or no.\nAnswer:"),
    "compare": ("cmp_a", "Does it make sense to ask whether {q0} is larger than {q1}? Answer yes or no.\nAnswer:"),
    "compare2": ("cmp_b", "Question: Is it meaningful to compare {q0} with {q1}? Answer yes or no.\nAnswer:"),
    "equate": ("eq_a", "Could {q0} and {q1} be the same amount of the same physical quantity? Answer yes or no.\nAnswer:"),
    "convert": ("conv_a", "Can {q0} be converted into a value in {u1}? Answer yes or no.\nAnswer:"),
}

# near-miss pairs: (unit id or expr spec, unit id or expr spec, same_dim)
NEAR_MISS = [
    ("joule", ("newton", "meter"), True), ("hertz", ("second", -1), True), ("pascal", ("newton", ("meter", -2)), True),
    ("gray", "sievert", True), ("calorie", "joule", True), ("psi", "pascal", True), ("becquerel", "hertz", True),
    ("kilowatt", "kilowatt_hour", False), ("pound", "pound_force", False), ("kilogram", "kilogram_force", False),
    ("watt", "joule", False), ("pascal", "newton", False), ("coulomb", "ampere", False), ("meter", ("meter", 2), False),
    (("meter", ("second", -1)), ("meter", ("second", -2)), False), ("newton", "joule", False), ("hertz", "second", False),
]


def _expr(reg, spec) -> UnitExpr:
    if isinstance(spec, str):
        return UnitExpr.of(reg[spec])
    fs = []
    for s in spec:
        if isinstance(s, str):
            fs.append((reg[s], 1))
        else:
            fs.append((reg[s[0]], s[1]))
    return UnitExpr.of(*fs)


CONDITIONS = ("FAM-SAME-UNIT", "FAM-SAME-DIM", "FAM-DIFF-FAR", "FAM-NEAR-MISS", "INV-LEX", "INV-BASE")


def generate(n_per_condition: int, seed: int, *, conditions=CONDITIONS) -> list[Item]:
    rng = random.Random(seed)
    reg = get_registry()
    pool = LexemePool(seed + 1)
    items: list[Item] = []
    c = 0
    base_dims = [Dimension.parse(s) for s in ("L", "M", "T")]
    comp_dims = [Dimension.parse(s) for s in ("L/T", "M L/T2", "M L2/T2", "L2", "L3", "M/(L T2)", "M L2/T3")]
    for cond in conditions:
        for i in range(n_per_condition):
            want_yes = (i % 2 == 0)
            op_key = rng.choice(list(OPS))
            tid, text = OPS[op_key]
            defs_units: list[Unit] = []
            if cond == "FAM-SAME-UNIT":
                d = rng.choice(base_dims + comp_dims)
                e0 = pick_unit(rng, d)
                e1 = e0 if want_yes else pick_unit(rng, rng.choice([x for x in base_dims + comp_dims if x != d]))
            elif cond == "FAM-SAME-DIM":
                d = rng.choice(base_dims + comp_dims)
                e0 = pick_unit(rng, d)
                if want_yes:
                    for _ in range(20):
                        e1 = pick_unit(rng, d)
                        if e1 != e0:
                            break
                else:
                    e1 = pick_unit(rng, rng.choice([x for x in base_dims + comp_dims if x.l1(d) == 1] or base_dims))
            elif cond == "FAM-DIFF-FAR":
                d0 = rng.choice(base_dims + comp_dims)
                e0 = pick_unit(rng, d0)
                if want_yes:
                    e1 = pick_unit(rng, d0)
                else:
                    far = [x for x in base_dims + comp_dims if x.l1(d0) >= 2]
                    e1 = pick_unit(rng, rng.choice(far))
            elif cond == "FAM-NEAR-MISS":
                cands = [p for p in NEAR_MISS if p[2] == want_yes]
                a, b, _ = rng.choice(cands)
                e0, e1 = _expr(reg, a), _expr(reg, b)
                if rng.random() < 0.5:
                    e0, e1 = e1, e0
            elif cond == "INV-LEX":
                d0 = rng.choice(base_dims)
                d1 = d0 if want_yes else rng.choice([x for x in base_dims if x != d0])
                u0, u1 = pool.invented_units([d0, d1])
                e0, e1 = UnitExpr.of(u0), UnitExpr.of(u1)
                defs_units = [u0, u1]
            elif cond == "INV-BASE":
                xd, xu, _ = pool.invented_base("X")
                e0 = UnitExpr.of(xu)
                if want_yes:
                    (u1,) = pool.invented_units([xd])
                    u1 = Unit(**{**u1.__dict__, "definition": f"A {u1.long_sg} is another unit of {xu.definition.split(' is a basic')[0].lower()}."})
                    e1 = UnitExpr.of(u1)
                    defs_units = [xu, u1]
                else:
                    e1 = pick_unit(rng, rng.choice(base_dims))
                    defs_units = [xu]
            else:
                raise ValueError(cond)
            if rng.random() < 0.5 and cond not in ("FAM-NEAR-MISS",):
                e0, e1 = e1, e0
            invented = any(u.invented for ex in (e0, e1) for u in ex.units)
            style = sample_style(rng, invented=invented)
            q0 = Quantity(float(rng.randint(2, 99)), e0)
            q1 = Quantity(float(rng.randint(2, 99)), e1)
            same = e0.dim == e1.dim
            body = text.format(q0=q0.render(style), q1=q1.render(style), u1=e1.render(style, plural=True))
            prompt = (definitions_prefix(defs_units or [u for ex in (e0, e1) for u in ex.units]) + body).strip()
            cands = [" yes", " no"]
            idx = 0 if same else 1
            items.append(Item(
                item_id=f"T2-{cond}-{c:05d}", task="T2", condition=cond, template_id=tid, prompt=prompt,
                candidates=cands, answer_index=idx, candidate_roles=["yes", "no"], **dim_fields(e0.dim),
                unit_strings=e0.renderings() + e1.renderings(),
                meta=dict(op=op_key, same_dim=same, dim0=str(e0.dim), dim1=str(e1.dim), l1=e0.dim.l1(e1.dim),
                          u0=e0.render("symbol"), u1=e1.render("symbol"), style=style, invented=invented),
            ))
            c += 1
    return items
