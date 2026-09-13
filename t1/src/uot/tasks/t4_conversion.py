"""T4 — conversion acceptability (yes/no): can a value in u0 be expressed in u1?"""
from __future__ import annotations

import random

from ..dims import Dimension
from ..units import Unit, UnitExpr, get_registry
from ..quantity import sample_style
from ..lexicon import LexemePool
from .base import Item, dim_fields, pick_unit, definitions_prefix
from .t2_consistency import NEAR_MISS, _expr

TEMPLATES = [
    ("c4_a", "Can a quantity measured in {u0} be converted into {u1}? Answer yes or no.\nAnswer:"),
    ("c4_b", "Question: Is there a conversion factor from {u0} to {u1}? Answer yes or no.\nAnswer:"),
    ("c4_c", "A value of {v} {u0} is to be expressed in {u1}. Is this possible? Answer yes or no.\nAnswer:"),
    ("c4_d", "Can {u0} be converted to {u1}? Answer yes or no.\nAnswer:"),
]
CONDITIONS = ("FAM-SAME-DIM", "FAM-DIFF-NEAR", "FAM-DIFF-FAR", "FAM-NEAR-MISS", "INV-LEX")


def generate(n_per_condition: int, seed: int, *, conditions=CONDITIONS) -> list[Item]:
    rng = random.Random(seed)
    reg = get_registry()
    pool = LexemePool(seed + 3)
    items: list[Item] = []
    c = 0
    dims = [Dimension.parse(s) for s in ("L", "M", "T", "L/T", "M L/T2", "M L2/T2", "L2", "L3", "M/(L T2)", "M L2/T3")]
    for cond in conditions:
        for i in range(n_per_condition):
            want_yes = (i % 2 == 0)
            defs: list[Unit] = []
            if cond == "FAM-SAME-DIM":
                d = rng.choice(dims)
                e0 = pick_unit(rng, d)
                if want_yes:
                    e1 = e0
                    for _ in range(20):
                        e1 = pick_unit(rng, d)
                        if e1 != e0:
                            break
                else:
                    e1 = pick_unit(rng, rng.choice([x for x in dims if x != d]))
            elif cond == "FAM-DIFF-NEAR":
                d = rng.choice(dims)
                e0 = pick_unit(rng, d)
                if want_yes:
                    e1 = pick_unit(rng, d)
                else:
                    near = [x for x in dims if x.l1(d) == 1] or [x for x in dims if x != d]
                    e1 = pick_unit(rng, rng.choice(near))
            elif cond == "FAM-DIFF-FAR":
                d = rng.choice(dims)
                e0 = pick_unit(rng, d)
                if want_yes:
                    e1 = pick_unit(rng, d)
                else:
                    far = [x for x in dims if x.l1(d) >= 3] or [x for x in dims if x != d]
                    e1 = pick_unit(rng, rng.choice(far))
            elif cond == "FAM-NEAR-MISS":
                a, b, _ = rng.choice([p for p in NEAR_MISS if p[2] == want_yes])
                e0, e1 = _expr(reg, a), _expr(reg, b)
                if rng.random() < 0.5:
                    e0, e1 = e1, e0
            elif cond == "INV-LEX":
                d0 = rng.choice(dims[:3])
                d1 = d0 if want_yes else rng.choice([x for x in dims[:3] if x != d0])
                u0, u1 = pool.invented_units([d0, d1])
                e0, e1 = UnitExpr.of(u0), UnitExpr.of(u1)
                defs = [u0, u1]
            else:
                raise ValueError(cond)
            invented = any(u.invented for ex in (e0, e1) for u in ex.units)
            style = sample_style(rng, invented=invented, allow=("symbol", "long", "slash", "per"))
            tid, text = rng.choice(TEMPLATES)
            same = e0.dim == e1.dim
            body = text.format(u0=e0.render(style, plural=True), u1=e1.render(style, plural=True), v=rng.randint(2, 99))
            prompt = (definitions_prefix(defs) + body).strip()
            items.append(Item(
                item_id=f"T4-{cond}-{c:05d}", task="T4", condition=cond, template_id=tid, prompt=prompt,
                candidates=[" yes", " no"], answer_index=0 if same else 1, candidate_roles=["yes", "no"],
                **dim_fields(e0.dim), unit_strings=e0.renderings() + e1.renderings(),
                meta=dict(same_dim=same, dim0=str(e0.dim), dim1=str(e1.dim), l1=e0.dim.l1(e1.dim),
                          u0=e0.render("symbol"), u1=e1.render("symbol"), style=style, invented=invented),
            ))
            c += 1
    return items
