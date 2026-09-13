"""T5 — error spotting: three quantity statements, one with a dimensionally wrong unit."""
from __future__ import annotations

import random

from ..dims import Dimension
from ..units import Unit, UnitExpr
from ..quantity import Quantity, sample_style
from ..lexicon import LexemePool
from .base import Item, dim_fields, pick_unit, definitions_prefix

# quantity noun -> dimension
NOUNS = {
    "length": "L", "height": "L", "width": "L", "distance": "L", "mass": "M", "duration": "T", "time taken": "T",
    "speed": "L/T", "area": "L2", "volume": "L3", "force": "M L/T2", "energy": "M L2/T2", "power": "M L2/T3",
    "pressure": "M/(L T2)", "acceleration": "L/T2", "density": "M/L3",
}
OBJECTS = ["the bridge", "the truck", "the river", "the satellite", "the pump", "the tank", "the engine", "the runner",
           "the drone", "the cable", "the beam", "the balloon", "the crane", "the boat", "the field", "the pipe"]
TEMPLATES = [
    ("e5_a", "{s}\nExactly one of the sentences above uses a unit that cannot measure the stated quantity. Which one? Answer A, B, or C.\nAnswer:"),
    ("e5_b", "Consider the following three statements.\n{s}\nWhich statement contains a unit error? Answer A, B, or C.\nAnswer:"),
    ("e5_c", "{s}\nQuestion: In which statement does the unit not match the quantity? (A, B, or C)\nAnswer:"),
]
CONDITIONS = ("FAM-FAR", "FAM-NEAR", "INV-LEX")


def generate(n_per_condition: int, seed: int, *, conditions=CONDITIONS) -> list[Item]:
    rng = random.Random(seed)
    pool = LexemePool(seed + 4)
    items: list[Item] = []
    c = 0
    nouns = list(NOUNS)
    for cond in conditions:
        for i in range(n_per_condition):
            picked = rng.sample(nouns, 3)
            wrong_k = i % 3
            dims = [Dimension.parse(NOUNS[n]) for n in picked]
            defs: list[Unit] = []
            exprs: list[UnitExpr] = []
            if cond == "INV-LEX":
                base = [d for d in dims]
                # only base-dimension nouns for invented lexemes
                picked = rng.sample([n for n in nouns if Dimension.parse(NOUNS[n]).distance() == 1], 3)
                dims = [Dimension.parse(NOUNS[n]) for n in picked]
                wrong_dim = rng.choice([Dimension.parse(s) for s in ("L", "M", "T") if Dimension.parse(s) != dims[wrong_k]])
                inv_dims = [wrong_dim if k == wrong_k else d for k, d in enumerate(dims)]
                us = pool.invented_units(inv_dims)
                exprs = [UnitExpr.of(u) for u in us]
                defs = us
                wrong_l1 = dims[wrong_k].l1(wrong_dim)
            else:
                for k, d in enumerate(dims):
                    if k != wrong_k:
                        exprs.append(pick_unit(rng, d))
                    else:
                        alld = [Dimension.parse(s) for s in set(NOUNS.values())]
                        if cond == "FAM-NEAR":
                            cands = [x for x in alld if x.l1(d) == 1] or [x for x in alld if x != d]
                        else:
                            cands = [x for x in alld if x.l1(d) >= 3] or [x for x in alld if x != d]
                        wd = rng.choice(cands)
                        exprs.append(pick_unit(rng, wd))
                        wrong_l1 = d.l1(wd)
            invented = cond == "INV-LEX"
            style = sample_style(rng, invented=invented)
            objs = rng.sample(OBJECTS, 3)
            sents = []
            for L, n, o, ex in zip("ABC", picked, objs, exprs):
                q = Quantity(float(rng.randint(2, 99)), ex)
                sents.append(f"({L}) The {n} of {o} is {q.render(style)}.")
            tid, text = rng.choice(TEMPLATES)
            prompt = (definitions_prefix(defs) + text.format(s="\n".join(sents))).strip()
            items.append(Item(
                item_id=f"T5-{cond}-{c:05d}", task="T5", condition=cond, template_id=tid, prompt=prompt,
                candidates=[" A", " B", " C"], answer_index=wrong_k, candidate_roles=["A", "B", "C"],
                **dim_fields(dims[wrong_k]), unit_strings=exprs[wrong_k].renderings(),
                meta=dict(nouns=picked, units=[e.render("symbol") for e in exprs], wrong_l1=wrong_l1, style=style, invented=invented),
            ))
            c += 1
    return items
