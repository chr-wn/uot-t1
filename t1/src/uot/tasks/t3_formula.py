"""T3 — formula selection: which expression over defined symbols has the units of the target?"""
from __future__ import annotations

import random

from ..dims import Dimension
from ..units import UnitExpr
from ..quantity import sample_style
from ..lexicon import LexemePool
from .base import Item, dim_fields, pick_unit, finalize_candidates, definitions_prefix, compose_from_base

# (target name or None, symbol dims, exps)
TARGETS = [
    ("speed", ("L", "T"), (1, -1)), ("acceleration", ("L", "T"), (1, -2)), ("area", ("L", "L"), (1, 1)),
    ("density", ("M", "L"), (1, -3)), ("force", ("M", "L", "T"), (1, 1, -2)), ("energy", ("M", "L", "T"), (1, 2, -2)),
    ("power", ("M", "L", "T"), (1, 2, -3)), ("pressure", ("M", "L", "T"), (1, -1, -2)), ("momentum", ("M", "L", "T"), (1, 1, -1)),
    ("frequency", ("T",), (-1,)),
    (None, ("M", "L"), (2, 1)), (None, ("L", "T"), (3, -5)), (None, ("M", "T"), (1, -2)), (None, ("L", "T"), (1, 1)),
    (None, ("M", "L"), (1, 3)), (None, ("T", "L"), (2, -1)), (None, ("M", "T"), (2, -1)),
]
SYMS = {"L": ("x", "length"), "M": ("m", "mass"), "T": ("t", "time"), "X1": ("w", None)}
TEMPLATES = [
    ("f3_a", "{defs} Which of the following expressions has the units of {target}, that is, {unit}?\n{opts}\nAnswer:"),
    ("f3_b", "{defs} Exactly one of these expressions is measured in {unit}{tname}. Which one?\n{opts}\nAnswer:"),
    ("f3_c", "{defs} Question: which expression below has units of {unit}?\n{opts}\nAnswer:"),
]


def _expr_str(syms: tuple[str, ...], exps: tuple[int, ...]) -> str:
    num = [f"{SYMS[s][0]}{'^' + str(e) if e != 1 else ''}" for s, e in zip(syms, exps) if e > 0]
    den = [f"{SYMS[s][0]}{'^' + str(-e) if e != -1 else ''}" for s, e in zip(syms, exps) if e < 0]
    n = "*".join(num) if num else "1"
    if not den:
        return n
    d = "*".join(den)
    return f"{n}/({d})" if len(den) > 1 else f"{n}/{d}"


CONDITIONS = ("FAM-NAMED", "FAM-UNNAMED", "INV-LEX")


def generate(n_per_condition: int, seed: int, *, conditions=CONDITIONS) -> list[Item]:
    rng = random.Random(seed)
    pool = LexemePool(seed + 2)
    items: list[Item] = []
    c = 0
    for cond in conditions:
        targets = [t for t in TARGETS if (t[0] is not None) == (cond != "FAM-UNNAMED")] if cond != "INV-LEX" else list(TARGETS)
        for i in range(n_per_condition):
            tname, syms, exps = targets[i % len(targets)]
            uniq = list(dict.fromkeys(syms))
            # exponent per unique symbol
            ev = {s: 0 for s in uniq}
            for s, e in zip(syms, exps):
                ev[s] += e
            if cond == "INV-LEX":
                units = dict(zip(uniq, pool.invented_units([Dimension({s: 1}) for s in uniq])))
            else:
                units = {s: pick_unit(rng, Dimension({s: 1})).factors[0][0] for s in uniq}
            style = sample_style(rng, invented=(cond == "INV-LEX"))
            defs = " ".join(f"Let {SYMS[s][0]} be a {SYMS[s][1]} measured in {units[s].long_pl}." for s in uniq)
            target = UnitExpr.of(*[(units[s], ev[s]) for s in uniq if ev[s] != 0]).canonical()
            # candidate formulas: correct + 3 distinct alternatives over the same symbols
            corr = tuple(ev[s] for s in uniq)
            alts = set()
            pool_exps = [-3, -2, -1, 1, 2, 3]
            tries = 0
            while len(alts) < 3 and tries < 500:
                tries += 1
                a = tuple(rng.choice(pool_exps) for _ in uniq)
                if a != corr and any(x != 0 for x in a):
                    alts.add(a)
            if len(alts) < 3:
                continue
            forms = [(corr, "correct")] + [(a, "alt") for a in alts]
            rng.shuffle(forms)
            letters = "ABCD"
            opts = "\n".join(f"({L}) {_expr_str(tuple(uniq), f)}" for L, (f, _) in zip(letters, forms))
            idx = [r for _, r in forms].index("correct")
            tid, text = rng.choice(TEMPLATES)
            prefix = definitions_prefix(units.values())
            prompt = (prefix + text.format(defs=defs, target=tname or "the quantity Q", unit=target.render(style, plural=True),
                                            tname=f" ({tname})" if tname else "", opts=opts)).strip()
            items.append(Item(
                item_id=f"T3-{cond}-{c:05d}", task="T3", condition=cond, template_id=tid, prompt=prompt,
                candidates=[f" {L}" for L in letters], answer_index=idx, candidate_roles=[r for _, r in forms],
                **dim_fields(target.dim), unit_strings=target.renderings(),
                meta=dict(target=tname, style=style, invented=(cond == "INV-LEX"), formulas=[_expr_str(tuple(uniq), f) for f, _ in forms]),
            ))
            c += 1
    return items
