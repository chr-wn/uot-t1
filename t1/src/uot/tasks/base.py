"""Shared item schema and helpers for task generators."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

from ..dims import Dimension, BASE_SYMBOLS
from ..units import Unit, UnitExpr, get_registry
from ..quantity import Quantity, fmt_value


@dataclass
class Item:
    item_id: str
    task: str
    condition: str
    template_id: str
    prompt: str                       # text up to the answer slot (no trailing space)
    candidates: list[str]             # continuation strings, each with a leading space
    answer_index: int
    candidate_roles: list[str]
    answer_dimension: str             # str(Dimension)
    answer_vector: list[int]          # over BASE_SYMBOLS + ("X",)
    lattice_distance: int
    named_point: str | None
    unit_strings: list[str]           # renderings of the answer unit (frequency lookup / generation match)
    meta: dict = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return self.candidates[self.answer_index]


VEC_BASIS = BASE_SYMBOLS + ("X",)


def dim_fields(d: Dimension) -> dict:
    return dict(answer_dimension=str(d), answer_vector=list(d.vector(VEC_BASIS)), lattice_distance=d.distance(),
                named_point=d.name())


def items_to_jsonl(items: Iterable[Item], path: str) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def items_from_jsonl(path: str) -> list[Item]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(Item(**json.loads(line)))
    return out


# ---- unit pools -------------------------------------------------------------

# Familiar, high-frequency units per base dimension used in headline cells.
CORE_UNITS = {
    "L": ["millimeter", "centimeter", "meter", "kilometer", "inch", "foot", "yard", "mile"],
    "M": ["milligram", "gram", "kilogram", "tonne", "ounce", "pound"],
    "T": ["millisecond", "second", "minute", "hour", "day", "year"],
}
# Named-derived units used as inputs for composite slots (FAM conditions).
NAMED_FOR_DIM = {
    "M L/T2": ["newton", "kilonewton", "pound_force"],
    "M L2/T2": ["joule", "kilojoule", "calorie", "kilowatt_hour"],
    "M L2/T3": ["watt", "kilowatt", "horsepower"],
    "M/(L T2)": ["pascal", "kilopascal", "psi", "bar"],
    "L3": ["liter", "milliliter", "gallon"],
    "L2": ["hectare", "acre"],
    "L/T": ["knot"],
}


def pick_unit(rng: random.Random, dim: Dimension, *, pool: str = "core") -> UnitExpr:
    """A familiar unit expression with dimension ``dim``.

    For base dimensions: a single unit from CORE_UNITS (or the whole registry if pool='all').
    For composite dimensions: a named-derived unit if one exists (50%) else a composed expression
    from base units.
    """
    reg = get_registry()
    syms = dim.symbols()
    if dim.distance() == 1 and len(syms) == 1:
        if syms[0] in CORE_UNITS and pool == "core":
            ids = CORE_UNITS[syms[0]]
        else:
            ids = [u.id for u in reg.base_units(syms[0], invented=False)]
        if not ids:
            raise ValueError(f"no real unit for base dimension {syms[0]}")
        return UnitExpr.of(reg[rng.choice(ids)])
    named = NAMED_FOR_DIM.get(str(dim))
    if named and (rng.random() < 0.5 or dim.distance() > 4):
        return UnitExpr.of(reg[rng.choice(named)])
    return compose_from_base(rng, dim, lambda s: pick_unit(rng, Dimension({s: 1}), pool=pool).factors[0][0])


def compose_from_base(rng: random.Random, dim: Dimension, base_unit_for) -> UnitExpr:
    """Compose a UnitExpr for ``dim`` from one unit per base symbol (numerators first)."""
    e = dim.as_dict()
    num = [(base_unit_for(s), k) for s, k in e.items() if k > 0]
    den = [(base_unit_for(s), k) for s, k in e.items() if k < 0]
    return UnitExpr.of(*(num + den))


# ---- candidate construction --------------------------------------------------

def lattice_neighbours(units: Sequence[Unit], correct: UnitExpr, rng: random.Random, k: int = 3,
                       exps: Sequence[int] = (-2, -1, 1, 2)) -> list[tuple[UnitExpr, str]]:
    """Distractor unit expressions built from the same lexemes at other lattice points.

    Priority: inverted (all exponents negated), 'copy' (all exponents ±1 in the correct sign
    pattern — equals correct for simple quotients, then skipped), product (all +1), single first
    unit, then random other points.  Returns (expr, role) pairs, distinct and != correct.
    """
    corr = correct.canonical()
    cu = {u.id: e for u, e in corr.factors}
    out: list[tuple[UnitExpr, str]] = []
    seen = {corr}

    def push(expr: UnitExpr, role: str) -> None:
        if len(out) >= k:
            return
        ex = expr.canonical()
        if not ex.factors or ex in seen:
            return
        seen.add(ex)
        out.append((ex, role))

    push(UnitExpr.of(*[(u, -e) for u, e in corr.factors]), "inverted")
    push(UnitExpr.of(*[(u, 1 if e > 0 else -1) for u, e in corr.factors]), "exp_dropped")
    push(UnitExpr.of(*[(u, 1) for u in units]), "product")
    push(UnitExpr.of((units[0], 1)), "single")
    if len(units) > 1:
        push(UnitExpr.of((units[1], 1)), "single")
    tries = 0
    while len(out) < k and tries < 200:
        tries += 1
        fs = [(u, rng.choice(exps)) for u in units]
        push(UnitExpr.of(*fs), "random_point")
    if len(out) < k:
        raise RuntimeError("could not build enough distractors")
    return out


def finalize_candidates(rng: random.Random, correct: str, distractors: list[tuple[str, str]]) -> tuple[list[str], int, list[str]]:
    cands = [(" " + correct, "correct")] + [(" " + s, r) for s, r in distractors]
    rng.shuffle(cands)
    idx = [r for _, r in cands].index("correct")
    return [c for c, _ in cands], idx, [r for _, r in cands]


def definitions_prefix(units: Iterable[Unit]) -> str:
    seen = []
    for u in units:
        if u.invented and u.definition and u.definition not in seen:
            seen.append(u.definition)
    return (" ".join(seen) + " ") if seen else ""
