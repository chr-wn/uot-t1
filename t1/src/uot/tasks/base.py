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
# Natural, everyday units used inside compound expressions and in T1 (keeps prompts plausible).
NATURAL_UNITS = {
    "L": ["meter", "kilometer", "centimeter", "foot", "mile", "inch"],
    "M": ["kilogram", "gram", "pound", "tonne"],
    "T": ["second", "minute", "hour"],
}
# System-consistent sets for composing compound expressions (no "foot kilograms").
NATURAL_SYSTEMS = {
    "metric": {"L": ["meter", "kilometer", "centimeter"], "M": ["kilogram", "gram"], "T": ["second", "minute", "hour"]},
    "imperial": {"L": ["foot", "mile", "inch"], "M": ["pound"], "T": ["second", "minute", "hour"]},
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
        elif syms[0] in NATURAL_UNITS and pool == "natural":
            ids = NATURAL_UNITS[syms[0]]
        else:
            ids = [u.id for u in reg.base_units(syms[0], invented=False)]
        if not ids:
            raise ValueError(f"no real unit for base dimension {syms[0]}")
        return UnitExpr.of(reg[rng.choice(ids)])
    named = NAMED_FOR_DIM.get(str(dim))
    if named and (rng.random() < 0.5 or dim.distance() > 4):
        return UnitExpr.of(reg[rng.choice(named)])
    # compound expressions are composed from natural, system-consistent units
    sysname = rng.choice(list(NATURAL_SYSTEMS))
    table = NATURAL_SYSTEMS[sysname]

    def base_unit_for(s: str) -> Unit:
        ids = table.get(s) or [u.id for u in reg.base_units(s, invented=False)]
        return reg[rng.choice(ids)]
    return compose_from_base(rng, dim, base_unit_for)


def compose_from_base(rng: random.Random, dim: Dimension, base_unit_for) -> UnitExpr:
    """Compose a UnitExpr for ``dim`` from one unit per base symbol (numerators first)."""
    e = dim.as_dict()
    num = [(base_unit_for(s), k) for s, k in e.items() if k > 0]
    den = [(base_unit_for(s), k) for s, k in e.items() if k < 0]
    return UnitExpr.of(*(num + den))


# ---- candidate construction --------------------------------------------------

def lattice_neighbours(units: Sequence[Unit], correct: UnitExpr, rng: random.Random, k: int = 3,
                       exps: Sequence[int] = (-3, -2, -1, 1, 2, 3)) -> list[tuple[UnitExpr, str]]:
    """Distractor unit expressions built from the same lexemes at other lattice points.

    Length-matched by construction: every distractor uses *all* the lexemes of the correct
    expression with nonzero exponents, so candidates differ only in exponents/arrangement.
    Priority: inverted (all exponents negated), exp_swapped (exponents permuted across units, or
    ±1 for a single unit), then random points over the same lexemes.  Returns (expr, role) pairs,
    distinct and != correct.
    """
    corr = correct.canonical()
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

    fs = list(corr.factors)
    push(UnitExpr.of(*[(u, -e) for u, e in fs]), "inverted")
    if len(fs) >= 2:
        es = [e for _, e in fs]
        perm = es[1:] + es[:1]
        push(UnitExpr.of(*[(u, e) for (u, _), e in zip(fs, perm)]), "exp_swapped")
    else:
        u, e = fs[0]
        push(UnitExpr.of((u, e + 1 if e > 0 else e - 1)), "exp_swapped")
        push(UnitExpr.of((u, 1 if abs(e) != 1 else 2)), "exp_dropped")
    tries = 0
    while len(out) < k and tries < 500:
        tries += 1
        push(UnitExpr.of(*[(u, rng.choice(exps)) for u, _ in fs]), "random_point")
    if len(out) < k:
        push(UnitExpr.of(*[(u, 1) for u, _ in fs]), "product")
        push(UnitExpr.of((fs[0][0], 1)), "single")
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
