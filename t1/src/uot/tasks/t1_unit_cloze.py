"""T1 — unit-of-answer cloze.

A scenario states input quantities and names (or defines) a derived quantity; the model must
complete the *unit* of the derived quantity.  Ground truth is the composed unit expression built
from the input lexemes (dimension arithmetic on the exponent lattice).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from ..dims import Dimension
from ..units import Unit, UnitExpr, get_registry
from ..quantity import Quantity, fmt_value, sample_style
from ..lexicon import LexemePool
from .base import Item, dim_fields, pick_unit, compose_from_base, lattice_neighbours, finalize_candidates, definitions_prefix, select_distractor_renderings


@dataclass(frozen=True)
class Relation:
    name: str                 # "speed"
    slots: tuple[str, ...]    # slot dimensions as parse strings, e.g. ("L", "T")
    exps: tuple[int, ...]     # output = prod slot_i ^ exps_i
    named: bool               # has a conventional name usable in text
    templates: tuple[tuple[str, str, bool], ...]  # (template_id, text, formula_given)
    # text placeholders: {q0} {q1} (input quantities), {v} (output value), {name} (quantity name)

    @property
    def out_dim(self) -> Dimension:
        d = Dimension()
        for s, e in zip(self.slots, self.exps):
            d = d * (Dimension.parse(s) ** e)
        return d


# Templates: each relation has in-order and swapped surface orders (the generator computes which).
RELATIONS: list[Relation] = [
    Relation("speed", ("L", "T"), (1, -1), True, (
        ("speed_a", "A runner covers {q0} in {q1}, so the runner's speed is {v}", False),
        ("speed_b", "Over {q1} the vehicle travels {q0}, giving an average speed of {v}", False),
        ("speed_c", "The train needs {q1} to travel {q0}. Its speed is therefore {v}", False),
        ("speed_d", "Speed is distance divided by time. A cyclist rides {q0} in {q1}, so the speed is {v}", True),
        ("speed_e", "Speed is distance divided by time. It takes {q1} to walk {q0}, so the speed is {v}", True),
        ("speed_f", "Distance: {q0}. Time: {q1}. Speed (distance per time): {v}", True),
    )),
    Relation("acceleration", ("L/T", "T"), (1, -1), True, (
        ("acc_a", "A car's speed increases by {q0} over {q1}, so its acceleration is {v}", False),
        ("acc_b", "Within {q1}, the speed rises by {q0}; the acceleration is {v}", False),
        ("acc_c", "Acceleration is change in speed divided by time. The speed changes by {q0} in {q1}, so the acceleration is {v}", True),
        ("acc_d", "Acceleration is change in speed divided by time. In {q1} the speed changes by {q0}, so the acceleration is {v}", True),
    )),
    Relation("area", ("L",), (2,), True, (
        ("area_a", "A square has sides of {q0}, so its area is {v}", False),
        ("area_b", "The area of a square with side length {q0} is {v}", False),
        ("area_c", "Area is length times length. A square with side {q0} has area {v}", True),
    )),
    Relation("volume", ("L",), (3,), True, (
        ("vol_a", "A cube has edges of {q0}, so its volume is {v}", False),
        ("vol_b", "The volume of a cube with edge length {q0} is {v}", False),
        ("vol_c", "Volume is length cubed. A cube with edge {q0} has volume {v}", True),
    )),
    Relation("density", ("M", "L"), (1, -3), True, (
        ("dens_a", "A cube with mass {q0} and edge length {q1} has a density of {v}", False),
        ("dens_b", "A cube with edge length {q1} has mass {q0}; its density is {v}", False),
        ("dens_c", "Density is mass divided by volume, and volume is edge length cubed. A cube of mass {q0} with edge {q1} has density {v}", True),
        ("dens_d", "Density is mass divided by volume, and volume is edge length cubed. A cube with edge {q1} and mass {q0} has density {v}", True),
    )),
    Relation("force", ("M", "L/T2"), (1, 1), True, (
        ("force_a", "A mass of {q0} is accelerated at {q1}, so the force on it is {v}", False),
        ("force_b", "An acceleration of {q1} is given to a mass of {q0}; the force is {v}", False),
        ("force_c", "Force is mass times acceleration. A mass of {q0} accelerating at {q1} feels a force of {v}", True),
        ("force_d", "Force is mass times acceleration. With acceleration {q1} and mass {q0}, the force is {v}", True),
    )),
    Relation("energy", ("M L/T2", "L"), (1, 1), True, (
        ("work_a", "A force of {q0} pushes an object through {q1}, doing work of {v}", False),
        ("work_b", "Over a distance of {q1}, a force of {q0} does work equal to {v}", False),
        ("work_c", "Work is force times distance. A force of {q0} over {q1} does work of {v}", True),
        ("work_d", "Work is force times distance. Over {q1} with a force of {q0}, the work done is {v}", True),
    )),
    Relation("power", ("M L2/T2", "T"), (1, -1), True, (
        ("pow_a", "A motor delivers {q0} of energy in {q1}, so its power output is {v}", False),
        ("pow_b", "In {q1} the heater releases {q0} of energy; its power is {v}", False),
        ("pow_c", "Power is energy divided by time. Delivering {q0} in {q1} corresponds to a power of {v}", True),
        ("pow_d", "Power is energy divided by time. In {q1}, {q0} is delivered, so the power is {v}", True),
    )),
    Relation("pressure", ("M L/T2", "L"), (1, -2), True, (
        ("pres_a", "A force of {q0} acts on a square plate with side {q1}, so the pressure is {v}", False),
        ("pres_b", "A square plate with side {q1} carries a force of {q0}; the pressure on it is {v}", False),
        ("pres_c", "Pressure is force divided by area, and the area of a square is its side squared. A force of {q0} on a square of side {q1} gives a pressure of {v}", True),
        ("pres_d", "Pressure is force divided by area, and the area of a square is its side squared. On a square of side {q1}, a force of {q0} gives a pressure of {v}", True),
    )),
    Relation("momentum", ("M", "L/T"), (1, 1), True, (
        ("mom_a", "A {q0} ball moving at {q1} has a momentum of {v}", False),
        ("mom_b", "Moving at {q1}, a ball of mass {q0} has momentum {v}", False),
        ("mom_c", "Momentum is mass times velocity. A mass of {q0} at {q1} has momentum {v}", True),
        ("mom_d", "Momentum is mass times velocity. At {q1}, a mass of {q0} has momentum {v}", True),
    )),
    Relation("flow rate", ("L3", "T"), (1, -1), True, (
        ("flow_a", "A pump moves {q0} of water in {q1}, so its flow rate is {v}", False),
        ("flow_b", "In {q1} the tap delivers {q0}; its flow rate is {v}", False),
        ("flow_c", "Flow rate is volume divided by time. Moving {q0} in {q1} is a flow rate of {v}", True),
        ("flow_d", "Flow rate is volume divided by time. In {q1}, {q0} flows, so the flow rate is {v}", True),
    )),
    # ---- unnamed lattice points: relation always stated explicitly ----
    Relation("Q(M2L)", ("M", "L"), (2, 1), False, (
        ("u_m2l_a", "Define the quantity Q as mass squared times length. With mass {q0} and length {q1}, Q equals {v}", True),
        ("u_m2l_b", "Define the quantity Q as mass squared times length. With length {q1} and mass {q0}, Q equals {v}", True),
        ("u_m2l_c", "Let Q be the square of the mass multiplied by the length. For a mass of {q0} and a length of {q1}, Q is {v}", True),
    )),
    Relation("Q(L3/T5)", ("L", "T"), (3, -5), False, (
        ("u_l3t5_a", "Define the quantity Q as length cubed divided by time to the fifth power. With length {q0} and time {q1}, Q equals {v}", True),
        ("u_l3t5_b", "Define the quantity Q as length cubed divided by time to the fifth power. With time {q1} and length {q0}, Q equals {v}", True),
    )),
    Relation("Q(M/T2)", ("M", "T"), (1, -2), False, (
        ("u_mt2_a", "Define the quantity Q as mass divided by time squared. With mass {q0} and time {q1}, Q equals {v}", True),
        ("u_mt2_b", "Define the quantity Q as mass divided by time squared. With time {q1} and mass {q0}, Q equals {v}", True),
        ("u_mt2_c", "Let Q be the mass divided by the square of the time. For a mass of {q0} and a time of {q1}, Q is {v}", True),
    )),
    Relation("Q(LT)", ("L", "T"), (1, 1), False, (
        ("u_lt_a", "Define the quantity Q as length multiplied by time. With length {q0} and time {q1}, Q equals {v}", True),
        ("u_lt_b", "Define the quantity Q as length multiplied by time. With time {q1} and length {q0}, Q equals {v}", True),
    )),
    Relation("Q(ML3)", ("M", "L"), (1, 3), False, (
        ("u_ml3_a", "Define the quantity Q as mass times length cubed. With mass {q0} and length {q1}, Q equals {v}", True),
        ("u_ml3_b", "Define the quantity Q as mass times length cubed. With length {q1} and mass {q0}, Q equals {v}", True),
    )),
    Relation("Q(T2/L)", ("T", "L"), (2, -1), False, (
        ("u_t2l_a", "Define the quantity Q as time squared divided by length. With time {q0} and length {q1}, Q equals {v}", True),
        ("u_t2l_b", "Define the quantity Q as time squared divided by length. With length {q1} and time {q0}, Q equals {v}", True),
    )),
    Relation("Q(M2/T)", ("M", "T"), (2, -1), False, (
        ("u_m2t_a", "Define the quantity Q as mass squared divided by time. With mass {q0} and time {q1}, Q equals {v}", True),
        ("u_m2t_b", "Define the quantity Q as mass squared divided by time. With time {q1} and mass {q0}, Q equals {v}", True),
    )),
]

# Relations for an invented base dimension X1 (always explicit; the quantity name is invented too).
INV_BASE_RELATIONS: list[Relation] = [
    Relation("X1 rate", ("X", "T"), (1, -1), False, (
        ("xb_rate_a", "A device produces {q0} of {name} in {q1}, so its rate of {name} production is {v}", False),
        ("xb_rate_b", "In {q1}, a device produces {q0} of {name}; its rate of {name} production is {v}", False),
        ("xb_rate_c", "The rate of {name} is the amount of {name} divided by time. Producing {q0} in {q1} is a rate of {v}", True),
    )),
    Relation("X1 per length", ("X", "L"), (1, -1), False, (
        ("xb_perlen_a", "Define the {name} density as {name} divided by length. A rod of length {q1} carrying {q0} has a {name} density of {v}", True),
        ("xb_perlen_b", "Define the {name} density as {name} divided by length. A rod carrying {q0} with length {q1} has a {name} density of {v}", True),
    )),
    Relation("X1 times length", ("X", "L"), (1, 1), False, (
        ("xb_timeslen_a", "Define the quantity Q as {name} multiplied by length. With {q0} and a length of {q1}, Q equals {v}", True),
        ("xb_timeslen_b", "Define the quantity Q as {name} multiplied by length. With a length of {q1} and {q0}, Q equals {v}", True),
    )),
    Relation("X1 squared per time", ("X", "T"), (2, -1), False, (
        ("xb_sq_a", "Define the quantity Q as {name} squared divided by time. With {q0} and a time of {q1}, Q equals {v}", True),
        ("xb_sq_b", "Define the quantity Q as {name} squared divided by time. With a time of {q1} and {q0}, Q equals {v}", True),
    )),
]

# Definition-by-example ("arith") templates: the relation is given only as arithmetic over the
# quantities; no relation name, no verbal formula (E1.0b, red-team Attack 1).
ARITH_RELATIONS: list[Relation] = [
    Relation("arith L/T", ("L", "T"), (1, -1), False, (
        ("ar_div_a", "Compute: {q0} divided by {q1} equals {v}", False),
        ("ar_div_b", "{q0} / {q1} = {v}", False),
        ("ar_div_c", "Dividing {q0} by {q1} gives {v}", False),
    )),
    Relation("arith M*L", ("M", "L"), (1, 1), False, (
        ("ar_mul_a", "Compute: {q0} multiplied by {q1} equals {v}", False),
        ("ar_mul_b", "{q0} × {q1} = {v}", False),
        ("ar_mul_c", "Multiplying {q0} by {q1} gives {v}", False),
    )),
    Relation("arith M/L3", ("M", "L"), (1, -3), False, (
        ("ar_cube_a", "Compute: {q0} divided by the cube of {q1} equals {v}", False),
        ("ar_cube_b", "{q0} / ({q1})^3 = {v}", False),
    )),
    Relation("arith L2", ("L",), (2,), False, (
        ("ar_sq_a", "Compute: {q0} squared equals {v}", False),
        ("ar_sq_b", "({q0})^2 = {v}", False),
    )),
    Relation("arith T/M", ("T", "M"), (1, -1), False, (
        ("ar_div2_a", "Compute: {q0} divided by {q1} equals {v}", False),
        ("ar_div2_b", "{q0} / {q1} = {v}", False),
    )),
    Relation("arith M2*T", ("M", "T"), (2, 1), False, (
        ("ar_sqmul_a", "Compute: the square of {q0} multiplied by {q1} equals {v}", False),
        ("ar_sqmul_b", "({q0})^2 × {q1} = {v}", False),
    )),
    Relation("arith L*T", ("L", "T"), (1, 1), False, (
        ("ar_mul2_a", "Compute: {q0} multiplied by {q1} equals {v}", False),
        ("ar_mul2_b", "{q0} × {q1} = {v}", False),
    )),
]

CONDITIONS = ("FAM-NAMED", "FAM-UNNAMED", "INV-LEX", "INV-BASE")


def _surface_order(text: str) -> tuple[int, ...]:
    return tuple(int(m.group(1)) for m in re.finditer(r"\{q(\d)\}", text))


def _composition_order(rel: Relation) -> tuple[int, ...]:
    num = [i for i, e in enumerate(rel.exps) if e > 0]
    den = [i for i, e in enumerate(rel.exps) if e < 0]
    return tuple(num + den)


def _values(rng: random.Random, rel: Relation, integer_output: bool = False) -> tuple[list[int], float]:
    """Input values and the consistent output value. With integer_output, inputs are chosen so
    that the output is an integer (free-generation readout: non-integer outputs make models
    continue the digits instead of emitting a unit; Phase-0 anomaly A4)."""
    hi = 9 if max(abs(e) for e in rel.exps) >= 3 else 60
    vals = [rng.randint(2, hi) for _ in rel.slots]
    if len(rel.slots) == 2 and rel.exps == (1, -1):
        vals[1] = rng.randint(2, 12)
        vals[0] = vals[1] * rng.randint(2, 30)
    elif integer_output and len(rel.slots) == 2 and any(e < 0 for e in rel.exps):
        j = [i for i, e in enumerate(rel.exps) if e < 0][0]
        i = 1 - j
        k, p = -rel.exps[j], rel.exps[i]
        vals[j] = rng.randint(2, 3)
        base = vals[j] ** (-(-k // p))  # ceil(k/p)
        vals[i] = base * rng.randint(1, 4)
    out = 1.0
    for v, e in zip(vals, rel.exps):
        out *= float(v) ** e
    return vals, float(f"{out:.4g}")


def _slot_units(rng: random.Random, rel: Relation, condition: str, pool: LexemePool, name_dim: Dimension | None,
                x_unit: Unit | None) -> list[UnitExpr]:
    """One UnitExpr per slot."""
    exprs: list[UnitExpr] = []
    if condition.startswith("FAM") or condition == "ARITH-FAM":
        for s in rel.slots:
            exprs.append(pick_unit(rng, Dimension.parse(s), pool="natural"))
        return exprs
    if condition in ("INV-LEX", "ARITH-INV", "ARITH-INV-NODEF"):
        # every base symbol appearing in any slot gets one invented unit (shared across slots)
        syms: list[str] = []
        for s in rel.slots:
            for b in Dimension.parse(s).symbols():
                if b not in syms:
                    syms.append(b)
        inv = dict(zip(syms, pool.invented_units([Dimension({b: 1}) for b in syms])))
        for s in rel.slots:
            exprs.append(compose_from_base(rng, Dimension.parse(s), lambda b: inv[b]))
        return exprs
    if condition == "INV-BASE":
        for s in rel.slots:
            d = Dimension.parse(s)
            if d == name_dim:
                exprs.append(UnitExpr.of(x_unit))
            else:
                exprs.append(pick_unit(rng, d, pool="natural"))
        return exprs
    raise ValueError(condition)


def generate(n_per_condition: int, seed: int, *, conditions=CONDITIONS, styles=None, integer_output: bool = False) -> list[Item]:
    rng = random.Random(seed)
    pool = LexemePool(seed)
    items: list[Item] = []
    counter = 0
    for cond in conditions:
        if cond == "FAM-NAMED":
            rels = [r for r in RELATIONS if r.named]
        elif cond == "FAM-UNNAMED":
            rels = [r for r in RELATIONS if not r.named]
        elif cond == "INV-LEX":
            rels = list(RELATIONS)
        elif cond == "INV-BASE":
            rels = list(INV_BASE_RELATIONS)
        elif cond == "ARITH-FAM":
            rels = list(ARITH_RELATIONS)
        elif cond in ("ARITH-INV", "ARITH-INV-NODEF"):
            rels = list(ARITH_RELATIONS)
        else:
            raise ValueError(cond)
        for i in range(n_per_condition):
            rel = rels[i % len(rels)]
            tid, text, formula_given = rel.templates[rng.randrange(len(rel.templates))]
            name_dim = x_unit = qname = None
            if cond == "INV-BASE":
                name_dim, x_unit, qname = pool.invented_base("X")
            exprs = _slot_units(rng, rel, cond, pool, name_dim, x_unit)
            vals, vout = _values(rng, rel, integer_output=integer_output)
            invented = any(u.invented for ex in exprs for u in ex.units)
            style = sample_style(rng, invented=invented) if styles is None else rng.choice(styles)
            qs = [Quantity(float(v), ex) for v, ex in zip(vals, exprs)]
            rendered = {f"q{k}": q.render(style) for k, q in enumerate(qs)}
            # correct answer: product of slot units ^ exps
            correct = UnitExpr.of(*[(u, e * k) for ex, k in zip(exprs, rel.exps) for u, e in ex.factors]).canonical()
            lexemes = list(dict.fromkeys(u for ex in exprs for u in ex.units))
            distractors = lattice_neighbours(lexemes, correct, rng, k=8)
            cstr = correct.render(style, plural=vout != 1)
            dstr = select_distractor_renderings(cstr, [(d.render(style, plural=vout != 1), r) for d, r in distractors])
            if len(dstr) < 3:
                raise RuntimeError(f"not enough non-prefix distractors for {cstr}")
            cands, idx, roles = finalize_candidates(rng, cstr, dstr)
            prefix = "" if cond.endswith("NODEF") else definitions_prefix(lexemes)
            body = text.format(v=fmt_value(vout), name=qname or "", **rendered)
            prompt = (prefix + body).strip()
            surf, comp = _surface_order(text), _composition_order(rel)
            swapped = len(surf) > 1 and surf != comp
            d = correct.dim
            items.append(Item(
                item_id=f"T1-{cond}-{counter:05d}", task="T1", condition=cond, template_id=tid, prompt=prompt,
                candidates=cands, answer_index=idx, candidate_roles=roles, **dim_fields(d),
                unit_strings=correct.renderings(),
                meta=dict(relation=rel.name, formula_given=formula_given, order_swapped=swapped, style=style,
                          values=vals, out_value=vout, slot_units=[ex.render("symbol") for ex in exprs],
                          slot_dims=[str(ex.dim) for ex in exprs], invented=invented,
                          n_lexemes=len(lexemes), twin_of=None, x_quantity=qname),
            ))
            counter += 1
    return items


def make_familiar_twins(items: list[Item], seed: int) -> list[Item]:
    """For every invented item, build a familiar twin: same relation/template/values/style, real units."""
    rng = random.Random(seed + 7)
    twins: list[Item] = []
    rel_by_name = {r.name: r for r in RELATIONS + INV_BASE_RELATIONS + ARITH_RELATIONS}
    for it in items:
        if not it.meta.get("invented"):
            continue
        rel = rel_by_name[it.meta["relation"]]
        tid, text, formula_given = next(t for t in rel.templates if t[0] == it.template_id)
        # replace X1 slots with a real base dimension (mass) so the twin stays in the real lattice
        exprs = []
        for s in rel.slots:
            d = Dimension.parse(s)
            if "X" in d.symbols():
                d = Dimension.parse(s.replace("X", "M"))
            exprs.append(pick_unit(rng, d, pool="natural"))
        vals, vout = it.meta["values"], it.meta["out_value"]
        style = rng.choice(("symbol", "long", "slash", "per"))
        qs = [Quantity(float(v), ex) for v, ex in zip(vals, exprs)]
        rendered = {f"q{k}": q.render(style) for k, q in enumerate(qs)}
        correct = UnitExpr.of(*[(u, e * k) for ex, k in zip(exprs, rel.exps) for u, e in ex.factors]).canonical()
        lexemes = list(dict.fromkeys(u for ex in exprs for u in ex.units))
        distractors = lattice_neighbours(lexemes, correct, rng, k=8)
        cstr = correct.render(style, plural=vout != 1)
        dstr = select_distractor_renderings(cstr, [(d.render(style, plural=vout != 1), r) for d, r in distractors])
        if len(dstr) < 3:
            raise RuntimeError(f"not enough non-prefix distractors for {cstr}")
        cands, idx, roles = finalize_candidates(rng, cstr, dstr)
        body = text.format(v=fmt_value(vout), name="mass", **rendered)
        d = correct.dim
        twins.append(Item(
            item_id=it.item_id.replace("T1-", "T1-TWIN-"), task="T1", condition=it.condition + "-TWIN",
            template_id=tid, prompt=body.strip(), candidates=cands, answer_index=idx, candidate_roles=roles,
            **dim_fields(d), unit_strings=correct.renderings(),
            meta=dict(it.meta, style=style, slot_units=[ex.render("symbol") for ex in exprs],
                      slot_dims=[str(ex.dim) for ex in exprs], invented=False, twin_of=it.item_id, x_quantity=None),
        ))
    return twins
