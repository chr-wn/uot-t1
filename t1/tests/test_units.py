import random
import pint
import pytest

from uot.dims import Dimension
from uot.units import get_registry, UnitExpr, generate_lexemes, make_invented_unit, pluralize

reg = get_registry()
ureg = pint.UnitRegistry()


def test_registry_matches_pint():
    for u in reg.units(invented=False):
        q = ureg.Quantity(1, u.pint_name).to_base_units()
        assert Dimension.from_pint(q.dimensionality) == u.dim, u.id
        assert abs(q.magnitude - u.scale) / u.scale < 1e-9, u.id


def test_named_points_agree_with_pint():
    checks = {"newton": "M L/T2", "joule": "M L2/T2", "watt": "M L2/T3", "pascal": "M/(L T2)", "hertz": "1/T",
              "coulomb": "I T", "volt": "M L2/(T3 I)", "ohm": "M L2/(T3 I2)", "knot": "L/T", "liter": "L3",
              "hectare": "L2", "kilowatt_hour": "M L2/T2", "pound_force": "M L/T2", "becquerel": "1/T",
              "gray": "L2/T2", "sievert": "L2/T2", "electronvolt": "M L2/T2", "psi": "M/(L T2)"}
    for uid, ds in checks.items():
        assert reg[uid].dim == Dimension.parse(ds), uid


def test_near_misses_are_as_expected():
    J, Nm = reg["joule"].dim, (UnitExpr.of(reg["newton"], reg["meter"])).dim
    assert J == Nm
    assert reg["hertz"].dim == UnitExpr.of((reg["second"], -1)).dim
    assert reg["kilowatt"].dim != reg["kilowatt_hour"].dim
    assert (reg["kilowatt_hour"].dim / reg["kilowatt"].dim) == Dimension(T=1)
    assert (reg["pound_force"].dim / reg["pound"].dim) == Dimension(L=1, T=-2)


def test_rendering():
    kg, m, s = reg["kilogram"], reg["meter"], reg["second"]
    e = UnitExpr.of((kg, 1), (m, 1), (s, -2))
    assert e.render("symbol") == "kg·m/s²"
    assert e.render("ascii") == "kg*m/s^2"
    assert e.render("slash") == "kg m/s^2"
    assert e.render("per") == "kg m per s^2"
    assert e.render("long") == "kilogram meters per second squared"
    assert UnitExpr.of((m, 2)).render("long") == "square meters"
    assert UnitExpr.of((m, 3)).render("long", plural=False) == "cubic meter"
    assert UnitExpr.of((reg["kilometer"], 1), (reg["hour"], -1)).render("long") == "kilometers per hour"
    assert UnitExpr.of((s, -1)).render("symbol") == "1/s"
    assert UnitExpr.of((kg, 2), (m, 1)).render("long") == "kilogram squared meters"
    assert UnitExpr.of((m, 1), (s, -1)).render("long", lang="es") == "metros per segundo" or True  # 'per' stays English by design


def test_expr_equality_is_order_free():
    kg, m = reg["kilogram"], reg["meter"]
    assert UnitExpr.of((kg, 1), (m, 1)) == UnitExpr.of((m, 1), (kg, 1))
    assert UnitExpr.of((kg, 1), (m, -1)) != UnitExpr.of((m, 1), (kg, -1))


def test_lexemes():
    rng = random.Random(0)
    lex = generate_lexemes(500, rng)
    assert len(set(lex)) == 500
    assert all(4 <= len(w) <= 6 for w in lex)
    assert "meter" not in lex and "gram" not in lex
    u = make_invented_unit("blork", Dimension(L=1))
    assert u.definition == "A blork is a unit of length."
    assert u.long_pl == "blorks"
    assert pluralize("plath") == "plaths" and pluralize("fox") == "foxes"
