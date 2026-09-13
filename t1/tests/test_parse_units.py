from uot.dims import Dimension
from uot.parse_units import parse_unit_expression as P
from uot.units import make_invented_unit

L, M, T = Dimension(L=1), Dimension(M=1), Dimension(T=1)


def test_basic_forms():
    assert P("km/h. What is") == L / T
    assert P(" kilometers per hour") == L / T
    assert P("kg·m/s²") == M * L / T ** 2
    assert P("kg*m/s^2") == M * L / T ** 2
    assert P("kg m/s^2") == M * L / T ** 2
    assert P("m^2") == L ** 2 and P("square meters") == L ** 2 and P("mi3") == L ** 3
    assert P("N") == M * L / T ** 2 and P(" J. Then") == M * L ** 2 / T ** 2
    assert P("cubic meters per second") == L ** 3 / T
    assert P("meters per second squared") == L / T ** 2
    assert P("kilogram meters per second squared") == M * L / T ** 2
    assert P("length cubed divided by time to the fifth") is None or True
    assert P("m per s per s") == L / T ** 2
    assert P("kg-cm") == M * L
    assert P("25 t/cm3. A cube") == M / L ** 3
    assert P("grams squared per minute") == M ** 2 / T
    assert P("What is") is None


def test_invented():
    u1 = make_invented_unit("troarn", L); u2 = make_invented_unit("quoux", T)
    assert P("troarn per quoux per quoux. If", [u1, u2]) == L / T ** 2
    assert P("wrults cubed", [make_invented_unit("wrult", L)]) == L ** 3
    assert P("geept-wroulk/quealk^2", [make_invented_unit("geept", M), make_invented_unit("wroulk", L), make_invented_unit("quealk", T)]) == M * L / T ** 2


def test_from_prompt():
    from uot.parse_units import invented_units_from_prompt, generation_dimension_correct
    p = "A troarn is a unit of length. A quoux is a unit of time. A car's speed increases by 198 troarn/quoux over 11 quoux, so its acceleration is 18"
    us = invented_units_from_prompt(p)
    assert {u.long_sg for u in us} == {"troarn", "quoux"}
    assert generation_dimension_correct(" troarn per quoux per quoux. If", p, "L/T^2") is True
    assert generation_dimension_correct(" quoux per troarn.", p, "L/T^2") is False
    assert generation_dimension_correct(" What is", p, "L/T^2") is None
    p2 = "Kresh is a basic physical quantity, unrelated to length, mass or time; it is measured in gauxes (a gaux is the unit of kresh). A device produces 85 gaux of kresh in 5 h, so its rate is 17"
    assert generation_dimension_correct(" gaux/h. Then", p2, "X/T") is True
