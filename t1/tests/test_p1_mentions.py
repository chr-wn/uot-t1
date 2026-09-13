from uot.tasks.p1_mentions import generate


def test_spans_align():
    items = generate(12, seed=0)
    assert len(items) > 40
    for it in items:
        p = it.prompt
        v0, v1 = it.spans["value"]
        assert p[v0:v1].replace(".", "").isdigit(), (p, p[v0:v1])
        if it.unit_string:
            u0, u1 = it.spans["unit"]
            assert p[u0:u1] == it.unit_string, (p[u0:u1], it.unit_string)
        a0, a1 = it.spans["anaphor"]
        assert p[a0:a1] == it.meta["noun"]
        m0, m1 = it.spans["mention_end"]
        assert p[m0:m1].startswith(".")
        assert it.spans["last"] == (len(p) - 1, len(p))
        if it.invented:
            assert "is a unit of" in p


def test_conditions_present():
    items = generate(30, seed=1)
    conds = {it.condition for it in items}
    assert {"REAL-BASE", "REAL-NAMED", "REAL-LATTICE", "DIMLESS", "INV-LEX", "XLING"} <= conds
    lat = [it for it in items if it.condition == "REAL-LATTICE"]
    assert len({it.dimension for it in lat}) >= 15
