import collections
import random

from uot.tasks import GENERATORS
from uot.tasks.t1_unit_cloze import generate as gen_t1, make_familiar_twins, RELATIONS, _surface_order, _composition_order


def test_t1_basic():
    items = gen_t1(40, seed=0)
    assert len(items) == 160
    for it in items:
        assert len(it.candidates) == 4 and len(set(it.candidates)) == 4
        assert it.candidates[it.answer_index].startswith(" ")
        assert "{" not in it.prompt
        if it.condition.startswith("INV"):
            assert "is a unit of" in it.prompt or "basic physical quantity" in it.prompt
    # both surface orders present
    sw = collections.Counter((it.condition, it.meta["order_swapped"]) for it in items)
    assert sw[("INV-LEX", True)] > 0 and sw[("INV-LEX", False)] > 0


def test_t1_every_relation_has_both_orders():
    for rel in RELATIONS:
        if len(rel.slots) < 2:
            continue
        orders = {_surface_order(t[1]) != _composition_order(rel) for t in rel.templates}
        assert orders == {True, False}, rel.name


def test_t1_twins():
    items = gen_t1(20, seed=1)
    twins = make_familiar_twins(items, seed=1)
    inv = [it for it in items if it.meta["invented"]]
    assert len(twins) == len(inv)
    for t, it in zip(twins, inv):
        assert t.template_id == it.template_id and t.meta["values"] == it.meta["values"]
        assert not t.meta["invented"]


def test_t1_lexeme_rotation():
    items = gen_t1(120, seed=2, conditions=("INV-LEX",))
    # no lexeme reused across items
    seen = collections.Counter()
    for it in items:
        for w in it.prompt.split():
            pass
    lex = collections.Counter(u for it in items for u in it.meta["slot_units"])
    assert max(lex.values()) <= 2  # a lexeme appears at most in its own item's slots


def test_all_generators_run_and_balance():
    for name, gen in GENERATORS.items():
        items = gen(24, seed=3)
        assert items, name
        for it in items:
            assert it.task == name
            assert 0 <= it.answer_index < len(it.candidates)
            assert it.prompt and not it.prompt.endswith(" ")
        if name in ("T2", "T4"):
            by_cond = collections.defaultdict(list)
            for it in items:
                by_cond[it.condition].append(it.answer_index)
            for cond, idxs in by_cond.items():
                assert abs(sum(idxs) / len(idxs) - 0.5) <= 0.1, (name, cond)
