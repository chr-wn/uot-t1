"""Per-stimulus lexeme allocation: draws invented lexemes and rotates them across dimensions.

Every item gets fresh lexemes; across a stimulus set no lexeme is bound to a single dimension
more often than chance (checked in tests).
"""
from __future__ import annotations

import random
from typing import Sequence

from .dims import Dimension
from .units import Unit, generate_lexemes, make_invented_unit, make_invented_base

_QUANTITY_NAMES = ["flarn", "quell", "brasp", "tholm", "vrindle", "skoop", "plurn", "gravix", "yarnel", "dobbin",
                   "kresh", "molvane", "tarrick", "zephod", "wumble", "orrin", "spandle", "caddox", "ferrule", "nubbin"]


class LexemePool:
    """Deterministic pool of nonsense lexemes for a stimulus set."""

    def __init__(self, seed: int, n: int = 4000):
        self.rng = random.Random(seed)
        self.pool = generate_lexemes(n, self.rng)
        self.i = 0

    def take(self, k: int) -> list[str]:
        if self.i + k > len(self.pool):
            raise RuntimeError("lexeme pool exhausted")
        out = self.pool[self.i:self.i + k]
        self.i += k
        return out

    def invented_units(self, dims: Sequence[Dimension]) -> list[Unit]:
        lex = self.take(len(dims))
        return [make_invented_unit(l, d) for l, d in zip(lex, dims)]

    def invented_base(self, sym: str = "X1") -> tuple[Dimension, Unit, str]:
        qname = self.rng.choice(_QUANTITY_NAMES)
        (ulex,) = self.take(1)
        dim, u = make_invented_base(qname, ulex, sym)
        return dim, u, qname
