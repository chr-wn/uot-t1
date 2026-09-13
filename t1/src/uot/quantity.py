"""Quantities and their surface forms."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .dims import Dimension
from .units import Unit, UnitExpr


def fmt_value(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: UnitExpr

    @classmethod
    def of(cls, value: float, unit: Unit | UnitExpr) -> "Quantity":
        if isinstance(unit, Unit):
            unit = UnitExpr.of(unit)
        return cls(float(value), unit)

    @property
    def dim(self) -> Dimension:
        return self.unit.dim

    def render(self, style: str = "symbol", lang: str = "en", sep: str = " ") -> str:
        """style ∈ {symbol, ascii, slash, per, long}; long uses plural iff value != 1."""
        plural = self.value != 1
        u = self.unit.render(style, lang=lang, plural=plural)
        return f"{fmt_value(self.value)}{sep}{u}"


# Surface-form sampling ------------------------------------------------------

SURFACE_STYLES = ("symbol", "ascii", "slash", "per", "long")


def sample_style(rng: random.Random, *, allow: tuple[str, ...] = SURFACE_STYLES, invented: bool = False) -> str:
    """Invented units have no symbols distinct from their names; 'long'/'per' are natural for them."""
    if invented:
        allow = tuple(s for s in allow if s in ("per", "long", "slash"))
        if not allow:
            allow = ("long",)
    return rng.choice(allow)


def sample_value(rng: random.Random, lo: int = 2, hi: int = 99) -> int:
    return rng.randint(lo, hi)
