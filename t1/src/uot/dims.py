"""Dimension algebra: the free abelian group of exponent vectors.

A ``Dimension`` is an immutable map from base-dimension symbols to integer
exponents.  Real physics uses the SI-7 basis; invented base dimensions may be
added by name (e.g. ``"X1"``).  Multiplying dimensions adds exponent vectors.
"""
from __future__ import annotations

from typing import Iterable, Mapping

# SI base quantities: length, mass, time, current, temperature, amount, luminous intensity
BASE_SYMBOLS: tuple[str, ...] = ("L", "M", "T", "I", "Th", "N", "J")

_PINT_TO_SYMBOL = {
    "[length]": "L",
    "[mass]": "M",
    "[time]": "T",
    "[current]": "I",
    "[temperature]": "Th",
    "[substance]": "N",
    "[luminosity]": "J",
}


class Dimension:
    """Immutable exponent vector over a (possibly extended) basis."""

    __slots__ = ("_e",)

    def __init__(self, exps: Mapping[str, int] | None = None, **kw: int):
        e: dict[str, int] = {}
        for src in (exps or {}, kw):
            for k, v in src.items():
                if v == 0:
                    continue
                if not isinstance(v, int):
                    if float(v).is_integer():
                        v = int(v)
                    else:
                        raise ValueError(f"non-integer exponent {v} for {k}")
                e[k] = e.get(k, 0) + v
        self._e: tuple[tuple[str, int], ...] = tuple(sorted((k, v) for k, v in e.items() if v != 0))

    # ---- constructors -------------------------------------------------
    @classmethod
    def dimensionless(cls) -> "Dimension":
        return cls()

    @classmethod
    def from_pint(cls, dimensionality) -> "Dimension":
        """Build from a ``pint`` UnitsContainer of dimensionality."""
        e = {}
        for k, v in dict(dimensionality).items():
            if k not in _PINT_TO_SYMBOL:
                raise ValueError(f"unknown pint dimension {k}")
            e[_PINT_TO_SYMBOL[k]] = v
        return cls(e)

    @classmethod
    def parse(cls, s: str) -> "Dimension":
        """Parse strings like ``"L T^-2"`` or ``"M L2 T-2"`` or ``"L/T"``."""
        s = s.strip()
        if s in ("1", "", "dimensionless"):
            return cls()
        num, _, den = s.partition("/")
        e: dict[str, int] = {}

        def add(part: str, sign: int) -> None:
            part = part.replace("(", " ").replace(")", " ")
            for tok in part.replace("*", " ").replace("·", " ").split():
                tok = tok.replace("^", "")
                if tok == "1":
                    continue
                i = 0
                while i < len(tok) and (tok[i].isalpha()):
                    i += 1
                sym, exp = tok[:i], tok[i:]
                if not sym:
                    raise ValueError(f"bad token {tok!r} in {s!r}")
                e[sym] = e.get(sym, 0) + sign * (int(exp) if exp else 1)

        add(num, +1)
        if den:
            add(den, -1)
        return cls(e)

    # ---- algebra ------------------------------------------------------
    def as_dict(self) -> dict[str, int]:
        return dict(self._e)

    def __mul__(self, other: "Dimension") -> "Dimension":
        e = self.as_dict()
        for k, v in other._e:
            e[k] = e.get(k, 0) + v
        return Dimension(e)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        e = self.as_dict()
        for k, v in other._e:
            e[k] = e.get(k, 0) - v
        return Dimension(e)

    def __pow__(self, n: int) -> "Dimension":
        return Dimension({k: v * n for k, v in self._e})

    def inverse(self) -> "Dimension":
        return self ** -1

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Dimension) and self._e == other._e

    def __hash__(self) -> int:
        return hash(self._e)

    # ---- geometry -----------------------------------------------------
    def symbols(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self._e)

    def vector(self, basis: Iterable[str] = BASE_SYMBOLS) -> tuple[int, ...]:
        d = self.as_dict()
        return tuple(d.get(b, 0) for b in basis)

    def distance(self) -> int:
        """L1 lattice distance from the origin (dimensionless)."""
        return sum(abs(v) for _, v in self._e)

    def l1(self, other: "Dimension") -> int:
        return (self / other).distance()

    @property
    def is_dimensionless(self) -> bool:
        return not self._e

    # ---- display ------------------------------------------------------
    def __repr__(self) -> str:
        if not self._e:
            return "Dim(1)"
        return "Dim(" + " ".join(f"{k}{v if v != 1 else ''}" for k, v in self._e) + ")"

    def __str__(self) -> str:
        if not self._e:
            return "1"
        num = [f"{k}{'^' + str(v) if v != 1 else ''}" for k, v in self._e if v > 0]
        den = [f"{k}{'^' + str(-v) if v != -1 else ''}" for k, v in self._e if v < 0]
        s = " ".join(num) if num else "1"
        if den:
            s += "/" + (" ".join(den) if len(den) == 1 else "(" + " ".join(den) + ")")
        return s

    def name(self) -> str | None:
        """Conventional physical name for common lattice points (None if unnamed)."""
        return NAMED_POINTS.get(self)


# Conventional names for lattice points (SI-7 basis).  Used for
# lattice-distance/"named vs unnamed" bookkeeping, not for ground truth.
def _d(s: str) -> Dimension:
    return Dimension.parse(s)


NAMED_POINTS: dict[Dimension, str] = {
    _d("1"): "dimensionless",
    _d("L"): "length",
    _d("M"): "mass",
    _d("T"): "time",
    _d("I"): "current",
    _d("Th"): "temperature",
    _d("N"): "amount",
    _d("J"): "luminous intensity",
    _d("L2"): "area",
    _d("L3"): "volume",
    _d("L/T"): "speed",
    _d("L/T2"): "acceleration",
    _d("M L/T2"): "force",
    _d("M L2/T2"): "energy",
    _d("M L2/T3"): "power",
    _d("M/(L T2)"): "pressure",
    _d("M L/T"): "momentum",
    _d("1/T"): "frequency",
    _d("M/L3"): "density",
    _d("L3/T"): "volumetric flow rate",
    _d("M/T"): "mass flow rate",
    _d("I T"): "charge",
    _d("M L2/(T3 I)"): "voltage",
    _d("M L2/(T3 I2)"): "resistance",
    _d("M L2/T"): "action",
    _d("1/L"): "wavenumber",
    _d("L2/T"): "kinematic viscosity",
    _d("M/(L T)"): "dynamic viscosity",
    _d("M/T2"): "surface tension",
    _d("M L2/(T2 Th)"): "entropy",
    _d("L2/T2"): "specific energy",
    _d("M/L2"): "area density",
    _d("M/L"): "linear density",
    _d("T/L"): "slowness",
}
