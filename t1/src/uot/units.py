"""Unit registry, compound unit expressions, surface rendering, invented lexemes.

Ground truth for real units (dimension + SI scale) is taken from Pint and every
registry row is checked against it in ``tests/test_units.py``.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

from .dims import Dimension

# ----------------------------------------------------------------------------
# Unit
# ----------------------------------------------------------------------------

SYSTEMS = ("SI", "SI-prefixed", "cgs", "imperial", "archaic", "nautical", "astronomical", "named-derived", "invented")


@dataclass(frozen=True)
class Unit:
    id: str                      # canonical id, e.g. "kilometer"
    dim: Dimension
    scale: float                 # multiplicative factor to SI base units (1.0 for invented)
    short: tuple[str, ...]       # symbols, first is preferred ("km",)
    long_sg: str                 # "kilometer"
    long_pl: str                 # "kilometers"
    system: str = "SI"
    translations: dict = field(default_factory=dict, hash=False, compare=False)  # lang -> (sg, pl)
    named_derived: bool = False  # N, J, W, ... (a single lexeme for a composite dimension)
    invented: bool = False
    pint_name: str | None = None
    definition: str | None = None  # in-context definition sentence for invented units

    @property
    def symbol(self) -> str:
        return self.short[0]

    def forms(self, lang: str = "en") -> dict[str, str]:
        if lang == "en":
            return {"short": self.symbol, "long_sg": self.long_sg, "long_pl": self.long_pl}
        sg, pl = self.translations[lang]
        return {"short": self.symbol, "long_sg": sg, "long_pl": pl}

    def __repr__(self) -> str:
        return f"Unit({self.id}, {self.dim})"


# ----------------------------------------------------------------------------
# Registry table.  (id, pint_name, short symbols, long sg, long pl, system, translations)
# Translations: es / de / fr (sg, pl) where a natural word exists.
# ----------------------------------------------------------------------------

_T = lambda es=None, de=None, fr=None: {k: v for k, v in (("es", es), ("de", de), ("fr", fr)) if v}

_ROWS: list[tuple] = [
    # ---- length ----
    ("millimeter", "millimeter", ("mm",), "millimeter", "millimeters", "SI-prefixed", _T(("milímetro", "milímetros"), ("Millimeter", "Millimeter"), ("millimètre", "millimètres"))),
    ("centimeter", "centimeter", ("cm",), "centimeter", "centimeters", "SI-prefixed", _T(("centímetro", "centímetros"), ("Zentimeter", "Zentimeter"), ("centimètre", "centimètres"))),
    ("meter", "meter", ("m",), "meter", "meters", "SI", _T(("metro", "metros"), ("Meter", "Meter"), ("mètre", "mètres"))),
    ("kilometer", "kilometer", ("km",), "kilometer", "kilometers", "SI-prefixed", _T(("kilómetro", "kilómetros"), ("Kilometer", "Kilometer"), ("kilomètre", "kilomètres"))),
    ("micrometer", "micrometer", ("µm", "um"), "micrometer", "micrometers", "SI-prefixed", _T(("micrómetro", "micrómetros"), ("Mikrometer", "Mikrometer"), ("micromètre", "micromètres"))),
    ("inch", "inch", ("in",), "inch", "inches", "imperial", _T(("pulgada", "pulgadas"), ("Zoll", "Zoll"), ("pouce", "pouces"))),
    ("foot", "foot", ("ft",), "foot", "feet", "imperial", _T(("pie", "pies"), ("Fuß", "Fuß"), ("pied", "pieds"))),
    ("yard", "yard", ("yd",), "yard", "yards", "imperial", _T(("yarda", "yardas"), None, None)),
    ("mile", "mile", ("mi",), "mile", "miles", "imperial", _T(("milla", "millas"), ("Meile", "Meilen"), ("mille", "milles"))),
    ("nautical_mile", "nautical_mile", ("nmi",), "nautical mile", "nautical miles", "nautical", _T()),
    ("fathom", "fathom", ("ftm",), "fathom", "fathoms", "nautical", _T()),
    ("furlong", "furlong", ("fur",), "furlong", "furlongs", "archaic", _T()),
    ("league", "league", ("lea",), "league", "leagues", "archaic", _T()),
    ("rod", "rod", ("rd",), "rod", "rods", "archaic", _T()),
    ("chain", "chain", ("ch",), "chain", "chains", "archaic", _T()),
    ("angstrom", "angstrom", ("Å",), "angstrom", "angstroms", "cgs", _T()),
    ("parsec", "parsec", ("pc",), "parsec", "parsecs", "astronomical", _T()),
    ("light_year", "light_year", ("ly",), "light-year", "light-years", "astronomical", _T(("año luz", "años luz"), ("Lichtjahr", "Lichtjahre"), ("année-lumière", "années-lumière"))),
    ("astronomical_unit", "astronomical_unit", ("au", "AU"), "astronomical unit", "astronomical units", "astronomical", _T()),
    # ---- mass ----
    ("milligram", "milligram", ("mg",), "milligram", "milligrams", "SI-prefixed", _T(("miligramo", "miligramos"), ("Milligramm", "Milligramm"), ("milligramme", "milligrammes"))),
    ("gram", "gram", ("g",), "gram", "grams", "SI", _T(("gramo", "gramos"), ("Gramm", "Gramm"), ("gramme", "grammes"))),
    ("kilogram", "kilogram", ("kg",), "kilogram", "kilograms", "SI", _T(("kilogramo", "kilogramos"), ("Kilogramm", "Kilogramm"), ("kilogramme", "kilogrammes"))),
    ("tonne", "metric_ton", ("t",), "tonne", "tonnes", "SI-prefixed", _T(("tonelada", "toneladas"), ("Tonne", "Tonnen"), ("tonne", "tonnes"))),
    ("ounce", "ounce", ("oz",), "ounce", "ounces", "imperial", _T(("onza", "onzas"), ("Unze", "Unzen"), ("once", "onces"))),
    ("pound", "pound", ("lb",), "pound", "pounds", "imperial", _T(("libra", "libras"), ("Pfund", "Pfund"), ("livre", "livres"))),
    ("stone", "stone", ("st",), "stone", "stone", "imperial", _T()),
    ("grain", "grain", ("gr",), "grain", "grains", "archaic", _T()),
    ("slug", "slug", ("slug",), "slug", "slugs", "imperial", _T()),
    ("carat", "carat", ("ct",), "carat", "carats", "archaic", _T()),
    ("dalton", "dalton", ("Da",), "dalton", "daltons", "named-derived", _T()),
    # ---- time ----
    ("millisecond", "millisecond", ("ms",), "millisecond", "milliseconds", "SI-prefixed", _T(("milisegundo", "milisegundos"), ("Millisekunde", "Millisekunden"), ("milliseconde", "millisecondes"))),
    ("second", "second", ("s",), "second", "seconds", "SI", _T(("segundo", "segundos"), ("Sekunde", "Sekunden"), ("seconde", "secondes"))),
    ("minute", "minute", ("min",), "minute", "minutes", "SI", _T(("minuto", "minutos"), ("Minute", "Minuten"), ("minute", "minutes"))),
    ("hour", "hour", ("h", "hr"), "hour", "hours", "SI", _T(("hora", "horas"), ("Stunde", "Stunden"), ("heure", "heures"))),
    ("day", "day", ("d",), "day", "days", "SI", _T(("día", "días"), ("Tag", "Tage"), ("jour", "jours"))),
    ("week", "week", ("wk",), "week", "weeks", "SI", _T(("semana", "semanas"), ("Woche", "Wochen"), ("semaine", "semaines"))),
    ("fortnight", "fortnight", ("fn",), "fortnight", "fortnights", "archaic", _T()),
    ("year", "year", ("yr",), "year", "years", "SI", _T(("año", "años"), ("Jahr", "Jahre"), ("an", "ans"))),
    # ---- current / temperature / amount / luminosity ----
    ("ampere", "ampere", ("A",), "ampere", "amperes", "SI", _T(("amperio", "amperios"), ("Ampere", "Ampere"), ("ampère", "ampères"))),
    ("milliampere", "milliampere", ("mA",), "milliampere", "milliamperes", "SI-prefixed", _T()),
    ("kelvin", "kelvin", ("K",), "kelvin", "kelvins", "SI", _T()),
    ("mole", "mole", ("mol",), "mole", "moles", "SI", _T(("mol", "moles"), ("Mol", "Mol"), ("mole", "moles"))),
    ("candela", "candela", ("cd",), "candela", "candelas", "SI", _T()),
    # ---- named derived ----
    ("newton", "newton", ("N",), "newton", "newtons", "named-derived", _T(("newton", "newtons"), ("Newton", "Newton"), ("newton", "newtons"))),
    ("kilonewton", "kilonewton", ("kN",), "kilonewton", "kilonewtons", "named-derived", _T()),
    ("dyne", "dyne", ("dyn",), "dyne", "dynes", "cgs", _T()),
    ("pound_force", "force_pound", ("lbf",), "pound-force", "pounds-force", "imperial", _T()),
    ("kilogram_force", "kilogram_force", ("kgf",), "kilogram-force", "kilograms-force", "archaic", _T()),
    ("joule", "joule", ("J",), "joule", "joules", "named-derived", _T(("julio", "julios"), ("Joule", "Joule"), ("joule", "joules"))),
    ("kilojoule", "kilojoule", ("kJ",), "kilojoule", "kilojoules", "named-derived", _T()),
    ("erg", "erg", ("erg",), "erg", "ergs", "cgs", _T()),
    ("calorie", "calorie", ("cal",), "calorie", "calories", "archaic", _T(("caloría", "calorías"), ("Kalorie", "Kalorien"), ("calorie", "calories"))),
    ("kilowatt_hour", "kilowatt_hour", ("kWh",), "kilowatt-hour", "kilowatt-hours", "named-derived", _T()),
    ("electronvolt", "electron_volt", ("eV",), "electronvolt", "electronvolts", "named-derived", _T()),
    ("watt", "watt", ("W",), "watt", "watts", "named-derived", _T(("vatio", "vatios"), ("Watt", "Watt"), ("watt", "watts"))),
    ("kilowatt", "kilowatt", ("kW",), "kilowatt", "kilowatts", "named-derived", _T()),
    ("horsepower", "horsepower", ("hp",), "horsepower", "horsepower", "archaic", _T()),
    ("pascal", "pascal", ("Pa",), "pascal", "pascals", "named-derived", _T()),
    ("kilopascal", "kilopascal", ("kPa",), "kilopascal", "kilopascals", "named-derived", _T()),
    ("psi", "psi", ("psi",), "pound per square inch", "pounds per square inch", "imperial", _T()),
    ("bar", "bar", ("bar",), "bar", "bars", "named-derived", _T()),
    ("atmosphere", "atmosphere", ("atm",), "atmosphere", "atmospheres", "archaic", _T()),
    ("hertz", "hertz", ("Hz",), "hertz", "hertz", "named-derived", _T()),
    ("kilohertz", "kilohertz", ("kHz",), "kilohertz", "kilohertz", "named-derived", _T()),
    ("becquerel", "becquerel", ("Bq",), "becquerel", "becquerels", "named-derived", _T()),
    ("coulomb", "coulomb", ("C",), "coulomb", "coulombs", "named-derived", _T()),
    ("volt", "volt", ("V",), "volt", "volts", "named-derived", _T(("voltio", "voltios"), ("Volt", "Volt"), ("volt", "volts"))),
    ("ohm", "ohm", ("Ω", "ohm"), "ohm", "ohms", "named-derived", _T()),
    ("farad", "farad", ("F",), "farad", "farads", "named-derived", _T()),
    ("tesla", "tesla", ("T",), "tesla", "teslas", "named-derived", _T()),
    ("weber", "weber", ("Wb",), "weber", "webers", "named-derived", _T()),
    ("gray", "gray", ("Gy",), "gray", "grays", "named-derived", _T()),
    ("sievert", "sievert", ("Sv",), "sievert", "sieverts", "named-derived", _T()),
    ("knot", "knot", ("kn",), "knot", "knots", "nautical", _T()),
    ("liter", "liter", ("L", "l"), "liter", "liters", "SI", _T(("litro", "litros"), ("Liter", "Liter"), ("litre", "litres"))),
    ("milliliter", "milliliter", ("mL", "ml"), "milliliter", "milliliters", "SI-prefixed", _T(("mililitro", "mililitros"), ("Milliliter", "Milliliter"), ("millilitre", "millilitres"))),
    ("gallon", "gallon", ("gal",), "gallon", "gallons", "imperial", _T(("galón", "galones"), ("Gallone", "Gallonen"), ("gallon", "gallons"))),
    ("hectare", "hectare", ("ha",), "hectare", "hectares", "SI", _T(("hectárea", "hectáreas"), ("Hektar", "Hektar"), ("hectare", "hectares"))),
    ("acre", "acre", ("ac",), "acre", "acres", "imperial", _T()),
    # ---- additional named units (single-word lexemes) to densify derived lattice points ----
    ("british_thermal_unit", "british_thermal_unit", ("BTU",), "British thermal unit", "British thermal units", "imperial", _T()),
    ("therm", "therm", ("thm",), "therm", "therms", "imperial", _T()),
    ("foot_pound", "foot_pound", ("ft·lbf",), "foot-pound", "foot-pounds", "imperial", _T()),
    ("kilocalorie", "kilocalorie", ("kcal",), "kilocalorie", "kilocalories", "archaic", _T()),
    ("megajoule", "megajoule", ("MJ",), "megajoule", "megajoules", "named-derived", _T()),
    ("watt_hour", "watt_hour", ("Wh",), "watt-hour", "watt-hours", "named-derived", _T()),
    ("poundal", "poundal", ("pdl",), "poundal", "poundals", "archaic", _T()),
    ("kip", "kip", ("kip",), "kip", "kips", "imperial", _T()),
    ("meganewton", "meganewton", ("MN",), "meganewton", "meganewtons", "named-derived", _T()),
    ("megawatt", "megawatt", ("MW",), "megawatt", "megawatts", "named-derived", _T()),
    ("gigawatt", "gigawatt", ("GW",), "gigawatt", "gigawatts", "named-derived", _T()),
    ("milliwatt", "milliwatt", ("mW",), "milliwatt", "milliwatts", "named-derived", _T()),
    ("torr", "torr", ("Torr",), "torr", "torr", "archaic", _T()),
    ("millimeter_mercury", "millimeter_Hg", ("mmHg",), "millimetre of mercury", "millimetres of mercury", "archaic", _T()),
    ("inch_mercury", "inch_Hg", ("inHg",), "inch of mercury", "inches of mercury", "archaic", _T()),
    ("kilopound_per_square_inch", "kip_per_square_inch", ("ksi",), "kilopound per square inch", "kilopounds per square inch", "imperial", _T()),
    ("megapascal", "megapascal", ("MPa",), "megapascal", "megapascals", "named-derived", _T()),
    ("hectopascal", "hectopascal", ("hPa",), "hectopascal", "hectopascals", "named-derived", _T()),
    ("millibar", "millibar", ("mbar",), "millibar", "millibars", "named-derived", _T()),
    ("megahertz", "megahertz", ("MHz",), "megahertz", "megahertz", "named-derived", _T()),
    ("gigahertz", "gigahertz", ("GHz",), "gigahertz", "gigahertz", "named-derived", _T()),
    ("rpm", "revolutions_per_minute", ("rpm",), "revolution per minute", "revolutions per minute", "archaic", _T()),
    ("curie", "curie", ("Ci",), "curie", "curies", "archaic", _T()),
    ("mph", "mile_per_hour", ("mph",), "mile per hour", "miles per hour", "imperial", _T()),
    ("kph", "kilometer_per_hour", ("kph",), "kilometre per hour", "kilometres per hour", "SI-prefixed", _T()),
    ("are", "are", ("a",), "are", "ares", "archaic", _T()),
    ("barn", "barn", ("b",), "barn", "barns", "named-derived", _T()),
    ("square_mile", "square_mile", ("sq mi",), "square mile", "square miles", "imperial", _T()),
    ("pint", "pint", ("pt",), "pint", "pints", "imperial", _T()),
    ("quart", "quart", ("qt",), "quart", "quarts", "imperial", _T()),
    ("barrel", "barrel", ("bbl",), "barrel", "barrels", "imperial", _T()),
    ("bushel", "bushel", ("bu",), "bushel", "bushels", "imperial", _T()),
    ("cubic_centimeter", "cubic_centimeter", ("cc",), "cubic centimetre", "cubic centimetres", "SI-prefixed", _T()),
    ("teaspoon", "teaspoon", ("tsp",), "teaspoon", "teaspoons", "imperial", _T()),
    ("tablespoon", "tablespoon", ("tbsp",), "tablespoon", "tablespoons", "imperial", _T()),
    ("fluid_ounce", "fluid_ounce", ("fl oz",), "fluid ounce", "fluid ounces", "imperial", _T()),
    ("hectoliter", "hectoliter", ("hL",), "hectolitre", "hectolitres", "SI-prefixed", _T()),
    ("stokes", "stokes", ("St",), "stokes", "stokes", "cgs", _T()),
    ("poise", "poise", ("P",), "poise", "poise", "cgs", _T()),
    ("galileo", "galileo", ("Gal",), "galileo", "galileos", "cgs", _T()),
    ("sverdrup", "sverdrup", ("Sv_",), "sverdrup", "sverdrups", "named-derived", _T()),
    ("kilogray", "kilogray", ("kGy",), "kilogray", "kilograys", "named-derived", _T()),
    ("millisievert", "millisievert", ("mSv",), "millisievert", "millisieverts", "named-derived", _T()),
]


_NAMED_DERIVED_IDS = {
    "newton", "kilonewton", "dyne", "pound_force", "kilogram_force", "joule", "kilojoule", "erg", "calorie",
    "kilowatt_hour", "electronvolt", "watt", "kilowatt", "horsepower", "pascal", "kilopascal", "psi", "bar",
    "atmosphere", "hertz", "kilohertz", "becquerel", "coulomb", "volt", "ohm", "farad", "tesla", "weber",
    "gray", "sievert", "knot", "liter", "milliliter", "gallon", "hectare", "acre", "dalton",
    "british_thermal_unit", "therm", "foot_pound", "kilocalorie", "megajoule", "watt_hour", "poundal", "kip", "meganewton",
    "megawatt", "gigawatt", "milliwatt", "torr", "millimeter_mercury", "inch_mercury", "kilopound_per_square_inch", "megapascal",
    "hectopascal", "millibar", "megahertz", "gigahertz", "rpm", "curie", "mph", "kph", "are", "barn", "square_mile", "pint", "quart",
    "barrel", "bushel", "cubic_centimeter", "teaspoon", "tablespoon", "fluid_ounce", "hectoliter", "stokes", "poise", "galileo",
    "sverdrup", "kilogray", "millisievert",
}

# Base dimension -> "kind" word used in in-context definitions ("a unit of length")
KIND_WORDS = {
    "L": "length", "M": "mass", "T": "time", "I": "electric current", "Th": "temperature",
    "N": "amount of substance", "J": "luminous intensity",
}


class Registry:
    def __init__(self) -> None:
        self._units: dict[str, Unit] = {}
        self._by_dim: dict[Dimension, list[Unit]] = {}
        self._build_real()

    # ---- construction -------------------------------------------------
    def _build_real(self) -> None:
        import pint
        ureg = pint.UnitRegistry()
        for uid, pname, short, sg, pl, system, tr in _ROWS:
            q = ureg.Quantity(1, pname).to_base_units()
            dim = Dimension.from_pint(q.dimensionality)
            u = Unit(id=uid, dim=dim, scale=float(q.magnitude), short=tuple(short), long_sg=sg, long_pl=pl,
                     system=system, translations=tr, named_derived=uid in _NAMED_DERIVED_IDS, pint_name=pname)
            self.add(u)

    def add(self, u: Unit) -> Unit:
        if u.id in self._units:
            raise KeyError(f"duplicate unit id {u.id}")
        self._units[u.id] = u
        self._by_dim.setdefault(u.dim, []).append(u)
        return u

    # ---- lookup -------------------------------------------------------
    def __getitem__(self, uid: str) -> Unit:
        return self._units[uid]

    def __contains__(self, uid: str) -> bool:
        return uid in self._units

    def __len__(self) -> int:
        return len(self._units)

    def units(self, dim: Dimension | None = None, *, system: str | Iterable[str] | None = None,
              named_derived: bool | None = None, invented: bool | None = None) -> list[Unit]:
        us = list(self._units.values()) if dim is None else list(self._by_dim.get(dim, []))
        if system is not None:
            systems = {system} if isinstance(system, str) else set(system)
            us = [u for u in us if u.system in systems]
        if named_derived is not None:
            us = [u for u in us if u.named_derived == named_derived]
        if invented is not None:
            us = [u for u in us if u.invented == invented]
        return us

    def base_units(self, sym: str, **kw) -> list[Unit]:
        """Units whose dimension is exactly the base symbol ``sym``."""
        return self.units(Dimension({sym: 1}), **kw)

    def all_symbols(self) -> set[str]:
        s: set[str] = set()
        for u in self._units.values():
            s.update(u.short)
            s.add(u.long_sg)
            s.add(u.long_pl)
        return s


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    return Registry()


# ----------------------------------------------------------------------------
# Invented lexemes
# ----------------------------------------------------------------------------

_ONSETS = ["b", "bl", "br", "d", "dr", "fl", "fr", "g", "gl", "gr", "k", "kl", "kr", "m", "n", "p", "pl", "pr",
           "sk", "sl", "sn", "sp", "st", "str", "t", "tr", "v", "vr", "z", "th", "sh", "ch", "wr", "j", "qu", "y", "w", "h", "l", "r"]
_NUCLEI = ["a", "e", "i", "o", "u", "ai", "au", "ea", "ee", "oo", "ou", "oa"]
_CODAS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "ck", "ft", "lk", "lm", "lp", "lt",
          "mp", "nd", "nk", "nt", "pt", "rb", "rd", "rk", "rm", "rn", "rp", "rt", "sk", "sp", "st", "th", "sh", "ch", "x", "ng"]

_COMMON_ENGLISH = None


def _english_words() -> set[str]:
    """A small guard list: real English words and unit-ish strings we must avoid."""
    global _COMMON_ENGLISH
    if _COMMON_ENGLISH is None:
        words = set()
        from pathlib import Path
        for wl in (Path(__file__).resolve().parents[2] / "data" / "wordlists" / "words_alpha.txt", Path("/usr/share/dict/words")):
            try:
                with open(wl) as f:
                    words |= {w.strip().lower() for w in f if w.strip()}
            except OSError:
                pass
        words |= {"bar", "gram", "ton", "watt", "volt", "ohm", "mole", "foot", "inch", "yard", "mile", "stone",
                  "grain", "slug", "knot", "day", "week", "year", "hour", "gal", "pound", "ounce", "cup", "pint",
                  "quart", "dram", "chain", "rod", "link", "span", "hand", "palm", "ell", "pace", "step"}
        _COMMON_ENGLISH = words
    return _COMMON_ENGLISH


def pluralize(w: str) -> str:
    if re.search(r"(s|x|z|ch|sh)$", w):
        return w + "es"
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    return w + "s"


def generate_lexemes(n: int, rng: random.Random, *, avoid: Iterable[str] = (), min_len: int = 4, max_len: int = 6) -> list[str]:
    """Pronounceable CVC(C) nonsense words not in the English wordlist / unit namespace."""
    avoid_set = {a.lower() for a in avoid} | _english_words() | get_registry().all_symbols()
    out: list[str] = []
    seen: set[str] = set()
    tries = 0
    while len(out) < n and tries < 100000:
        tries += 1
        w = rng.choice(_ONSETS) + rng.choice(_NUCLEI) + rng.choice(_CODAS)
        if not (min_len <= len(w) <= max_len):
            continue
        if w in seen or w in avoid_set or pluralize(w) in avoid_set:
            continue
        # avoid words that are a real word plus 's' etc.
        if any(w.startswith(p) for p in ("meter", "metr", "gram", "sec", "min")):
            continue
        seen.add(w)
        out.append(w)
    if len(out) < n:
        raise RuntimeError("could not generate enough lexemes")
    return out


def make_invented_unit(lexeme: str, dim: Dimension, *, kind_word: str | None = None) -> Unit:
    """An invented unit of a (real or invented) dimension, defined in context."""
    if kind_word is None:
        if dim.distance() == 1 and len(dim.symbols()) == 1:
            kind_word = KIND_WORDS.get(dim.symbols()[0], dim.name() or str(dim))
        else:
            kind_word = dim.name() or str(dim)
    art = "an" if lexeme[0] in "aeiou" else "a"
    definition = f"{art.capitalize()} {lexeme} is a unit of {kind_word}."
    return Unit(id=f"inv:{lexeme}", dim=dim, scale=1.0, short=(lexeme,), long_sg=lexeme, long_pl=pluralize(lexeme),
                system="invented", invented=True, definition=definition)


def make_invented_base(quantity_lexeme: str, unit_lexeme: str, sym: str) -> tuple[Dimension, Unit]:
    """A new base dimension (e.g. 'flarn') with its unit (e.g. 'glorp')."""
    dim = Dimension({sym: 1})
    art = "an" if unit_lexeme[0] in "aeiou" else "a"
    definition = (f"{quantity_lexeme.capitalize()} is a basic physical quantity, unrelated to length, mass or time; "
                  f"it is measured in {pluralize(unit_lexeme)} ({art} {unit_lexeme} is the unit of {quantity_lexeme}).")
    u = Unit(id=f"inv:{unit_lexeme}", dim=dim, scale=1.0, short=(unit_lexeme,), long_sg=unit_lexeme,
             long_pl=pluralize(unit_lexeme), system="invented", invented=True, definition=definition)
    return dim, u


# ----------------------------------------------------------------------------
# Compound unit expressions and rendering
# ----------------------------------------------------------------------------

SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


@dataclass(frozen=True)
class UnitExpr:
    """Product of (Unit, exponent) factors, in a fixed order (numerator first)."""
    factors: tuple[tuple[Unit, int], ...]

    @classmethod
    def of(cls, *pairs: tuple[Unit, int] | Unit) -> "UnitExpr":
        fs = []
        for p in pairs:
            u, e = (p, 1) if isinstance(p, Unit) else p
            if e != 0:
                fs.append((u, int(e)))
        return cls(tuple(fs))

    @property
    def dim(self) -> Dimension:
        d = Dimension()
        for u, e in self.factors:
            d = d * (u.dim ** e)
        return d

    @property
    def units(self) -> tuple[Unit, ...]:
        return tuple(u for u, _ in self.factors)

    def canonical(self) -> "UnitExpr":
        """Merge duplicate units, numerators first, then denominators (stable)."""
        acc: dict[str, list] = {}
        order: list[str] = []
        for u, e in self.factors:
            if u.id not in acc:
                acc[u.id] = [u, 0]
                order.append(u.id)
            acc[u.id][1] += e
        fs = [(acc[i][0], acc[i][1]) for i in order if acc[i][1] != 0]
        num = [f for f in fs if f[1] > 0]
        den = [f for f in fs if f[1] < 0]
        return UnitExpr(tuple(num + den))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnitExpr):
            return False
        a = sorted((u.id, e) for u, e in self.canonical().factors)
        b = sorted((u.id, e) for u, e in other.canonical().factors)
        return a == b

    def __hash__(self) -> int:
        return hash(tuple(sorted((u.id, e) for u, e in self.canonical().factors)))

    def is_single(self) -> bool:
        return len(self.factors) == 1 and self.factors[0][1] == 1

    # ---- rendering ----------------------------------------------------
    def render(self, style: str = "symbol", lang: str = "en", plural: bool = True) -> str:
        """Render the compound unit.

        styles: 'symbol'  -> kg·m/s²   (unicode dot + superscripts)
                'ascii'   -> kg*m/s^2
                'slash'   -> kg m/s^2   (space product, caret powers)
                'per'     -> kg m per s^2  (symbols with 'per')
                'long'    -> kilogram meters per second squared  (words)
        Single units render as their symbol / long form.
        """
        ex = self.canonical()
        num = [(u, e) for u, e in ex.factors if e > 0]
        den = [(u, -e) for u, e in ex.factors if e < 0]
        if style == "long":
            return _render_long(num, den, lang=lang, plural=plural)

        def sym(u: Unit) -> str:
            return u.symbol

        if style == "symbol":
            def pw(u, e):
                return sym(u) + (str(e).translate(SUP) if e != 1 else "")
            n = "·".join(pw(u, e) for u, e in num) if num else "1"
            if not den:
                return n
            d = "·".join(pw(u, e) for u, e in den)
            if len(den) > 1:
                d = "(" + d + ")"
            return f"{n}/{d}"
        if style in ("ascii", "slash"):
            joiner = "*" if style == "ascii" else " "

            def pw(u, e):
                return sym(u) + (f"^{e}" if e != 1 else "")
            n = joiner.join(pw(u, e) for u, e in num) if num else "1"
            if not den:
                return n
            d = joiner.join(pw(u, e) for u, e in den)
            if len(den) > 1:
                d = "(" + d + ")"
            return f"{n}/{d}"
        if style == "per":
            def pw(u, e):
                return sym(u) + (f"^{e}" if e != 1 else "")
            n = " ".join(pw(u, e) for u, e in num) if num else "1"
            if not den:
                return n
            return n + " per " + " ".join(pw(u, e) for u, e in den)
        raise ValueError(style)

    def renderings(self, lang: str = "en") -> list[str]:
        """All acceptable renderings (for matching generated text)."""
        out = []
        for st in ("symbol", "ascii", "slash", "per", "long"):
            out.append(self.render(st, lang=lang))
        out.append(self.render("long", lang=lang, plural=False))
        return list(dict.fromkeys(out))


_POWER_WORDS = {2: "squared", 3: "cubed", 4: "to the fourth", 5: "to the fifth", 6: "to the sixth"}
_PREFIX_WORDS = {2: "square", 3: "cubic"}


def _word(u: Unit, lang: str, plural: bool) -> str:
    f = u.forms(lang)
    return f["long_pl"] if plural else f["long_sg"]


def _render_long(num, den, *, lang: str = "en", plural: bool = True) -> str:
    parts: list[str] = []
    for i, (u, e) in enumerate(num):
        last = i == len(num) - 1
        w = _word(u, lang, plural and last)  # only the last numerator word is pluralised: "kilogram meters"
        if e == 1:
            parts.append(w)
        elif e in _PREFIX_WORDS and u.dim == Dimension(L=1):
            parts.append(f"{_PREFIX_WORDS[e]} {w}")
        else:
            parts.append(f"{w} {_POWER_WORDS.get(e, 'to the power ' + str(e))}")
    s = " ".join(parts) if parts else ""
    if den:
        dparts = []
        for u, e in den:
            w = _word(u, lang, False)
            if e == 1:
                dparts.append(w)
            elif e in _PREFIX_WORDS and u.dim == Dimension(L=1):
                dparts.append(f"{_PREFIX_WORDS[e]} {w}")
            else:
                dparts.append(f"{w} {_POWER_WORDS.get(e, 'to the power ' + str(e))}")
        s = (s + " " if s else "") + "per " + " per ".join(dparts)
        if not parts:
            s = "inverse " + " per ".join(dparts) if len(dparts) == 1 and den[0][1] == 1 else s
    return s
