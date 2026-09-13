from .base import Item, items_to_jsonl, items_from_jsonl
from . import t1_unit_cloze, t2_consistency, t3_formula, t4_conversion, t5_error_spotting

GENERATORS = {
    "T1": t1_unit_cloze.generate,
    "T2": t2_consistency.generate,
    "T3": t3_formula.generate,
    "T4": t4_conversion.generate,
    "T5": t5_error_spotting.generate,
}

__all__ = ["Item", "GENERATORS", "items_to_jsonl", "items_from_jsonl"]
