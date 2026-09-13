"""Corpus-frequency instrument: infini-gram counts (Dolma 1.7 index) for unit strings.

Results are cached on disk (data/frequency/counts.json) so every string is queried once.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

API = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_dolma-v1_7_llama"


class FrequencyCache:
    def __init__(self, path: str | Path, index: str = DEFAULT_INDEX):
        self.path = Path(path)
        self.index = index
        self.counts: dict[str, int] = {}
        if self.path.exists():
            self.counts = json.load(open(self.path))

    def _query(self, s: str) -> int:
        for attempt in range(5):
            try:
                r = requests.post(API, json={"index": self.index, "query_type": "count", "query": s}, timeout=60)
                j = r.json()
                if "count" in j:
                    return int(j["count"])
                if "error" in j and "Invalid" in j["error"]:
                    raise RuntimeError(j["error"])
            except (requests.RequestException, ValueError):
                pass
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"infini-gram query failed for {s!r}")

    def count(self, s: str) -> int:
        s = s.strip()
        if s not in self.counts:
            self.counts[s] = self._query(s)
        return self.counts[s]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.counts, open(self.path, "w"), ensure_ascii=False, indent=0)

    def log_freq(self, strings: list[str]) -> float:
        """log10(1 + max count over renderings)."""
        import math
        return math.log10(1 + max(self.count(s) for s in strings))
