from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class BestEventKey:
    best_version: int
    model_sha256: str


class AsyncEvalDedup:
    def __init__(self, history_csv: Path) -> None:
        self._history_csv = history_csv
        self._seen: Dict[Tuple[int, str], bool] = {}
        self._load()

    def _load(self) -> None:
        if not self._history_csv.is_file():
            return
        for ln in self._history_csv.read_text(encoding="utf-8").splitlines()[1:]:
            cols = [x.strip() for x in ln.split(",")]
            if len(cols) < 2:
                continue
            try:
                key = (int(cols[0]), str(cols[1]))
            except ValueError:
                continue
            self._seen[key] = True

    def seen(self, key: BestEventKey) -> bool:
        return self._seen.get((int(key.best_version), str(key.model_sha256)), False)

    def mark(self, key: BestEventKey) -> None:
        self._seen[(int(key.best_version), str(key.model_sha256))] = True
