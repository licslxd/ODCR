"""Single writer surface for ODCR AI_analysis artifacts."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
AI_ANALYSIS_DIRS: Mapping[str, str] = {
    "root": ".",
    "index": "00_index",
    "raw_logs": "01_raw_logs",
    "search_hits": "02_search_hits",
    "evidence_ledgers": "03_evidence_ledgers",
    "phase_summaries": "04_phase_summaries",
    "final_reports": "05_final_reports",
}


def _base(base_dir: str | Path | None = None) -> Path:
    return Path(base_dir).resolve() if base_dir is not None else (REPO_ROOT / "AI_analysis").resolve()


def ensure_ai_analysis_tree(base_dir: str | Path | None = None) -> dict[str, Path]:
    base = _base(base_dir)
    paths = {"root": base}
    base.mkdir(parents=True, exist_ok=True)
    for key, rel in AI_ANALYSIS_DIRS.items():
        if key == "root":
            continue
        path = base / rel
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = path
    return paths


def _safe_name(name: str, *, suffix: str | None = None) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("AI_analysis artifact name must be non-empty")
    text = text.replace("\\", "/")
    if "/" in text or text in {".", ".."} or ".." in text:
        raise ValueError(f"AI_analysis artifact name must not contain path traversal: {name!r}")
    if suffix and not text.endswith(suffix):
        text += suffix
    return text


def atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(str(text), encoding="utf-8")
    tmp.replace(target)
    return target


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return atomic_write_text(path, json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write(kind: str, name: str, content: str, *, base_dir: str | Path | None = None, suffix: str = ".md") -> Path:
    paths = ensure_ai_analysis_tree(base_dir)
    if kind not in paths:
        raise KeyError(f"unknown AI_analysis artifact kind: {kind}")
    return atomic_write_text(paths[kind] / _safe_name(name, suffix=suffix), content)


def write_raw_log(name: str, content: str, *, base_dir: str | Path | None = None) -> Path:
    return _write("raw_logs", name, content, base_dir=base_dir, suffix=".log")


def write_search_hit(name: str, content: str, *, base_dir: str | Path | None = None) -> Path:
    return _write("search_hits", name, content, base_dir=base_dir, suffix=".md")


def write_ledger(name: str, content: str, *, base_dir: str | Path | None = None) -> Path:
    return _write("evidence_ledgers", name, content, base_dir=base_dir, suffix=".md")


def write_phase_summary(name: str, content: str, *, base_dir: str | Path | None = None) -> Path:
    return _write("phase_summaries", name, content, base_dir=base_dir, suffix=".md")


def write_final_report(name: str, content: str, *, base_dir: str | Path | None = None) -> Path:
    return _write("final_reports", name, content, base_dir=base_dir, suffix=".md")


def write_index(name: str, content: str, *, base_dir: str | Path | None = None) -> Path:
    return _write("index", name, content, base_dir=base_dir, suffix=".md")

