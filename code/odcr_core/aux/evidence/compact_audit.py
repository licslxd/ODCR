"""Compact audit facade for the unified AI_analysis writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ai_analysis_writer import AIAnalysisWriteResult, get_writer


def write_compact_audit(name: str, body: str, *, repo_root: str | Path | None = None, **kwargs: Any) -> AIAnalysisWriteResult:
    return get_writer(repo_root).compact_audit(name, body, **kwargs)
