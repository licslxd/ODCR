"""Runtime diagnostic facade for AI_analysis raw-log JSON payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .ai_analysis_writer import AIAnalysisWriteResult, get_writer


def write_runtime_diagnostic(
    name: str,
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    **kwargs: Any,
) -> AIAnalysisWriteResult:
    return get_writer(repo_root).runtime_diagnostic(name, payload, **kwargs)
