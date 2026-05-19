"""Runtime report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from odcr_core.aux.evidence.ai_analysis_writer import get_writer


def write_runtime_report(
    name: str,
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    stage: str | None = None,
    task: int | str | None = None,
    source: str = "runtime",
) -> Path:
    body = "# ODCR Runtime Report\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n```\n"
    result = get_writer(repo_root).final_report(name, body, source=source, stage=stage, task=task, validation_result=payload.get("success"))
    return result.path
