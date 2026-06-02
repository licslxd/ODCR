"""Schema file presence checks for ablation control artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from odcr_core.ablation.registry import AblationValidationError


SCHEMA_FILES = (
    "registry.schema.json",
    "ablation_manifest.schema.json",
    "result_snapshot.schema.json",
)


def validate_schema_files(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    schema_dir = root / "ablations" / "schemas"
    results = []
    for name in SCHEMA_FILES:
        path = schema_dir / name
        if not path.is_file():
            raise AblationValidationError(f"missing ablation schema: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AblationValidationError(f"invalid ablation schema JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise AblationValidationError(f"ablation schema must be a JSON object: {path}")
        for field in ("$schema", "type"):
            if field not in payload:
                raise AblationValidationError(f"{path} missing schema field {field}")
        results.append({"path": path.relative_to(root).as_posix(), "status": "present"})
    return {
        "schema_version": "odcr_ablation_schema_file_validation/1",
        "status": "pass",
        "schemas": results,
    }
