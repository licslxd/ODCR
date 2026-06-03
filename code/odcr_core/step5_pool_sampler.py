"""Minimal Step4-to-retrieval pool contract constants.

The old Step5 sample-plan builder has been removed with the FLAN/T5 generator.
Step4 still imports these names to validate existing RCR pool artifacts, and
RACER-C1 consumes the train-only routing table/pools directly. No sampling,
prompt construction, or generator fallback lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


STEP5_POOL_MANIFEST_SCHEMA_VERSION = "odcr_step5_pool_manifest/1"
STEP5_POOL_SAMPLER_SCHEMA_VERSION = "odcr_racer_c1_pool_contract/1"
STEP5_SAMPLING_CONTRACT_SCHEMA_VERSION = "odcr_step5_sampling_contract/1"
STEP5_POOLS_DIRNAME = "step5_pools"
STEP5_POOL_MANIFEST = "step5_pool_manifest.json"
STEP5_SAMPLING_CONTRACT = "step5_sampling_contract.json"
STEP5_POOL_DISTRIBUTION_REPORT = "step5_pool_distribution_report.json"
STEP5_POOL_EXPORTS_STATUS = "step5_pool_exports_status.json"

_EXPLANATION_POOL_PREFIX = "step5_explanation"
_OLD_EXPLAINER_POOL_PREFIX = "step5" + "B"

POOL_NAMES: tuple[str, ...] = (
    f"{_EXPLANATION_POOL_PREFIX}_target_gold_anchor_high",
    f"{_EXPLANATION_POOL_PREFIX}_target_gold_anchor_medium",
    f"{_EXPLANATION_POOL_PREFIX}_aux_gold_anchor_high",
    f"{_EXPLANATION_POOL_PREFIX}_aux_gold_anchor_medium",
    f"{_EXPLANATION_POOL_PREFIX}_cf_explainer_high",
    f"{_EXPLANATION_POOL_PREFIX}_cf_explainer_medium",
    f"{_EXPLANATION_POOL_PREFIX}_cf_explainer_low_weighted",
    f"{_EXPLANATION_POOL_PREFIX}_cf_explainer_reject",
)
POOL_PARQUET_NAMES: dict[str, str] = {name: f"{name}.parquet" for name in POOL_NAMES}
LEGACY_POOL_ALIASES: dict[str, str] = {
    name: name.replace(_EXPLANATION_POOL_PREFIX, _OLD_EXPLAINER_POOL_PREFIX, 1) for name in POOL_NAMES
}

POOL_ALIAS_COLUMNS: dict[str, str] = {
    "cf_tier_step5_explanation": f"cf_tier_{_OLD_EXPLAINER_POOL_PREFIX}",
    "cf_tier_reason_step5_explanation": f"cf_tier_reason_{_OLD_EXPLAINER_POOL_PREFIX}",
    "cf_quality_score_step5_explanation": f"cf_quality_score_{_OLD_EXPLAINER_POOL_PREFIX}",
    "recommended_sampling_weight_step5_explanation": f"recommended_sampling_weight_{_OLD_EXPLAINER_POOL_PREFIX}",
}


def required_columns_with_legacy_aliases(required: Sequence[str], available: Sequence[str]) -> list[str]:
    """Return required columns missing from a pool, allowing old Step4 alias names.

    This is a read-only compatibility check for already-produced Step4 artifacts.
    It does not route execution back to the retired Step5 generator.
    """

    available_set = set(str(x) for x in available)
    missing: list[str] = []
    for column in required:
        name = str(column)
        alias = POOL_ALIAS_COLUMNS.get(name)
        if name not in available_set and (not alias or alias not in available_set):
            missing.append(name)
    return missing


def validate_step5_formal_sample_plan(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Fail-fast replacement for the retired Step5 sampler preflight."""

    return {
        "schema_version": STEP5_POOL_SAMPLER_SCHEMA_VERSION,
        "status": "retired",
        "ready": False,
        "message": "Old Step5 sampling is deleted; use RACER-C1 train-only evidence retrieval.",
    }


def validate_step5_formal_sample_plan_for_source(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return validate_step5_formal_sample_plan()


def load_step5_pool_manifest(run_dir: str | Path) -> Mapping[str, Any]:
    import json

    path = Path(run_dir) / STEP5_POOLS_DIRNAME / STEP5_POOL_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"Step4 pool manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Step4 pool manifest must be a JSON object: {path}")
    return payload


__all__ = [
    "LEGACY_POOL_ALIASES",
    "POOL_ALIAS_COLUMNS",
    "POOL_NAMES",
    "POOL_PARQUET_NAMES",
    "STEP5_POOL_DISTRIBUTION_REPORT",
    "STEP5_POOL_EXPORTS_STATUS",
    "STEP5_POOL_MANIFEST",
    "STEP5_POOL_MANIFEST_SCHEMA_VERSION",
    "STEP5_POOL_SAMPLER_SCHEMA_VERSION",
    "STEP5_POOLS_DIRNAME",
    "STEP5_SAMPLING_CONTRACT",
    "STEP5_SAMPLING_CONTRACT_SCHEMA_VERSION",
    "load_step5_pool_manifest",
    "required_columns_with_legacy_aliases",
    "validate_step5_formal_sample_plan",
    "validate_step5_formal_sample_plan_for_source",
]
