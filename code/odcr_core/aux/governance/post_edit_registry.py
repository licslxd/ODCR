"""Single source for post-edit scopes, reasons, and command planning policy."""

from __future__ import annotations

SCOPES = (
    "governance-fast",
    "docs",
    "governance",
    "config",
    "logging",
    "preprocess",
    "step3",
    "step4",
    "step5",
    "eval",
    "all",
)

BUSINESS_STAGE_SCOPES = ("preprocess", "step3", "step4", "step5", "eval")
SCOPE_ORDER = ("governance-fast", "governance", "config", "logging", *BUSINESS_STAGE_SCOPES, "all")
DEFAULT_SCOPE = "governance"
DEFAULT_MANUAL_MAX_SECONDS = 900
DEFAULT_HOOK_CHILD_MAX_SECONDS = 120
DEFAULT_WRAPPER_TIMEOUT_SECONDS = 180
MANUAL_ALL_FOLLOWUP_COMMAND = "python code/tools/odcr_post_edit_check.py --scope all --max-seconds 900"

SCOPE_REASONS: dict[str, str] = {
    "governance-fast": "fast/static governance path for hooks, docs, guardrail and post-edit tooling",
    "docs": "docs-only alias for the fast/static governance path",
    "governance": "governance contracts plus doctor; no stage execution",
    "config": "One-Control config/schema/resolver/runners checks plus governance contracts",
    "logging": "run-summary, tail, log-path and artifact-boundary checks",
    "preprocess": "preprocess contract/runtime unit checks with no real preprocess execution",
    "step3": "Step3 dry-run and contract tests only; no formal training",
    "step4": "Step4 entrypoint smoke, export validator, readiness and upstream contract tests only",
    "step5": "Step5 resolver dry-run, graph/loss and cache-admission tests only",
    "eval": "eval/rerank contract checks only; no real eval or rerank run",
    "all": "manual deep lightweight sweep across scopes; still no real training/eval/rerank",
}

LOGGING_SCOPE_PATH_HINTS = (
    "logging",
    "log_",
    "_log",
    "metrics",
    "cache",
    "report",
    "run_summary",
    "latest.json",
    "AI_analysis",
    "path_layout",
)

CROSS_STAGE_SCOPE_FILES = {
    "code/data_contract.py",
    "code/odcr_core/index_contract.py",
    "code/odcr_core/manifests.py",
    "code/odcr_core/training_checkpoint.py",
}


def validate_scope(scope: str) -> str:
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of: {', '.join(SCOPES)}")
    return scope


def scope_reason(scope: str) -> str:
    return SCOPE_REASONS.get(scope, "manual scope selected by caller")
