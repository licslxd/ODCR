"""Single metadata registry for ODCR guardrails, post-edit scopes, and hook scope rules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


GUARDRAIL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("control-plane", ("R002", "R003", "R005", "R006", "R009", "R025", "R026", "R027", "R028")),
    ("data-contract", ("R008", "R024")),
    ("lineage-cache", ("R031", "R032", "R033", "R034", "R035", "R041")),
    ("ddp-loss", ("R036", "R037", "R038", "R040")),
    ("legacy-cleanup", ("R001", "R004", "R095")),
    (
        "step3-mainline",
        (
            "R010",
            "R011",
            "R029",
            "R097",
            "R099",
            "R100",
            "R101",
            "R102",
            "R103",
            "R104",
            "R105",
            "R106",
            "R107",
            "R108",
            "R109",
            "R110",
            "R111",
        ),
    ),
    ("stage-truth-upstream", ("R112", "R113")),
    ("step4-runtime-preflight", ("R114", "R115")),
    ("step4-evidence-level", ("R116",)),
    ("no-accum-architecture", ("R117",)),
    ("step4-rcr", ("R012", "R013", "R014", "R015")),
    ("step5-innovation", ("R016", "R017", "R018", "R019", "R020", "R021", "R022", "R023", "R030", "R039")),
    ("code-hygiene", ("R007",)),
    ("evolution-protocol", ("R042", "R043", "R044", "R045", "R046", "R047", "R048", "R049", "R050", "R096")),
    ("post-edit-workflow", ("R051", "R052", "R053", "R054", "R055", "R056", "R089")),
    ("run-summary-logging", ("R057", "R058", "R059", "R090", "R091")),
    ("p0-cache-hard-gates", ("R092", "R093", "R094", "R098")),
    ("logging-console-file", ("R060", "R061", "R062")),
    ("logging-artifact-evolution", ("R068", "R069", "R070", "R071", "R072")),
    ("logging-directory-boundaries", ("R078", "R079", "R080", "R081", "R082", "R083")),
    ("logging-old-layout-tail", ("R084", "R085", "R086", "R087", "R088")),
    ("post-edit-fast-path", ("R063", "R064", "R065", "R066", "R067", "R073", "R074", "R075", "R076", "R077")),
)

RULE_GROUP_BY_ID = {rule_id: group for group, rule_ids in GUARDRAIL_GROUPS for rule_id in rule_ids}

POST_EDIT_SCOPES = (
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
CROSS_STAGE_SCOPE_FILES = {
    "code/data_contract.py",
    "code/odcr_core/index_contract.py",
    "code/odcr_core/manifests.py",
    "code/odcr_core/training_checkpoint.py",
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


def scope_sort_key(scope: str) -> int:
    try:
        return SCOPE_ORDER.index(scope)
    except ValueError:
        return len(SCOPE_ORDER)


def _looks_like(rel_path: str, needles: Iterable[str]) -> bool:
    lowered = rel_path.lower()
    return any(needle in lowered for needle in needles)


def suggest_scope_for_paths(paths: Iterable[str]) -> str | None:
    normalized = [str(path).replace("\\", "/") for path in paths]
    if any(any(hint in path for hint in LOGGING_SCOPE_PATH_HINTS) for path in normalized):
        return "logging"
    return None


def hook_scope_for_path(rel_path: str) -> str | None:
    rel = rel_path.replace("\\", "/")
    name = Path(rel).name
    if rel.startswith("_archive/"):
        return None
    if rel in {"README.md", "AGENTS.md"} or rel.startswith("docs/"):
        return "governance-fast"
    if rel.startswith(".codex/"):
        return "governance-fast"
    if rel in CROSS_STAGE_SCOPE_FILES or _looks_like(rel, ("lineage", "cache_manifest", "checkpoint_lineage", "export_contract")):
        return "all"
    if rel in {
        "code/tools/check_one_control_guardrails.py",
        "code/tools/odcr_post_edit_check.py",
        "code/tests/test_one_control_guardrails.py",
        "code/tests/test_post_edit_check.py",
    }:
        return "governance-fast"
    if rel in {
        "code/odcr_core/logging_meta.py",
        "code/tests/test_run_summary_logging.py",
        "code/tests/test_logging_console_file.py",
        "code/train_logging.py",
    }:
        return "logging"
    if rel.startswith("configs/") or name in {
        "config_schema.py",
        "config_resolver.py",
        "config.py",
        "paths_config.py",
        "runners.py",
        "path_layout.py",
    }:
        return "config"
    if name in {
        "preprocess_data.py",
        "split_data.py",
        "combine_data.py",
        "compute_embeddings.py",
        "infer_domain_semantics.py",
        "preprocess_schema.py",
        "preprocess_runtime.py",
        "preprocess_status.py",
        "preprocess_registry.py",
    } or name.startswith("preprocess_") or _looks_like(rel, ("preprocess",)):
        return "preprocess"
    if name in {"step3_train_core.py", "step3_entry.py", "odcr_representation.py", "odcr_losses.py"} or _looks_like(rel, ("step3", "test_step3")):
        return "step3"
    if name in {"step4_engine.py", "step4_entry.py", "odcr_cf_routing.py", "step4_training_export.py"} or _looks_like(rel, ("step4", "test_step4", "test_index_contract")):
        return "step4"
    if name in {"step5_engine.py", "step5_entry.py", "step5_innovation.py", "step5_native_lora.py", "step5_word_losses.py"} or _looks_like(rel, ("step5", "test_step5")):
        return "step5"
    if _looks_like(rel, ("eval", "rerank", "bleu", "bert_score", "bertscore", "decode")):
        return "eval"
    return None

