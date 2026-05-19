"""ODCR one-control runtime layout.

The user-facing control plane now writes run artifacts to
``runs/{stage}/task{T}/{run_id}/`` for task stages and
``runs/preprocess/{unit}/{run_id}/`` for preprocess units.  Logs and manifests
live under each run's ``meta/`` directory; data remains outside ``runs/``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from odcr_core import run_naming


def _safe_namespace_component(value: str, *, label: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} must be non-empty")
    if raw in {".", ".."} or "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError(f"invalid {label}: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw):
        raise ValueError(f"invalid {label}: {value!r}")
    return raw


@dataclass(frozen=True)
class ArtifactRoleSpec:
    role: str
    default_directory: str
    filename_convention: str
    producer: str
    consumer: str
    retention_note: str
    ai_analysis_may_copy: bool


_ARTIFACT_ROLE_SPECS: dict[str, ArtifactRoleSpec] = {
    "run_meta": ArtifactRoleSpec(
        role="run_meta",
        default_directory="runs/<stage>/<unit>/<run_id>/meta",
        filename_convention="directory containing canonical run metadata files",
        producer="./odcr or python code/odcr.py launcher",
        consumer="doctor/show/latest handoff, humans, downstream validators",
        retention_note="keep with the formal run; do not place reusable cache payloads here",
        ai_analysis_may_copy=False,
    ),
    "console_log": ArtifactRoleSpec(
        role="console_log",
        default_directory="runs/<stage>/<unit>/<run_id>/meta",
        filename_convention="console.log",
        producer="run logging layer",
        consumer="human console replay, run_summary",
        retention_note="compact summary log only",
        ai_analysis_may_copy=False,
    ),
    "full_log": ArtifactRoleSpec(
        role="full_log",
        default_directory="runs/<stage>/<unit>/<run_id>/meta",
        filename_convention="full.log",
        producer="run logging layer and child process capture",
        consumer="debugging, postmortem, stage owner",
        retention_note="full run detail stays in run meta; AI handoff may cite paths or excerpts",
        ai_analysis_may_copy=False,
    ),
    "errors_log": ArtifactRoleSpec(
        role="errors_log",
        default_directory="runs/<stage>/<unit>/<run_id>/meta",
        filename_convention="errors.log",
        producer="run logging layer",
        consumer="doctor, postmortem, handoff digest",
        retention_note="warnings/errors only; AI_analysis should store digests rather than mirror the file",
        ai_analysis_may_copy=False,
    ),
    "metrics": ArtifactRoleSpec(
        role="metrics",
        default_directory="runs/<stage>/<unit>/<run_id>/meta or eval/rerank run root",
        filename_convention="metrics.jsonl, eval_metrics.json, rerank_summary.json, epoch_summary.csv, loss_breakdown.jsonl, timing_profile.jsonl, gpu_profile.jsonl, rcr_distribution.json",
        producer="stage/eval/rerank metric writers",
        consumer="summaries, analysis packs, baselines, humans",
        retention_note="small structured metrics may be retained with the producing run",
        ai_analysis_may_copy=True,
    ),
    "lineage": ArtifactRoleSpec(
        role="lineage",
        default_directory="runs/<stage>/<unit>/<run_id>/state or meta",
        filename_convention="checkpoint_lineage.json or stage_status.json",
        producer="checkpoint/export/cache writers",
        consumer="reuse gates, downstream stages, doctor",
        retention_note="required for fail-fast reuse decisions",
        ai_analysis_may_copy=True,
    ),
    "manifest": ArtifactRoleSpec(
        role="manifest",
        default_directory="runs/<stage>/<unit>/<run_id>/meta or cache/<producer>/<cache_key>",
        filename_convention="manifest.json or cache_manifest.json",
        producer="run/cache/export writers",
        consumer="doctor, reuse gates, downstream stages",
        retention_note="structured metadata only; large payloads stay outside manifests",
        ai_analysis_may_copy=True,
    ),
    "cache": ArtifactRoleSpec(
        role="cache",
        default_directory="cache/<producer>/<cache_key>",
        filename_convention="manifest.json plus reusable shards/payloads",
        producer="preprocess/tokenize/embed cache writers",
        consumer="same-stage cache readers with lineage/fingerprint validation",
        retention_note="reusable and evictable; never a full run log sink",
        ai_analysis_may_copy=False,
    ),
    "ai_analysis": ArtifactRoleSpec(
        role="ai_analysis",
        default_directory="AI_analysis/<bucket>",
        filename_convention="phase summaries, ledgers, final reports, search hits, compact raw audit logs",
        producer="Codex/AI-assisted workflow",
        consumer="future AI handoff, reviewers, governance audits",
        retention_note="do not mirror full training logs; store digests, excerpts, key paths, and evidence ledgers",
        ai_analysis_may_copy=True,
    ),
    "data_artifact": ArtifactRoleSpec(
        role="data_artifact",
        default_directory="data/<dataset>",
        filename_convention="processed.csv, train.csv, valid.csv, test.csv, profile/domain npy outputs",
        producer="preprocess data producers",
        consumer="preprocess B/C and training/eval stages",
        retention_note="canonical data contract outputs only; no logs",
        ai_analysis_may_copy=False,
    ),
    "merged_artifact": ArtifactRoleSpec(
        role="merged_artifact",
        default_directory="merged/<task>",
        filename_convention="aug_train.csv, aug_valid.csv",
        producer="preprocess combine stage",
        consumer="training/eval stages",
        retention_note="canonical merged data contract outputs only; no logs",
        ai_analysis_may_copy=False,
    ),
}

_METRIC_FILENAMES: dict[str, str] = {
    "metrics": "metrics.jsonl",
    "epoch_summary": "epoch_summary.csv",
    "loss_breakdown": "loss_breakdown.jsonl",
    "timing_profile": "timing_profile.jsonl",
    "gpu_profile": "gpu_profile.jsonl",
    "scheduler_events": "scheduler_events.jsonl",
    "damping_events": "damping_events.jsonl",
    "objective_drift": "objective_drift.jsonl",
    "recovery_events": "recovery_events.jsonl",
    "training_effectiveness": "training_effectiveness.jsonl",
    "training_effectiveness_summary": "training_effectiveness_summary.json",
    "loss_component_epoch_summary": "loss_component_epoch_summary.csv",
    "loss_component_trends": "loss_component_trends.json",
    "component_contribution_summary": "component_contribution_summary.md",
    "step3_eval_status": "step3_eval_status.json",
    "samples": "samples.jsonl",
    "collapse_stats": "collapse_stats.json",
    "eval_summary": "eval_summary.json",
    "eval_protocol": "eval_protocol.json",
    "sample_integrity_report": "sample_integrity_report.json",
    "quality_audit": "quality_audit.json",
    "rcr_distribution": "rcr_distribution.json",
    "eval_metrics": "eval_metrics.json",
    "rerank_summary": "rerank_summary.json",
    "data_audit": "data_audit.json",
    "data_audit_summary": "data_audit_summary.csv",
}


def artifact_role_registry() -> Mapping[str, ArtifactRoleSpec]:
    return dict(_ARTIFACT_ROLE_SPECS)


def get_artifact_role(role: str) -> ArtifactRoleSpec:
    key = str(role).strip()
    try:
        return _ARTIFACT_ROLE_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"unknown artifact role {role!r}; register it before writing artifacts") from exc


def validate_artifact_role_spec(spec: ArtifactRoleSpec) -> ArtifactRoleSpec:
    required = {
        "role": spec.role,
        "default_directory": spec.default_directory,
        "filename_convention": spec.filename_convention,
        "producer": spec.producer,
        "consumer": spec.consumer,
        "retention_note": spec.retention_note,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"artifact role spec is incomplete for {spec.role!r}: {missing}")
    return spec


def register_artifact_role(
    registry: Mapping[str, ArtifactRoleSpec],
    spec: ArtifactRoleSpec,
) -> dict[str, ArtifactRoleSpec]:
    checked = validate_artifact_role_spec(spec)
    out = dict(registry)
    out[checked.role] = checked
    return out


def metrics_filename(kind: str) -> str:
    key = str(kind).strip()
    try:
        return _METRIC_FILENAMES[key]
    except KeyError as exc:
        raise KeyError(f"unknown metrics filename kind {kind!r}; register a role/filename before writing it") from exc


def eval_metrics_filename(*, rerank: bool = False) -> str:
    return metrics_filename("rerank_summary" if rerank else "eval_metrics")


def eval_metrics_path(run_dir: Path, *, rerank: bool = False) -> Path:
    return Path(run_dir).expanduser().resolve() / eval_metrics_filename(rerank=rerank)


def runs_root(repo_root: Path) -> Path:
    return (repo_root / "runs").resolve()


def cache_root(repo_root: Path) -> Path:
    return (repo_root / "cache").resolve()


def preprocess_cache_root(repo_root: Path, unit: str) -> Path:
    key = str(unit).strip().lower()
    if key in ("b", "preprocess_b"):
        return cache_root(repo_root) / "preprocess_b"
    if key in ("c", "preprocess_c"):
        return cache_root(repo_root) / "preprocess_c"
    raise ValueError(f"preprocess cache unit must be b/c, got {unit!r}")


def preprocess_cache_entry_dir(repo_root: Path, unit: str, cache_key: str) -> Path:
    key = str(cache_key).strip()
    if not key:
        raise ValueError("cache_key must be non-empty")
    return preprocess_cache_root(repo_root, unit) / key


def get_global_iteration_root(repo_root: Path, iteration_id: str) -> Path:
    """Compatibility helper for cross-task metadata."""
    it = run_naming.normalize_iteration_id(iteration_id)
    return runs_root(repo_root) / "global" / it


def get_global_meta_dir(repo_root: Path, iteration_id: str) -> Path:
    """跨任务全局 meta：如 ``eval_registry_all.*``、多任务 ``shell_logs``。"""
    return get_global_iteration_root(repo_root, iteration_id) / "meta"


def get_task_root(repo_root: Path, task_id: int) -> Path:
    return runs_root(repo_root) / f"task{int(task_id)}"


def get_stage_task_root(repo_root: Path, stage: str, task_id: int) -> Path:
    canonical = {
        "train_step3": "step3",
        "train_step4": "step4",
        "train_step5": "step5",
        "eval-rerank": "rerank",
    }.get(stage, stage)
    return runs_root(repo_root) / canonical / f"task{int(task_id)}"


def get_preprocess_unit_root(repo_root: Path, unit: str) -> Path:
    return runs_root(repo_root) / "preprocess" / str(unit).lower()


def get_iteration_root(repo_root: Path, task_id: int, iteration_id: str) -> Path:
    _ = run_naming.normalize_iteration_id(iteration_id)
    return get_task_root(repo_root, task_id)


def get_task_meta_dir(repo_root: Path, task_id: int, iteration_id: str) -> Path:
    """Task-level meta outside a single run."""
    return get_iteration_root(repo_root, task_id, iteration_id) / "meta"


def get_multiseed_root(repo_root: Path, task_id: int, iteration_id: str, run_id: str) -> Path:
    """Task-level multi-seed metadata."""
    rid = run_naming.parse_run_id(run_id)
    return get_task_meta_dir(repo_root, task_id, iteration_id) / "multi_seed" / rid


def get_train_step3_run_root(
    repo_root: Path, task_id: int, iteration_id: str, run_id: str
) -> Path:
    return get_stage_task_root(repo_root, "step3", task_id) / run_id


def get_train_step5_run_root(
    repo_root: Path, task_id: int, iteration_id: str, run_id: str
) -> Path:
    return get_stage_task_root(repo_root, "step5", task_id) / run_id


def get_train_step4_run_root(
    repo_root: Path, task_id: int, iteration_id: str, run_id: str
) -> Path:
    return get_stage_task_root(repo_root, "step4", task_id) / run_id


def get_eval_run_root(repo_root: Path, task_id: int, iteration_id: str, run_id: str) -> Path:
    return get_stage_task_root(repo_root, "eval", task_id) / run_id


def get_rerank_run_root(repo_root: Path, task_id: int, iteration_id: str, run_id: str) -> Path:
    return get_stage_task_root(repo_root, "rerank", task_id) / run_id


def get_matrix_run_root(repo_root: Path, task_id: int, iteration_id: str, run_id: str) -> Path:
    return get_stage_task_root(repo_root, "matrix", task_id) / run_id


def get_baselines_root(repo_root: Path, task_id: int, iteration_id: str) -> Path:
    """``runs/task{T}/vN/baselines/``：注册基线、默认基线索引与 metrics 快照（不修改原 eval 目录）。"""
    return get_iteration_root(repo_root, task_id, iteration_id) / "baselines"


def get_analysis_root(repo_root: Path, task_id: int, iteration_id: str) -> Path:
    return get_iteration_root(repo_root, task_id, iteration_id) / "analysis"


def get_analysis_pack_root(
    repo_root: Path, task_id: int, iteration_id: str, pack_id: str
) -> Path:
    return get_analysis_root(repo_root, task_id, iteration_id) / pack_id


def get_stage_run_root(
    repo_root: Path,
    task_id: int,
    iteration_id: str,
    stage_name: str,
    run_id: str,
) -> Path:
    """
    stage_name: step3 | step4 | step5 | train_step3 | train_step4 | train_step5 | eval | rerank | matrix
    """
    if stage_name in ("step3", "train_step3"):
        return get_train_step3_run_root(repo_root, task_id, iteration_id, run_id)
    if stage_name in ("step4", "train_step4"):
        return get_train_step4_run_root(repo_root, task_id, iteration_id, run_id)
    if stage_name in ("step5", "train_step5"):
        return get_train_step5_run_root(repo_root, task_id, iteration_id, run_id)
    if stage_name == "eval":
        return get_eval_run_root(repo_root, task_id, iteration_id, run_id)
    if stage_name == "rerank":
        return get_rerank_run_root(repo_root, task_id, iteration_id, run_id)
    if stage_name == "matrix":
        return get_matrix_run_root(repo_root, task_id, iteration_id, run_id)
    raise ValueError(f"未知 stage_name: {stage_name!r}")


def best_model_path(stage_run_root: Path) -> Path:
    """Step3 / Step5 共用 canonical 权重路径 ``model/best.pth``。"""
    return (stage_run_root / "model" / "best.pth").resolve()


def model_file_path(stage_run_root: Path) -> Path:
    """与 :func:`best_model_path` 同义（唯一模型权重文件）。"""
    return best_model_path(stage_run_root)


def logs_dir(stage_run_root: Path) -> Path:
    return (stage_run_root / "meta").resolve()


def state_dir(stage_run_root: Path) -> Path:
    return (stage_run_root / "state").resolve()


def hf_cache_root(repo_root: Path, task_id: int) -> Path:
    return (repo_root / "cache" / f"task{int(task_id)}" / "hf").resolve()


def step3_tokenizer_cache_root(repo_root: Path, formal_cache_namespace: str) -> Path:
    namespace = str(formal_cache_namespace or "").strip().strip("/")
    if not namespace:
        raise ValueError("step3 tokenizer formal_cache_namespace must be non-empty")
    if Path(namespace).is_absolute() or ".." in Path(namespace).parts:
        raise ValueError(f"invalid step3 tokenizer formal_cache_namespace: {formal_cache_namespace!r}")
    return (repo_root / namespace).resolve()


def step3_tokenizer_cache_entry_dir(
    repo_root: Path,
    *,
    formal_cache_namespace: str,
    task_id: int,
    source_domain: str,
    target_domain: str,
    compatibility_key: str,
) -> Path:
    key = str(compatibility_key or "").strip()
    if not key:
        raise ValueError("step3 tokenizer cache compatibility_key must be non-empty")
    domain_ns = f"{str(source_domain).replace('/', '_')}_to_{str(target_domain).replace('/', '_')}"
    return (
        step3_tokenizer_cache_root(repo_root, formal_cache_namespace)
        / f"task{int(task_id)}"
        / domain_ns
        / key
    ).resolve()


def get_step3_validation_root(repo_root: Path, validation_slug: str) -> Path:
    slug = _safe_namespace_component(str(validation_slug), label="step3 validation slug")
    return repo_root / "test_artifacts" / "runs_like" / "step3_validation" / slug


def get_step3_validation_run_root(repo_root: Path, validation_slug: str, run_id: str) -> Path:
    rid = _safe_namespace_component(str(run_id), label="step3 validation run_id")
    return get_step3_validation_root(repo_root, validation_slug) / rid


def get_step3_validation_meta_dir(repo_root: Path, validation_slug: str, run_id: str) -> Path:
    return (get_step3_validation_run_root(repo_root, validation_slug, run_id) / "meta").resolve()


def step3_validation_evidence_root(repo_root: Path, validation_slug: str, run_id: str) -> Path:
    slug = _safe_namespace_component(str(validation_slug), label="step3 validation slug")
    rid = _safe_namespace_component(str(run_id), label="step3 validation run_id")
    return (repo_root / "AI_analysis" / "01_raw_logs" / slug / rid).resolve()


def step3_validation_tokenizer_cache_entry_dir(
    repo_root: Path,
    *,
    validation_slug: str,
    run_id: str,
    task_id: int,
    source_domain: str,
    target_domain: str,
    compatibility_key: str,
) -> Path:
    key = str(compatibility_key or "").strip()
    if not key:
        raise ValueError("step3 validation tokenizer cache compatibility_key must be non-empty")
    domain_ns = f"{str(source_domain).replace('/', '_')}_to_{str(target_domain).replace('/', '_')}"
    return (
        step3_validation_evidence_root(repo_root, validation_slug, run_id)
        / "cache"
        / "step3"
        / "tokenizer"
        / f"task{int(task_id)}"
        / domain_ns
        / key
    ).resolve()
