"""Pre-GPU runner for RACER-C1.

This module prepares the retrieval-first path without running GPU training or
official evaluation. The future train/eval implementation must enter here
after CUDA is available and after the user explicitly authorizes it.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from odcr_core import run_naming
from odcr_core.config_schema import OneControlConfigError, ResolvedConfig, as_plain_dict, fingerprint

from .contracts import (
    RACER_C1_BOTTLENECK_SCHEMA_VERSION,
    RACER_C1_LEAKAGE_SCHEMA_VERSION,
    RACER_C1_RUN_SCHEMA_VERSION,
    RACER_C1_SOURCE_TABLE_SCHEMA_VERSION,
    RACER_C1_STAGE_STATUS_SCHEMA_VERSION,
    RacerC1Paths,
    planned_relative_outputs,
)
from .evidence_pool import build_train_only_evidence_pool
from .innovation_alignment import write_innovation_alignment
from .logging import RacerC1RunLogger, cpu_snapshot, cuda_snapshot
from .contrastive_trainer import run_train_eval
from .train_labels import write_contrastive_pair_manifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_racer_config(cfg: ResolvedConfig) -> dict[str, Any]:
    try:
        out = json.loads(str(getattr(cfg, "step5_racer_c1_config_json", "") or "{}"))
    except json.JSONDecodeError as exc:
        raise OneControlConfigError("ResolvedConfig.step5_racer_c1_config_json is invalid JSON") from exc
    if not isinstance(out, dict) or not out:
        raise OneControlConfigError("RACER-C1 config is missing from the resolved Step5 payload")
    if not bool(out.get("enabled")):
        raise OneControlConfigError("RACER-C1 is disabled in configs/odcr.yaml: step5.racer_c1.enabled=false")
    if int(getattr(cfg, "task_id", 0)) not in {int(x) for x in out.get("task_allowlist", [])}:
        raise OneControlConfigError(f"RACER-C1 is not enabled for task {getattr(cfg, 'task_id', None)}")
    return out


def _select_run_id(parent: Path, requested: str | None, *, dry_run: bool) -> str:
    raw = str(requested or "").strip()
    if raw and raw not in {"auto", ""}:
        return run_naming.parse_run_id(raw)
    if dry_run:
        return "dry_run"
    parent.mkdir(parents=True, exist_ok=True)
    return run_naming.allocate_child_dir(parent, requested=None, kind="run")


def _run_paths(repo_root: Path, task_id: int, run_id: str) -> RacerC1Paths:
    return RacerC1Paths.from_root(repo_root / "runs" / "racer_c1" / f"task{int(task_id)}" / run_id)


def _cache_identity(cfg: ResolvedConfig, racer_cfg: dict[str, Any]) -> dict[str, Any]:
    evidence = racer_cfg["evidence_pool"]
    cache = racer_cfg["cache"]
    return {
        "schema_version": cache["cache_schema_version"],
        "task_id": int(cfg.task_id),
        "source_split": evidence["source_split"],
        "evidence_pool_schema_version": evidence["schema_version"],
        "clean_explanation_max_tokens": evidence["clean_explanation_max_tokens"],
        "sentence_embed_model": str(cfg.sentence_embed_model),
        "embed_dim": int(cfg.embed_dim),
        "identity_fields": list(cache["identity_fields"]),
    }


def _resource_plan(cfg: ResolvedConfig, racer_cfg: dict[str, Any]) -> dict[str, Any]:
    train = racer_cfg["train"]
    return {
        "target_gpu_memory_gb_per_card": float(train["target_gpu_memory_gb"]),
        "global_batch_size": int(train["global_batch_size"]),
        "per_gpu_batch_size": int(train["per_gpu_batch_size"]),
        "ddp_world_size": int(cfg.ddp_world_size),
        "batch_semantics": train["batch_semantics"],
        "expected_memory_explanation": [
            "RACER-C1 trains a compact dual-encoder projection, not a large decoder.",
            "If observed GPU memory is far below target, first suspect CPU/embedding-cache/dataloader bottlenecks or a too-small pair batch.",
            "If throughput is high but memory stays low, increase in-batch negatives/global batch before increasing model size.",
            "If GPU utilization is low and CPU utilization is high, precompute embeddings and raise dataloader workers within the One-Control CPU budget.",
        ],
        "throughput_targets": {
            "primary": "maximize query-evidence pairs/sec under stable loss and no dataloader starvation",
            "not_required": "do not chase exact 35GB allocation if CPU-bound or embedding-cache-bound",
        },
    }


def _bottleneck_template(cfg: ResolvedConfig, racer_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RACER_C1_BOTTLENECK_SCHEMA_VERSION,
        "status": "prepared_no_gpu_measurements",
        "target_gpu_memory_gb_per_card": float(racer_cfg["train"]["target_gpu_memory_gb"]),
        "observed_gpu_memory_gb_per_card": None,
        "observed_gpu_utilization": None,
        "observed_cpu_utilization": None,
        "observed_pairs_per_second": None,
        "observed_tokens_per_second": None,
        "token_length_stats_path": "diagnostics/token_length_stats.json",
        "analysis_rules": _resource_plan(cfg, racer_cfg)["expected_memory_explanation"],
        "next_optimization_order": [
            "verify embedding cache hit and no text model recomputation",
            "increase global contrastive batch within no-accum semantics",
            "increase dataloader workers/prefetch within hardware.max_parallel_cpu",
            "inspect hard-negative mining CPU time",
            "only then enlarge projection model",
        ],
    }


def _leakage_template() -> dict[str, Any]:
    return {
        "schema_version": RACER_C1_LEAKAGE_SCHEMA_VERSION,
        "status": "prepared_not_executed",
        "checks": {
            "valid_test_evidence_source_split_train": "required",
            "train_positive_excludes_current_interaction": "required",
            "valid_test_reference_not_in_query_or_pool": "required",
            "prediction_provenance_required": "required",
        },
    }


def _step4_train_csv(cfg: ResolvedConfig, snapshot: dict[str, Any]) -> Path:
    root = Path(cfg.repo_root)
    task = int(cfg.task_id)
    candidates: list[str] = []
    for name in ("from_run", "step4_run"):
        raw = str(getattr(cfg, name, "") or "").strip()
        if raw and raw != "latest":
            candidates.append(raw)
    upstream = snapshot.get("upstream_resolution")
    if isinstance(upstream, dict):
        for value in json.dumps(upstream, ensure_ascii=False, default=str).split('"'):
            marker = f"runs/step4/task{task}/"
            if marker in value:
                tail = value.split(marker, 1)[1].split("/", 1)[0]
                if tail:
                    candidates.append(tail)
    candidates.append("1")
    for run in dict.fromkeys(candidates):
        path = root / "runs" / "step4" / f"task{task}" / str(run) / "odcr_routing_train.csv"
        if path.is_file():
            return path
    raise OneControlConfigError(
        f"RACER-C1 could not locate task{task} Step4 odcr_routing_train.csv from candidates={candidates}"
    )


def build_racer_c1_plan(
    cfg: ResolvedConfig,
    snapshot: dict[str, Any],
    *,
    mode: str,
    run_id: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    racer_cfg = _load_racer_config(cfg)
    parent = Path(cfg.repo_root) / "runs" / "racer_c1" / f"task{int(cfg.task_id)}"
    resolved_run_id = _select_run_id(parent, run_id, dry_run=dry_run)
    paths = _run_paths(Path(cfg.repo_root), int(cfg.task_id), resolved_run_id)
    output_files = planned_relative_outputs(racer_cfg)
    return {
        "schema_version": RACER_C1_RUN_SCHEMA_VERSION,
        "method_name": racer_cfg["method_name"],
        "paper_method_name": racer_cfg["paper_method_name"],
        "command": "racer-c1",
        "mode": mode,
        "dry_run": bool(dry_run),
        "task_id": int(cfg.task_id),
        "task": {
            "source": cfg.auxiliary,
            "target": cfg.target,
            "scenario": cfg.scenario,
            "direction": cfg.direction,
        },
        "run_id": resolved_run_id,
        "paths": paths.as_dict(),
        "planned_relative_outputs": output_files,
        "cache_identity": _cache_identity(cfg, racer_cfg),
        "cache_identity_fingerprint": fingerprint(_cache_identity(cfg, racer_cfg)),
        "resource_plan": _resource_plan(cfg, racer_cfg),
        "epoch_policy": {
            "max_epochs": int(racer_cfg["train"]["max_epochs"]),
            "min_epochs": int(racer_cfg["train"]["min_epochs"]),
            "early_stopping_patience": int(racer_cfg["train"]["early_stopping_patience"]),
            "selection": "valid BLEU-4, then METEOR, then ROUGE-L",
            "rationale": "Contrastive retrieval should converge faster than generative SFT, but Task2 needs enough epochs for hard negatives to stabilize.",
        },
        "legacy_cleanup_policy": {
            "big_model_generator": racer_cfg["legacy_generator_policy"],
            "formal_path_invokes_big_model": False,
            "flan_t5_lora_status": "deleted_not_available",
        },
        "guardrails": racer_cfg["guardrails"],
        "upstream_resolution": snapshot.get("upstream_resolution"),
        "resolved_step5_run_dir": snapshot.get("run", {}).get("stage_run_dir") if isinstance(snapshot.get("run"), dict) else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _touch_text(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _prepare_run_skeleton(plan: dict[str, Any], cfg: ResolvedConfig, snapshot: dict[str, Any]) -> dict[str, Any]:
    paths = RacerC1Paths.from_root(Path(plan["paths"]["run_root"]))
    started_at = _utc_now()
    racer_cfg = _load_racer_config(cfg)
    logger = RacerC1RunLogger(paths, log_interval_steps=int(racer_cfg["logging"]["log_interval_steps"]))
    logger.initialize()
    logger.console("RACER-C1 prepare started: building real train-only evidence pool; no GPU training launched.")
    for directory in (
        paths.meta_dir,
        paths.evidence_dir,
        paths.train_dir,
        paths.predictions_dir,
        paths.metrics_dir,
        paths.diagnostics_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for rel in plan["planned_relative_outputs"]:
        if not str(rel).startswith("meta/"):
            continue
        target = paths.run_root / rel
        if target.suffix == ".json":
            continue
        _touch_text(target)
    source_csv = _step4_train_csv(cfg, snapshot)
    evidence_path = paths.evidence_dir / "train_evidence_pool.jsonl"
    cross_domain_path = paths.diagnostics_dir / "cross_domain_evidence_distribution.json"
    pool_started = time.monotonic()
    if evidence_path.is_file() and cross_domain_path.is_file():
        cached_cross_domain = json.loads(cross_domain_path.read_text(encoding="utf-8"))
        cached_count = int(cached_cross_domain.get("record_count") or 0)
        source_rows = int(
            ((snapshot.get("upstream_resolution") or {}).get("stage_status") or {})
            .get("export_readiness", {})
            .get("row_count")
            or 0
        )
        if cached_count > 0 and (source_rows <= 0 or cached_count == source_rows):
            evidence_summary = {
                "schema_version": racer_cfg["evidence_pool"]["schema_version"],
                "status": "reused_cached_full_pool",
                "source_csv": source_csv.as_posix(),
                "output_jsonl": evidence_path.as_posix(),
                "record_count": cached_count,
                "source_row_count": cached_count,
                "streaming_chunksize": 100_000,
                "target_gold_count": int(cached_cross_domain.get("target_gold_count") or 0),
                "aux_gold_count": int(cached_cross_domain.get("aux_gold_count") or 0),
                "cf_total_count": int(cached_cross_domain.get("cf_total_count") or 0),
                "cf_accepted_count": int(cached_cross_domain.get("cf_accepted_count") or 0),
                "cf_rejected_count": int(cached_cross_domain.get("cf_rejected_count") or 0),
                "cf_template_count": int(cached_cross_domain.get("cf_template_count") or 0),
                "cf_quarantine_count": int(cached_cross_domain.get("cf_quarantine_count") or 0),
                "cross_domain_evidence_distribution": cached_cross_domain,
            }
            logger.console(
                "RACER-C1 evidence pool cache reused: "
                f"records={cached_count} path={evidence_path.relative_to(Path(cfg.repo_root)).as_posix()}"
            )
        else:
            evidence_summary = build_train_only_evidence_pool(
                source_csv=source_csv,
                output_jsonl=evidence_path,
                diagnostics_dir=paths.diagnostics_dir,
                max_tokens=int(racer_cfg["evidence_pool"]["clean_explanation_max_tokens"]),
            )
    else:
        evidence_summary = build_train_only_evidence_pool(
            source_csv=source_csv,
            output_jsonl=evidence_path,
            diagnostics_dir=paths.diagnostics_dir,
            max_tokens=int(racer_cfg["evidence_pool"]["clean_explanation_max_tokens"]),
        )
    pool_elapsed = max(0.000001, time.monotonic() - pool_started)
    logger.console(
        "RACER-C1 evidence pool built: "
        f"records={evidence_summary['record_count']} "
        f"records/sec={evidence_summary['record_count'] / pool_elapsed:.2f} "
        f"source={source_csv.relative_to(Path(cfg.repo_root)).as_posix()}"
    )
    pair_manifest = write_contrastive_pair_manifest(
        paths.train_dir / "contrastive_pairs_manifest.json",
        evidence_summary=evidence_summary,
        racer_cfg=racer_cfg,
        train_evidence_path=paths.evidence_dir / "train_evidence_pool.jsonl",
    )
    innovation_alignment = write_innovation_alignment(
        paths.diagnostics_dir / "innovation_alignment.json",
        racer_cfg=racer_cfg,
        evidence_summary=evidence_summary,
        pair_manifest=pair_manifest,
    )
    logger.console(
        "RACER-C1 innovation alignment recorded: "
        f"{innovation_alignment['summary']['implemented']}/{innovation_alignment['summary']['total']} implemented"
    )
    logger.resource({"phase": "prepare", "cpu": cpu_snapshot(), "cuda": cuda_snapshot()})
    logger.throughput(
        {
            "phase": "prepare",
            "records_built": evidence_summary["record_count"],
            "elapsed_sec": round(pool_elapsed, 6),
            "records_per_sec": round(evidence_summary["record_count"] / pool_elapsed, 6),
            "pairs_per_sec": None,
            "tokens_per_sec": None,
            "status": "not_training",
        }
    )
    _write_json(
        paths.meta_dir / "run_summary.json",
        {
            "schema_version": RACER_C1_RUN_SCHEMA_VERSION,
            "method_name": plan["method_name"],
            "paper_method_name": plan["paper_method_name"],
            "task_id": int(cfg.task_id),
            "run_id": plan["run_id"],
            "status": "prepared_no_gpu",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "real_training": "not_run",
            "latest_pointer_updated": False,
            "planned_outputs": plan["planned_relative_outputs"],
            "evidence_pool": evidence_summary,
            "contrastive_pairs_manifest": {
                "path": "train/contrastive_pairs_manifest.json",
                "schema_version": pair_manifest["schema_version"],
                "status": pair_manifest["status"],
            },
            "innovation_alignment": {
                "path": "diagnostics/innovation_alignment.json",
                "schema_version": innovation_alignment["schema_version"],
                "status": innovation_alignment["status"],
                "summary": innovation_alignment["summary"],
            },
        },
    )
    _write_json(
        paths.meta_dir / "source_table.json",
        {
            "schema_version": RACER_C1_SOURCE_TABLE_SCHEMA_VERSION,
            "config": "configs/odcr.yaml:step5.racer_c1",
            "resolved_payload_key": "step5_racer_c1",
            "paper_method_name": plan["paper_method_name"],
            "experiment_method_name": plan["method_name"],
            "step3_rating_source": json.loads(cfg.rating_source_config_json or "{}"),
            "upstream_resolution": plan.get("upstream_resolution"),
            "cache_identity": plan["cache_identity"],
            "cache_identity_fingerprint": plan["cache_identity_fingerprint"],
            "train_evidence_source": source_csv.relative_to(Path(cfg.repo_root)).as_posix(),
        },
    )
    _write_json(
        paths.meta_dir / "stage_status.json",
        {
            "schema_version": RACER_C1_STAGE_STATUS_SCHEMA_VERSION,
            "producer_stage": "racer_c1",
            "task": int(cfg.task_id),
            "run_id": plan["run_id"],
            "final_status": "prepared_no_gpu",
            "ready_for_train_eval": False,
            "reason": "GPU training/eval not run in this preparation turn",
        },
    )
    _write_json(paths.diagnostics_dir / "bottleneck_analysis.json", _bottleneck_template(cfg, racer_cfg))
    _write_json(paths.meta_dir / "resolved_config.json", as_plain_dict(cfg))
    _write_json(paths.meta_dir / "resolved_snapshot.json", snapshot)
    logger.console("RACER-C1 prepare finished: train_eval remains CUDA-gated.")
    return {
        **plan,
        "status": "prepared_no_gpu",
        "written": True,
        "run_summary": str(paths.meta_dir / "run_summary.json"),
    }


def _cuda_unavailable_message() -> str:
    return (
        "Current tmux does not expose CUDA. Please manually run `odcr-enter-gpu <JOBID>` "
        "in this same tmux to enter the GPU node, then rerun the probe."
    )


def _has_cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) >= 1
    except Exception:
        return False


def run_racer_c1(
    cfg: ResolvedConfig,
    snapshot: dict[str, Any],
    *,
    mode: str,
    run_id: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    if mode not in {"prepare", "train_eval"}:
        raise OneControlConfigError("racer-c1 --mode must be prepare or train_eval")
    plan = build_racer_c1_plan(cfg, snapshot, mode=mode, run_id=run_id, dry_run=dry_run)
    if dry_run:
        return {
            **plan,
            "status": "dry_run_ok",
            "written": False,
            "elapsed_sec": round(time.monotonic() - started, 6),
        }
    if mode == "train_eval":
        if not _has_cuda():
            raise OneControlConfigError(_cuda_unavailable_message())
        prepared = _prepare_run_skeleton(plan, cfg, snapshot)
        paths = RacerC1Paths.from_root(Path(plan["paths"]["run_root"]))
        logger = RacerC1RunLogger(paths, log_interval_steps=int(_load_racer_config(cfg)["logging"]["log_interval_steps"]))
        logger.initialize()
        logger.console(f"RACER-C1 train_eval starting after evidence prepare: {prepared.get('run_summary')}")
        result = run_train_eval(cfg=cfg, paths=paths, racer_cfg=_load_racer_config(cfg), logger=logger)
        return {
            **plan,
            "status": result.status,
            "checkpoint_path": result.checkpoint_path,
            "best_epoch": result.best_epoch,
            "metrics": result.metrics,
            "elapsed_sec": round(time.monotonic() - started, 6),
        }
    result = _prepare_run_skeleton(plan, cfg, snapshot)
    result["elapsed_sec"] = round(time.monotonic() - started, 6)
    return result
