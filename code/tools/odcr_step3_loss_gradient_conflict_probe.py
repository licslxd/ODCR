#!/usr/bin/env python3
"""Bounded Step3 loss-group gradient conflict probe entrypoint.

``--dry-run`` validates the loss-group schema. A real run is intentionally
checkpoint-neutral: it loads the live Step3 model/data/checkpoint, runs at most
four training batches through the real forward/loss path, measures group
gradients, and writes only AI_analysis evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.file_atomic import atomic_write_json  # noqa: E402
from odcr_core.step3_v3_policy import (  # noqa: E402
    LOSS_GROUPS,
    STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
    validate_loss_group_mapping,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "AI_analysis" / "06_probe_evidence" / "step3_loss_gradient_conflict_probe"
DEFAULT_RUNTIME_OUTPUT_DIR = (
    REPO_ROOT / "AI_analysis" / "06_probe_evidence" / "step3_v3_gradient_conflict_runtime_probe"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "step3" / "task2" / "2" / "model" / "best_observed.pth"
DEFAULT_LINEAGE = REPO_ROOT / "runs" / "step3" / "task2" / "2" / "model" / "best_observed.pth.lineage.json"
DEFAULT_RUN_SUMMARY = REPO_ROOT / "runs" / "step3" / "task2" / "2" / "meta" / "run_summary.json"
MAX_REAL_BATCHES = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        import torch

        if torch.is_tensor(value):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
    except Exception:
        pass
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(payload))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_state(path: Path) -> dict[str, Any]:
    p = path.expanduser().resolve()
    out: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if p.is_file():
        st = p.stat()
        out.update({"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
        if st.st_size <= 64 * 1024 * 1024:
            out["sha256"] = _sha256_file(p)
    return out


@contextmanager
def _patched_env(updates: Mapping[str, str]):
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update({str(k): str(v) for k, v in updates.items()})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _torchrun_cmd() -> list[str]:
    import shutil

    if shutil.which("torchrun"):
        return ["torchrun"]
    return [sys.executable, "-m", "torch.distributed.run"]


def _is_distributed_child() -> bool:
    return all(name in os.environ for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))


def _canonical_output_dir(raw: str | None, *, real_run: bool = False) -> Path:
    default = DEFAULT_RUNTIME_OUTPUT_DIR if real_run else DEFAULT_OUTPUT_DIR
    out = Path(raw or default).expanduser()
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_for_probe(task_id: int):
    from odcr_core.config_resolver import resolve_config

    cfg, sources, snapshot = resolve_config(
        config_path=REPO_ROOT / "configs" / "odcr.yaml",
        command="step3",
        task_id=int(task_id),
        set_overrides=[],
        dry_run=True,
        run_id="auto",
        mode="full",
    )
    return cfg, sources, snapshot


def _probe_env(cfg: Any, output_dir: Path) -> dict[str, str]:
    from odcr_core.runners import _odcr_layout_env, _torchrun_hardware_env

    meta_dir = output_dir / "meta"
    stage_dir = output_dir / "validation_stage_run"
    meta_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    env = dict(_odcr_layout_env(cfg))
    env.update(_torchrun_hardware_env(cfg))
    env.update(
        {
            "ODCR_STAGE_RUN_DIR": str(stage_dir.resolve()),
            "ODCR_MANIFEST_DIR": str(meta_dir.resolve()),
            "ODCR_LOG_DIR": str(meta_dir.resolve()),
            "ODCR_SUMMARY_LOG": str((output_dir / "console_probe.log").resolve()),
            "ODCR_LOG_CONSOLE": "0",
            "ODCR_STEP3_TOKENIZER_CACHE_STARTUP_JSON": str(
                (output_dir / "step3_tokenizer_cache_startup.json").resolve()
            ),
        }
    )
    return env


def _spawn_torchrun_if_needed(args: argparse.Namespace, output_dir: Path) -> int | None:
    if _is_distributed_child():
        return None
    from odcr_core.manifests import write_resolved_config_artifacts

    cfg, _sources, _snapshot = _resolve_for_probe(int(args.task))
    world_size = int(getattr(cfg, "ddp_world_size", 1) or 1)
    if world_size <= 1:
        return None
    env = dict(os.environ)
    env.update(_probe_env(cfg, output_dir))
    write_resolved_config_artifacts(
        Path(env["ODCR_MANIFEST_DIR"]),
        _snapshot,
        formal_only_source_table=True,
        write_verbose_source_table=True,
    )
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(CODE_DIR) if not prev else f"{CODE_DIR}{os.pathsep}{prev}"
    cmd = [
        *_torchrun_cmd(),
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(Path(__file__).resolve()),
        "--task",
        str(int(args.task)),
        "--max-batches",
        str(int(args.max_batches)),
        "--output-dir",
        str(output_dir),
        "--allow-real-run",
        "--real-data",
        "--real-length",
        "--no-synthetic",
        "--validation-namespace",
        "--no-checkpoint-write",
        "--distributed-child",
    ]
    if str(args.real_data_manifest or "").strip():
        cmd.extend(["--real-data-manifest", str(args.real_data_manifest)])
    (output_dir / "command.sh").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    child_log = output_dir / "torchrun_child_output.log"
    child_log.write_text((proc.stdout or "")[-700_000:], encoding="utf-8")
    if proc.returncode != 0:
        _write_json(
            output_dir / "status.json",
            {
                "schema_version": STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
                "status": "failed",
                "phase": "torchrun_parent",
                "returncode": int(proc.returncode),
                "elapsed_s": round(elapsed, 6),
                "child_output": str(child_log),
                "writes_formal_checkpoint": False,
                "formal_namespace_guard": True,
                "updated_at_utc": _utc_now(),
            },
        )
    return int(proc.returncode)


def build_dry_run_payload(*, task_id: int, max_batches: int) -> dict[str, Any]:
    mapping = validate_loss_group_mapping()
    return {
        "schema_version": STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
        "status": "dry_run_ready",
        "task_id": int(task_id),
        "bounded_max_batches": int(max_batches),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "real_data_only": True,
        "real_length_required": True,
        "real_step3_forward_loss_required": True,
        "synthetic_benchmark_forbidden": True,
        "writes_formal_checkpoint": False,
        "formal_namespace_guard": True,
        "outputs": {
            "group_grad_norm": "required_when_run",
            "group_cosine_similarity_matrix": "required_when_run",
            "conflict_rate": "required_when_run",
            "rating_explainer_conflicts": "required_when_run",
            "recommendation": ["no_conflict", "dynamic_weighting", "PCGrad", "GradNorm", "adapter/gating"],
        },
        "loss_group_mapping": mapping,
    }


def _cosine(a: Any, b: Any) -> float:
    import torch

    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    denom = float((na * nb).detach().cpu().item())
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return float(torch.dot(a, b).detach().cpu().item() / denom)


def _mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _max(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(max(clean)) if clean else 0.0


def _recommend(conflict_rate: float, pair_stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if conflict_rate < 0.10:
        base = "no_conflict"
        formal = "keep_phase_schedule"
    elif conflict_rate < 0.30:
        base = "dynamic_weighting"
        formal = "phase_schedule_or_dynamic_weighting_enough"
    elif conflict_rate < 0.50:
        base = "consider_PCGrad_or_GradNorm"
        formal = "consider_conflict_handling_before_formal_enablement"
    else:
        base = "strong_PCGrad_adapter_gating_audit"
        formal = "strong_recommendation_for_gradient_surgery_or_adapter_audit"

    structural = {"easd_content", "hss_style", "disentangle_geometry"}
    rating_negative = [
        key
        for key, value in pair_stats.items()
        if "rating" in key.split("|")
        and bool(set(key.split("|")) & structural)
        and float(value.get("cosine_mean", 0.0) or 0.0) < 0.0
    ]
    explanation_negative = [
        key
        for key, value in pair_stats.items()
        if "explanation" in key.split("|")
        and bool(set(key.split("|")) & {"hss_style", "disentangle_geometry"})
        and float(value.get("cosine_mean", 0.0) or 0.0) < 0.0
    ]
    final = base
    if len(rating_negative) >= 2:
        final = "rating_adapter_or_gradient_surgery"
    elif explanation_negative:
        final = "explainer_specific_adapter_or_phase_weighting"
    return {
        "schema_version": STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
        "status": "completed",
        "conflict_rate": float(conflict_rate),
        "base_recommendation": base,
        "recommendation": final,
        "formal_enable_now": False,
        "formal_enablement_note": "Do not enable PCGrad/GradNorm/adapter in formal from this tool alone; wire through One-Control after review.",
        "strategy_bucket": formal,
        "rating_negative_structural_pairs": rating_negative,
        "explanation_negative_style_geometry_pairs": explanation_negative,
    }


def _flatten_grads(
    *,
    loss: Any,
    params: Sequence[Any],
    device: Any,
    world_size: int,
) -> Any:
    import torch
    import torch.distributed as dist

    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = sum(int(param.numel()) for param in params)
    flat = torch.empty(total, dtype=torch.float32, device=device)
    offset = 0
    for param, grad in zip(params, grads):
        n = int(param.numel())
        part = flat[offset : offset + n]
        if grad is None:
            part.zero_()
        else:
            part.copy_(grad.detach().reshape(-1).to(dtype=torch.float32, device=device))
        offset += n
    if int(world_size) > 1:
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(float(world_size))
    return flat


def _run_real_probe(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace

    from config import resolve_task_idx_from_aux_target
    from executors.step3_train_core import (
        apply_step3_phase_loss_multipliers,
        build_step3_training_components,
        compose_step3_loss_from_forward_output,
        get_underlying_model,
        require_gathered_batch,
        step3_loss_semantics_from_config,
        step3_structured_loss_weights_from_config,
        step3_trainable_parameters,
    )
    from odcr_core.manifests import write_resolved_config_artifacts
    from odcr_core.step3_v3_policy import resolve_phase_for_epoch
    from train_diagnostics import odcr_cuda_bf16_autocast

    if int(args.max_batches) < 1 or int(args.max_batches) > MAX_REAL_BATCHES:
        raise RuntimeError(f"real Step3 conflict probe requires 1 <= max_batches <= {MAX_REAL_BATCHES}")
    if not bool(args.allow_real_run):
        raise RuntimeError("real probe requires --allow-real-run")
    if bool(args.no_synthetic) is False:
        raise RuntimeError("real probe refuses synthetic benchmark mode; pass --no-synthetic")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError("real Step3 conflict probe requires visible CUDA in the current process")

    cfg, _sources, snapshot = _resolve_for_probe(int(args.task))
    env = _probe_env(cfg, output_dir)
    meta_dir = Path(env["ODCR_MANIFEST_DIR"])
    if not (meta_dir / "resolved_config.json").is_file() or not (meta_dir / "source_table.json").is_file():
        write_resolved_config_artifacts(
            meta_dir,
            snapshot,
            formal_only_source_table=True,
            write_verbose_source_table=True,
        )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", str(getattr(cfg, "ddp_world_size", 1) or 1)))
    env["ODCR_STEP3_TOKENIZER_CACHE_STARTUP_JSON"] = str(
        (output_dir / f"step3_tokenizer_cache_startup_rank{rank}.json").resolve()
    )
    if int(world_size) != int(getattr(cfg, "ddp_world_size", world_size) or world_size):
        raise RuntimeError(
            f"torchrun WORLD_SIZE={world_size} does not match resolved Step3 ddp_world_size={cfg.ddp_world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.sh").write_text(" ".join(shlex.quote(x) for x in sys.argv) + "\n", encoding="utf-8")
    checkpoint_path = Path(args.checkpoint or DEFAULT_CHECKPOINT).expanduser().resolve()
    lineage_path = Path(args.lineage or DEFAULT_LINEAGE).expanduser().resolve()
    run_summary_path = Path(args.real_data_manifest or DEFAULT_RUN_SUMMARY).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"checkpoint missing: {checkpoint_path}")
    if not lineage_path.is_file():
        raise RuntimeError(f"checkpoint lineage missing: {lineage_path}")
    if not run_summary_path.is_file():
        raise RuntimeError(f"real-data manifest/run_summary missing: {run_summary_path}")
    run_summary = _read_json(run_summary_path)
    lineage = _read_json(lineage_path)
    before_states = {
        "checkpoint": _file_state(checkpoint_path),
        "lineage": _file_state(lineage_path),
        "latest_json": _file_state(REPO_ROOT / "runs" / "step3" / "task2" / "latest.json"),
    }
    mapping = validate_loss_group_mapping()
    if mapping.get("status") != "pass":
        raise RuntimeError(f"loss group mapping failed: {mapping}")

    preflight = {
        "schema_version": STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
        "status": "running",
        "task_id": int(args.task),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(local_rank),
        "max_batches": int(args.max_batches),
        "real_data_only": True,
        "real_length_required": True,
        "real_step3_forward_loss_required": True,
        "synthetic_benchmark_forbidden": True,
        "writes_formal_checkpoint": False,
        "formal_namespace_guard": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_hash": str(run_summary.get("selected_checkpoint_hash") or lineage.get("checkpoint_file_hash") or ""),
        "run_summary": str(run_summary_path),
        "resolved_profile": snapshot.get("task") or {},
        "loss_group_mapping": mapping,
        "created_at_utc": _utc_now(),
    }
    if rank == 0:
        _write_json(output_dir / "preflight.json", preflight)

    args_ns = SimpleNamespace(
        auxiliary=str(cfg.auxiliary),
        target=str(cfg.target),
        num_proc=int(cfg.num_proc),
        seed=int(cfg.seed),
        checkpoint_metric="valid_loss",
        log_file=str(output_dir / "probe_full.log"),
        save_file=str(checkpoint_path),
        nlayers=int(getattr(cfg, "nlayers", 2) or 2),
        batch_size=None,
        per_device_batch_size=None,
        scheduler_initial_lr=None,
        learning_rate=None,
        epochs=None,
        warmup_steps=None,
        warmup_ratio=None,
        warmup_epochs=None,
        min_lr_ratio=None,
        lr_scheduler=None,
        eval_batch_size=None,
        quick_eval_max_samples=None,
        early_stop_patience_full=None,
        early_stop_patience_loss=None,
        min_epochs=None,
        early_stop_patience=None,
        bleu4_max_samples=None,
    )

    with _patched_env(env):
        final_cfg, train_dataloader, _valid_dataloader, ddp_model, sampler = build_step3_training_components(
            args_ns,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
        model = get_underlying_model(ddp_model)
        state = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
        load_result = model.load_state_dict(state, strict=False)
        if getattr(load_result, "unexpected_keys", None):
            raise RuntimeError(f"unexpected checkpoint keys: {load_result.unexpected_keys[:10]}")
        missing_keys = list(getattr(load_result, "missing_keys", []) or [])
        model.train()
        sampler.set_epoch(0)
        structured_weights = step3_structured_loss_weights_from_config(final_cfg)
        loss_semantics = step3_loss_semantics_from_config(final_cfg)
        checkpoint_epoch = int(lineage.get("checkpoint_epoch") or 1)
        phase_record = resolve_phase_for_epoch(
            epoch=checkpoint_epoch,
            config=json.loads(str(getattr(final_cfg, "phase_loss_schedule_config_json", "") or "{}")),
        )
        phase_weights = apply_step3_phase_loss_multipliers(
            structured_weights,
            phase_record.get("loss_multipliers") if bool(phase_record.get("enabled", True)) else {},
        )
        params = step3_trainable_parameters(model)
        group_names = list(LOSS_GROUPS.keys())
        norm_rows: dict[str, list[float]] = {name: [] for name in group_names}
        pair_values: dict[str, list[float]] = {}
        per_batch: list[dict[str, Any]] = []
        total_negative = 0
        total_pairs = 0
        started = time.monotonic()
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= int(args.max_batches):
                break
            g = require_gathered_batch(model.gather(batch, local_rank))
            with odcr_cuda_bf16_autocast():
                forward_out = model(
                    g.user_idx,
                    g.item_idx,
                    g.tgt_input,
                    g.domain_idx,
                    content_anchor=g.content_anchor_score,
                    style_anchor=g.style_anchor_score,
                    content_evidence_ids=g.content_evidence_ids,
                    style_evidence_ids=g.style_evidence_ids,
                    domain_style_anchor_ids=g.domain_style_anchor_ids,
                    local_style_hint_ids=g.local_style_hint_ids,
                    polarity_ids=g.polarity_ids,
                    evidence_quality_prior=g.evidence_quality_prior,
                )
                loss_bundle = compose_step3_loss_from_forward_output(
                    forward_output=forward_out,
                    batch=g,
                    final_cfg=final_cfg,
                    weights=phase_weights,
                    semantics=loss_semantics,
                )
            group_losses = {
                group: sum(
                    (loss_bundle.weighted_components[name] for name in components),
                    loss_bundle.total_loss * 0.0,
                )
                for group, components in LOSS_GROUPS.items()
            }
            vectors: dict[str, Any] = {}
            batch_norms: dict[str, float] = {}
            for group in group_names:
                vec = _flatten_grads(
                    loss=group_losses[group],
                    params=params,
                    device=device,
                    world_size=world_size,
                )
                vectors[group] = vec
                norm = float(torch.linalg.vector_norm(vec).detach().cpu().item())
                batch_norms[group] = norm
                norm_rows[group].append(norm)
            batch_pairs: list[dict[str, Any]] = []
            for i, left in enumerate(group_names):
                for right in group_names[i + 1 :]:
                    c = _cosine(vectors[left], vectors[right])
                    key = f"{left}|{right}"
                    pair_values.setdefault(key, []).append(c)
                    neg = c < 0.0
                    total_negative += 1 if neg else 0
                    total_pairs += 1
                    batch_pairs.append(
                        {
                            "group_i": left,
                            "group_j": right,
                            "cosine": c,
                            "negative": bool(neg),
                        }
                    )
            component_values = {
                key: float(value.detach().cpu().item())
                for key, value in loss_bundle.weighted_components.items()
            }
            per_batch.append(
                {
                    "batch_index": int(batch_idx),
                    "rank0_local_batch_size": int(g.user_idx.shape[0]),
                    "effective_world_size": int(world_size),
                    "phase": phase_record,
                    "group_grad_norm": batch_norms,
                    "pairs": batch_pairs,
                    "weighted_components": component_values,
                    "total_loss": float(loss_bundle.total_loss.detach().cpu().item()),
                }
            )
            del vectors, group_losses, loss_bundle, forward_out
            torch.cuda.empty_cache()

        if int(world_size) > 1:
            dist.barrier()
        batches_completed = len(per_batch)
        if batches_completed <= 0:
            raise RuntimeError("real probe completed zero batches")
        pair_stats: dict[str, dict[str, Any]] = {}
        for key, values in sorted(pair_values.items()):
            left, right = key.split("|", 1)
            neg_count = sum(1 for value in values if float(value) < 0.0)
            pair_stats[key] = {
                "group_i": left,
                "group_j": right,
                "cosine_mean": _mean(values),
                "cosine_min": min(float(v) for v in values),
                "cosine_max": max(float(v) for v in values),
                "conflict_rate": float(neg_count / max(1, len(values))),
                "negative_batches": int(neg_count),
                "batches": int(len(values)),
                "verdict": "conflict" if neg_count > 0 else "aligned_or_neutral",
            }
        conflict_rate = float(total_negative / max(1, total_pairs))
        recommendation = _recommend(conflict_rate, pair_stats)
        norm_summary = {
            group: {
                "grad_norm_mean": _mean(values),
                "grad_norm_max": _max(values),
                "batches": int(len(values)),
            }
            for group, values in norm_rows.items()
        }
        matrix = {
            left: {
                right: 1.0 if left == right else (
                    pair_stats.get(f"{left}|{right}") or pair_stats.get(f"{right}|{left}") or {}
                ).get("cosine_mean", 0.0)
                for right in group_names
            }
            for left in group_names
        }
        rating_vs = {
            key.replace("rating|", "").replace("|rating", ""): value
            for key, value in pair_stats.items()
            if "rating" in key.split("|")
        }
        explanation_vs = {
            key.replace("explanation|", "").replace("|explanation", ""): value
            for key, value in pair_stats.items()
            if "explanation" in key.split("|")
        }
        after_states = {
            "checkpoint": _file_state(checkpoint_path),
            "lineage": _file_state(lineage_path),
            "latest_json": _file_state(REPO_ROOT / "runs" / "step3" / "task2" / "latest.json"),
        }
        status = {
            "schema_version": STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
            "status": "completed",
            "task_id": int(args.task),
            "real_data": True,
            "real_length": True,
            "real_step3_forward_loss": True,
            "synthetic_benchmark_used": False,
            "max_batches": int(args.max_batches),
            "batches_completed": int(batches_completed),
            "world_size": int(world_size),
            "rank0_device": torch.cuda.get_device_name(local_rank) if rank == 0 else "",
            "checkpoint": str(checkpoint_path),
            "writes_formal_checkpoint": False,
            "formal_namespace_guard": True,
            "formal_namespace_pollution": before_states != after_states,
            "elapsed_s": round(time.monotonic() - started, 6),
            "missing_checkpoint_keys_count": len(missing_keys),
            "missing_checkpoint_keys_sample": missing_keys[:20],
            "updated_at_utc": _utc_now(),
        }
        outputs = {
            "status": status,
            "gradient_conflict_matrix": {
                "schema_version": STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
                "groups": group_names,
                "matrix": matrix,
                "pair_stats": pair_stats,
                "per_batch": per_batch,
            },
            "loss_group_grad_norms": {
                "schema_version": STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
                "groups": norm_summary,
            },
            "conflict_summary": {
                "schema_version": STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
                "conflict_rate": conflict_rate,
                "negative_pair_count": int(total_negative),
                "total_pair_observations": int(total_pairs),
                "negative_cosine_pairs": [
                    value for value in pair_stats.values() if float(value.get("cosine_mean", 0.0) or 0.0) < 0.0
                ],
                "rating_vs_each_group": rating_vs,
                "explanation_vs_each_group": explanation_vs,
                "per_batch_summary": per_batch,
            },
            "recommendation": recommendation,
        }
        if rank == 0:
            for name, payload in outputs.items():
                _write_json(output_dir / f"{name}.json", payload)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return outputs["status"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--output-dir", "--output", dest="output_dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--real-data-manifest", default="")
    parser.add_argument("--allow-real-run", action="store_true")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--real-length", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--validation-namespace", action="store_true")
    parser.add_argument("--no-checkpoint-write", action="store_true")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--lineage", default=str(DEFAULT_LINEAGE))
    parser.add_argument("--distributed-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if int(args.max_batches) > MAX_REAL_BATCHES:
        parser.error(f"--max-batches must be <= {MAX_REAL_BATCHES}")
    if not args.dry_run and not args.allow_real_run:
        parser.error("real probe requires --allow-real-run and a validated real-data/GPU context; use --dry-run for schema validation")
    if args.allow_real_run and not args.no_synthetic:
        parser.error("real probe requires --no-synthetic")
    if args.allow_real_run and not args.no_checkpoint_write:
        parser.error("real probe requires --no-checkpoint-write")

    if args.dry_run:
        payload = build_dry_run_payload(task_id=int(args.task), max_batches=int(args.max_batches))
        out_dir = _canonical_output_dir(args.output_dir, real_run=False)
        out = out_dir / "dry_run_schema.json"
        atomic_write_json(out, payload)
        print(json.dumps({"status": payload["status"], "output": str(out), "writes_formal_checkpoint": False}, indent=2, sort_keys=True))
        return 0

    out_dir = _canonical_output_dir(args.output_dir, real_run=True)
    spawned = _spawn_torchrun_if_needed(args, out_dir)
    if spawned is not None:
        print(json.dumps({"status": "completed" if spawned == 0 else "failed", "output_dir": str(out_dir)}, indent=2, sort_keys=True))
        return int(spawned)
    try:
        status = _run_real_probe(args, out_dir)
    except Exception as exc:
        if int(os.environ.get("RANK", "0")) == 0:
            _write_json(
                out_dir / "status.json",
                {
                    "schema_version": STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
                    "status": "failed",
                    "error": repr(exc),
                    "writes_formal_checkpoint": False,
                    "formal_namespace_guard": True,
                    "updated_at_utc": _utc_now(),
                },
            )
        raise
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
