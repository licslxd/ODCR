"""Bounded, paper-ineligible runtime probes for task8 ablations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from odcr_core.ablation.binding import (
    AblationBindingError,
    apply_binding_to_step5_runtime_configs,
    load_ablation_binding,
)
from odcr_core.ablation.guards import paper_table_gate
from odcr_core.ablation.registry import load_config_override
from odcr_core.config_resolver import resolve_config
from odcr_core.file_atomic import atomic_write_json
from odcr_core.step5_explanation_flan_bridge import per_sample_decoder_ce_from_logits
from odcr_core.step5_innovation import (
    CCVControlPacket,
    Step5ExplanationGate,
    evidence_basis_fca_loss,
    parse_step5_innovation_config_json,
)
from odcr_core.step5_pool_sampler import (
    resolve_step5_pool_source,
    sample_effective_epochs_from_pools,
    validate_step5_formal_sample_plan_for_source,
)
from odcr_core.training_checkpoint import stable_hash


class AblationProbeError(AblationBindingError):
    """Raised when a bounded ablation probe fails safety or runtime checks."""


FORBIDDEN_PROBE_ARTIFACT_NAMES = {
    "latest.json",
    "run_summary.json",
    "stage_status.json",
    "eval_metrics.json",
    "official_eval_report.json",
    "best_observed.pth",
}


def _repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _sha256_or_none(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AblationProbeError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AblationProbeError(f"{label} invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AblationProbeError(f"{label} must be a JSON object: {path}")
    return payload


def _step4_run_id_from_override(repo_root: Path, *, task: int, variant: str) -> str:
    override = load_config_override(repo_root, task, variant)
    latest_rel = str(override.get("expected_step4_handoff_source") or "").strip()
    if not latest_rel:
        raise AblationProbeError("ablation override missing expected_step4_handoff_source")
    latest = repo_root / latest_rel
    payload = _load_json(latest, label="Step4 latest pointer")
    if int(payload.get("task_id") or 0) != int(task):
        raise AblationProbeError(f"Step4 latest task mismatch for task{task}: {latest}")
    run_id = str(payload.get("latest_run_id") or payload.get("active_run_id") or "").strip()
    if not run_id or run_id.startswith("ablation_"):
        raise AblationProbeError(f"unsafe Step4 run id for ablation probe: {run_id!r}")
    return run_id


def _json_payload(raw: str | Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        payload = json.loads(str(raw or "{}"))
    except json.JSONDecodeError as exc:
        raise AblationProbeError(f"{label} invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AblationProbeError(f"{label} root must be object")
    return payload


def _selected_global_batch(batch_candidates: Mapping[str, Any], tuning: Mapping[str, Any]) -> int:
    selected = str(tuning.get("batch_candidate") or batch_candidates.get("selected_default") or "").strip()
    for item in batch_candidates.get("candidates") or []:
        if isinstance(item, Mapping) and str(item.get("id") or "") == selected:
            value = int(item.get("global_batch_size") or 0)
            return value if value > 0 else 64
    return 64


def _counts(series: pd.Series) -> dict[str, int]:
    if series is None:
        return {}
    return {str(k): int(v) for k, v in series.fillna("").astype(str).value_counts(dropna=False).items()}


def _batch_composition_payload(
    *,
    df: pd.DataFrame,
    sample_stats: Mapping[str, Any],
    variant: str,
    max_steps: int,
    bounded_max_rows: int,
) -> dict[str, Any]:
    component_counts = _counts(df["sampler_component"]) if "sampler_component" in df.columns else {}
    origin_counts = _counts(df["sample_origin"]) if "sample_origin" in df.columns else {}
    sample_weight = pd.to_numeric(df["sample_weight_hint"], errors="coerce") if "sample_weight_hint" in df.columns else pd.Series(dtype="float64")
    posterior_weight = (
        pd.to_numeric(df["posterior_sample_weight_hint"], errors="coerce")
        if "posterior_sample_weight_hint" in df.columns
        else pd.Series(dtype="float64")
    )
    unique_weights = []
    if len(sample_weight) > 0:
        unique_weights = sorted({round(float(v), 8) for v in sample_weight.dropna().tolist()})[:16]
    payload = {
        "schema_version": "odcr_ablation_probe_batch_composition/1",
        "variant": str(variant),
        "max_steps": int(max_steps),
        "bounded_max_rows": int(bounded_max_rows),
        "row_count": int(len(df)),
        "batch_nonempty": bool(len(df) > 0),
        "component_counts": component_counts,
        "sample_origin_counts": origin_counts,
        "target_gold_count": int(component_counts.get("target_gold", 0)),
        "aux_gold_count": int(component_counts.get("aux_gold", 0)),
        "cf_count": int(component_counts.get("cf", 0)),
        "aux_cf_origin_count": int(origin_counts.get("aux_cf", 0)),
        "sample_weight_hint_unique_values": unique_weights,
        "sample_weight_hint_mean": None if len(sample_weight) == 0 else float(sample_weight.mean()),
        "posterior_sample_weight_hint_mean": None if len(posterior_weight) == 0 else float(posterior_weight.mean()),
        "sampler_stats": dict(sample_stats),
    }
    if variant == "wo_rcr":
        payload["rcr_route_pool_weighting"] = "disabled"
        payload["pool_weight_source"] = str(sample_stats.get("pool_weight_source") or "")
        payload["sampling_policy"] = str(sample_stats.get("sampling_policy") or "")
        payload["observed_uniform_weights"] = bool(unique_weights == [1.0])
        payload["route_filter_enabled"] = bool(sample_stats.get("route_filter_enabled"))
        payload["posterior_weight_preserved_for_audit"] = "posterior_sample_weight_hint" in df.columns
    if variant == "wo_cf":
        payload["cf_samples"] = "disabled"
        payload["aux_cf_samples"] = "disabled"
        payload["target_gold_only"] = bool(
            int(component_counts.get("target_gold", 0)) > 0
            and int(component_counts.get("aux_gold", 0)) == 0
            and int(component_counts.get("cf", 0)) == 0
            and int(origin_counts.get("aux_cf", 0)) == 0
        )
        payload["observed_cf_count"] = int(component_counts.get("cf", 0))
        payload["observed_aux_count"] = int(component_counts.get("aux_gold", 0))
    return payload


def _tiny_loss_probe(innovation_config: Mapping[str, Any], *, variant: str) -> dict[str, Any]:
    cfg = parse_step5_innovation_config_json(innovation_config)
    torch.manual_seed(7)
    batch = 2
    dim = 4
    tok = 5
    logits = torch.randn(batch, 3, tok)
    labels = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    ce = per_sample_decoder_ce_from_logits(logits, labels, ignore_index=0, label_smoothing=0.0)
    packet = CCVControlPacket(
        content_evidence_ids=torch.ones(batch, 2, dtype=torch.long),
        style_evidence_ids=torch.ones(batch, 2, dtype=torch.long),
        domain_style_anchor_ids=torch.ones(batch, 2, dtype=torch.long),
        local_style_hint_ids=torch.ones(batch, 2, dtype=torch.long),
        polarity_ids=torch.ones(batch, 2, dtype=torch.long),
        route_scorer_mask=torch.ones(batch),
        route_explainer_mask=torch.ones(batch),
        sample_weight_hint=torch.ones(batch),
        cf_reliability_score=torch.ones(batch),
        content_retention_score=torch.ones(batch),
        style_shift_score=torch.zeros(batch),
        rating_stability_score=torch.ones(batch),
        uncertainty_score=torch.zeros(batch),
        confidence_bucket=torch.ones(batch),
        evidence_quality_prior=torch.ones(batch),
        content_anchor_score=torch.ones(batch),
        style_anchor_score=torch.ones(batch),
    )
    gate = Step5ExplanationGate(
        explainer_weight=torch.ones(batch),
        fca_weight=torch.ones(batch),
        route_mask=torch.ones(batch),
        reliability=torch.ones(batch),
        uncertainty=torch.zeros(batch),
        confidence_bucket=torch.ones(batch),
        style_shift=torch.zeros(batch),
    )
    fca = evidence_basis_fca_loss(
        scorer_hidden=torch.ones(batch, dim),
        explainer_hidden=torch.zeros(batch, dim),
        shared_latent=torch.ones(batch, dim),
        content_profile=torch.ones(batch, dim),
        content_evidence_latent=torch.zeros(batch, dim),
        packet=packet,
        gate=gate,
        cfg=cfg,
    )
    return {
        "schema_version": "odcr_ablation_probe_loss_components/1",
        "variant": str(variant),
        "main_ce_loss_present": True,
        "main_ce_probe_mean": float(ce.mean().item()),
        "ccv": {
            "enabled": bool(cfg.ccv.enabled),
            "observed_loss_weight": 0.0,
            "active_loss_term": "not_applicable_no_standalone_ccv_loss",
            "numeric_control_weight": float(cfg.ccv.numeric_control_weight),
            "route_conditioning": bool(cfg.ccv.route_conditioning),
        },
        "fca": {
            "enabled": bool(cfg.fca.enabled),
            "observed_loss_weight": float(cfg.fca.weight),
            "probe_loss": float(fca.fca_loss.detach().item()),
            "probe_weighted_loss": float(fca.fca_weighted_loss.detach().item()),
        },
        "formal_forward_backward": False,
        "paper_metric": False,
    }


def _scan_forbidden_artifacts(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    out: list[str] = []
    for path in run_dir.rglob("*"):
        if path.is_file() and path.name in FORBIDDEN_PROBE_ARTIFACT_NAMES:
            out.append(path.as_posix())
    return sorted(out)


def run_ablation_probe(
    repo_root: str | Path,
    *,
    task: int,
    variant: str,
    max_steps: int,
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    if int(task) != 8:
        raise AblationProbeError("this runtime-binding phase only permits bounded real probes for task8")
    if int(max_steps) < 1 or int(max_steps) > 2:
        raise AblationProbeError("ablation probe max_steps must be between 1 and 2")
    binding = load_ablation_binding(root, task=int(task), variant=str(variant))
    binding_payload = binding.to_dict()
    if binding_payload.get("paper_table_allowed") is not False:
        raise AblationProbeError("probe binding must remain paper_table_allowed=false")
    step4_run_id = _step4_run_id_from_override(root, task=int(task), variant=str(variant))
    latest_path = root / "runs" / "step5" / f"task{int(task)}" / "latest.json"
    latest_before = _sha256_or_none(latest_path)
    cfg, _sources, _snapshot = resolve_config(
        config_path=root / "configs" / "odcr.yaml",
        command="step5",
        task_id=int(task),
        set_overrides=[],
        dry_run=True,
        from_step4=step4_run_id,
    )
    sampler = _json_payload(str(getattr(cfg, "step5_sampler_config_json", "{}")), label="step5.sampler")
    batch_candidates = _json_payload(
        str(getattr(cfg, "step5_batch_candidates_config_json", "{}")),
        label="step5.batch_candidates",
    )
    tuning = _json_payload(str(getattr(cfg, "step5_tuning_config_json", "{}")), label="step5.tuning")
    innovation = _json_payload(str(getattr(cfg, "step5_innovation_config_json", "{}")), label="step5.innovation")
    applied = apply_binding_to_step5_runtime_configs(
        binding,
        sampler_config=sampler,
        batch_candidates_config=batch_candidates,
        tuning_config=tuning,
        innovation_config=innovation,
    )
    bounded_rows = int(max_steps) * _selected_global_batch(
        applied["batch_candidates_config"],
        applied["tuning_config"],
    )
    step4_run_dir = root / "runs" / "step4" / f"task{int(task)}" / step4_run_id
    source = resolve_step5_pool_source(step4_run_dir=step4_run_dir, repo_root=root)
    preflight = validate_step5_formal_sample_plan_for_source(
        source,
        sampler_config=applied["sampler_config"],
        batch_candidates_config=applied["batch_candidates_config"],
        tuning_config=applied["tuning_config"],
        task_head="explanation",
        mode="bounded",
        bounded_max_rows=bounded_rows,
        fail_on_route_incompatible=True,
        no_write=True,
    )
    sampled = sample_effective_epochs_from_pools(
        source,
        sampler_config=applied["sampler_config"],
        batch_candidates_config=applied["batch_candidates_config"],
        tuning_config=applied["tuning_config"],
        mode="bounded",
        task_head="explanation",
        bounded_max_rows=bounded_rows,
    )
    batch_payload = _batch_composition_payload(
        df=sampled.train_df,
        sample_stats=sampled.stats,
        variant=str(variant),
        max_steps=int(max_steps),
        bounded_max_rows=bounded_rows,
    )
    loss_payload = _tiny_loss_probe(applied["innovation_config"], variant=str(variant))
    run_dir = root / binding.run_namespace
    probe_dir = run_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    latest_after_before_writes = _sha256_or_none(latest_path)
    if latest_after_before_writes != latest_before:
        raise AblationProbeError("P0 latest pollution detected before probe evidence writes")
    guardrail = {
        "schema_version": "odcr_ablation_probe_guardrail_status/1",
        "would_write_latest": False,
        "did_write_latest": False,
        "latest_path": latest_path.relative_to(root).as_posix(),
        "latest_sha256_before": latest_before,
        "latest_sha256_after": latest_after_before_writes,
        "paper_table_gate": paper_table_gate(
            {
                "valid_complete": False,
                "test_complete": False,
                "paper_greedy_25": True,
                "task_local_rating_source": True,
                "paper_table_allowed": False,
                "requires_manual_review": True,
            }
        ),
        "forbidden_artifacts_before_write": _scan_forbidden_artifacts(run_dir),
        "formal_training_started": False,
        "formal_eval_started": False,
        "paper_metric_written": False,
    }
    status = "pass"
    errors: list[str] = []
    if not bool(batch_payload.get("batch_nonempty")):
        errors.append("empty_batch")
    if str(variant) == "wo_rcr":
        if batch_payload.get("pool_weight_source") != "uniform":
            errors.append("wo_rcr_pool_weight_source_not_uniform")
        if batch_payload.get("observed_uniform_weights") is not True:
            errors.append("wo_rcr_weights_not_uniform")
        if batch_payload.get("route_filter_enabled") is not False:
            errors.append("wo_rcr_route_filter_not_disabled")
    if str(variant) == "wo_cf":
        if batch_payload.get("observed_cf_count") != 0 or batch_payload.get("observed_aux_count") != 0:
            errors.append("wo_cf_observed_cf_or_aux")
        if int(batch_payload.get("target_gold_count") or 0) <= 0:
            errors.append("wo_cf_missing_target_gold")
        if batch_payload.get("aux_cf_origin_count") != 0:
            errors.append("wo_cf_observed_aux_cf_origin")
    if str(variant) == "wo_ccv_fca":
        if loss_payload["ccv"]["enabled"] is not False:
            errors.append("wo_ccv_fca_ccv_not_disabled")
        if loss_payload["fca"]["enabled"] is not False:
            errors.append("wo_ccv_fca_fca_not_disabled")
        if float(loss_payload["fca"]["probe_weighted_loss"]) != 0.0:
            errors.append("wo_ccv_fca_fca_weighted_loss_nonzero")
        if loss_payload.get("main_ce_loss_present") is not True:
            errors.append("wo_ccv_fca_main_ce_missing")
    if guardrail["paper_table_gate"]["eligible"] is True:
        errors.append("paper_gate_unexpectedly_eligible")
    if errors:
        status = "fail"
    report = {
        "schema_version": "odcr_ablation_probe_report/1",
        "status": status,
        "errors": errors,
        "task": int(task),
        "variant": str(variant),
        "max_steps": int(max_steps),
        "bounded_max_rows": int(bounded_rows),
        "output_dir": probe_dir.relative_to(root).as_posix(),
        "step4_run_id": step4_run_id,
        "source_pool": source.to_summary(),
        "runtime_binding_hash": binding_payload["binding_hash"],
        "preflight_status": preflight.get("status"),
        "sample_plan_hash": stable_hash(
            {
                "sample_ids": list(sampled.train_df["sample_id"].astype(str)) if "sample_id" in sampled.train_df.columns else [],
                "components": list(sampled.train_df["sampler_component"].astype(str)) if "sampler_component" in sampled.train_df.columns else [],
                "variant": str(variant),
            }
        ),
        "no_formal_training": True,
        "no_formal_eval": True,
        "no_fake_metrics": True,
        "paper_table_allowed": False,
    }
    atomic_write_json(probe_dir / "runtime_binding.json", binding_payload)
    atomic_write_json(probe_dir / "batch_composition.json", batch_payload)
    atomic_write_json(probe_dir / "loss_components.json", loss_payload)
    guardrail["forbidden_artifacts_after_write"] = _scan_forbidden_artifacts(run_dir)
    latest_after = _sha256_or_none(latest_path)
    guardrail["latest_sha256_after_probe"] = latest_after
    guardrail["did_write_latest"] = bool(latest_after != latest_before)
    if guardrail["did_write_latest"]:
        status = "fail"
        report["status"] = "fail"
        report.setdefault("errors", []).append("latest_pollution")
    atomic_write_json(probe_dir / "guardrail_status.json", guardrail)
    atomic_write_json(probe_dir / "probe_report.json", report)
    if report["status"] != "pass":
        raise AblationProbeError("ablation probe failed: " + ", ".join(report["errors"]))
    return report


__all__ = ["AblationProbeError", "run_ablation_probe"]
