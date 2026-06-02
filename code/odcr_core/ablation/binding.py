"""Runtime binding for controlled task7/task8 ablations."""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from odcr_core.ablation.guards import paper_table_gate
from odcr_core.ablation.manifest import validate_manifest_file
from odcr_core.ablation.registry import (
    ABLATION_VARIANTS,
    AblationValidationError,
    SCENARIO,
    entry_key,
    load_config_override,
    normalize_variant,
    registry_entry,
    validate_config_override,
)
from odcr_core.training_checkpoint import stable_hash


class AblationBindingError(AblationValidationError):
    """Raised when an ablation cannot be bound to Step5 runtime controls."""


_ABLATION_RUN_ID_RE = re.compile(r"^ablation_(wo_rcr|wo_cf|wo_ccv_fca)_1$")


def is_ablation_run_id(run_id: str | None) -> bool:
    """Return whether a Step5 run id is a controlled task7/task8 ablation id."""

    return bool(_ABLATION_RUN_ID_RE.fullmatch(str(run_id or "").strip()))


def variant_from_ablation_run_id(run_id: str | None) -> str:
    """Extract the registry variant from a controlled ablation Step5 run id."""

    raw = str(run_id or "").strip()
    match = _ABLATION_RUN_ID_RE.fullmatch(raw)
    if not match:
        raise AblationBindingError(f"not a controlled ablation run id: {run_id!r}")
    return normalize_variant(match.group(1))


@dataclass(frozen=True)
class AblationBinding:
    task: int
    variant: str
    scenario: str
    direction: str
    run_namespace: str
    run_id: str
    source_full_run: str
    base_protocol: str
    step5_head: str
    explanation_only: bool
    rcr_controls: Mapping[str, Any]
    cf_controls: Mapping[str, Any]
    ccv_controls: Mapping[str, Any]
    fca_controls: Mapping[str, Any]
    forbidden_to_promote_latest: bool
    paper_table_allowed: bool
    requires_manual_review: bool
    registry_key: str
    manifest_path: str
    override_path: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "odcr_ablation_runtime_binding/1",
            "task": int(self.task),
            "variant": str(self.variant),
            "scenario": str(self.scenario),
            "direction": str(self.direction),
            "run_namespace": str(self.run_namespace),
            "run_id": str(self.run_id),
            "source_full_run": str(self.source_full_run),
            "base_protocol": str(self.base_protocol),
            "step5_head": str(self.step5_head),
            "explanation_only": bool(self.explanation_only),
            "rcr_controls": dict(self.rcr_controls),
            "cf_controls": dict(self.cf_controls),
            "ccv_controls": dict(self.ccv_controls),
            "fca_controls": dict(self.fca_controls),
            "forbidden_to_promote_latest": bool(self.forbidden_to_promote_latest),
            "paper_table_allowed": bool(self.paper_table_allowed),
            "requires_manual_review": bool(self.requires_manual_review),
            "registry_key": str(self.registry_key),
            "manifest_path": str(self.manifest_path),
            "override_path": str(self.override_path),
        }
        payload["binding_hash"] = stable_hash(payload)
        return payload


def _repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _run_id_from_namespace(path: str) -> str:
    return Path(str(path)).name


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AblationBindingError(f"ablation binding requires mapping field: {field}")
    return value


def _assert_variant_semantics(binding: AblationBinding) -> None:
    variant = str(binding.variant)
    if variant == "wo_rcr":
        if binding.rcr_controls.get("route_pool_weighting") != "disabled":
            raise AblationBindingError("wo_rcr requires rcr.route_pool_weighting=disabled")
        if binding.rcr_controls.get("pool_weight_source") != "uniform":
            raise AblationBindingError("wo_rcr requires rcr.pool_weight_source=uniform")
        if binding.rcr_controls.get("sampling_policy") != "flat_uniform_eligible_pool":
            raise AblationBindingError("wo_rcr requires flat_uniform_eligible_pool sampling")
    elif variant == "wo_cf":
        ratios = _require_mapping(binding.cf_controls.get("sampler_ratios"), field="cf.sampler_ratios")
        expected = {"target_gold": 1.0, "aux_gold": 0.0, "cf": 0.0}
        for key, value in expected.items():
            if float(ratios.get(key, -1.0)) != float(value):
                raise AblationBindingError(f"wo_cf requires sampler_ratios.{key}={value}")
        if binding.cf_controls.get("target_gold_only") is not True:
            raise AblationBindingError("wo_cf requires target_gold_only=true")
    elif variant == "wo_ccv_fca":
        if binding.ccv_controls.get("enabled") is not False:
            raise AblationBindingError("wo_ccv_fca requires ccv.enabled=false in the override")
        if binding.fca_controls.get("enabled") is not False:
            raise AblationBindingError("wo_ccv_fca requires fca.enabled=false in the override")
        if float(binding.fca_controls.get("weight", 0.0)) != 0.0:
            raise AblationBindingError("wo_ccv_fca requires fca.weight=0.0")
    else:  # pragma: no cover - normalize_variant prevents this.
        raise AblationBindingError(f"unsupported ablation variant for binding: {variant}")


def load_ablation_binding(repo_root: str | Path, *, task: int, variant: str) -> AblationBinding:
    root = _repo_root(repo_root)
    variant_s = normalize_variant(variant)
    if variant_s not in ABLATION_VARIANTS:
        raise AblationBindingError("runtime binding is only valid for ablation variants")
    if int(task) not in (7, 8):
        raise AblationBindingError("ablation runtime binding only supports task7/task8")
    key = entry_key(int(task), variant_s)
    entry = registry_entry(root, int(task), variant_s)
    override = load_config_override(root, int(task), variant_s)
    validate_config_override(override, registry_entry=entry, key=key)
    manifest_validation = validate_manifest_file(root, int(task), variant_s)
    output_run = str(entry.get("output_run") or "")
    if not output_run.startswith(f"runs/step5/task{int(task)}/ablation_"):
        raise AblationBindingError(f"unsafe ablation output namespace: {output_run}")
    if manifest_validation.get("paper_table_allowed") is not False:
        raise AblationBindingError("ablation manifest must stay paper_table_allowed=false")
    controls = _require_mapping(override.get("variant_controls"), field="variant_controls")
    step5 = _require_mapping(override.get("step5"), field="step5")
    safety = _require_mapping(override.get("safety"), field="safety")
    gate = paper_table_gate(
        {
            "valid_complete": False,
            "test_complete": False,
            "paper_greedy_25": True,
            "task_local_rating_source": True,
            "paper_table_allowed": False,
            "requires_manual_review": True,
        }
    )
    if gate.get("eligible") is True:
        raise AblationBindingError("planned/probe ablation must not pass the paper table gate")
    binding = AblationBinding(
        task=int(task),
        variant=variant_s,
        scenario=str(entry.get("scenario") or SCENARIO),
        direction=str(entry.get("direction") or ""),
        run_namespace=output_run,
        run_id=_run_id_from_namespace(output_run),
        source_full_run=str(override.get("source_full_run") or ""),
        base_protocol=str(override.get("base_protocol") or ""),
        step5_head=str(step5.get("head") or "explanation"),
        explanation_only=bool(step5.get("explanation_only") is True),
        rcr_controls=_require_mapping(controls.get("rcr"), field="variant_controls.rcr"),
        cf_controls=_require_mapping(controls.get("cf"), field="variant_controls.cf"),
        ccv_controls=_require_mapping(controls.get("ccv"), field="variant_controls.ccv"),
        fca_controls=_require_mapping(controls.get("fca"), field="variant_controls.fca"),
        forbidden_to_promote_latest=bool(safety.get("forbidden_to_promote_latest") is True),
        paper_table_allowed=False,
        requires_manual_review=True,
        registry_key=key,
        manifest_path=(
            root / output_run / "meta" / "ablation_manifest.json"
        ).relative_to(root).as_posix(),
        override_path=(
            root / "ablations" / "config_overrides" / f"task{int(task)}_{variant_s}.yaml"
        ).relative_to(root).as_posix(),
    )
    if binding.scenario != SCENARIO:
        raise AblationBindingError(f"unsupported ablation scenario: {binding.scenario}")
    if binding.step5_head != "explanation" or binding.explanation_only is not True:
        raise AblationBindingError("ablation runtime binding requires Step5_e explanation-only")
    if binding.forbidden_to_promote_latest is not True:
        raise AblationBindingError("ablation runtime binding requires forbidden_to_promote_latest=true")
    _assert_variant_semantics(binding)
    return binding


def _json_clone(raw: Mapping[str, Any] | str | None, *, label: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise AblationBindingError(f"{label} JSON root must be object")
        return obj
    return deepcopy(dict(raw))


def _head_cfg(sampler: dict[str, Any]) -> dict[str, Any]:
    head = sampler.get("explanation")
    if not isinstance(head, dict):
        raise AblationBindingError("Step5 sampler payload missing explanation head")
    return head


def apply_binding_to_step5_runtime_configs(
    binding: AblationBinding | Mapping[str, Any],
    *,
    sampler_config: Mapping[str, Any] | str,
    batch_candidates_config: Mapping[str, Any] | str | None = None,
    tuning_config: Mapping[str, Any] | str | None = None,
    innovation_config: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Apply an ablation binding to the same Step5 payloads used by training.

    The returned dictionaries are content-affecting runtime payloads. They do
    not update `configs/odcr.yaml`, do not write formal run state, and are
    consumed by bounded probes or a future explicit formal ablation launcher.
    """

    binding_payload = binding.to_dict() if isinstance(binding, AblationBinding) else deepcopy(dict(binding))
    variant = str(binding_payload.get("variant") or "")
    if variant not in ABLATION_VARIANTS:
        raise AblationBindingError(f"cannot apply unknown ablation variant: {variant!r}")
    sampler = _json_clone(sampler_config, label="step5.sampler")
    batch_candidates = _json_clone(batch_candidates_config, label="step5.batch_candidates")
    tuning = _json_clone(tuning_config, label="step5.tuning")
    innovation = _json_clone(innovation_config, label="step5.innovation")
    sampler_runtime = {
        "schema_version": "odcr_step5_ablation_runtime_overlay/1",
        "task": int(binding_payload["task"]),
        "variant": variant,
        "scenario": str(binding_payload.get("scenario") or SCENARIO),
        "run_namespace": str(binding_payload.get("run_namespace") or ""),
        "binding_hash": str(binding_payload.get("binding_hash") or stable_hash(binding_payload)),
    }
    sampler["ablation_runtime"] = deepcopy(sampler_runtime)
    sampler["task_override_source"] = "ablations/runtime_binding"
    evidence: dict[str, Any] = {
        "schema_version": "odcr_ablation_runtime_application/1",
        "variant": variant,
        "sampler_changed": False,
        "loss_config_changed": False,
        "notes": [],
    }
    if variant == "wo_rcr":
        rcr = dict(binding_payload.get("rcr_controls") or {})
        sampler["ablation_runtime"]["rcr"] = {
            "route_pool_weighting": "disabled",
            "route_filter": "disabled_for_ablation",
            "pool_weight_source": "uniform",
            "sampling_policy": "flat_uniform_eligible_pool",
        }
        head = _head_cfg(sampler)
        for key in ("aux_gold_weight", "cf_high_weight", "cf_medium_weight", "cf_low_weight"):
            head[key] = 1.0
        evidence.update(
            {
                "sampler_changed": True,
                "rcr": {
                    "declared": rcr,
                    "route_pool_weighting": "disabled",
                    "pool_weight_source": "uniform",
                    "sampling_policy": "flat_uniform_eligible_pool",
                },
            }
        )
    elif variant == "wo_cf":
        cf_controls = dict(binding_payload.get("cf_controls") or {})
        ratios = dict(cf_controls.get("sampler_ratios") or {})
        head = _head_cfg(sampler)
        head["target_gold_ratio"] = 1.0
        head["aux_gold_ratio"] = 0.0
        head["cf_ratio"] = 0.0
        sampler["components"] = {
            **dict(sampler.get("components") or {}),
            "target_gold": "enabled",
            "aux_gold": "disabled_for_ablation",
            "cf": "disabled_for_ablation",
            "aux_cf": "disabled_for_ablation",
        }
        sampler["ablation_runtime"]["cf"] = {
            "cf_samples": "disabled",
            "aux_cf_samples": "disabled",
            "target_gold_only": True,
            "sampler_ratios": {"target_gold": 1.0, "aux_gold": 0.0, "cf": 0.0},
        }
        evidence.update(
            {
                "sampler_changed": True,
                "cf": {
                    "declared_ratios": ratios,
                    "applied_ratios": {"target_gold": 1.0, "aux_gold": 0.0, "cf": 0.0},
                    "target_gold_only": True,
                },
            }
        )
        effective_samples = cf_controls.get("effective_samples")
        if isinstance(effective_samples, Mapping) and effective_samples.get("explanation"):
            capped = int(effective_samples["explanation"])
            if capped <= 0:
                raise AblationBindingError("wo_cf effective_samples.explanation must be positive")
            tuning.setdefault("effective_samples", {})["explanation"] = capped
            evidence["cf"]["effective_samples"] = {
                "explanation": capped,
                "reason": effective_samples.get("reason") or "target_gold_only_no_replacement_cap",
            }
    elif variant == "wo_ccv_fca":
        ccv = innovation.get("ccv")
        fca = innovation.get("fca")
        if not isinstance(ccv, dict) or not isinstance(fca, dict):
            raise AblationBindingError("wo_ccv_fca requires resolved step5.innovation ccv/fca mappings")
        ccv["enabled"] = False
        ccv["numeric_control_weight"] = 0.0
        ccv["route_conditioning"] = False
        fca["enabled"] = False
        fca["weight"] = 0.0
        evidence.update(
            {
                "loss_config_changed": True,
                "ccv": {
                    "enabled": False,
                    "observed_loss_weight": 0.0,
                    "active_loss_term": "not_applicable_no_standalone_ccv_loss",
                    "packet_required_for_main_ce": True,
                },
                "fca": {
                    "enabled": False,
                    "observed_loss_weight": 0.0,
                    "active_loss_term": "disabled_graph_safe_zero",
                },
            }
        )
    return {
        "schema_version": "odcr_ablation_step5_runtime_configs/1",
        "binding": binding_payload,
        "sampler_config": sampler,
        "batch_candidates_config": batch_candidates,
        "tuning_config": tuning,
        "innovation_config": innovation,
        "application_evidence": evidence,
    }


def apply_binding_to_resolved_step5_config(
    binding: AblationBinding,
    cfg: Any,
    snapshot: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Overlay ablation controls onto the resolved Step5/eval runtime payloads.

    This is the single binding point used by formal Step5 train, eval replay,
    and bounded probes. It keeps the source of truth in registry + override +
    manifest, while leaving the primary ``configs/odcr.yaml`` untouched.
    """

    applied = apply_binding_to_step5_runtime_configs(
        binding,
        sampler_config=getattr(cfg, "step5_sampler_config_json", "{}") or "{}",
        batch_candidates_config=getattr(cfg, "step5_batch_candidates_config_json", "{}") or "{}",
        tuning_config=getattr(cfg, "step5_tuning_config_json", "{}") or "{}",
        innovation_config=getattr(cfg, "step5_innovation_config_json", "{}") or "{}",
    )
    replacement_fields: dict[str, Any] = {
        "step5_sampler_config_json": json.dumps(applied["sampler_config"], sort_keys=True),
        "step5_batch_candidates_config_json": json.dumps(applied["batch_candidates_config"], sort_keys=True),
        "step5_tuning_config_json": json.dumps(applied["tuning_config"], sort_keys=True),
        "step5_innovation_config_json": json.dumps(applied["innovation_config"], sort_keys=True),
    }
    application_evidence = applied["application_evidence"]
    payload = _json_clone(getattr(cfg, "effective_training_payload_json", "{}") or "{}", label="effective_training_payload")
    if payload:
        payload["step5_sampler"] = deepcopy(applied["sampler_config"])
        payload["step5_batch_candidates"] = deepcopy(applied["batch_candidates_config"])
        payload["step5_tuning"] = deepcopy(applied["tuning_config"])
        payload["step5_innovation"] = deepcopy(applied["innovation_config"])
        payload["step5_effective_samples"] = dict(applied["tuning_config"].get("effective_samples") or {})
        payload["step5_optimizer_steps"] = dict(applied["tuning_config"].get("optimizer_steps") or {})
        payload["ablation_runtime_binding"] = deepcopy(applied["binding"])
        payload["ablation_runtime_application"] = deepcopy(application_evidence)
        replacement_fields["effective_training_payload_json"] = json.dumps(payload, sort_keys=True)
        replacement_fields["training_semantic_fingerprint"] = stable_hash(
            {
                "schema_version": "odcr_ablation_training_semantic_overlay/1",
                "payload": payload,
                "binding_hash": applied["binding"].get("binding_hash"),
            }
        )
    effective_samples = (
        ((application_evidence.get("cf") or {}).get("effective_samples") or {})
        if isinstance(application_evidence.get("cf"), Mapping)
        else {}
    )
    if isinstance(effective_samples, Mapping) and effective_samples.get("explanation"):
        capped = int(effective_samples["explanation"])
        global_batch_size = max(int(getattr(cfg, "global_batch_size", 1) or 1), 1)
        steps = int(math.ceil(capped / global_batch_size))
        replacement_fields["step5_effective_samples_json"] = json.dumps({"explanation": capped}, sort_keys=True)
        replacement_fields["step5_optimizer_steps_json"] = json.dumps({"explanation": steps}, sort_keys=True)
    cfg = replace(
        cfg,
        **replacement_fields,
    )
    snap = dict(snapshot or {})
    snap["ablation_runtime_binding"] = applied["binding"]
    snap["ablation_runtime_application"] = applied["application_evidence"]
    snap["step5_sampler"] = applied["sampler_config"]
    snap["step5_batch_candidates"] = applied["batch_candidates_config"]
    snap["step5_tuning"] = applied["tuning_config"]
    snap["step5_innovation"] = applied["innovation_config"]
    snap["step5_sampler_config_json"] = replacement_fields["step5_sampler_config_json"]
    snap["step5_batch_candidates_config_json"] = replacement_fields["step5_batch_candidates_config_json"]
    snap["step5_tuning_config_json"] = replacement_fields["step5_tuning_config_json"]
    snap["step5_innovation_config_json"] = replacement_fields["step5_innovation_config_json"]
    if "effective_training_payload_json" in replacement_fields:
        snap["effective_training_payload_json"] = replacement_fields["effective_training_payload_json"]
        snap["training_semantic_fingerprint"] = replacement_fields["training_semantic_fingerprint"]
    if isinstance(effective_samples, Mapping) and effective_samples.get("explanation"):
        capped = int(effective_samples["explanation"])
        global_batch_size = max(int(getattr(cfg, "global_batch_size", 1) or 1), 1)
        steps = int(math.ceil(capped / global_batch_size))
        train = dict(snap.get("train") or {})
        train["effective_samples"] = {"explanation": capped}
        train["optimizer_steps"] = {"explanation": steps}
        train["ablation_effective_sample_cap"] = dict(effective_samples)
        snap["train"] = train
        snap["step5_effective_samples_json"] = replacement_fields["step5_effective_samples_json"]
        snap["step5_optimizer_steps_json"] = replacement_fields["step5_optimizer_steps_json"]
    return cfg, snap


__all__ = [
    "AblationBinding",
    "AblationBindingError",
    "apply_binding_to_resolved_step5_config",
    "apply_binding_to_step5_runtime_configs",
    "is_ablation_run_id",
    "load_ablation_binding",
    "variant_from_ablation_run_id",
]
