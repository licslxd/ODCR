"""Canonical Step5 sample-plan intent.

This module is the single runtime interpretation point for the Step5 tuning
candidate tokens, sampler ratios, tier mixes, effective sample budget, and the
content-affecting sample-plan cache identity.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from odcr_core.training_checkpoint import stable_hash


STEP5_SAMPLE_PLAN_INTENT_SCHEMA_VERSION = "odcr_step5_sample_plan_intent/1"
STEP5_SAMPLE_PLAN_PRODUCER_CODE_VERSION = "odcr_step5_pool_sampler/2_sample_plan_intent"

_COMPONENTS: tuple[str, ...] = ("target_gold", "aux_gold", "cf")
_GOLD_TIERS: tuple[str, ...] = ("high", "medium")
_CF_TIERS: tuple[str, ...] = ("high", "medium", "low_weighted")


class Step5SamplePlanIntentError(RuntimeError):
    """Raised when the resolved Step5 sample-plan intent is inconsistent."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Step5SamplePlanIntentError(f"{label} must be an object")
    return value


def _float(value: Any, label: str, *, min_value: float = 0.0, max_value: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise Step5SamplePlanIntentError(f"{label} must be numeric") from exc
    if out < min_value:
        raise Step5SamplePlanIntentError(f"{label} must be >= {min_value}")
    if max_value is not None and out > max_value:
        raise Step5SamplePlanIntentError(f"{label} must be <= {max_value}")
    return out


def _int(value: Any, label: str, *, min_value: int = 0) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise Step5SamplePlanIntentError(f"{label} must be integer") from exc
    if out < min_value:
        raise Step5SamplePlanIntentError(f"{label} must be >= {min_value}")
    return out


def _bool_exact(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise Step5SamplePlanIntentError(f"{label} must be boolean")
    return bool(value)


def _close(actual: float, expected: float) -> bool:
    return abs(float(actual) - float(expected)) <= 1.0e-6


def _candidate_id_set(rows: Any, label: str) -> set[str]:
    if not isinstance(rows, list):
        raise Step5SamplePlanIntentError(f"{label} must be a resolved candidate list")
    out: set[str] = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Step5SamplePlanIntentError(f"{label}[{idx}] must be an object")
        candidate_id = str(row.get("id") or "").strip()
        if not candidate_id:
            raise Step5SamplePlanIntentError(f"{label}[{idx}].id must be non-empty")
        if candidate_id in out:
            raise Step5SamplePlanIntentError(f"{label} contains duplicate candidate id {candidate_id!r}")
        out.add(candidate_id)
    if not out:
        raise Step5SamplePlanIntentError(f"{label} must not be empty")
    return out


def _tokens_for_tuning_candidate(candidate: str, tuning: Mapping[str, Any]) -> dict[str, str]:
    tokens = {str(part).strip() for part in str(candidate or "").split("+") if str(part).strip()}
    ratio_rows = (tuning.get("ratio_candidates") or {}).get("explanation") if isinstance(tuning.get("ratio_candidates"), Mapping) else None
    cf_mix_rows = (tuning.get("cf_tier_mix_candidates") or {}).get("explanation") if isinstance(tuning.get("cf_tier_mix_candidates"), Mapping) else None
    gold_rows = tuning.get("gold_tier_mix_candidates")
    target_mix_rows = (gold_rows or {}).get("target_gold") if isinstance(gold_rows, Mapping) else None
    aux_mix_rows = (gold_rows or {}).get("aux_gold") if isinstance(gold_rows, Mapping) else None
    weight_rows = tuning.get("innovation_weight_candidates")
    out: dict[str, str] = {}
    for candidate_ids, key, label in (
        (_candidate_id_set(ratio_rows, "step5.tuning.ratio_candidates.explanation"), "ratio", "ratio candidate id"),
        (_candidate_id_set(cf_mix_rows, "step5.tuning.cf_tier_mix_candidates.explanation"), "cf_mix", "cf tier mix candidate id"),
        (_candidate_id_set(target_mix_rows, "step5.tuning.gold_tier_mix_candidates.target_gold"), "target_gold_mix", "target gold tier mix candidate id"),
        (_candidate_id_set(aux_mix_rows, "step5.tuning.gold_tier_mix_candidates.aux_gold"), "aux_gold_mix", "aux gold tier mix candidate id"),
        (_candidate_id_set(weight_rows, "step5.tuning.innovation_weight_candidates"), "weight", "innovation weight candidate id"),
    ):
        matches = sorted(token for token in tokens if token in candidate_ids)
        if len(matches) != 1:
            raise Step5SamplePlanIntentError(
                f"step5.tuning.selected_tuning_candidate must contain exactly one {label}; got {candidate!r}"
            )
        out[key] = matches[0]
    for matcher, key, label in (
        (lambda token: token.startswith("LR_"), "lr", "LR_"),
    ):
        matches = sorted(token for token in tokens if matcher(token))
        if len(matches) != 1:
            raise Step5SamplePlanIntentError(
                f"step5.tuning.selected_tuning_candidate must contain exactly one {label} token; got {candidate!r}"
            )
        out[key] = matches[0]
    return out


def step5_candidate_tokens_from_tuning(tuning_config: Mapping[str, Any]) -> dict[str, str]:
    tuning = _mapping(tuning_config, "step5.tuning")
    return _tokens_for_tuning_candidate(str(tuning.get("selected_tuning_candidate") or ""), tuning)


def _candidate_row_by_id(rows: Any, wanted: str, label: str) -> Mapping[str, Any]:
    if not isinstance(rows, list):
        raise Step5SamplePlanIntentError(f"{label} must be a resolved candidate list")
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("id")) == wanted:
            return row
    raise Step5SamplePlanIntentError(f"selected Step5 candidate id {wanted!r} missing from {label}")


def selected_step5_candidate_rows(tuning_config: Mapping[str, Any]) -> dict[str, Mapping[str, Any] | dict[str, str]]:
    tuning = _mapping(tuning_config, "step5.tuning")
    tokens = step5_candidate_tokens_from_tuning(tuning)
    return {
        "tokens": tokens,
        "ratio": _candidate_row_by_id(
            (tuning.get("ratio_candidates") or {}).get("explanation") if isinstance(tuning.get("ratio_candidates"), Mapping) else None,
            tokens["ratio"],
            "step5.tuning.ratio_candidates.explanation",
        ),
        "cf_mix": _candidate_row_by_id(
            (tuning.get("cf_tier_mix_candidates") or {}).get("explanation") if isinstance(tuning.get("cf_tier_mix_candidates"), Mapping) else None,
            tokens["cf_mix"],
            "step5.tuning.cf_tier_mix_candidates.explanation",
        ),
        "target_gold_mix": _candidate_row_by_id(
            ((tuning.get("gold_tier_mix_candidates") or {}).get("target_gold") if isinstance(tuning.get("gold_tier_mix_candidates"), Mapping) else None),
            tokens["target_gold_mix"],
            "step5.tuning.gold_tier_mix_candidates.target_gold",
        ),
        "aux_gold_mix": _candidate_row_by_id(
            ((tuning.get("gold_tier_mix_candidates") or {}).get("aux_gold") if isinstance(tuning.get("gold_tier_mix_candidates"), Mapping) else None),
            tokens["aux_gold_mix"],
            "step5.tuning.gold_tier_mix_candidates.aux_gold",
        ),
        "weights": _candidate_row_by_id(
            tuning.get("innovation_weight_candidates"),
            tokens["weight"],
            "step5.tuning.innovation_weight_candidates",
        ),
    }


def cf_mix_is_low_weighted_only(row: Mapping[str, Any]) -> bool:
    return (
        _close(_float(row.get("high"), "step5.tuning.cf_tier_mix_candidates.selected.high"), 0.0)
        and _close(_float(row.get("medium"), "step5.tuning.cf_tier_mix_candidates.selected.medium"), 0.0)
        and _float(row.get("low_weighted"), "step5.tuning.cf_tier_mix_candidates.selected.low_weighted") > 0.0
    )


def gold_mix_is_medium_only(row: Mapping[str, Any]) -> bool:
    return (
        _close(_float(row.get("high"), "step5.tuning.gold_tier_mix_candidates.selected.high"), 0.0)
        and _float(row.get("medium"), "step5.tuning.gold_tier_mix_candidates.selected.medium") > 0.0
    )


def _assert_close_mapping(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    fields: tuple[str, ...],
    label: str,
) -> None:
    for field in fields:
        actual_value = _float(actual.get(field), f"{label}.{field}")
        expected_value = _float(expected.get(field), f"{label}.selected.{field}")
        if not _close(actual_value, expected_value):
            raise Step5SamplePlanIntentError(
                f"{label}.{field}={actual_value} does not match selected Step5 sample-plan candidate value {expected_value}"
            )


def normalize_step5_intent_head(raw: Any) -> str:
    head = str(raw or "explanation").strip()
    if head != "explanation":
        raise Step5SamplePlanIntentError(f"unsupported Step5 sample-plan head: {head}")
    return head


def _selected_batch_payload(batch_candidates_config: Mapping[str, Any] | None, batch_candidate: str) -> dict[str, Any]:
    batch_candidates = _mapping(batch_candidates_config, "step5.batch_candidates")
    selected = str(batch_candidate or "").strip()
    if not selected:
        raise Step5SamplePlanIntentError("step5.tuning.batch_candidate must be non-empty")
    candidates = batch_candidates.get("candidates")
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, Mapping) and str(item.get("id") or "") == selected:
                return {
                    "id": selected,
                    "per_gpu_batch_size": int(item.get("per_gpu_batch_size") or 0),
                    "global_batch_size": int(item.get("global_batch_size") or 0),
                }
    raise Step5SamplePlanIntentError(f"step5.tuning.batch_candidate {selected!r} missing from step5.batch_candidates")


def _component_ratios(head_cfg: Mapping[str, Any]) -> dict[str, float]:
    ratios = {
        "target_gold": _float(head_cfg.get("target_gold_ratio"), "step5.sampler.explanation.target_gold_ratio", max_value=1.0),
        "aux_gold": _float(head_cfg.get("aux_gold_ratio"), "step5.sampler.explanation.aux_gold_ratio", max_value=1.0),
        "cf": _float(head_cfg.get("cf_ratio"), "step5.sampler.explanation.cf_ratio", max_value=1.0),
    }
    total = sum(ratios.values())
    if not _close(total, 1.0):
        raise Step5SamplePlanIntentError(f"step5.sampler.explanation ratios must sum to 1.0, got {total}")
    return ratios


def _component_mix(head_cfg: Mapping[str, Any], component: str) -> dict[str, float]:
    key = {
        "target_gold": "target_gold_tier_mix",
        "aux_gold": "aux_gold_tier_mix",
        "cf": "cf_tier_mix",
    }[component]
    tiers = _GOLD_TIERS if component in {"target_gold", "aux_gold"} else _CF_TIERS
    raw = _mapping(head_cfg.get(key), f"step5.sampler.explanation.{key}")
    out = {
        tier: _float(raw.get(tier), f"step5.sampler.explanation.{key}.{tier}", max_value=1.0)
        for tier in tiers
    }
    if sum(out.values()) <= 0.0:
        raise Step5SamplePlanIntentError(f"step5.sampler.explanation.{key} must have at least one positive tier")
    return out


def build_step5_sample_plan_intent(
    *,
    sampler_config: Mapping[str, Any],
    tuning_config: Mapping[str, Any],
    batch_candidates_config: Mapping[str, Any] | None = None,
    task_head: str = "explanation",
) -> dict[str, Any]:
    """Build and validate the content-affecting Step5 sample-plan intent."""

    sampler = _mapping(sampler_config, "step5.sampler")
    tuning = _mapping(tuning_config, "step5.tuning")
    head = normalize_step5_intent_head(task_head)
    if sampler.get("enabled") is not True:
        raise Step5SamplePlanIntentError("step5.sampler.enabled must be true")
    if sampler.get("contract_source") != "step4_pool_manifest":
        raise Step5SamplePlanIntentError("step5.sampler.contract_source must be step4_pool_manifest")

    selected_rows = selected_step5_candidate_rows(tuning)
    tokens = dict(selected_rows["tokens"])
    head_cfg = _mapping(sampler.get(head), f"step5.sampler.{head}")
    ratios = _component_ratios(head_cfg)
    target_gold_mix = _component_mix(head_cfg, "target_gold")
    aux_gold_mix = _component_mix(head_cfg, "aux_gold")
    cf_mix = _component_mix(head_cfg, "cf")
    weights = {
        "aux_gold_weight": _float(head_cfg.get("aux_gold_weight"), "step5.sampler.explanation.aux_gold_weight"),
        "cf_high_weight": _float(head_cfg.get("cf_high_weight"), "step5.sampler.explanation.cf_high_weight"),
        "cf_medium_weight": _float(head_cfg.get("cf_medium_weight"), "step5.sampler.explanation.cf_medium_weight"),
        "cf_low_weight": _float(head_cfg.get("cf_low_weight"), "step5.sampler.explanation.cf_low_weight"),
    }

    ratio_row = _mapping(selected_rows["ratio"], "step5.tuning.ratio_candidates.selected")
    _assert_close_mapping(ratios, ratio_row, fields=_COMPONENTS, label="step5.sampler.explanation")

    cf_mix_row = _mapping(selected_rows["cf_mix"], "step5.tuning.cf_tier_mix_candidates.selected")
    _assert_close_mapping(cf_mix, cf_mix_row, fields=_CF_TIERS, label="step5.sampler.explanation.cf_tier_mix")

    target_mix_row = _mapping(selected_rows["target_gold_mix"], "step5.tuning.gold_tier_mix_candidates.target_gold.selected")
    _assert_close_mapping(
        target_gold_mix,
        target_mix_row,
        fields=_GOLD_TIERS,
        label="step5.sampler.explanation.target_gold_tier_mix",
    )
    aux_mix_row = _mapping(selected_rows["aux_gold_mix"], "step5.tuning.gold_tier_mix_candidates.aux_gold.selected")
    _assert_close_mapping(
        aux_gold_mix,
        aux_mix_row,
        fields=_GOLD_TIERS,
        label="step5.sampler.explanation.aux_gold_tier_mix",
    )

    effective_samples = {
        str(key): _int(value, f"step5.tuning.effective_samples.{key}", min_value=1)
        for key, value in (_mapping(tuning.get("effective_samples"), "step5.tuning.effective_samples")).items()
    }
    optimizer_steps = {
        str(key): _int(value, f"step5.tuning.optimizer_steps.{key}", min_value=1)
        for key, value in (_mapping(tuning.get("optimizer_steps"), "step5.tuning.optimizer_steps")).items()
    }
    batch_candidate = str(tuning.get("batch_candidate") or "").strip()
    selected_batch = _selected_batch_payload(batch_candidates_config, batch_candidate)
    epochs_cfg = _mapping(sampler.get("epochs"), "step5.sampler.epochs")
    max_effective_epochs = _int(epochs_cfg.get("max_effective_epochs"), "step5.sampler.epochs.max_effective_epochs", min_value=1)
    content_identity = {
        "schema_version": STEP5_SAMPLE_PLAN_INTENT_SCHEMA_VERSION,
        "producer_code_version": STEP5_SAMPLE_PLAN_PRODUCER_CODE_VERSION,
        "head": head,
        "selected_tuning_candidate": str(tuning.get("selected_tuning_candidate") or ""),
        "candidate_parts": dict(tokens),
        "sampler_protocol": tokens["cf_mix"],
        "component_ratios": ratios,
        "target_gold_tier_mix": target_gold_mix,
        "aux_gold_tier_mix": aux_gold_mix,
        "cf_tier_mix": cf_mix,
        "weights": weights,
        "seed": _int(sampler.get("seed"), "step5.sampler.seed", min_value=0),
        "rotate_across_epochs": _bool_exact(sampler.get("rotate_across_epochs"), "step5.sampler.rotate_across_epochs"),
        "effective_epoch_enabled": _bool_exact(sampler.get("effective_epoch_enabled"), "step5.sampler.effective_epoch_enabled"),
        "max_effective_epochs": int(max_effective_epochs),
        "contract_source": str(sampler.get("contract_source") or ""),
        "selected_budget_candidate": str(tuning.get("selected_budget_candidate") or ""),
        "batch_candidate": batch_candidate,
        "selected_batch": selected_batch,
        "effective_samples": effective_samples,
        "optimizer_steps": optimizer_steps,
        "auto_budget": dict(_mapping(sampler.get("auto_budget"), "step5.sampler.auto_budget")),
    }
    intent_hash = stable_hash(content_identity)
    return {
        **content_identity,
        "sample_plan_intent_hash": intent_hash,
        "content_identity": content_identity,
        "weak_cross_platform_protocol": cf_mix_is_low_weighted_only(cf_mix_row),
        "full_audit_default_forbidden": True,
        "step4_sampling_contract_role": "pool_lineage_only",
        "active_sampler_source": (
            f"{sampler.get('task_override_source') or 'step5.sampler'} + "
            f"{tuning.get('selected_tuning_candidate_source') or 'step5.tuning.selected_tuning_candidate'}"
        ),
    }


def head_config_from_sample_plan_intent(intent: Mapping[str, Any]) -> dict[str, Any]:
    """Return the sampler head config that the runtime must actually execute."""

    ratios = _mapping(intent.get("component_ratios"), "sample_plan_intent.component_ratios")
    weights = _mapping(intent.get("weights"), "sample_plan_intent.weights")
    default_candidate = str(intent.get("selected_budget_candidate") or "").strip()
    if not default_candidate:
        raise Step5SamplePlanIntentError("sample_plan_intent.selected_budget_candidate must be non-empty")
    return {
        "default_candidate": default_candidate,
        "target_gold_ratio": _float(ratios.get("target_gold"), "sample_plan_intent.component_ratios.target_gold", max_value=1.0),
        "aux_gold_ratio": _float(ratios.get("aux_gold"), "sample_plan_intent.component_ratios.aux_gold", max_value=1.0),
        "cf_ratio": _float(ratios.get("cf"), "sample_plan_intent.component_ratios.cf", max_value=1.0),
        "target_gold_tier_mix": dict(_mapping(intent.get("target_gold_tier_mix"), "sample_plan_intent.target_gold_tier_mix")),
        "aux_gold_tier_mix": dict(_mapping(intent.get("aux_gold_tier_mix"), "sample_plan_intent.aux_gold_tier_mix")),
        "cf_tier_mix": dict(_mapping(intent.get("cf_tier_mix"), "sample_plan_intent.cf_tier_mix")),
        "aux_gold_weight": _float(weights.get("aux_gold_weight"), "sample_plan_intent.weights.aux_gold_weight"),
        "cf_high_weight": _float(weights.get("cf_high_weight"), "sample_plan_intent.weights.cf_high_weight"),
        "cf_medium_weight": _float(weights.get("cf_medium_weight"), "sample_plan_intent.weights.cf_medium_weight"),
        "cf_low_weight": _float(weights.get("cf_low_weight"), "sample_plan_intent.weights.cf_low_weight"),
    }


def component_counts_from_ratios(total: int, ratios: Mapping[str, Any]) -> dict[str, int]:
    total = int(total)
    ratio_map = _mapping(ratios, "sample_plan_intent.component_ratios")
    target = int(round(total * _float(ratio_map.get("target_gold"), "sample_plan_intent.component_ratios.target_gold", max_value=1.0)))
    aux = int(round(total * _float(ratio_map.get("aux_gold"), "sample_plan_intent.component_ratios.aux_gold", max_value=1.0)))
    cf = total - target - aux
    if cf < 0:
        cf = 0
        overflow = target + aux - total
        aux = max(0, aux - overflow)
    return {"target_gold": int(target), "aux_gold": int(aux), "cf": int(cf)}


def tier_counts_from_mix(total: int, mix: Mapping[str, Any], tiers: tuple[str, ...]) -> dict[str, int]:
    total = int(total)
    if total <= 0:
        return {tier: 0 for tier in tiers}
    weights = {tier: max(0.0, float(mix.get(tier, 0.0))) for tier in tiers}
    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        raise Step5SamplePlanIntentError("Step5 tier mix must contain at least one positive tier")
    raw = {tier: total * (weights[tier] / weight_sum) for tier in tiers}
    out = {tier: int(raw[tier] // 1) for tier in tiers}
    remainder = total - sum(out.values())
    if remainder > 0:
        order = sorted(
            [tier for tier in tiers if weights[tier] > 0.0],
            key=lambda tier: (-(raw[tier] - out[tier]), tiers.index(tier)),
        )
        for tier in order[:remainder]:
            out[tier] += 1
    return out


def assert_sampled_plan_matches_intent(
    df: pd.DataFrame,
    *,
    stats: Mapping[str, Any] | None,
    intent: Mapping[str, Any],
    context: str,
) -> None:
    """Fail if sampled rows do not match the resolved Step5 sample-plan intent."""

    if df is None or df.empty:
        raise Step5SamplePlanIntentError(f"{context}: sampled plan is empty")
    missing = [col for col in ("sampler_component", "sampler_tier", "effective_epoch") if col not in df.columns]
    if missing:
        raise Step5SamplePlanIntentError(f"{context}: sampled plan missing columns: {', '.join(missing)}")
    head = normalize_step5_intent_head(intent.get("head") or intent.get("task_head") or "explanation")
    effective_samples = int((intent.get("effective_samples") or {}).get(head) or 0)
    if effective_samples <= 0 and isinstance(stats, Mapping):
        effective_samples = int(stats.get("effective_samples_per_epoch") or 0)
    if effective_samples <= 0:
        raise Step5SamplePlanIntentError(f"{context}: missing effective sample budget in Step5 sample-plan intent")
    ratios = _mapping(intent.get("component_ratios"), "sample_plan_intent.component_ratios")
    expected_components = component_counts_from_ratios(effective_samples, ratios)
    expected_tiers = {
        "target_gold": tier_counts_from_mix(expected_components["target_gold"], _mapping(intent.get("target_gold_tier_mix"), "sample_plan_intent.target_gold_tier_mix"), _GOLD_TIERS),
        "aux_gold": tier_counts_from_mix(expected_components["aux_gold"], _mapping(intent.get("aux_gold_tier_mix"), "sample_plan_intent.aux_gold_tier_mix"), _GOLD_TIERS),
        "cf": tier_counts_from_mix(expected_components["cf"], _mapping(intent.get("cf_tier_mix"), "sample_plan_intent.cf_tier_mix"), _CF_TIERS),
    }
    actual_epochs = sorted(int(epoch) for epoch in pd.to_numeric(df["effective_epoch"], errors="raise").unique())
    expected_epoch_count = int((stats or {}).get("max_effective_epochs") or intent.get("max_effective_epochs") or len(actual_epochs))
    if len(actual_epochs) != expected_epoch_count:
        raise Step5SamplePlanIntentError(
            f"{context}: effective epoch count {len(actual_epochs)} does not match resolved intent {expected_epoch_count}"
        )
    for epoch in actual_epochs:
        epoch_df = df.loc[pd.to_numeric(df["effective_epoch"], errors="raise").astype(int) == epoch]
        if len(epoch_df) != effective_samples:
            raise Step5SamplePlanIntentError(
                f"{context}: epoch {epoch} row count {len(epoch_df)} does not match resolved effective_samples {effective_samples}"
            )
        actual_components = {
            component: int((epoch_df["sampler_component"].astype(str) == component).sum())
            for component in _COMPONENTS
        }
        if actual_components != expected_components:
            raise Step5SamplePlanIntentError(
                f"{context}: epoch {epoch} component counts {actual_components} do not match resolved intent {expected_components}"
            )
        for component, tiers in expected_tiers.items():
            comp_df = epoch_df.loc[epoch_df["sampler_component"].astype(str) == component]
            actual_tiers = {
                tier: int((comp_df["sampler_tier"].astype(str) == tier).sum())
                for tier in (_GOLD_TIERS if component in {"target_gold", "aux_gold"} else _CF_TIERS)
            }
            if actual_tiers != tiers:
                raise Step5SamplePlanIntentError(
                    f"{context}: epoch {epoch} {component} tier counts {actual_tiers} do not match resolved intent {tiers}"
                )


def sample_plan_intent_cache_identity(intent: Mapping[str, Any]) -> dict[str, Any]:
    content_identity = intent.get("content_identity")
    if isinstance(content_identity, Mapping):
        return dict(content_identity)
    raise Step5SamplePlanIntentError("sample_plan_intent.content_identity is required for cache identity")
