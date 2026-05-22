"""Step5 effective-epoch sample budget planning.

The planner is intentionally CPU-only and side-effect free: it reads the
Step4 pool manifest plus resolved One-Control sampler/batch/tuning payloads and
returns bounded/formal preparation metadata.  It never reads the full audit
table and never writes formal run state.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping


STEP5_AUTO_BUDGET_SCHEMA_VERSION = "odcr_step5_auto_budget/1"
STEP5_TUNING_SCHEMA_VERSION = "odcr_step5_tuning/1"

HEADS: tuple[str, ...] = ("explanation",)
BUDGET_ORDER: tuple[str, ...] = ("small", "medium", "full", "large")
COMPONENTS: tuple[str, ...] = ("target_gold", "aux_gold", "cf")
GOLD_TIERS: tuple[str, ...] = ("high", "medium")
CF_TIERS: tuple[str, ...] = ("high", "medium", "low_weighted")


class Step5utoBudgetError(RuntimeError):
    """Raised when Step5 auto-budget inputs are missing or inconsistent."""


def _pool_name(head: str, component: str, tier: str) -> str:
    if head != "explanation":
        raise Step5utoBudgetError(f"unsupported Step5 head: {head}")
    prefix = "step5_explanation"
    if component == "target_gold":
        return f"{prefix}_target_gold_anchor_{tier}"
    if component == "aux_gold":
        return f"{prefix}_aux_gold_anchor_{tier}"
    if component == "cf":
        return f"{prefix}_cf_explainer_{tier}"
    raise Step5utoBudgetError(f"unknown Step5 component: {component}")

def _pool_count(manifest: Mapping[str, Any], name: str) -> int:
    pools = manifest.get("pools")
    if not isinstance(pools, Mapping):
        raise Step5utoBudgetError("Step5 pool manifest missing pools object")
    item = pools.get(name)
    if not isinstance(item, Mapping):
        item = pools.get(name.replace("step5_explanation", "step5" + "B", 1))
    if not isinstance(item, Mapping):
        raise Step5utoBudgetError(f"Step5 pool manifest missing pool: {name}")
    try:
        count = int(item.get("row_count") or 0)
    except Exception as exc:
        raise Step5utoBudgetError(f"Step5 pool row_count must be integer for {name}") from exc
    if count < 0:
        raise Step5utoBudgetError(f"Step5 pool row_count must be non-negative for {name}")
    return count


def _default_tier_mix(component: str, tiers: Mapping[str, int]) -> dict[str, float]:
    return {str(tier): 1.0 for tier in tiers}


def _tier_mix_for_component(
    head_cfg: Mapping[str, Any] | None,
    component: str,
    tiers: Mapping[str, int],
) -> dict[str, float]:
    """Return the active tier mix for a component.

    Auto-budget capacity must use the same active tier policy as the sampler.
    A zero tier weight is therefore unavailable for mainline effective epochs,
    even when the Step4 pool manifest contains rows for that tier.
    """

    if not isinstance(head_cfg, Mapping):
        return _default_tier_mix(component, tiers)
    key = {
        "target_gold": "target_gold_tier_mix",
        "aux_gold": "aux_gold_tier_mix",
        "cf": "cf_tier_mix",
    }[component]
    raw = head_cfg.get(key)
    if not isinstance(raw, Mapping):
        return _default_tier_mix(component, tiers)
    mix: dict[str, float] = {}
    for tier in tiers:
        value = float(raw.get(tier) or 0.0)
        if value < 0.0:
            raise Step5utoBudgetError(f"Step5 {component} tier mix must be non-negative for {tier}")
        mix[str(tier)] = value
    if not any(value > 0.0 for value in mix.values()):
        raise Step5utoBudgetError(f"Step5 {component} tier mix has no active tier")
    return mix


def _active_tier_total(tiers: Mapping[str, int], tier_mix: Mapping[str, float]) -> int:
    return int(sum(int(count or 0) for tier, count in tiers.items() if float(tier_mix.get(tier) or 0.0) > 0.0))


def pool_capacity_from_manifest(
    manifest: Mapping[str, Any],
    head: str,
    *,
    head_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the capacity components required by the Step5 auto-budget rule."""

    if head not in HEADS:
        raise Step5utoBudgetError(f"unsupported Step5 head: {head}")
    target_gold = {
        "high": _pool_count(manifest, _pool_name(head, "target_gold", "high")),
        "medium": _pool_count(manifest, _pool_name(head, "target_gold", "medium")),
    }
    aux_gold = {
        "high": _pool_count(manifest, _pool_name(head, "aux_gold", "high")),
        "medium": _pool_count(manifest, _pool_name(head, "aux_gold", "medium")),
    }
    cf = {
        "high": _pool_count(manifest, _pool_name(head, "cf", "high")),
        "medium": _pool_count(manifest, _pool_name(head, "cf", "medium")),
        "low_weighted": _pool_count(manifest, _pool_name(head, "cf", "low_weighted")),
    }
    tier_available = {"target_gold": target_gold, "aux_gold": aux_gold, "cf": cf}
    tier_mix = {
        key: _tier_mix_for_component(head_cfg, key, tiers)
        for key, tiers in tier_available.items()
    }
    active_tiers = {
        key: [tier for tier, value in mix.items() if float(value) > 0.0]
        for key, mix in tier_mix.items()
    }
    return {
        "target_gold": target_gold,
        "aux_gold": aux_gold,
        "cf": cf,
        "tier_available": tier_available,
        "tier_mix": tier_mix,
        "active_tiers": active_tiers,
        "available_all": {
            "target_gold": int(sum(target_gold.values())),
            "aux_gold": int(sum(aux_gold.values())),
            "cf": int(sum(cf.values())),
        },
        "available": {
            key: _active_tier_total(tiers, tier_mix[key])
            for key, tiers in tier_available.items()
        },
    }


def _ratio_payload(head_cfg: Mapping[str, Any], ratio_override: Mapping[str, Any] | None = None) -> dict[str, float]:
    raw = dict(ratio_override or {})
    if raw:
        ratios = {
            "target_gold": float(raw.get("target_gold", raw.get("target_gold_ratio", 0.0))),
            "aux_gold": float(raw.get("aux_gold", raw.get("aux_gold_ratio", 0.0))),
            "cf": float(raw.get("cf", raw.get("cf_ratio", raw.get("CF", 0.0)))),
        }
    else:
        ratios = {
            "target_gold": float(head_cfg.get("target_gold_ratio", 0.0)),
            "aux_gold": float(head_cfg.get("aux_gold_ratio", 0.0)),
            "cf": float(head_cfg.get("cf_ratio", 0.0)),
        }
    total = sum(ratios.values())
    if total <= 0.0:
        raise Step5utoBudgetError("Step5 sampler ratios must have positive sum")
    if abs(total - 1.0) > 1e-6:
        raise Step5utoBudgetError(f"Step5 sampler ratios must sum to 1.0, got {total}")
    if any(value < 0.0 for value in ratios.values()):
        raise Step5utoBudgetError("Step5 sampler ratios must be non-negative")
    return ratios


def _ceil_div(numerator: int, denominator: int) -> int:
    return int(math.ceil(float(numerator) / float(max(denominator, 1))))


def _component_requests(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    requested: dict[str, int] = {}
    running = 0
    keys = list(COMPONENTS)
    for key in keys[:-1]:
        count = int(round(int(total) * float(ratios[key])))
        requested[key] = count
        running += count
    requested[keys[-1]] = max(0, int(total) - running)
    return requested


def _weighted_tier_requests(component_total: int, tier_mix: Mapping[str, float]) -> dict[str, int]:
    active = [(str(tier), float(weight)) for tier, weight in tier_mix.items() if float(weight) > 0.0]
    if not active:
        raise Step5utoBudgetError("Step5 tier request needs at least one active tier")
    total_weight = sum(weight for _tier, weight in active)
    if total_weight <= 0.0:
        raise Step5utoBudgetError("Step5 tier mix weights must sum to a positive value")
    assigned: dict[str, int] = {}
    running = 0
    for tier, weight in active[:-1]:
        count = int(round(int(component_total) * (weight / total_weight)))
        assigned[tier] = count
        running += count
    assigned[active[-1][0]] = max(0, int(component_total) - running)
    for tier in tier_mix:
        assigned.setdefault(str(tier), 0)
    return assigned


def _tier_replacement_payload(
    requested: Mapping[str, int],
    tier_available: Mapping[str, Mapping[str, int]],
    tier_mix: Mapping[str, Mapping[str, float]],
) -> tuple[float, int, dict[str, float], dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    replacement_extra = 0
    by_component: dict[str, float] = {}
    requested_by_tier: dict[str, dict[str, int]] = {}
    by_tier: dict[str, dict[str, float]] = {}
    for key in COMPONENTS:
        req = int(requested.get(key) or 0)
        tier_requests = _weighted_tier_requests(req, tier_mix.get(key) or {})
        requested_by_tier[key] = tier_requests
        component_extra = 0
        component_by_tier: dict[str, float] = {}
        for tier, tier_req in tier_requests.items():
            avail = int((tier_available.get(key) or {}).get(tier) or 0)
            extra = max(0, int(tier_req) - avail)
            component_extra += extra
            component_by_tier[tier] = float(extra) / float(max(int(tier_req), 1))
        replacement_extra += component_extra
        by_component[key] = float(component_extra) / float(max(req, 1))
        by_tier[key] = component_by_tier
    total = int(sum(int(requested.get(key) or 0) for key in COMPONENTS))
    return (
        float(replacement_extra) / float(max(total, 1)),
        int(replacement_extra),
        by_component,
        requested_by_tier,
        by_tier,
    )


def _replacement_payload(
    requested: Mapping[str, int],
    available: Mapping[str, int],
) -> tuple[float, int, dict[str, float]]:
    replacement_extra = 0
    by_component: dict[str, float] = {}
    for key in COMPONENTS:
        req = int(requested.get(key) or 0)
        avail = int(available.get(key) or 0)
        extra = max(0, req - avail)
        replacement_extra += extra
        by_component[key] = float(extra) / float(max(req, 1))
    total = int(sum(int(requested.get(key) or 0) for key in COMPONENTS))
    return float(replacement_extra) / float(max(total, 1)), int(replacement_extra), by_component


def _max_samples_for_replacement(
    available: Mapping[str, int],
    ratios: Mapping[str, float],
    max_replacement_rate: float,
) -> int:
    caps: list[float] = []
    keep_rate = max(1.0e-9, 1.0 - float(max_replacement_rate))
    for key in COMPONENTS:
        ratio = float(ratios.get(key) or 0.0)
        if ratio <= 0.0:
            continue
        caps.append(float(available.get(key) or 0) / (ratio * keep_rate))
    if not caps:
        raise Step5utoBudgetError("Step5 auto-budget has no positive ratio component")
    return max(1, int(math.floor(min(caps))))


def _selected_batch(
    batch_candidates_config: Mapping[str, Any],
    tuning_config: Mapping[str, Any] | None,
    batch_candidate: str | None = None,
) -> dict[str, Any]:
    tuning = tuning_config if isinstance(tuning_config, Mapping) else {}
    selected = str(batch_candidate or tuning.get("batch_candidate") or batch_candidates_config.get("selected_default") or "").strip()
    candidates = batch_candidates_config.get("candidates")
    if not isinstance(candidates, list):
        raise Step5utoBudgetError("step5.batch_candidates.candidates must be resolved before auto-budget planning")
    for item in candidates:
        if isinstance(item, Mapping) and str(item.get("id") or "") == selected:
            return {
                "id": selected,
                "per_gpu_batch_size": int(item.get("per_gpu_batch_size") or 0),
                "global_batch_size": int(item.get("global_batch_size") or 0),
            }
    raise Step5utoBudgetError(f"Step5 tuning batch candidate is not resolved: {selected}")


def compute_head_auto_budget(
    manifest: Mapping[str, Any],
    *,
    head: str,
    head_cfg: Mapping[str, Any],
    auto_budget_cfg: Mapping[str, Any],
    global_batch_size: int,
    selected_budget_candidate: str | None = None,
    ratio_override: Mapping[str, Any] | None = None,
    throughput_samples_per_sec: float | None = None,
) -> dict[str, Any]:
    """Compute balanced capacity and budget candidates for one Step5 head."""

    if auto_budget_cfg.get("enabled") is not True:
        raise Step5utoBudgetError("step5.sampler.auto_budget.enabled must be true")
    if str(auto_budget_cfg.get("capacity_basis") or "") != "balanced_capacity":
        raise Step5utoBudgetError("step5.sampler.auto_budget.capacity_basis must be balanced_capacity")
    global_batch = int(global_batch_size)
    if global_batch <= 0:
        raise Step5utoBudgetError("global_batch_size must be positive for Step5 auto-budget")
    capacity = pool_capacity_from_manifest(manifest, head, head_cfg=head_cfg)
    available = dict(capacity["available"])
    tier_available = {
        key: dict(value)
        for key, value in (capacity.get("tier_available") or {}).items()
    }
    tier_mix = {
        key: dict(value)
        for key, value in (capacity.get("tier_mix") or {}).items()
    }
    ratios = _ratio_payload(head_cfg, ratio_override)
    balanced_terms = {
        key: float(available[key]) / float(ratios[key])
        for key in COMPONENTS
        if float(ratios[key]) > 0.0
    }
    balanced_capacity = int(math.floor(min(balanced_terms.values())))
    multipliers = auto_budget_cfg.get("budget_multipliers")
    if not isinstance(multipliers, Mapping):
        raise Step5utoBudgetError("step5.sampler.auto_budget.budget_multipliers missing")
    min_steps = int(auto_budget_cfg.get("min_steps_per_effective_epoch") or 1)
    preferred = auto_budget_cfg.get("preferred_steps_per_effective_epoch")
    if not isinstance(preferred, list) or len(preferred) != 2:
        raise Step5utoBudgetError("step5.sampler.auto_budget.preferred_steps_per_effective_epoch must be a pair")
    preferred_low = int(preferred[0])
    preferred_high = int(preferred[1])
    max_steps = int(auto_budget_cfg.get("max_steps_per_effective_epoch") or preferred_high)
    max_replacement = float(auto_budget_cfg.get("max_replacement_rate") or 0.0)
    replacement_cap = _max_samples_for_replacement(available, ratios, max_replacement)
    candidates: dict[str, Any] = {}
    for name in BUDGET_ORDER:
        if name not in multipliers:
            raise Step5utoBudgetError(f"missing Step5 budget multiplier: {name}")
        raw_samples = max(1, int(round(float(balanced_capacity) * float(multipliers[name]))))
        adjusted = raw_samples
        adjustments: list[str] = []
        min_samples = int(min_steps) * global_batch
        max_samples = int(max_steps) * global_batch
        if adjusted < min_samples:
            adjusted = min_samples
            adjustments.append("raised_to_min_steps")
        if adjusted > max_samples:
            adjusted = max_samples
            adjustments.append("clamped_to_max_steps")
        if adjusted > replacement_cap:
            adjusted = replacement_cap
            adjustments.append("clamped_to_replacement_guard")
        steps = _ceil_div(adjusted, global_batch)
        requested = _component_requests(adjusted, ratios)
        repl_rate, repl_extra, repl_by_component = _replacement_payload(requested, available)
        tier_shortage_rate, tier_shortage_extra, tier_shortage_by_component, requested_by_tier, tier_shortage_by_tier = _tier_replacement_payload(
            requested,
            tier_available,
            tier_mix,
        )
        if steps < min_steps:
            adjustments.append("below_min_steps_after_guards")
        if steps > max_steps:
            adjustments.append("above_max_steps_after_guards")
        if repl_rate > max_replacement + 1e-9:
            adjustments.append("replacement_rate_exceeds_guard")
        estimated_seconds = None
        if throughput_samples_per_sec is not None and float(throughput_samples_per_sec) > 0.0:
            estimated_seconds = float(adjusted) / float(throughput_samples_per_sec)
        candidates[name] = {
            "budget_candidate": name,
            "multiplier": float(multipliers[name]),
            "raw_effective_samples": int(raw_samples),
            "effective_samples": int(adjusted),
            "global_batch_size": int(global_batch),
            "optimizer_steps": int(steps),
            "preferred_steps_ok": bool(preferred_low <= steps <= preferred_high),
            "min_steps_ok": bool(steps >= min_steps),
            "max_steps_ok": bool(steps <= max_steps),
            "requested_counts": requested,
            "requested_tier_counts": requested_by_tier,
            "unique_sample_count": int(adjusted - repl_extra),
            "replacement_extra": int(repl_extra),
            "replacement_rate": float(repl_rate),
            "replacement_rate_by_component": repl_by_component,
            "tier_shortage_extra_before_reallocation": int(tier_shortage_extra),
            "tier_shortage_rate_before_reallocation": float(tier_shortage_rate),
            "tier_shortage_rate_by_component_before_reallocation": tier_shortage_by_component,
            "tier_shortage_rate_by_tier_before_reallocation": tier_shortage_by_tier,
            "replacement_rate_ok": bool(repl_rate <= max_replacement + 1e-9),
            "pool_coverage": {
                key: float(requested.get(key, 0)) / float(max(int(available.get(key) or 0), 1))
                for key in COMPONENTS
            },
            "estimated_epoch_time_seconds": estimated_seconds,
            "adjustments": adjustments,
        }
    selected = str(selected_budget_candidate or head_cfg.get("default_candidate") or "medium").strip()
    if selected not in candidates:
        raise Step5utoBudgetError(f"selected Step5 budget candidate is not available: {selected}")
    return {
        "schema_version": STEP5_AUTO_BUDGET_SCHEMA_VERSION,
        "head": head,
        "capacity_basis": "balanced_capacity",
        "available": available,
        "available_all": dict(capacity.get("available_all") or {}),
        "active_tiers": {
            key: list(value)
            for key, value in (capacity.get("active_tiers") or {}).items()
        },
        "tier_mix": tier_mix,
        "tier_available": {
            "target_gold": dict(capacity["target_gold"]),
            "aux_gold": dict(capacity["aux_gold"]),
            "cf": dict(capacity["cf"]),
        },
        "ratios": ratios,
        "balanced_capacity_terms": balanced_terms,
        "balanced_capacity": int(balanced_capacity),
        "replacement_guard_effective_sample_cap": int(replacement_cap),
        "constraints": {
            "min_steps_per_effective_epoch": int(min_steps),
            "preferred_steps_per_effective_epoch": [int(preferred_low), int(preferred_high)],
            "max_steps_per_effective_epoch": int(max_steps),
            "max_replacement_rate": float(max_replacement),
        },
        "budget_candidates": candidates,
        "selected_budget_candidate": selected,
        "selected": deepcopy(candidates[selected]),
    }


def replacement_summary_for_effective_samples(
    manifest: Mapping[str, Any],
    *,
    head: str,
    head_cfg: Mapping[str, Any],
    effective_samples: int,
    ratio_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate replacement pressure for an explicit sample budget."""

    capacity = pool_capacity_from_manifest(manifest, head, head_cfg=head_cfg)
    ratios = _ratio_payload(head_cfg, ratio_override)
    requested = _component_requests(int(effective_samples), ratios)
    repl_rate, repl_extra, repl_by_component = _replacement_payload(requested, dict(capacity.get("available") or {}))
    tier_shortage_rate, tier_shortage_extra, tier_shortage_by_component, requested_by_tier, tier_shortage_by_tier = _tier_replacement_payload(
        requested,
        {
            key: dict(value)
            for key, value in (capacity.get("tier_available") or {}).items()
        },
        {
            key: dict(value)
            for key, value in (capacity.get("tier_mix") or {}).items()
        },
    )
    return {
        "effective_samples": int(effective_samples),
        "available": dict(capacity.get("available") or {}),
        "available_all": dict(capacity.get("available_all") or {}),
        "active_tiers": {
            key: list(value)
            for key, value in (capacity.get("active_tiers") or {}).items()
        },
        "tier_available": {
            key: dict(value)
            for key, value in (capacity.get("tier_available") or {}).items()
        },
        "tier_mix": {
            key: dict(value)
            for key, value in (capacity.get("tier_mix") or {}).items()
        },
        "ratios": ratios,
        "requested_counts": requested,
        "requested_tier_counts": requested_by_tier,
        "unique_sample_count": int(int(effective_samples) - repl_extra),
        "replacement_extra": int(repl_extra),
        "replacement_rate": float(repl_rate),
        "replacement_rate_by_component": repl_by_component,
        "tier_shortage_extra_before_reallocation": int(tier_shortage_extra),
        "tier_shortage_rate_before_reallocation": float(tier_shortage_rate),
        "tier_shortage_rate_by_component_before_reallocation": tier_shortage_by_component,
        "tier_shortage_rate_by_tier_before_reallocation": tier_shortage_by_tier,
    }


def compute_step5_auto_budget_report(
    manifest: Mapping[str, Any],
    *,
    sampler_config: Mapping[str, Any],
    batch_candidates_config: Mapping[str, Any],
    tuning_config: Mapping[str, Any] | None = None,
    batch_candidate: str | None = None,
    selected_budget_candidate: str | None = None,
    throughput_samples_per_sec: float | None = None,
) -> dict[str, Any]:
    """Compute Step5 explanation budgets using the resolved tuning batch."""

    auto_budget_cfg = sampler_config.get("auto_budget")
    if not isinstance(auto_budget_cfg, Mapping):
        raise Step5utoBudgetError("step5.sampler.auto_budget missing from resolved sampler config")
    batch = _selected_batch(batch_candidates_config, tuning_config, batch_candidate=batch_candidate)
    tuning = tuning_config if isinstance(tuning_config, Mapping) else {}
    budget_name = str(selected_budget_candidate or tuning.get("selected_budget_candidate") or "medium")
    expected_samples = tuning.get("effective_samples") if isinstance(tuning.get("effective_samples"), Mapping) else {}
    expected_steps = tuning.get("optimizer_steps") if isinstance(tuning.get("optimizer_steps"), Mapping) else {}
    heads: dict[str, Any] = {}
    for head in HEADS:
        head_cfg = sampler_config.get(head)
        if not isinstance(head_cfg, Mapping):
            raise Step5utoBudgetError(f"step5.sampler.{head} missing")
        head_report = compute_head_auto_budget(
            manifest,
            head=head,
            head_cfg=head_cfg,
            auto_budget_cfg=auto_budget_cfg,
            global_batch_size=int(batch["global_batch_size"]),
            selected_budget_candidate=budget_name,
            throughput_samples_per_sec=throughput_samples_per_sec,
        )
        if head in expected_samples:
            effective = int(expected_samples[head])
            override = replacement_summary_for_effective_samples(
                manifest,
                head=head,
                head_cfg=head_cfg,
                effective_samples=effective,
            )
            steps = int(expected_steps.get(head) or _ceil_div(effective, int(batch["global_batch_size"])))
            selected = dict(head_report["budget_candidates"].get(budget_name) or {})
            constraints = head_report.get("constraints") or {}
            preferred = constraints.get("preferred_steps_per_effective_epoch") or [0, 10**12]
            max_replacement = float(constraints.get("max_replacement_rate") or 0.0)
            selected.update(
                {
                    "budget_candidate": budget_name,
                    "raw_effective_samples": int(effective),
                    "effective_samples": int(effective),
                    "global_batch_size": int(batch["global_batch_size"]),
                    "optimizer_steps": int(steps),
                    "preferred_steps_ok": bool(int(preferred[0]) <= steps <= int(preferred[1])),
                    "min_steps_ok": bool(steps >= int(constraints.get("min_steps_per_effective_epoch") or 1)),
                    "max_steps_ok": bool(steps <= int(constraints.get("max_steps_per_effective_epoch") or steps)),
                    "requested_counts": dict(override["requested_counts"]),
                    "requested_tier_counts": dict(override["requested_tier_counts"]),
                    "unique_sample_count": int(override["unique_sample_count"]),
                    "replacement_extra": int(override["replacement_extra"]),
                    "replacement_rate": float(override["replacement_rate"]),
                    "replacement_rate_by_component": dict(override["replacement_rate_by_component"]),
                    "tier_shortage_extra_before_reallocation": int(override["tier_shortage_extra_before_reallocation"]),
                    "tier_shortage_rate_before_reallocation": float(override["tier_shortage_rate_before_reallocation"]),
                    "tier_shortage_rate_by_component_before_reallocation": dict(override["tier_shortage_rate_by_component_before_reallocation"]),
                    "tier_shortage_rate_by_tier_before_reallocation": dict(override["tier_shortage_rate_by_tier_before_reallocation"]),
                    "replacement_rate_ok": bool(float(override["replacement_rate"]) <= max_replacement + 1e-9),
                    "pool_coverage": {
                        key: float((override["requested_counts"] or {}).get(key, 0)) / float(max(int((override["available"] or {}).get(key) or 0), 1))
                        for key in COMPONENTS
                    },
                    "estimated_epoch_time_seconds": (
                        float(effective) / float(throughput_samples_per_sec)
                        if throughput_samples_per_sec is not None and float(throughput_samples_per_sec) > 0.0
                        else None
                    ),
                    "adjustments": ["explicit_reconciled_effective_samples"],
                    "explicit_reconciled_effective_samples": True,
                }
            )
            head_report["budget_candidates"][budget_name] = selected
            head_report["selected"] = deepcopy(selected)
            head_report["selected_budget_candidate"] = budget_name
        heads[head] = head_report
    return {
        "schema_version": STEP5_AUTO_BUDGET_SCHEMA_VERSION,
        "auto_budget_enabled": True,
        "batch_candidate": batch["id"],
        "per_gpu_batch_size": int(batch["per_gpu_batch_size"]),
        "global_batch_size": int(batch["global_batch_size"]),
        "selected_budget_candidate": budget_name,
        "heads": heads,
    }


def effective_samples_for_head(
    manifest: Mapping[str, Any],
    *,
    sampler_config: Mapping[str, Any],
    batch_candidates_config: Mapping[str, Any],
    tuning_config: Mapping[str, Any] | None,
    head: str,
    bounded_max_rows: int | None,
    mode: str,
    throughput_samples_per_sec: float | None = None,
) -> tuple[int, dict[str, Any]]:
    """Return the effective sample count used by the sampler plus metadata."""

    batch = _selected_batch(batch_candidates_config, tuning_config)
    tuning = tuning_config if isinstance(tuning_config, Mapping) else {}
    selected_budget = str(tuning.get("selected_budget_candidate") or "medium")
    report = compute_head_auto_budget(
        manifest,
        head=head,
        head_cfg=sampler_config[head],
        auto_budget_cfg=sampler_config["auto_budget"],
        global_batch_size=int(batch["global_batch_size"]),
        selected_budget_candidate=selected_budget,
        throughput_samples_per_sec=throughput_samples_per_sec,
    )
    budget = int(report["selected"]["effective_samples"])
    expected_samples = tuning.get("effective_samples") if isinstance(tuning.get("effective_samples"), Mapping) else {}
    expected_steps = tuning.get("optimizer_steps") if isinstance(tuning.get("optimizer_steps"), Mapping) else {}
    if str(mode).lower() != "bounded" and head in expected_samples:
        expected_budget = int(expected_samples[head])
        budget = expected_budget
        report["selected"] = {
            **dict(report.get("selected") or {}),
            **replacement_summary_for_effective_samples(
                manifest,
                head=head,
                head_cfg=sampler_config[head],
                effective_samples=budget,
            ),
            "budget_candidate": selected_budget,
            "raw_effective_samples": int(budget),
            "effective_samples": int(budget),
            "global_batch_size": int(batch["global_batch_size"]),
            "explicit_reconciled_effective_samples": True,
        }
    if str(mode).lower() != "bounded" and head in expected_steps:
        report["selected"]["optimizer_steps"] = int(expected_steps[head])
        actual_steps = int(report["selected"].get("optimizer_steps") or 0)
        expected_step_count = int(expected_steps[head])
        if actual_steps != expected_step_count:
            raise Step5utoBudgetError(
                f"resolved Step5 {head} optimizer_steps={actual_steps} does not match "
                f"step5.tuning.optimizer_steps.{head}={expected_step_count}"
            )
    if str(mode).lower() == "bounded" and bounded_max_rows is not None:
        budget = min(budget, max(1, int(bounded_max_rows)))
    return max(1, int(budget)), report
