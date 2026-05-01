from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from odcr_core import path_layout, run_naming
from odcr_core.config_schema import (
    OneControlConfigError,
    ResolvedConfig,
    SAFE_DECODE_PLACEHOLDER,
    SourceRecord,
    TOP_LEVEL_BLOCKS,
    fingerprint,
    json_dumps,
)
from odcr_core.training_diagnostics import runtime_diagnostics_fingerprint_source

_CODE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CODE_DIR.parent


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise OneControlConfigError("PyYAML is required to read configs/odcr.yaml") from exc
    p = Path(path)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    if not p.is_file():
        raise OneControlConfigError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise OneControlConfigError(f"{p} must contain a mapping at the top level")
    missing = [k for k in TOP_LEVEL_BLOCKS if k not in raw]
    if missing:
        raise OneControlConfigError(f"{p} is missing required top-level blocks: {missing}")
    extra = sorted(set(raw) - set(TOP_LEVEL_BLOCKS))
    if extra:
        raise OneControlConfigError(f"{p} has unsupported top-level blocks: {extra}")
    return raw


def parse_set_value(raw: str) -> Any:
    s = str(raw).strip()
    lower = s.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("none", "null", "~"):
        return None
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    try:
        if "." not in s and "e" not in lower:
            return int(s)
        return float(s)
    except ValueError:
        return raw


def apply_cli_sets(cfg: dict[str, Any], raw_sets: Iterable[str]) -> tuple[dict[str, Any], dict[str, str]]:
    out = deepcopy(cfg)
    sources: dict[str, str] = {}
    for item in raw_sets:
        if "=" not in item:
            raise OneControlConfigError(f"--set expects key=value, got {item!r}")
        key, value_raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise OneControlConfigError("--set key cannot be empty")
        parts = key.split(".")
        cur: Any = out
        for part in parts[:-1]:
            if not isinstance(cur, dict):
                raise OneControlConfigError(f"--set cannot descend through non-object key {part!r} in {key!r}")
            if part not in cur or cur[part] is None:
                cur[part] = {}
            cur = cur[part]
        if not isinstance(cur, dict):
            raise OneControlConfigError(f"--set target parent is not an object for {key!r}")
        cur[parts[-1]] = parse_set_value(value_raw)
        sources[key] = "cli --set"
    return out, sources


def _get(obj: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _merge_dicts(*items: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if not item:
            continue
        for key, value in item.items():
            if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
                out[key] = _merge_dicts(out[key], value)
            else:
                out[key] = deepcopy(value)
    return out


_STEP3_RETIRED_CONTROL_FIELDS = frozenset(
    {
        "adv",
        "eta",
        "adversarial_coef",
        "adversarial_alpha",
        "adversarial_beta",
        "adversarial_schedule",
        "adversarial_schedule_enabled",
        "adversarial_start_epoch",
        "adversarial_warmup_epochs",
        "adversarial_coef_target",
    }
)
_TASK_RETIRED_CONTROL_FIELDS = frozenset({"adv", "eta"})
_STEP5_RETIRED_TRAIN_FIELDS = frozenset({"adv", "eta"})


def _reject_step3_retired_controls(train: Mapping[str, Any]) -> None:
    bad = sorted(k for k in _STEP3_RETIRED_CONTROL_FIELDS if k in train)
    if bad:
        raise OneControlConfigError(
            "step3.train contains retired adversarial/counterfactual controls "
            f"{bad}; Step3 is resolved as structured shared/specific disentanglement. "
            "Remove these keys or place downstream counterfactual weights under step5."
        )


def _reject_step5_retired_controls(train: Mapping[str, Any]) -> None:
    bad = sorted(k for k in _STEP5_RETIRED_TRAIN_FIELDS if k in train)
    if bad:
        raise OneControlConfigError(
            "step5.train contains retired ambiguous controls "
            f"{bad}; use step5.train.explainer_loss_weight for the Step5B explainer loss multiplier."
        )


def _task_row(cfg: Mapping[str, Any], task_id: int) -> dict[str, Any]:
    tasks = _get(cfg, "tasks", {})
    raw = None
    if isinstance(tasks, Mapping):
        raw = tasks.get(str(task_id), tasks.get(task_id))
    if not isinstance(raw, Mapping):
        raise OneControlConfigError(f"tasks.{task_id} must be configured in configs/odcr.yaml")
    retired = sorted(k for k in _TASK_RETIRED_CONTROL_FIELDS if k in raw)
    if retired:
        raise OneControlConfigError(
            f"tasks.{task_id} contains retired ambiguous controls {retired}; "
            "task rows may define source/target/lr/coef only."
        )
    source = raw.get("source", raw.get("auxiliary"))
    target = raw.get("target")
    if not source or not target:
        raise OneControlConfigError(f"tasks.{task_id} must define source and target")
    return dict(raw)


def _stage_task_override(stage_cfg: Mapping[str, Any], task_id: int) -> dict[str, Any]:
    raw = stage_cfg.get("tasks", {})
    if not isinstance(raw, Mapping):
        return {}
    item = raw.get(str(task_id), raw.get(task_id, {}))
    return dict(item) if isinstance(item, Mapping) else {}


def _set_nested(obj: dict[str, Any], dotted: str, value: Any) -> None:
    cur = obj
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = deepcopy(value)


def _apply_train_cli_overrides(
    *,
    cfg: Mapping[str, Any],
    cli_sources: Mapping[str, str],
    stage: str,
    train: dict[str, Any],
) -> None:
    prefix = f"{stage}.train."
    for key in cli_sources:
        if key.startswith(prefix):
            _set_nested(train, key[len(prefix) :], _get(cfg, key))


def _active_hardware(cfg: Mapping[str, Any], stage_profile: str | None = None) -> tuple[str, dict[str, Any]]:
    hardware = _get(cfg, "hardware", {})
    if not isinstance(hardware, Mapping):
        raise OneControlConfigError("hardware must be a mapping")
    stem = str(stage_profile or hardware.get("active") or "default")
    profiles = hardware.get("profiles")
    if not isinstance(profiles, Mapping) or stem not in profiles:
        raise OneControlConfigError(f"hardware profile {stem!r} not found under hardware.profiles")
    profile = profiles[stem]
    if not isinstance(profile, Mapping):
        raise OneControlConfigError(f"hardware.profiles.{stem} must be a mapping")
    return stem, dict(profile)


def _positive_int(value: Any, key: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise OneControlConfigError(f"{key} must be an integer, got {value!r}") from exc
    if out < 1:
        raise OneControlConfigError(f"{key} must be >= 1, got {out}")
    return out


def _positive_float(value: Any, key: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise OneControlConfigError(f"{key} must be a number, got {value!r}") from exc
    return out


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return int(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _resolve_repo_path(value: Any, key: str, repo_root: Path) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise OneControlConfigError(f"{key} must be configured in configs/odcr.yaml")
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = repo_root / p
    return str(p.resolve())


def _resolve_global_runtime_roots(cfg: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    project = _mapping(_get(cfg, "project"), "project")
    env = _mapping(_get(cfg, "env"), "env")
    roots = {
        "runs_dir": _resolve_repo_path(project.get("run_root"), "project.run_root", repo_root),
        "cache_dir": _resolve_repo_path(project.get("cache_dir"), "project.cache_dir", repo_root),
        "data_dir": _resolve_repo_path(project.get("data_dir"), "project.data_dir", repo_root),
        "merged_dir": _resolve_repo_path(project.get("merged_dir"), "project.merged_dir", repo_root),
        "models_dir": _resolve_repo_path(env.get("models_dir"), "env.models_dir", repo_root),
        "step5_text_model": _resolve_repo_path(env.get("step5_text_model"), "env.step5_text_model", repo_root),
        "sentence_embed_model": _resolve_repo_path(env.get("sentence_embed_model"), "env.sentence_embed_model", repo_root),
        "embed_dim": _positive_int(env.get("embed_dim"), "env.embed_dim"),
        "offline": _bool(env.get("offline", True)),
        "local_files_only": _bool(env.get("local_files_only", True)),
    }
    expected_by_env = {
        "ODCR_DATA_DIR": ("data_dir", roots["data_dir"]),
        "ODCR_MERGED_DATA_DIR": ("merged_dir", roots["merged_dir"]),
        "ODCR_MODELS_DIR": ("models_dir", roots["models_dir"]),
        "ODCR_STEP5_TEXT_MODEL": ("step5_text_model", roots["step5_text_model"]),
        "ODCR_SENTENCE_EMBED_MODEL": ("sentence_embed_model", roots["sentence_embed_model"]),
    }
    conflicts: list[str] = []
    for env_name, (root_key, expected) in expected_by_env.items():
        raw = (os.environ.get(env_name) or "").strip()
        if not raw:
            continue
        observed = str(Path(os.path.expanduser(raw)).resolve())
        if observed != str(expected):
            conflicts.append(f"{env_name}={observed} conflicts with configs/odcr.yaml {root_key}={expected}")
    raw_embed = (os.environ.get("ODCR_EMBED_DIM") or "").strip()
    if raw_embed:
        try:
            observed_embed = int(raw_embed)
        except ValueError as exc:
            raise OneControlConfigError("ODCR_EMBED_DIM must be an integer when present") from exc
        if observed_embed != int(roots["embed_dim"]):
            conflicts.append(
                f"ODCR_EMBED_DIM={observed_embed} conflicts with configs/odcr.yaml env.embed_dim={roots['embed_dim']}"
            )
    if conflicts:
        raise OneControlConfigError(
            "Legacy ODCR_* root/model/embed_dim environment variables cannot override One-Control config: "
            + "; ".join(conflicts)
        )
    return roots


_STEP4_RCR_REQUIRED_EXPORT_FIELDS = (
    "train_keep",
    "sample_weight_hint",
    "route_scorer",
    "route_explainer",
    "route_reason_scorer",
    "route_reason_explainer",
    "content_retention_score",
    "style_shift_score",
    "rating_stability_score",
    "cf_reliability_score",
    "uncertainty_score",
    "entropy_score",
    "text_quality_score",
    "confidence_bucket",
    "preprocess_route_scorer_prior",
    "preprocess_route_explainer_prior",
)


def _mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OneControlConfigError(f"{key} must be a mapping")
    return value


def _rcr_float(value: Any, key: str, *, min_value: float | None = None, max_value: float | None = None) -> float:
    out = _positive_float(value, key)
    if min_value is not None and out < min_value:
        raise OneControlConfigError(f"{key} must be >= {min_value}, got {out}")
    if max_value is not None and out > max_value:
        raise OneControlConfigError(f"{key} must be <= {max_value}, got {out}")
    return float(out)


def _rcr_int(value: Any, key: str, *, min_value: int = 0) -> int:
    out = int(_positive_int(value, key))
    if out < min_value:
        raise OneControlConfigError(f"{key} must be >= {min_value}, got {out}")
    return out


def _resolve_rcr_weights(raw: Mapping[str, Any], key: str, names: tuple[str, ...]) -> dict[str, float]:
    obj = _mapping(raw.get(key), f"step4.rcr.{key}")
    missing = [name for name in names if name not in obj]
    extra = sorted(set(obj) - set(names))
    if missing or extra:
        raise OneControlConfigError(f"step4.rcr.{key} mismatch; missing={missing}, extra={extra}")
    out = {name: _rcr_float(obj[name], f"step4.rcr.{key}.{name}", min_value=0.0, max_value=1.0) for name in names}
    total = sum(out.values())
    if abs(total - 1.0) > 1e-6:
        raise OneControlConfigError(f"step4.rcr.{key} weights must sum to 1.0, got {total}")
    return out


def _resolve_step4_rcr_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step4.rcr"), "step4.rcr")
    scorer = _mapping(raw.get("route_scorer"), "step4.rcr.route_scorer")
    explainer = _mapping(raw.get("route_explainer"), "step4.rcr.route_explainer")
    bucket = _mapping(raw.get("confidence_bucket"), "step4.rcr.confidence_bucket")
    high = _mapping(bucket.get("high"), "step4.rcr.confidence_bucket.high")
    medium = _mapping(bucket.get("medium"), "step4.rcr.confidence_bucket.medium")
    train_keep = _mapping(raw.get("train_keep"), "step4.rcr.train_keep")
    sample_weight = _mapping(raw.get("sample_weight_hint"), "step4.rcr.sample_weight_hint")
    export = _mapping(raw.get("export"), "step4.rcr.export")
    required = export.get("required_fields")
    if not isinstance(required, list) or not all(isinstance(x, str) and x for x in required):
        raise OneControlConfigError("step4.rcr.export.required_fields must be a non-empty string list")
    missing_required = [x for x in _STEP4_RCR_REQUIRED_EXPORT_FIELDS if x not in required]
    if missing_required:
        raise OneControlConfigError(
            "step4.rcr.export.required_fields must include the Step4 RCR contract columns: "
            + ", ".join(missing_required)
        )
    return {
        "cf_reliability_weights": _resolve_rcr_weights(
            raw,
            "cf_reliability_weights",
            ("content_retention", "rating_stability", "style_shift", "text_quality"),
        ),
        "uncertainty_weights": _resolve_rcr_weights(
            raw,
            "uncertainty_weights",
            ("rating_instability", "content_weakness", "text_quality_weakness", "entropy"),
        ),
        "rating_delta_soft_cap": _rcr_float(
            raw.get("rating_delta_soft_cap"), "step4.rcr.rating_delta_soft_cap", min_value=1e-6
        ),
        "route_scorer": {
            "min_reliability": _rcr_float(scorer.get("min_reliability"), "step4.rcr.route_scorer.min_reliability", min_value=0.0, max_value=1.0),
            "min_content_retention": _rcr_float(scorer.get("min_content_retention"), "step4.rcr.route_scorer.min_content_retention", min_value=0.0, max_value=1.0),
            "min_rating_stability": _rcr_float(scorer.get("min_rating_stability"), "step4.rcr.route_scorer.min_rating_stability", min_value=0.0, max_value=1.0),
            "max_uncertainty": _rcr_float(scorer.get("max_uncertainty"), "step4.rcr.route_scorer.max_uncertainty", min_value=0.0, max_value=1.0),
            "max_rating_delta": _rcr_float(scorer.get("max_rating_delta"), "step4.rcr.route_scorer.max_rating_delta", min_value=0.0),
            "min_text_quality": _rcr_float(scorer.get("min_text_quality"), "step4.rcr.route_scorer.min_text_quality", min_value=0.0, max_value=1.0),
        },
        "route_explainer": {
            "min_reliability": _rcr_float(explainer.get("min_reliability"), "step4.rcr.route_explainer.min_reliability", min_value=0.0, max_value=1.0),
            "relaxed_min_reliability": _rcr_float(explainer.get("relaxed_min_reliability"), "step4.rcr.route_explainer.relaxed_min_reliability", min_value=0.0, max_value=1.0),
            "min_content_retention": _rcr_float(explainer.get("min_content_retention"), "step4.rcr.route_explainer.min_content_retention", min_value=0.0, max_value=1.0),
            "min_style_shift": _rcr_float(explainer.get("min_style_shift"), "step4.rcr.route_explainer.min_style_shift", min_value=0.0, max_value=1.0),
            "max_uncertainty": _rcr_float(explainer.get("max_uncertainty"), "step4.rcr.route_explainer.max_uncertainty", min_value=0.0, max_value=1.0),
            "min_text_quality": _rcr_float(explainer.get("min_text_quality"), "step4.rcr.route_explainer.min_text_quality", min_value=0.0, max_value=1.0),
        },
        "confidence_bucket": {
            "high": {
                "bucket": _rcr_int(high.get("bucket"), "step4.rcr.confidence_bucket.high.bucket"),
                "min_reliability": _rcr_float(high.get("min_reliability"), "step4.rcr.confidence_bucket.high.min_reliability", min_value=0.0, max_value=1.0),
                "max_uncertainty": _rcr_float(high.get("max_uncertainty"), "step4.rcr.confidence_bucket.high.max_uncertainty", min_value=0.0, max_value=1.0),
                "min_rating_stability": _rcr_float(high.get("min_rating_stability"), "step4.rcr.confidence_bucket.high.min_rating_stability", min_value=0.0, max_value=1.0),
                "min_content_retention": _rcr_float(high.get("min_content_retention"), "step4.rcr.confidence_bucket.high.min_content_retention", min_value=0.0, max_value=1.0),
            },
            "medium": {
                "bucket": _rcr_int(medium.get("bucket"), "step4.rcr.confidence_bucket.medium.bucket"),
                "min_reliability": _rcr_float(medium.get("min_reliability"), "step4.rcr.confidence_bucket.medium.min_reliability", min_value=0.0, max_value=1.0),
                "max_uncertainty": _rcr_float(medium.get("max_uncertainty"), "step4.rcr.confidence_bucket.medium.max_uncertainty", min_value=0.0, max_value=1.0),
                "min_content_retention": _rcr_float(medium.get("min_content_retention"), "step4.rcr.confidence_bucket.medium.min_content_retention", min_value=0.0, max_value=1.0),
            },
            "low_bucket": int(bucket.get("low_bucket", 0)),
        },
        "train_keep": {
            "reject_when_both_routes_zero": _bool(train_keep.get("reject_when_both_routes_zero", True)),
            "reject_reason": str(train_keep.get("reject_reason", "rcr_route_reject")),
        },
        "sample_weight_hint": {
            "scorer_route_multiplier": _rcr_float(sample_weight.get("scorer_route_multiplier"), "step4.rcr.sample_weight_hint.scorer_route_multiplier", min_value=0.0),
            "explainer_only_route_multiplier": _rcr_float(sample_weight.get("explainer_only_route_multiplier"), "step4.rcr.sample_weight_hint.explainer_only_route_multiplier", min_value=0.0),
            "reliability_floor": _rcr_float(sample_weight.get("reliability_floor"), "step4.rcr.sample_weight_hint.reliability_floor", min_value=0.0, max_value=1.0),
            "reliability_scale": _rcr_float(sample_weight.get("reliability_scale"), "step4.rcr.sample_weight_hint.reliability_scale", min_value=0.0, max_value=1.0),
            "uncertainty_base": _rcr_float(sample_weight.get("uncertainty_base"), "step4.rcr.sample_weight_hint.uncertainty_base", min_value=0.0, max_value=1.0),
            "uncertainty_scale": _rcr_float(sample_weight.get("uncertainty_scale"), "step4.rcr.sample_weight_hint.uncertainty_scale", min_value=0.0, max_value=1.0),
        },
        "export": {
            "required_fields": list(required),
        },
    }


def _resolve_step3_structured_losses_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.structured_losses"), "step3.structured_losses")
    orth = _mapping(raw.get("orthogonal"), "step3.structured_losses.orthogonal")
    return {
        "orthogonal": {
            "weight": _rcr_float(
                orth.get("weight"),
                "step3.structured_losses.orthogonal.weight",
                min_value=0.0,
            ),
            "xcov_weight": _rcr_float(
                orth.get("xcov_weight"),
                "step3.structured_losses.orthogonal.xcov_weight",
                min_value=0.0,
            ),
            "cosine_weight": _rcr_float(
                orth.get("cosine_weight"),
                "step3.structured_losses.orthogonal.cosine_weight",
                min_value=0.0,
            ),
        },
        "variance_weight": _rcr_float(raw.get("variance_weight"), "step3.structured_losses.variance_weight", min_value=0.0),
        "shared_invariance_weight": _rcr_float(raw.get("shared_invariance_weight"), "step3.structured_losses.shared_invariance_weight", min_value=0.0),
        "specific_separation_weight": _rcr_float(raw.get("specific_separation_weight"), "step3.structured_losses.specific_separation_weight", min_value=0.0),
        "anchor_alignment_weight": _rcr_float(raw.get("anchor_alignment_weight"), "step3.structured_losses.anchor_alignment_weight", min_value=0.0),
        "content_alignment_weight": _rcr_float(raw.get("content_alignment_weight"), "step3.structured_losses.content_alignment_weight", min_value=0.0),
        "style_alignment_weight": _rcr_float(raw.get("style_alignment_weight"), "step3.structured_losses.style_alignment_weight", min_value=0.0),
        "shared_prototype_weight": _rcr_float(raw.get("shared_prototype_weight"), "step3.structured_losses.shared_prototype_weight", min_value=0.0),
        "domain_style_alignment_weight": _rcr_float(raw.get("domain_style_alignment_weight"), "step3.structured_losses.domain_style_alignment_weight", min_value=0.0),
        "local_style_alignment_weight": _rcr_float(raw.get("local_style_alignment_weight"), "step3.structured_losses.local_style_alignment_weight", min_value=0.0),
        "polarity_alignment_weight": _rcr_float(raw.get("polarity_alignment_weight"), "step3.structured_losses.polarity_alignment_weight", min_value=0.0),
        "residual_specific_weight": _rcr_float(raw.get("residual_specific_weight"), "step3.structured_losses.residual_specific_weight", min_value=0.0),
        "prototype_separation_weight": _rcr_float(raw.get("prototype_separation_weight"), "step3.structured_losses.prototype_separation_weight", min_value=0.0),
        "light_explainer_weight": _rcr_float(raw.get("light_explainer_weight"), "step3.structured_losses.light_explainer_weight", min_value=0.0),
    }


_STEP5_CCV_REQUIRED_CONTROL_FIELDS = (
    "content_evidence",
    "style_evidence",
    "domain_style_anchor",
    "local_style_residual_hint",
    "cf_reliability_score",
    "uncertainty_score",
    "confidence_bucket",
    "route_explainer",
    "route_scorer",
    "sample_weight_hint",
)
_STEP5_RETIRED_BACKEND_KEYS = frozenset(
    {
        "lambda_lci",
        "lambda_fca",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
    }
)


def _resolve_named_float_map(
    raw: Mapping[str, Any],
    key: str,
    names: tuple[str, ...],
    *,
    min_value: float = 0.0,
    max_value: float | None = None,
) -> dict[str, float]:
    lookup_key = key.split(".")[-1]
    obj = _mapping(raw.get(lookup_key), key)
    missing = [name for name in names if name not in obj]
    extra = sorted(set(obj) - set(names))
    if missing or extra:
        raise OneControlConfigError(f"{key} mismatch; missing={missing}, extra={extra}")
    return {
        name: _rcr_float(obj[name], f"{key}.{name}", min_value=min_value, max_value=max_value)
        for name in names
    }


def _resolve_step5_innovation_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5"), "step5")
    train = raw.get("train")
    backend = train.get("backend") if isinstance(train, Mapping) else None
    if isinstance(backend, Mapping):
        retired = sorted(k for k in _STEP5_RETIRED_BACKEND_KEYS if k in backend)
        if retired:
            raise OneControlConfigError(
                "step5.train.backend contains retired Step5 LCI/FCA/LoRA keys "
                f"{retired}; use step5.lci / step5.fca / step5.ccv.native_lora instead."
            )
    lci = _mapping(raw.get("lci"), "step5.lci")
    uci = _mapping(raw.get("uci"), "step5.uci")
    explainer_gate = _mapping(raw.get("explainer_gate"), "step5.explainer_gate")
    ccv = _mapping(raw.get("ccv"), "step5.ccv")
    fca = _mapping(raw.get("fca"), "step5.fca")
    native_lora = _mapping(ccv.get("native_lora"), "step5.ccv.native_lora")
    control_fields = ccv.get("control_fields")
    if not isinstance(control_fields, list) or not all(isinstance(x, str) and x for x in control_fields):
        raise OneControlConfigError("step5.ccv.control_fields must be a non-empty string list")
    missing_control = [x for x in _STEP5_CCV_REQUIRED_CONTROL_FIELDS if x not in control_fields]
    if missing_control:
        raise OneControlConfigError(
            "step5.ccv.control_fields missing required Step5B control fields: "
            + ", ".join(missing_control)
        )
    mode = str(fca.get("evidence_alignment_mode", "")).strip().lower()
    if mode not in ("evidence_basis",):
        raise OneControlConfigError("step5.fca.evidence_alignment_mode must be 'evidence_basis'")
    packet_policy = str(ccv.get("control_packet_field_policy", "")).strip().lower()
    if packet_policy not in ("strict_required",):
        raise OneControlConfigError("step5.ccv.control_packet_field_policy must be 'strict_required'")
    adapter_policy = str(ccv.get("verbalizer_adapter_policy", "")).strip().lower()
    if adapter_policy not in ("ccv_control_adapter",):
        raise OneControlConfigError("step5.ccv.verbalizer_adapter_policy must be 'ccv_control_adapter'")
    soft_prompt_len = _positive_int(ccv.get("soft_prompt_len"), "step5.ccv.soft_prompt_len")
    numeric_control_dim = _positive_int(ccv.get("numeric_control_dim"), "step5.ccv.numeric_control_dim")
    if numeric_control_dim != 13:
        raise OneControlConfigError("step5.ccv.numeric_control_dim must be 13 for the current CCVControlPacket")
    control_adapter_blocks = _positive_int(
        ccv.get("control_adapter_input_blocks"),
        "step5.ccv.control_adapter_input_blocks",
    )
    if control_adapter_blocks != 6:
        raise OneControlConfigError("step5.ccv.control_adapter_input_blocks must be 6 for the current CCV adapter")
    lora_targets = native_lora.get("target_modules", [])
    if not isinstance(lora_targets, list) or not all(isinstance(x, str) for x in lora_targets):
        raise OneControlConfigError("step5.ccv.native_lora.target_modules must be a string list")
    explainer_min_weight = _rcr_float(
        explainer_gate.get("min_weight"),
        "step5.explainer_gate.min_weight",
        min_value=0.0,
    )
    explainer_max_weight = _rcr_float(
        explainer_gate.get("max_weight"),
        "step5.explainer_gate.max_weight",
        min_value=0.0,
    )
    if explainer_max_weight < explainer_min_weight:
        raise OneControlConfigError("step5.explainer_gate.max_weight must be >= min_weight")
    return {
        "lci": {
            "enabled": _bool(lci.get("enabled", True)),
            "weight": _rcr_float(lci.get("weight"), "step5.lci.weight", min_value=0.0),
            "confidence_schedule": _resolve_named_float_map(
                lci,
                "step5.lci.confidence_schedule",
                ("high", "medium", "low"),
                min_value=0.0,
            ),
            "min_reliability": _rcr_float(lci.get("min_reliability"), "step5.lci.min_reliability", min_value=0.0, max_value=1.0),
            "max_uncertainty": _rcr_float(lci.get("max_uncertainty"), "step5.lci.max_uncertainty", min_value=0.0, max_value=1.0),
            "perturb_std": _rcr_float(lci.get("perturb_std"), "step5.lci.perturb_std", min_value=0.0),
            "counterfactual_label_weight": _rcr_float(lci.get("counterfactual_label_weight"), "step5.lci.counterfactual_label_weight", min_value=0.0),
            "robustness_weight": _rcr_float(lci.get("robustness_weight"), "step5.lci.robustness_weight", min_value=0.0),
        },
        "uci": {
            "enabled": _bool(uci.get("enabled", True)),
            "bucket_weights": _resolve_named_float_map(
                uci,
                "step5.uci.bucket_weights",
                ("high", "medium", "low"),
                min_value=0.0,
            ),
            "uncertainty_temperature": _rcr_float(uci.get("uncertainty_temperature"), "step5.uci.uncertainty_temperature", min_value=1e-6),
            "low_confidence_floor": _rcr_float(uci.get("low_confidence_floor", 0.0), "step5.uci.low_confidence_floor", min_value=0.0, max_value=1.0),
        },
        "explainer_gate": {
            "bucket_weights": _resolve_named_float_map(
                explainer_gate,
                "step5.explainer_gate.bucket_weights",
                ("high", "medium", "low"),
                min_value=0.0,
            ),
            "uncertainty_exponent": _rcr_float(
                explainer_gate.get("uncertainty_exponent"),
                "step5.explainer_gate.uncertainty_exponent",
                min_value=1e-6,
            ),
            "style_shift_diversity_boost": _rcr_float(
                explainer_gate.get("style_shift_diversity_boost"),
                "step5.explainer_gate.style_shift_diversity_boost",
                min_value=0.0,
            ),
            "min_weight": explainer_min_weight,
            "max_weight": explainer_max_weight,
            "explainer_only_multiplier": _rcr_float(
                explainer_gate.get("explainer_only_multiplier"),
                "step5.explainer_gate.explainer_only_multiplier",
                min_value=0.0,
            ),
        },
        "ccv": {
            "enabled": _bool(ccv.get("enabled", True)),
            "control_fields": list(control_fields),
            "uncertainty_tone_control": _bool(ccv.get("uncertainty_tone_control", True)),
            "route_conditioning": _bool(ccv.get("route_conditioning", True)),
            "numeric_control_weight": _rcr_float(ccv.get("numeric_control_weight"), "step5.ccv.numeric_control_weight", min_value=0.0),
            "control_packet_field_policy": packet_policy,
            "verbalizer_adapter_policy": adapter_policy,
            "soft_prompt_len": soft_prompt_len,
            "numeric_control_dim": numeric_control_dim,
            "control_adapter_input_blocks": control_adapter_blocks,
            "native_lora": {
                "enabled": _bool(native_lora.get("enabled", True)),
                "r": _positive_int(native_lora.get("r"), "step5.ccv.native_lora.r"),
                "alpha": _rcr_float(native_lora.get("alpha"), "step5.ccv.native_lora.alpha", min_value=0.0),
                "dropout": _rcr_float(
                    native_lora.get("dropout"),
                    "step5.ccv.native_lora.dropout",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "target_modules": [x for x in lora_targets if x.strip()],
            },
        },
        "fca": {
            "enabled": _bool(fca.get("enabled", True)),
            "weight": _rcr_float(fca.get("weight"), "step5.fca.weight", min_value=0.0),
            "min_reliability": _rcr_float(fca.get("min_reliability"), "step5.fca.min_reliability", min_value=0.0, max_value=1.0),
            "max_uncertainty": _rcr_float(fca.get("max_uncertainty"), "step5.fca.max_uncertainty", min_value=0.0, max_value=1.0),
            "evidence_alignment_mode": mode,
        },
    }


def _validate_train_batch(stage: str, train: Mapping[str, Any], ddp_world_size: int) -> tuple[int, int, int, int]:
    batch_size = _positive_int(train.get("batch_size"), f"{stage}.train.batch_size")
    micro = _positive_int(train.get("micro_batch_size"), f"{stage}.train.micro_batch_size")
    accum = _positive_int(train.get("grad_accum"), f"{stage}.train.grad_accum")
    expected = micro * int(ddp_world_size) * accum
    if batch_size != expected:
        raise OneControlConfigError(
            f"{stage}.train batch formula failed: batch_size={batch_size} but "
            f"micro_batch_size({micro}) * ddp_world_size({ddp_world_size}) * grad_accum({accum}) = {expected}"
        )
    return batch_size, micro, accum, expected


def _validate_eval_batch(eval_batch_size: int | None, ddp_world_size: int) -> int | None:
    if eval_batch_size is None:
        return None
    ebs = _positive_int(eval_batch_size, "eval_batch_size")
    if ebs % int(ddp_world_size) != 0:
        raise OneControlConfigError(
            f"eval_batch_size={ebs} must be divisible by ddp_world_size={ddp_world_size}"
        )
    return ebs // int(ddp_world_size)


def needs_decode_layer(command: str, *, step5_train_only: bool = False) -> bool:
    """Return whether the resolver command consumes generation/decode config."""
    if command == "step5" and step5_train_only:
        return False
    return command in ("step4", "step5", "eval")


def _latest_run(repo_root: Path, task_id: int, stage: str, *, dry_run: bool) -> str:
    parent = path_layout.get_stage_task_root(repo_root, stage, task_id)
    latest = parent / "latest.json"
    if not latest.is_file():
        raise OneControlConfigError(
            f"missing {latest}; latest resolution requires the formal latest.json -> meta/run_summary.json handoff. "
            f"Specify an explicit {stage} run id or rerun the upstream stage."
        )
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OneControlConfigError(f"latest.json is not valid JSON: {latest}") from exc
    latest_run_id = str(payload.get("latest_run_id") or "").strip()
    latest_summary_path = str(payload.get("latest_summary_path") or "").strip()
    if not latest_run_id or not latest_summary_path:
        raise OneControlConfigError(
            f"latest.json pointer is incomplete for {stage} task {task_id}: {latest}; "
            "expected latest_run_id and latest_summary_path"
        )
    run_id = run_naming.parse_run_id(latest_run_id)
    summary = Path(latest_summary_path)
    if not summary.is_absolute():
        summary = (repo_root / summary).resolve()
    if summary.name != "run_summary.json" or summary.parent.name != "meta":
        raise OneControlConfigError(
            f"latest.json pointer is damaged for {stage} task {task_id}: "
            f"latest_summary_path must target meta/run_summary.json, got {summary}"
        )
    if not summary.is_file():
        raise OneControlConfigError(
            f"latest.json pointer is damaged for {stage} task {task_id}: "
            f"missing run_summary.json at {summary}"
        )
    try:
        summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OneControlConfigError(f"run_summary.json is not valid JSON: {summary}") from exc
    summary_run_id = str(summary_payload.get("run_id") or "").strip()
    if summary_run_id and run_naming.parse_run_id(summary_run_id) != run_id:
        raise OneControlConfigError(
            f"latest.json pointer is damaged for {stage} task {task_id}: "
            f"latest_run_id={run_id!r} but run_summary run_id={summary_run_id!r}"
        )
    return run_id


def _alloc_run(repo_root: Path, task_id: int, stage: str, requested: str | None, *, dry_run: bool) -> str:
    if dry_run:
        return run_naming.parse_run_id(requested) if requested and requested not in ("auto", "") else "dry_run"
    parent = path_layout.get_stage_task_root(repo_root, stage, task_id)
    parent.mkdir(parents=True, exist_ok=True)
    return run_naming.allocate_child_dir(parent, requested=requested, kind="run")


def _stage_root(repo_root: Path, task_id: int, stage: str, run_id: str) -> Path:
    return path_layout.get_stage_run_root(repo_root, task_id, "v1", stage, run_id).resolve()


def _default_decode_placeholder() -> dict[str, Any]:
    return deepcopy(SAFE_DECODE_PLACEHOLDER)


def _resolve_eval_profile(cfg: Mapping[str, Any], profile_name: str | None) -> tuple[str, dict[str, Any]]:
    eval_cfg = _get(cfg, "eval", {})
    if not isinstance(eval_cfg, Mapping):
        raise OneControlConfigError("eval must be a mapping")
    name = str(profile_name or eval_cfg.get("profile") or "balanced_2gpu")
    profiles = eval_cfg.get("profiles", {})
    if not isinstance(profiles, Mapping) or name not in profiles:
        raise OneControlConfigError(f"eval profile {name!r} not found under eval.profiles")
    raw = profiles[name]
    if not isinstance(raw, Mapping):
        raise OneControlConfigError(f"eval.profiles.{name} must be a mapping")
    return name, dict(raw)


def _resolve_decode(cfg: Mapping[str, Any], name: str | None, *, need_decode: bool) -> tuple[str, dict[str, Any]]:
    if not need_decode:
        return "", _default_decode_placeholder()
    eval_cfg = _get(cfg, "eval", {})
    decode_cfg = eval_cfg.get("decode", {}) if isinstance(eval_cfg, Mapping) else {}
    if not isinstance(decode_cfg, Mapping):
        raise OneControlConfigError("eval.decode must be a mapping")
    base = decode_cfg.get("default", {})
    if not isinstance(base, Mapping):
        raise OneControlConfigError("eval.decode.default must be a mapping")
    stem = str(name or "mainline")
    overlay = decode_cfg.get(stem, {})
    if stem != "default" and not isinstance(overlay, Mapping):
        raise OneControlConfigError(f"eval.decode.{stem} must be a mapping")
    merged = _merge_dicts(base, overlay if stem != "default" else {})
    return stem, merged


def _resolve_rerank(cfg: Mapping[str, Any], name: str | None, *, need_rerank: bool) -> tuple[str, dict[str, Any]]:
    if not need_rerank:
        return "", {}
    eval_cfg = _get(cfg, "eval", {})
    rerank_cfg = eval_cfg.get("rerank", {}) if isinstance(eval_cfg, Mapping) else {}
    if not isinstance(rerank_cfg, Mapping):
        raise OneControlConfigError("eval.rerank must be a mapping")
    base = rerank_cfg.get("default", {})
    if not isinstance(base, Mapping):
        raise OneControlConfigError("eval.rerank.default must be a mapping")
    stem = str(name or "quality")
    overlay = rerank_cfg.get(stem, {})
    if stem != "default" and not isinstance(overlay, Mapping):
        raise OneControlConfigError(f"eval.rerank.{stem} must be a mapping")
    return stem, _merge_dicts(base, overlay if stem != "default" else {})


def _training_row(stage: str, train: Mapping[str, Any], task: Mapping[str, Any], *, eval_batch_size: int | None) -> dict[str, Any]:
    backend = train.get("backend", {})
    if backend is not None and not isinstance(backend, Mapping):
        raise OneControlConfigError(f"{stage}.train.backend must be a mapping")
    if stage == "step3":
        _reject_step3_retired_controls(train)
    if stage == "step5":
        _reject_step5_retired_controls(train)
    row = dict(backend or {})
    public_keys = {"batch_size", "micro_batch_size", "grad_accum", "backend", "mode"}
    for key, value in train.items():
        if key not in public_keys:
            row[key] = deepcopy(value)
    row.update(
        {
            "train_batch_size": int(train["batch_size"]),
            "per_device_train_batch_size": int(train["micro_batch_size"]),
            "gradient_accumulation_steps": int(train["grad_accum"]),
            "epochs": int(train["epochs"]),
            "train_label_max_length": int(train.get("train_label_max_length", 64)),
            "lr": float(train.get("lr", task.get("lr", 1e-3))),
            "coef": float(train.get("coef", task.get("coef", 0.5))),
        }
    )
    if eval_batch_size is not None:
        row["eval_batch_size"] = int(eval_batch_size)
    return row


def _apply_step5_native_lora_row(row: dict[str, Any], step5_innovation_config: Mapping[str, Any]) -> None:
    ccv = _mapping(step5_innovation_config.get("ccv"), "step5.ccv")
    native_lora = _mapping(ccv.get("native_lora"), "step5.ccv.native_lora")
    lci = _mapping(step5_innovation_config.get("lci"), "step5.lci")
    fca = _mapping(step5_innovation_config.get("fca"), "step5.fca")
    row["step5_lci_weight"] = _rcr_float(lci.get("weight"), "step5.lci.weight", min_value=0.0)
    row["step5_fca_weight"] = _rcr_float(fca.get("weight"), "step5.fca.weight", min_value=0.0)
    enabled = _bool(native_lora.get("enabled", True))
    row["train_mode"] = "lora" if enabled else "full"
    row["lora_r"] = _positive_int(native_lora.get("r"), "step5.ccv.native_lora.r")
    row["lora_alpha"] = _rcr_float(native_lora.get("alpha"), "step5.ccv.native_lora.alpha", min_value=0.0)
    row["lora_dropout"] = _rcr_float(
        native_lora.get("dropout"),
        "step5.ccv.native_lora.dropout",
        min_value=0.0,
        max_value=1.0,
    )
    row["lora_target_modules"] = list(native_lora.get("target_modules", []) or [])


def _resolve_step5_model_config(cfg: Mapping[str, Any], runtime_roots: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.model"), "step5.model")
    out = {
        "emsize": _positive_int(raw.get("emsize"), "step5.model.emsize"),
        "nlayers": _positive_int(raw.get("nlayers"), "step5.model.nlayers"),
        "nhead": _positive_int(raw.get("nhead"), "step5.model.nhead"),
        "nhid": _positive_int(raw.get("nhid"), "step5.model.nhid"),
        "dropout": _rcr_float(raw.get("dropout"), "step5.model.dropout", min_value=0.0, max_value=1.0),
    }
    if int(out["emsize"]) != int(runtime_roots["embed_dim"]):
        raise OneControlConfigError(
            "step5.model.emsize must equal env.embed_dim; "
            f"got step5.model.emsize={out['emsize']} env.embed_dim={runtime_roots['embed_dim']}"
        )
    if int(out["emsize"]) % int(out["nhead"]) != 0:
        raise OneControlConfigError("step5.model.emsize must be divisible by step5.model.nhead")
    return out


def _resolve_step5_ddp_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.ddp"), "step5.ddp")
    find_unused = _bool(raw.get("find_unused_parameters", True))
    preflight = str(raw.get("find_unused_false_preflight", "synthetic_one_batch")).strip().lower()
    if preflight not in ("synthetic_one_batch", "fail_fast"):
        raise OneControlConfigError(
            "step5.ddp.find_unused_false_preflight must be 'synthetic_one_batch' or 'fail_fast'"
        )
    if not find_unused and preflight != "synthetic_one_batch":
        raise OneControlConfigError(
            "step5.ddp.find_unused_parameters=false requires "
            "step5.ddp.find_unused_false_preflight=synthetic_one_batch"
        )
    return {
        "ddp_find_unused_parameters": find_unused,
        "ddp_find_unused_false_preflight": preflight,
    }


def _lineage_for_step5(step4_run: str) -> str:
    if step4_run == "latest":
        return "latest"
    parts = run_naming.parse_run_id(step4_run).split("_")
    return parts[0]


def _lineage_for_eval(step5_run: str) -> tuple[str, str]:
    if step5_run == "latest":
        return "latest", "latest"
    parts = run_naming.parse_run_id(step5_run).split("_")
    if len(parts) < 3:
        raise OneControlConfigError("step5 run must look like {step3}_{step4_child}_{step5_child}, e.g. 2_1_1")
    return parts[0], "_".join(parts[:-1])


def resolve_config(
    *,
    config_path: str | Path,
    command: str,
    task_id: int | None,
    set_overrides: Iterable[str],
    dry_run: bool,
    run_id: str | None = None,
    from_step3: str | None = None,
    from_step4: str | None = None,
    from_step5: str | None = None,
    eval_profile: str | None = None,
    mode: str | None = None,
) -> tuple[ResolvedConfig, list[SourceRecord], dict[str, Any]]:
    base = load_yaml_config(config_path)
    cfg, cli_sources = apply_cli_sets(base, set_overrides)
    project = cfg["project"]
    if not isinstance(project, Mapping):
        raise OneControlConfigError("project must be a mapping")
    repo_root = _REPO_ROOT
    runtime_roots = _resolve_global_runtime_roots(cfg, repo_root)
    tid = int(task_id or project.get("default_task") or 4)
    task = _task_row(cfg, tid)
    auxiliary = str(task["source"])
    target = str(task["target"])

    stage_for_train = "step5" if command == "eval" else command
    stage_cfg = _get(cfg, stage_for_train, {})
    if not isinstance(stage_cfg, Mapping):
        stage_cfg = {}
    train_base = stage_cfg.get("train", {})
    if train_base is None:
        train_base = {}
    if not isinstance(train_base, Mapping):
        raise OneControlConfigError(f"{stage_for_train}.train must be a mapping")
    train = _merge_dicts(train_base, _stage_task_override(stage_cfg, tid))
    _apply_train_cli_overrides(cfg=cfg, cli_sources=cli_sources, stage=stage_for_train, train=train)

    eval_profile_name = ""
    eval_profile_obj: dict[str, Any] = {}
    need_eval = command in ("step4", "step5", "eval")
    if need_eval:
        eval_profile_name, eval_profile_obj = _resolve_eval_profile(cfg, eval_profile)
    hw_name_from_profile = eval_profile_obj.get("hardware") if eval_profile_obj else None
    hw_name, hw = _active_hardware(cfg, str(hw_name_from_profile) if hw_name_from_profile else None)
    ddp_world_size = _positive_int(hw.get("ddp_world_size", 1), "hardware.ddp_world_size")
    num_proc = _positive_int(hw.get("num_proc", 1), "hardware.num_proc")
    batch_size, micro, accum, eff = _validate_train_batch(stage_for_train, train, ddp_world_size)

    eval_batch_size: int | None = None
    eval_per_gpu: int | None = None
    if need_eval:
        eval_batch_size = _positive_int(eval_profile_obj.get("eval_batch_size"), "eval.profile.eval_batch_size")
        eval_per_gpu = _validate_eval_batch(eval_batch_size, ddp_world_size)

    need_decode = needs_decode_layer(command)
    decode_name = str(eval_profile_obj.get("decode")) if eval_profile_obj.get("decode") else None
    decode_id, decode = _resolve_decode(cfg, decode_name, need_decode=need_decode)
    need_rerank = command == "eval" and bool(eval_profile_obj.get("rerank"))
    rerank_name = str(eval_profile_obj.get("rerank")) if eval_profile_obj.get("rerank") else None
    rerank_id, rerank = _resolve_rerank(cfg, rerank_name, need_rerank=need_rerank)

    iteration_id = "v1"
    run_name: str | None = None
    from_run: str | None = None
    step4_run: str | None = None
    step5_run: str | None = None
    step3_checkpoint_dir: str | None = None
    eval_run_dir: str | None = None
    model_path: str | None = None

    if command == "step3":
        run_name = _alloc_run(repo_root, tid, "step3", run_id or "auto", dry_run=dry_run)
        run_root = _stage_root(repo_root, tid, "step3", run_name)
    elif command == "step4":
        src = from_step3 or "latest"
        from_run = _latest_run(repo_root, tid, "step3", dry_run=dry_run) if src == "latest" else run_naming.parse_run_id(src)
        step4_run = _alloc_run(repo_root, tid, "step4", run_id or "auto", dry_run=dry_run)
        run_root = _stage_root(repo_root, tid, "step4", step4_run)
        step3_checkpoint_dir = str(_stage_root(repo_root, tid, "step3", from_run))
    elif command == "step5":
        src = from_step4 or "latest"
        step4_run = _latest_run(repo_root, tid, "step4", dry_run=dry_run) if src == "latest" else run_naming.parse_run_id(src)
        from_run = _lineage_for_step5(step4_run)
        if run_id and run_id not in ("", "auto"):
            step5_run = _alloc_run(repo_root, tid, "step5", run_id, dry_run=dry_run)
        else:
            step5_parent = path_layout.get_stage_task_root(repo_root, "step5", tid)
            if not dry_run:
                step5_parent.mkdir(parents=True, exist_ok=True)
            step5_run = "dry_run" if dry_run else run_naming.allocate_step5_run_id(step5_parent, step4_run)
        run_root = _stage_root(repo_root, tid, "step5", step5_run)
        eval_run_dir = str((run_root / "post_train_eval").resolve())
    elif command == "eval":
        src = from_step5 or "latest"
        step5_run = _latest_run(repo_root, tid, "step5", dry_run=dry_run) if src == "latest" else run_naming.parse_run_id(src)
        from_run, step4_run = _lineage_for_eval(step5_run)
        eval_stage = "rerank" if need_rerank else "eval"
        eval_run_id = _alloc_run(repo_root, tid, eval_stage, run_id or "auto", dry_run=dry_run)
        run_root = _stage_root(repo_root, tid, "step5", step5_run)
        eval_run_dir = str(_stage_root(repo_root, tid, eval_stage, eval_run_id))
    else:
        raise OneControlConfigError(f"unsupported command for resolver: {command}")

    log_dir = str((Path(eval_run_dir) if command == "eval" and eval_run_dir else run_root) / "meta")
    manifest_dir = log_dir
    iteration_root = repo_root / "runs" / f"task{tid}"
    if not dry_run:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        stamp = Path(log_dir) / "resolved_config.generated_at"
        stamp.write_text(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ\n"), encoding="utf-8")

    step4_rcr_config = _resolve_step4_rcr_config(cfg)
    step3_structured_losses_config = _resolve_step3_structured_losses_config(cfg)
    step5_innovation_config = _resolve_step5_innovation_config(cfg)
    step5_model_config = _resolve_step5_model_config(cfg, runtime_roots)
    step5_ddp_config = _resolve_step5_ddp_config(cfg) if stage_for_train == "step5" else {}
    row = _training_row(stage_for_train, train, task, eval_batch_size=eval_batch_size if command in ("step5", "eval") else None)
    if stage_for_train == "step5":
        _apply_step5_native_lora_row(row, step5_innovation_config)
        row.update(step5_model_config)
        row.update(step5_ddp_config)
        if "explainer_loss_weight" not in row:
            raise OneControlConfigError("step5.train.explainer_loss_weight must be configured in configs/odcr.yaml")
        row["explainer_loss_weight"] = _rcr_float(
            row["explainer_loss_weight"],
            "step5.train.explainer_loss_weight",
            min_value=0.0,
        )
    payload = {
        "schema_version": 3,
        "task_id": tid,
        "preset_name": stage_for_train,
        "training_row": row,
        "explainer_loss_weight": 0.0 if stage_for_train == "step3" else float(row.get("explainer_loss_weight", 0.0)),
        "auxiliary": auxiliary,
        "target": target,
        "runtime_roots": runtime_roots,
    }
    payload["step3_structured_losses"] = step3_structured_losses_config
    payload["step4_rcr"] = step4_rcr_config
    if stage_for_train == "step5":
        payload["step5_innovation"] = step5_innovation_config
        payload["step5_model"] = step5_model_config

    decode_strategy = str(decode.get("decode_strategy", "greedy")).strip().lower()
    decode_seed = _optional_int(decode.get("decode_seed"))
    no_repeat = _optional_int(decode.get("no_repeat_ngram_size"))
    min_len = _optional_int(decode.get("min_len"))
    domain_fusion_mode = str(decode.get("domain_fusion_mode", "gate_cross_attn"))
    if decode_strategy not in ("greedy", "nucleus", "uncertainty_low_temp_top_k"):
        raise OneControlConfigError(f"unsupported decode_strategy: {decode_strategy!r}")

    hw_semantic = {
        k: v
        for k, v in hw.items()
        if k not in ("omp_num_threads", "mkl_num_threads", "tokenizers_parallelism", "cuda_visible_devices")
    }
    thread_env = {
        "OMP_NUM_THREADS": str(_positive_int(hw.get("omp_num_threads", 1), "hardware.omp_num_threads")),
        "MKL_NUM_THREADS": str(_positive_int(hw.get("mkl_num_threads", 1), "hardware.mkl_num_threads")),
        "TOKENIZERS_PARALLELISM": "true" if _bool(hw.get("tokenizers_parallelism", False)) else "false",
    }
    launcher_env = {}
    if hw.get("cuda_visible_devices") not in (None, ""):
        launcher_env["CUDA_VISIBLE_DEVICES"] = str(hw["cuda_visible_devices"])

    field_sources = {
        "config": str(Path(config_path)),
        "override_chain": "CLI --set > configs/odcr.yaml > resolver schema defaults",
        "task": f"tasks.{tid}",
        "hardware": f"hardware.profiles.{hw_name}",
        "train": f"{stage_for_train}.train",
        "eval_profile": f"eval.profiles.{eval_profile_name}" if eval_profile_name else None,
        "decode": f"eval.decode.{decode_id}" if decode_id else None,
        "rerank": f"eval.rerank.{rerank_id}" if rerank_id else None,
        "step4_rcr": "step4.rcr",
        "step3_structured_losses": "step3.structured_losses",
        "step5_lci": "step5.lci",
        "step5_uci": "step5.uci",
        "step5_explainer_gate": "step5.explainer_gate",
        "step5_ccv": "step5.ccv",
        "step5_fca": "step5.fca",
        "step5_native_lora": "step5.ccv.native_lora",
        "step5_model": "step5.model",
        "step5_ddp_find_unused_parameters": "step5.ddp.find_unused_parameters",
        "step5_ddp_find_unused_false_preflight": "step5.ddp.find_unused_false_preflight",
        "step5_train_explainer_loss_weight": "step5.train.explainer_loss_weight",
        "runs_dir": "project.run_root",
        "cache_dir": "project.cache_dir",
        "data_dir": "project.data_dir",
        "merged_dir": "project.merged_dir",
        "models_dir": "env.models_dir",
        "step5_text_model": "env.step5_text_model",
        "sentence_embed_model": "env.sentence_embed_model",
        "embed_dim": "env.embed_dim",
        "offline": "env.offline",
        "local_files_only": "env.local_files_only",
        **cli_sources,
    }
    consumed = {
        "single_config": str(Path(config_path)),
        "hardware_profile": hw_name,
        "eval_profile": eval_profile_name or None,
        "decode_profile": decode_id or None,
        "rerank_profile": rerank_id or None,
        "step4_rcr": "step4.rcr",
        "step3_structured_losses": "step3.structured_losses",
        "runtime_roots": {
            "data_dir": "project.data_dir",
            "merged_dir": "project.merged_dir",
            "runs_dir": "project.run_root",
            "cache_dir": "project.cache_dir",
            "models_dir": "env.models_dir",
            "step5_text_model": "env.step5_text_model",
            "sentence_embed_model": "env.sentence_embed_model",
            "embed_dim": "env.embed_dim",
            "offline": "env.offline",
            "local_files_only": "env.local_files_only",
        },
        "step5_innovation": {
            "lci": "step5.lci",
            "uci": "step5.uci",
            "explainer_gate": "step5.explainer_gate",
            "ccv": "step5.ccv",
            "fca": "step5.fca",
            "native_lora": "step5.ccv.native_lora",
            "model": "step5.model",
            "ddp": "step5.ddp",
            "explainer_loss_weight": "step5.train.explainer_loss_weight",
        },
    }
    train_fp = fingerprint({"payload": payload, "hardware": hw_semantic, "ddp_world_size": ddp_world_size})
    gen_fp = fingerprint({"decode": decode, "eval_batch_size": eval_batch_size, "rerank": rerank}) if need_decode else ""
    runtime_fp = fingerprint(runtime_diagnostics_fingerprint_source())
    sources = [SourceRecord(k, v, field_sources.get(k, "configs/odcr.yaml")) for k, v in sorted(field_sources.items())]

    resolved_snapshot = {
        "task": {"id": tid, "source": auxiliary, "target": target},
        "hardware": {"profile": hw_name, **hw_semantic},
        "train": {
            "stage": stage_for_train,
            "batch_size": batch_size,
            "micro_batch_size": micro,
            "grad_accum": accum,
            "ddp_world_size": ddp_world_size,
            "epochs": int(train["epochs"]),
            "lr": row["lr"],
            "coef": row["coef"],
            **(
                {"explainer_loss_weight": float(row["explainer_loss_weight"])}
                if stage_for_train == "step5"
                else {}
            ),
            **(
                {
                    "ddp_find_unused_parameters": bool(row["ddp_find_unused_parameters"]),
                    "ddp_find_unused_false_preflight": str(row["ddp_find_unused_false_preflight"]),
                }
                if stage_for_train == "step5"
                else {}
            ),
        },
        "eval": {
            "profile": eval_profile_name or None,
            "eval_batch_size": eval_batch_size,
            "eval_per_gpu_batch_size": eval_per_gpu,
            "decode": decode,
            "rerank": rerank if rerank else None,
            "rerank_source_table": (
                {
                    "eval_profile": f"eval.profiles.{eval_profile_name}.rerank",
                    "rerank_profile": f"eval.rerank.{rerank_id}",
                    "transport": "ResolvedConfig.rerank_profile_json + step5 eval-rerank CLI transport",
                }
                if rerank
                else None
            ),
        },
        "step4_rcr": step4_rcr_config if command == "step4" else None,
        "step3_structured_losses": step3_structured_losses_config if stage_for_train == "step3" else None,
        "step5": step5_innovation_config if stage_for_train == "step5" else None,
        "step5_model": step5_model_config if stage_for_train == "step5" else None,
        "step5_ddp": step5_ddp_config if stage_for_train == "step5" else None,
        "run": {
            "stage_run_dir": str(run_root),
            "meta_dir": log_dir,
            "from_step3": from_run,
            "from_step4": step4_run,
            "from_step5": step5_run,
            "eval_run_dir": eval_run_dir,
        },
        "roots": {
            "runs_dir": runtime_roots["runs_dir"],
            "cache_dir": runtime_roots["cache_dir"],
            "data_dir": runtime_roots["data_dir"],
            "merged_dir": runtime_roots["merged_dir"],
            "models_dir": runtime_roots["models_dir"],
            "source": {
                "runs_dir": "project.run_root",
                "cache_dir": "project.cache_dir",
                "data_dir": "project.data_dir",
                "merged_dir": "project.merged_dir",
                "models_dir": "env.models_dir",
            },
        },
        "models": {
            "step5_text_model": runtime_roots["step5_text_model"],
            "sentence_embed_model": runtime_roots["sentence_embed_model"],
            "source": {
                "step5_text_model": "env.step5_text_model",
                "sentence_embed_model": "env.sentence_embed_model",
            },
        },
        "embed_dim": {
            "value": int(runtime_roots["embed_dim"]),
            "source": "env.embed_dim",
        },
        "offline": {
            "value": bool(runtime_roots["offline"]),
            "source": "env.offline",
        },
        "local_files_only": {
            "value": bool(runtime_roots["local_files_only"]),
            "source": "env.local_files_only",
        },
        "field_sources": field_sources,
    }

    cfg_obj = ResolvedConfig(
        command="eval-rerank" if command == "eval" and need_rerank else command,
        repo_root=repo_root,
        code_dir=_CODE_DIR,
        task_id=tid,
        auxiliary=auxiliary,
        target=target,
        preset_name=stage_for_train,
        run_name=run_name,
        from_run=from_run,
        step5_run=step5_run,
        step4_run=step4_run,
        step3_checkpoint_dir=step3_checkpoint_dir,
        train_csv=None,
        model_path=model_path,
        learning_rate=float(row["lr"]),
        coef=float(row["coef"]),
        adv=0.0,
        eta=0.0,
        train_batch_size=batch_size,
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=accum,
        effective_global_batch_size=eff,
        epochs=int(train["epochs"]),
        num_proc=num_proc,
        ddp_world_size=ddp_world_size,
        seed=int(project.get("seed", 3407)),
        checkpoint_dir=str(run_root),
        log_dir=log_dir,
        iteration_root_dir=str(iteration_root),
        iteration_id=iteration_id,
        manifest_dir=manifest_dir,
        eval_run_dir=eval_run_dir,
        label_smoothing=float(decode["label_smoothing"]),
        repetition_penalty=float(decode["repetition_penalty"]),
        generate_temperature=float(decode["generate_temperature"]),
        generate_top_p=float(decode["generate_top_p"]),
        decode_strategy=decode_strategy,
        decode_seed=decode_seed,
        max_explanation_length=int(decode["max_explanation_length"]),
        train_label_max_length=int(train.get("train_label_max_length", 64)),
        no_repeat_ngram_size=no_repeat,
        min_len=min_len,
        domain_fusion_mode=domain_fusion_mode,
        step3_mode=str(mode or train.get("mode") or "full"),
        step5_train_only=False,
        hardware_preset_id=hw_name,
        decode_preset_id=decode_id,
        num_return_sequences=int(eval_profile_obj.get("num_return_sequences", rerank.get("num_return_sequences", 3) if rerank else 3)),
        rerank_method=str(rerank.get("rerank_method", "rule_v3")),
        rerank_top_k=int(rerank.get("rerank_top_k", 1)),
        rerank_weight_logprob=float(rerank.get("rerank_weight_logprob", 0.35)),
        rerank_weight_length=float(rerank.get("rerank_weight_length", 0.10)),
        rerank_weight_repeat=float(rerank.get("rerank_weight_repeat", 0.16)),
        rerank_weight_dirty=float(rerank.get("rerank_weight_dirty", 0.20)),
        rerank_target_len_ratio=float(rerank.get("rerank_target_len_ratio", 1.10)),
        export_examples_mode=str(rerank.get("export_examples_mode", "head50")),
        export_full_rerank_examples=str(rerank.get("export_examples_mode", "")).lower() == "full",
        rerank_malformed_tail_penalty=float(rerank.get("rerank_malformed_tail_penalty", 0.15)),
        rerank_malformed_token_penalty=float(rerank.get("rerank_malformed_token_penalty", 0.18)),
        decode_profile_json=json_dumps(decode),
        rerank_profile_json=json_dumps(rerank),
        rerank_preset_id=rerank_id,
        hardware_profile_json=json_dumps(hw_semantic),
        omp_num_threads=int(thread_env["OMP_NUM_THREADS"]),
        mkl_num_threads=int(thread_env["MKL_NUM_THREADS"]),
        tokenizers_parallelism=thread_env["TOKENIZERS_PARALLELISM"] == "true",
        thread_env_requested_json=json_dumps(thread_env),
        thread_env_effective_json=json_dumps(thread_env),
        launcher_env_requested_json=json_dumps(launcher_env),
        launcher_env_effective_json=json_dumps(launcher_env),
        training_preset_train_batch_size=batch_size,
        global_eval_batch_size=eval_batch_size,
        eval_per_gpu_batch_size=eval_per_gpu,
        eval_profile_id=eval_profile_name,
        consumed_presets_json=json_dumps(consumed),
        config_before_cli_json=json_dumps({"config_path": str(config_path)}),
        matrix_session_id=None,
        matrix_cell_id=None,
        invoked_command=command,
        resolved_command_kind=command,
        cell_command=None,
        effective_training_payload_json=json_dumps(payload),
        training_semantic_fingerprint=train_fp,
        generation_semantic_fingerprint=gen_fp,
        runtime_diagnostics_fingerprint=runtime_fp,
        config_field_sources_json=json_dumps(field_sources),
        eval_profile_resolution_json=json_dumps(eval_profile_obj),
        step4_rcr_config_json=json_dumps(step4_rcr_config),
        step5_innovation_config_json=json_dumps(step5_innovation_config),
        ddp_find_unused_parameters=bool(row.get("ddp_find_unused_parameters", True)),
        ddp_find_unused_false_preflight=str(row.get("ddp_find_unused_false_preflight", "synthetic_one_batch")),
        explainer_loss_weight=float(payload["explainer_loss_weight"]),
        data_dir=str(runtime_roots["data_dir"]),
        merged_dir=str(runtime_roots["merged_dir"]),
        runs_dir=str(runtime_roots["runs_dir"]),
        cache_dir=str(runtime_roots["cache_dir"]),
        models_dir=str(runtime_roots["models_dir"]),
        step5_text_model=str(runtime_roots["step5_text_model"]),
        sentence_embed_model=str(runtime_roots["sentence_embed_model"]),
        embed_dim=int(runtime_roots["embed_dim"]),
        offline=bool(runtime_roots["offline"]),
        local_files_only=bool(runtime_roots["local_files_only"]),
        full_bleu_eval_resolved=dict(row.get("full_bleu_eval", {"mode": "off"})),
        full_bleu_decode_strategy=str(row.get("full_bleu_decode_strategy", "inherit")),
        train_mode=str(row.get("train_mode", "lora" if stage_for_train == "step5" else "full")),
        train_precision=str(row.get("train_precision", "bf16")),
        per_device_eval_batch_size=int(row.get("per_device_eval_batch_size", 2)),
        lora_r=int(row.get("lora_r", 16)),
        lora_alpha=float(row.get("lora_alpha", 32.0)),
        lora_dropout=float(row.get("lora_dropout", 0.05)),
        lora_target_modules=tuple(row.get("lora_target_modules", ()) or ()),
        nlayers=int(row.get("nlayers", 2)),
        nhead=int(row.get("nhead", 2)),
        nhid=int(row.get("nhid", 2048)),
        dropout=float(row.get("dropout", 0.2)),
    )
    return cfg_obj, sources, resolved_snapshot


def write_resolved_config(cfg: ResolvedConfig, snapshot: Mapping[str, Any], *, dry_run: bool) -> Path | None:
    if dry_run:
        return None
    from odcr_core.manifests import write_resolved_config_artifacts

    out, _ = write_resolved_config_artifacts(Path(cfg.manifest_dir), snapshot)
    return out


def build_preprocess_config(
    *,
    config_path: str | Path,
    stage_letter: str,
    set_overrides: Iterable[str],
    dry_run: bool,
):
    base = load_yaml_config(config_path)
    cfg, _ = apply_cli_sets(base, set_overrides)
    repo_root = _REPO_ROOT
    runtime_roots = _resolve_global_runtime_roots(cfg, repo_root)
    pp = cfg["preprocess"]
    if not isinstance(pp, Mapping):
        raise OneControlConfigError("preprocess must be a mapping")
    letter = stage_letter.lower()
    if letter not in ("a", "b", "c"):
        raise OneControlConfigError("preprocess stage must be a, b, or c")
    raw = pp.get(letter, {})
    if not isinstance(raw, Mapping):
        raise OneControlConfigError(f"preprocess.{letter} must be a mapping")

    from odcr_core.preprocess_schema import (
        PreprocessAConfig,
        PreprocessBConfig,
        PreprocessCConfig,
        PreprocessHardwareConfig,
        PreprocessPathsConfig,
        PreprocessResolvedPayload,
        PreprocessRuntimeOptions,
        validate_preprocess_config,
    )

    def _preprocess_cache_path(raw_value: Any, *, default_name: str) -> str:
        raw_text = str(raw_value or default_name).strip()
        if not raw_text:
            raise OneControlConfigError("preprocess cache path must be non-empty")
        p = Path(raw_text)
        if p.is_absolute():
            return str(p.resolve())
        parts = p.parts
        if parts and parts[0] == "cache":
            return str((Path(runtime_roots["cache_dir"]) / Path(*parts[1:])).resolve())
        return str((repo_root / p).resolve())

    def _resolved_payload(
        *,
        gpu_ids: tuple[int, ...] = (),
        bf16: bool = False,
        tf32: bool = False,
    ) -> PreprocessResolvedPayload:
        return PreprocessResolvedPayload(
            data_dir=str(runtime_roots["data_dir"]),
            merged_dir=str(runtime_roots["merged_dir"]),
            runs_dir=str(runtime_roots["runs_dir"]),
            cache_dir=str(runtime_roots["cache_dir"]),
            models_dir=str(runtime_roots["models_dir"]),
            step5_text_model=str(runtime_roots["step5_text_model"]),
            sentence_embed_model=str(runtime_roots["sentence_embed_model"]),
            sentence_embed_model_path=str(runtime_roots["sentence_embed_model"]),
            embed_dim=int(runtime_roots["embed_dim"]),
            offline=bool(runtime_roots["offline"]),
            local_files_only=bool(runtime_roots["local_files_only"]),
            gpu_ids=tuple(int(item) for item in gpu_ids),
            bf16=bool(bf16),
            tf32=bool(tf32),
            sources={
                "data_dir": "project.data_dir",
                "merged_dir": "project.merged_dir",
                "runs_dir": "project.run_root",
                "cache_dir": "project.cache_dir",
                "models_dir": "env.models_dir",
                "step5_text_model": "env.step5_text_model",
                "sentence_embed_model": "env.sentence_embed_model",
                "sentence_embed_model_path": "env.sentence_embed_model",
                "embed_dim": "env.embed_dim",
                "offline": "env.offline",
                "local_files_only": "env.local_files_only",
                "gpu_ids": f"preprocess.{letter}.gpu_ids or hardware.preprocess.gpu_ids",
                "bf16": f"preprocess.{letter}.bf16_enabled",
                "tf32": f"preprocess.{letter}.tf32_enabled",
            },
        )

    datasets = tuple(raw.get("datasets", pp.get("datasets", ())))
    runtime = PreprocessRuntimeOptions(
        python_bin=str(_get(cfg, "env.python_bin", "python")),
        resume=_bool(raw.get("resume", True)),
        skip_completed=_bool(raw.get("skip_completed", True)),
        verify_only=_bool(raw.get("verify_only", False)),
        dry_run=bool(dry_run),
        workers=raw.get("workers"),
        force_datasets=tuple(raw.get("force_datasets", ())),
    )
    if dry_run:
        preprocess_run_id = "dry_run"
    else:
        preprocess_parent = Path(runtime_roots["runs_dir"]) / "preprocess" / letter
        preprocess_parent.mkdir(parents=True, exist_ok=True)
        preprocess_run_id = run_naming.allocate_child_dir(
            preprocess_parent,
            requested="auto",
            kind="run",
        )
    preprocess_run_root = Path(runtime_roots["runs_dir"]) / "preprocess" / letter / preprocess_run_id
    paths = PreprocessPathsConfig(
        meta_root=str((preprocess_run_root / "meta").resolve()),
        shell_log_dir=str((preprocess_run_root / "meta" / "shell_logs").resolve()),
    )
    if letter == "a":
        return validate_preprocess_config(
            PreprocessAConfig(
                preset_name="one_control_preprocess_a",
                description="One-control preprocess A",
                datasets=datasets,
                paths=paths,
                runtime=runtime,
                run_id=preprocess_run_id,
                resolved=_resolved_payload(),
            )
        )
    gpu_ids = tuple(int(x) for x in raw.get("gpu_ids", _get(cfg, "hardware.preprocess.gpu_ids", (0, 1))))
    if letter == "b":
        bf16_enabled = _bool(raw.get("bf16_enabled", True))
        tf32_enabled = _bool(raw.get("tf32_enabled", True))
        return validate_preprocess_config(
            PreprocessBConfig(
                preset_name="one_control_preprocess_b",
                description="One-control preprocess B",
                datasets=datasets,
                paths=paths,
                runtime=runtime,
                run_id=preprocess_run_id,
                hardware=PreprocessHardwareConfig(gpu_ids=gpu_ids),
                embed_batch_size=int(raw.get("batch_size", 512)),
                read_chunk_rows=int(raw.get("read_chunk_rows", 100_000)),
                group_shard_size=int(raw.get("group_shard_size", 4096)),
                grouped_text_cache_enabled=_bool(raw.get("grouped_text_cache_enabled", True)),
                grouped_text_cache_dir=_preprocess_cache_path(
                    raw.get("grouped_text_cache_dir", "cache/preprocess_b"),
                    default_name="cache/preprocess_b",
                ),
                grouped_text_cache_version=str(raw.get("grouped_text_cache_version", "preprocess_b_grouped_text_cache_v1")),
                bf16_enabled=bf16_enabled,
                tf32_enabled=tf32_enabled,
                verify_sample_size=int(raw.get("verify_sample_size", 8)),
                verify_seed=int(raw.get("verify_seed", 7)),
                resolved=_resolved_payload(gpu_ids=gpu_ids, bf16=bf16_enabled, tf32=tf32_enabled),
            )
        )
    bf16_enabled = _bool(raw.get("bf16_enabled", True))
    tf32_enabled = _bool(raw.get("tf32_enabled", True))
    return validate_preprocess_config(
        PreprocessCConfig(
            preset_name="one_control_preprocess_c",
            description="One-control preprocess C",
            datasets=datasets,
            paths=paths,
            runtime=runtime,
            run_id=preprocess_run_id,
            hardware=PreprocessHardwareConfig(gpu_ids=gpu_ids),
            chunk_batch_size=int(raw.get("chunk_batch_size", 512)),
            bf16_enabled=bf16_enabled,
            tf32_enabled=tf32_enabled,
            tokenizer_hotpath_enabled=_bool(raw.get("tokenizer_hotpath_enabled", True)),
            token_window_cache_enabled=_bool(raw.get("token_window_cache_enabled", True)),
            token_window_cache_dir=_preprocess_cache_path(
                raw.get("token_window_cache_dir", "cache/preprocess_c"),
                default_name="cache/preprocess_c",
            ),
            token_window_cache_version=str(raw.get("token_window_cache_version", "preprocess_c_token_windows_v3")),
            token_window_cache_shard_size=int(raw.get("token_window_cache_shard_size", 2048)),
            resolved=_resolved_payload(gpu_ids=gpu_ids, bf16=bf16_enabled, tf32=tf32_enabled),
        )
    )
