from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from odcr_core.gather_schema import GatheredBatch
from odcr_core.step5_word_losses import route_weighted_mean


EVIDENCE_QUALITY_PRIOR = 0
CF_RELIABILITY = 1
STYLE_SHIFT = 2
RATING_STABILITY = 3
CONTENT_RETENTION = 4
TEXT_QUALITY = 5
UNCERTAINTY = 6
RATING_POLARITY = 7
STEP5_EVIDENCE_FEATURE_DIM = 8


@dataclass(frozen=True)
class Step5LCIConfig:
    enabled: bool
    weight: float
    confidence_schedule: dict[str, float]
    min_reliability: float
    max_uncertainty: float
    perturb_std: float
    counterfactual_label_weight: float
    robustness_weight: float


@dataclass(frozen=True)
class Step5UCIConfig:
    enabled: bool
    bucket_weights: dict[str, float]
    uncertainty_temperature: float
    low_confidence_floor: float


@dataclass(frozen=True)
class Step5ExplainerGateConfig:
    bucket_weights: dict[str, float]
    uncertainty_exponent: float
    style_shift_diversity_boost: float
    min_weight: float
    max_weight: float
    explainer_only_multiplier: float


@dataclass(frozen=True)
class Step5NativeLoRAConfig:
    enabled: bool
    r: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class Step5CCVConfig:
    enabled: bool
    control_fields: tuple[str, ...]
    uncertainty_tone_control: bool
    route_conditioning: bool
    numeric_control_weight: float
    control_packet_field_policy: str
    verbalizer_adapter_policy: str
    soft_prompt_len: int
    numeric_control_dim: int
    control_adapter_input_blocks: int
    native_lora: Step5NativeLoRAConfig


@dataclass(frozen=True)
class Step5FCAConfig:
    enabled: bool
    weight: float
    min_reliability: float
    max_uncertainty: float
    evidence_alignment_mode: str


@dataclass(frozen=True)
class Step5InnovationConfig:
    lci: Step5LCIConfig
    uci: Step5UCIConfig
    explainer_gate: Step5ExplainerGateConfig
    ccv: Step5CCVConfig
    fca: Step5FCAConfig


@dataclass(frozen=True)
class RatingStabilityControlGate:
    scorer_weight: torch.Tensor
    lci_weight: torch.Tensor
    uci_weight: torch.Tensor
    route_mask: torch.Tensor
    reliability: torch.Tensor
    uncertainty: torch.Tensor
    confidence_bucket: torch.Tensor
    rating_stability: torch.Tensor


@dataclass(frozen=True)
class Step5ExplanationGate:
    explainer_weight: torch.Tensor
    fca_weight: torch.Tensor
    route_mask: torch.Tensor
    reliability: torch.Tensor
    uncertainty: torch.Tensor
    confidence_bucket: torch.Tensor
    style_shift: torch.Tensor


@dataclass(frozen=True)
class RatingStabilityControlLCILoss:
    lci_loss: torch.Tensor
    lci_weighted_loss: torch.Tensor
    lci_consistency_loss: torch.Tensor
    lci_cf_score_loss: torch.Tensor
    lci_robustness_loss: torch.Tensor
    uci_weight_mean: torch.Tensor
    scorer_weight_mean: torch.Tensor


@dataclass(frozen=True)
class Step5ExplanationFCALoss:
    fca_loss: torch.Tensor
    fca_weighted_loss: torch.Tensor
    fca_weight_mean: torch.Tensor
    scorer_evidence_basis: torch.Tensor
    explainer_evidence_basis: torch.Tensor


@dataclass(frozen=True)
class CCVControlPacket:
    content_evidence_ids: torch.Tensor
    style_evidence_ids: torch.Tensor
    domain_style_anchor_ids: torch.Tensor
    local_style_hint_ids: torch.Tensor
    polarity_ids: torch.Tensor
    route_scorer_mask: torch.Tensor
    route_explainer_mask: torch.Tensor
    sample_weight_hint: torch.Tensor
    cf_reliability_score: torch.Tensor
    content_retention_score: torch.Tensor
    style_shift_score: torch.Tensor
    rating_stability_score: torch.Tensor
    uncertainty_score: torch.Tensor
    confidence_bucket: torch.Tensor
    evidence_quality_prior: torch.Tensor
    content_anchor_score: torch.Tensor
    style_anchor_score: torch.Tensor

    def numeric_controls(self) -> torch.Tensor:
        tone = (1.0 - self.uncertainty_score).clamp(0.0, 1.0)
        bucket = (self.confidence_bucket / 2.0).clamp(0.0, 1.0)
        return torch.stack(
            [
                self.route_scorer_mask,
                self.route_explainer_mask,
                self.sample_weight_hint.clamp_min(0.0),
                self.cf_reliability_score.clamp(0.0, 1.0),
                self.content_retention_score.clamp(0.0, 1.0),
                self.style_shift_score.clamp(0.0, 1.0),
                self.rating_stability_score.clamp(0.0, 1.0),
                self.uncertainty_score.clamp(0.0, 1.0),
                bucket,
                tone,
                self.evidence_quality_prior.clamp(0.0, 1.0),
                self.content_anchor_score.clamp(0.0, 1.0),
                self.style_anchor_score.clamp(0.0, 1.0),
            ],
            dim=-1,
        )


_CCV_TEXT_CONTROL_ID_FIELDS: tuple[str, ...] = (
    "content_evidence_ids",
    "style_evidence_ids",
    "domain_style_anchor_ids",
    "local_style_hint_ids",
    "polarity_ids",
)


def _packet_batch_size(packet: Any) -> int | None:
    for name in (
        "route_scorer_mask",
        "route_explainer_mask",
        "sample_weight_hint",
        "content_anchor_score",
        "style_anchor_score",
    ):
        value = getattr(packet, name, None)
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            return int(value.shape[0])
    return None


def validate_ccv_control_packet_shapes(
    packet: Any,
    *,
    producer: str,
    head: str,
    strict: bool = True,
) -> None:
    """Fail fast on malformed CCV text-control tensors before model embedding."""

    batch_size = _packet_batch_size(packet)
    missing: list[str] = []
    for name in _CCV_TEXT_CONTROL_ID_FIELDS:
        ids = getattr(packet, name, None)
        if ids is None:
            missing.append(name)
            continue
        if not isinstance(ids, torch.Tensor):
            raise RuntimeError(
                f"producer={producer} head={head} field={name} "
                f"CCV control ids must be tensor [B,T], got {type(ids).__name__}."
            )
        if ids.dim() != 2:
            raise RuntimeError(
                f"producer={producer} head={head} field={name} "
                f"CCV control ids must be [B,T], got {tuple(ids.shape)}"
            )
        if batch_size is not None and int(ids.shape[0]) != int(batch_size):
            raise RuntimeError(
                f"producer={producer} head={head} field={name} "
                f"CCV control ids batch dim must be B={int(batch_size)}, got {tuple(ids.shape)}"
            )
        if int(ids.shape[1]) <= 0:
            raise RuntimeError(
                f"producer={producer} head={head} field={name} "
                f"CCV control ids must have non-empty token length T, got {tuple(ids.shape)}"
            )
        if ids.dtype not in (torch.long, torch.int64, torch.int32):
            raise RuntimeError(
                f"producer={producer} head={head} field={name} "
                f"CCV control ids must use integer token dtype, got {ids.dtype}"
            )
    if strict and missing:
        raise RuntimeError(
            f"producer={producer} head={head} CCV control packet missing text-control fields: "
            + ", ".join(missing)
        )


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _step5_test_default_mapping() -> dict[str, Any]:
    return {
        "lci": {
            "enabled": True,
            "weight": 0.12,
            "confidence_schedule": {"high": 1.25, "medium": 0.75, "low": 0.0},
            "min_reliability": 0.55,
            "max_uncertainty": 0.58,
            "perturb_std": 0.12,
            "counterfactual_label_weight": 0.25,
            "robustness_weight": 0.35,
        },
        "uci": {
            "enabled": True,
            "bucket_weights": {"high": 1.25, "medium": 0.75, "low": 0.0},
            "uncertainty_temperature": 1.35,
            "low_confidence_floor": 0.0,
        },
        "explainer_gate": {
            "bucket_weights": {"high": 1.10, "medium": 1.0, "low": 0.55},
            "uncertainty_exponent": 0.85,
            "style_shift_diversity_boost": 0.15,
            "min_weight": 0.0,
            "max_weight": 2.0,
            "explainer_only_multiplier": 0.7,
        },
        "ccv": {
            "enabled": True,
            "control_fields": (
                "content_evidence",
                "style_evidence",
                "domain_style_anchor",
                "local_style_residual_hint",
                "polarity_anchor",
                "cf_reliability_score",
                "content_retention_score",
                "style_shift_score",
                "rating_stability_score",
                "uncertainty_score",
                "confidence_bucket",
                "route_explainer",
                "route_scorer",
                "sample_weight_hint",
                "content_anchor_score",
                "style_anchor_score",
                "evidence_quality_prior",
            ),
            "uncertainty_tone_control": True,
            "route_conditioning": True,
            "numeric_control_weight": 1.0,
            "control_packet_field_policy": "strict_required",
            "verbalizer_adapter_policy": "ccv_control_adapter",
            "soft_prompt_len": 16,
            "numeric_control_dim": 13,
            "control_adapter_input_blocks": 6,
            "native_lora": {
                "enabled": True,
                "r": 16,
                "alpha": 32.0,
                "dropout": 0.05,
                "target_modules": (),
            },
        },
        "fca": {
            "enabled": True,
            "weight": 0.08,
            "min_reliability": 0.50,
            "max_uncertainty": 0.62,
            "evidence_alignment_mode": "evidence_basis",
        },
    }


def _merge_nested(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge_nested(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _required_mapping(obj: Mapping[str, Any], key: str, ctx: str, *, allow_test_defaults: bool) -> Mapping[str, Any]:
    value = obj.get(key)
    if isinstance(value, Mapping):
        return value
    if allow_test_defaults:
        return {}
    raise RuntimeError(f"Step5 active config missing required mapping {ctx}.{key}; use configs/odcr.yaml via resolver.")


def _required_value(obj: Mapping[str, Any], key: str, ctx: str, default: Any, *, allow_test_defaults: bool) -> Any:
    if key in obj:
        return obj[key]
    if allow_test_defaults:
        return default
    raise RuntimeError(f"Step5 active config missing required value {ctx}.{key}; use configs/odcr.yaml via resolver.")


def for_test_default_step5_innovation_config() -> Step5InnovationConfig:
    """Explicit unit-test fixture; active Step5 must pass resolver JSON."""
    return parse_step5_innovation_config_json(None, allow_test_defaults=True)


def parse_step5_innovation_config_json(
    raw: str | Mapping[str, Any] | None,
    *,
    allow_test_defaults: bool = False,
) -> Step5InnovationConfig:
    if isinstance(raw, Mapping):
        obj = dict(raw)
    elif raw and str(raw).strip():
        obj = json.loads(str(raw))
    else:
        if not allow_test_defaults:
            raise RuntimeError("Step5 active config JSON is required; parser None/empty fallback is test-only.")
        obj = {}
    if not isinstance(obj, Mapping):
        raise RuntimeError("Step5 innovation config JSON root must be an object.")
    if allow_test_defaults:
        obj = _merge_nested(_step5_test_default_mapping(), obj)
    elif not obj:
        raise RuntimeError("Step5 active config JSON must not be {}; use configs/odcr.yaml via resolver.")
    lci = _required_mapping(obj, "lci", "step5", allow_test_defaults=allow_test_defaults)
    uci = _required_mapping(obj, "uci", "step5", allow_test_defaults=allow_test_defaults)
    explainer_gate = _required_mapping(
        obj,
        "explainer_gate",
        "step5",
        allow_test_defaults=allow_test_defaults,
    )
    ccv = _required_mapping(obj, "ccv", "step5", allow_test_defaults=allow_test_defaults)
    fca = _required_mapping(obj, "fca", "step5", allow_test_defaults=allow_test_defaults)
    native_lora = _required_mapping(ccv, "native_lora", "step5.ccv", allow_test_defaults=allow_test_defaults)
    lora_targets = _required_value(
        native_lora,
        "target_modules",
        "step5.ccv.native_lora",
        (),
        allow_test_defaults=allow_test_defaults,
    )
    if not isinstance(lora_targets, (list, tuple)):
        lora_targets = ()
    lci_schedule = _required_mapping(lci, "confidence_schedule", "step5.lci", allow_test_defaults=allow_test_defaults)
    uci_buckets = _required_mapping(uci, "bucket_weights", "step5.uci", allow_test_defaults=allow_test_defaults)
    explainer_buckets = _required_mapping(
        explainer_gate,
        "bucket_weights",
        "step5.explainer_gate",
        allow_test_defaults=allow_test_defaults,
    )
    explainer_min = max(
        0.0,
        _float(
            _required_value(
                explainer_gate,
                "min_weight",
                "step5.explainer_gate",
                0.0,
                allow_test_defaults=allow_test_defaults,
            ),
            0.0,
        ),
    )
    explainer_max = max(
        explainer_min,
        _float(
            _required_value(
                explainer_gate,
                "max_weight",
                "step5.explainer_gate",
                2.0,
                allow_test_defaults=allow_test_defaults,
            ),
            2.0,
        ),
    )
    return Step5InnovationConfig(
        lci=Step5LCIConfig(
            enabled=_bool(_required_value(lci, "enabled", "step5.lci", True, allow_test_defaults=allow_test_defaults)),
            weight=max(0.0, _float(_required_value(lci, "weight", "step5.lci", 0.12, allow_test_defaults=allow_test_defaults), 0.12)),
            confidence_schedule={
                "high": max(0.0, _float(_required_value(lci_schedule, "high", "step5.lci.confidence_schedule", 1.25, allow_test_defaults=allow_test_defaults), 1.25)),
                "medium": max(0.0, _float(_required_value(lci_schedule, "medium", "step5.lci.confidence_schedule", 0.75, allow_test_defaults=allow_test_defaults), 0.75)),
                "low": max(0.0, _float(_required_value(lci_schedule, "low", "step5.lci.confidence_schedule", 0.0, allow_test_defaults=allow_test_defaults), 0.0)),
            },
            min_reliability=max(0.0, min(1.0, _float(_required_value(lci, "min_reliability", "step5.lci", 0.55, allow_test_defaults=allow_test_defaults), 0.55))),
            max_uncertainty=max(0.0, min(1.0, _float(_required_value(lci, "max_uncertainty", "step5.lci", 0.58, allow_test_defaults=allow_test_defaults), 0.58))),
            perturb_std=max(0.0, _float(_required_value(lci, "perturb_std", "step5.lci", 0.12, allow_test_defaults=allow_test_defaults), 0.12)),
            counterfactual_label_weight=max(0.0, _float(_required_value(lci, "counterfactual_label_weight", "step5.lci", 0.25, allow_test_defaults=allow_test_defaults), 0.25)),
            robustness_weight=max(0.0, _float(_required_value(lci, "robustness_weight", "step5.lci", 0.35, allow_test_defaults=allow_test_defaults), 0.35)),
        ),
        uci=Step5UCIConfig(
            enabled=_bool(_required_value(uci, "enabled", "step5.uci", True, allow_test_defaults=allow_test_defaults)),
            bucket_weights={
                "high": max(0.0, _float(_required_value(uci_buckets, "high", "step5.uci.bucket_weights", 1.25, allow_test_defaults=allow_test_defaults), 1.25)),
                "medium": max(0.0, _float(_required_value(uci_buckets, "medium", "step5.uci.bucket_weights", 0.75, allow_test_defaults=allow_test_defaults), 0.75)),
                "low": max(0.0, _float(_required_value(uci_buckets, "low", "step5.uci.bucket_weights", 0.0, allow_test_defaults=allow_test_defaults), 0.0)),
            },
            uncertainty_temperature=max(1e-6, _float(_required_value(uci, "uncertainty_temperature", "step5.uci", 1.35, allow_test_defaults=allow_test_defaults), 1.35)),
            low_confidence_floor=max(0.0, min(1.0, _float(_required_value(uci, "low_confidence_floor", "step5.uci", 0.0, allow_test_defaults=allow_test_defaults), 0.0))),
        ),
        explainer_gate=Step5ExplainerGateConfig(
            bucket_weights={
                "high": max(0.0, _float(_required_value(explainer_buckets, "high", "step5.explainer_gate.bucket_weights", 1.10, allow_test_defaults=allow_test_defaults), 1.10)),
                "medium": max(0.0, _float(_required_value(explainer_buckets, "medium", "step5.explainer_gate.bucket_weights", 1.0, allow_test_defaults=allow_test_defaults), 1.0)),
                "low": max(0.0, _float(_required_value(explainer_buckets, "low", "step5.explainer_gate.bucket_weights", 0.55, allow_test_defaults=allow_test_defaults), 0.55)),
            },
            uncertainty_exponent=max(1e-6, _float(_required_value(explainer_gate, "uncertainty_exponent", "step5.explainer_gate", 0.85, allow_test_defaults=allow_test_defaults), 0.85)),
            style_shift_diversity_boost=max(0.0, _float(_required_value(explainer_gate, "style_shift_diversity_boost", "step5.explainer_gate", 0.15, allow_test_defaults=allow_test_defaults), 0.15)),
            min_weight=explainer_min,
            max_weight=explainer_max,
            explainer_only_multiplier=max(0.0, _float(_required_value(explainer_gate, "explainer_only_multiplier", "step5.explainer_gate", 0.7, allow_test_defaults=allow_test_defaults), 0.7)),
        ),
        ccv=Step5CCVConfig(
            enabled=_bool(_required_value(ccv, "enabled", "step5.ccv", True, allow_test_defaults=allow_test_defaults)),
            control_fields=tuple(str(x) for x in _required_value(ccv, "control_fields", "step5.ccv", (), allow_test_defaults=allow_test_defaults) or ()),
            uncertainty_tone_control=_bool(_required_value(ccv, "uncertainty_tone_control", "step5.ccv", True, allow_test_defaults=allow_test_defaults)),
            route_conditioning=_bool(_required_value(ccv, "route_conditioning", "step5.ccv", True, allow_test_defaults=allow_test_defaults)),
            numeric_control_weight=max(0.0, _float(_required_value(ccv, "numeric_control_weight", "step5.ccv", 1.0, allow_test_defaults=allow_test_defaults), 1.0)),
            control_packet_field_policy=str(
                _required_value(ccv, "control_packet_field_policy", "step5.ccv", "strict_required", allow_test_defaults=allow_test_defaults)
            ).strip().lower(),
            verbalizer_adapter_policy=str(
                _required_value(ccv, "verbalizer_adapter_policy", "step5.ccv", "ccv_control_adapter", allow_test_defaults=allow_test_defaults)
            ).strip().lower(),
            soft_prompt_len=max(1, _int(_required_value(ccv, "soft_prompt_len", "step5.ccv", 16, allow_test_defaults=allow_test_defaults), 16)),
            numeric_control_dim=max(1, _int(_required_value(ccv, "numeric_control_dim", "step5.ccv", 13, allow_test_defaults=allow_test_defaults), 13)),
            control_adapter_input_blocks=max(1, _int(_required_value(ccv, "control_adapter_input_blocks", "step5.ccv", 6, allow_test_defaults=allow_test_defaults), 6)),
            native_lora=Step5NativeLoRAConfig(
                enabled=_bool(_required_value(native_lora, "enabled", "step5.ccv.native_lora", True, allow_test_defaults=allow_test_defaults)),
                r=max(1, _int(_required_value(native_lora, "r", "step5.ccv.native_lora", 16, allow_test_defaults=allow_test_defaults), 16)),
                alpha=max(0.0, _float(_required_value(native_lora, "alpha", "step5.ccv.native_lora", 32.0, allow_test_defaults=allow_test_defaults), 32.0)),
                dropout=max(0.0, min(1.0, _float(_required_value(native_lora, "dropout", "step5.ccv.native_lora", 0.05, allow_test_defaults=allow_test_defaults), 0.05))),
                target_modules=tuple(str(x).strip() for x in lora_targets if str(x).strip()),
            ),
        ),
        fca=Step5FCAConfig(
            enabled=_bool(_required_value(fca, "enabled", "step5.fca", True, allow_test_defaults=allow_test_defaults)),
            weight=max(0.0, _float(_required_value(fca, "weight", "step5.fca", 0.08, allow_test_defaults=allow_test_defaults), 0.08)),
            min_reliability=max(0.0, min(1.0, _float(_required_value(fca, "min_reliability", "step5.fca", 0.50, allow_test_defaults=allow_test_defaults), 0.50))),
            max_uncertainty=max(0.0, min(1.0, _float(_required_value(fca, "max_uncertainty", "step5.fca", 0.62, allow_test_defaults=allow_test_defaults), 0.62))),
            evidence_alignment_mode=str(_required_value(fca, "evidence_alignment_mode", "step5.fca", "evidence_basis", allow_test_defaults=allow_test_defaults)).strip().lower(),
        ),
    )


def _feature(batch: GatheredBatch, index: int, name: str) -> torch.Tensor:
    if batch.evidence_features is None:
        raise RuntimeError(f"Step5 batch missing evidence_features; cannot read {name}.")
    if int(batch.evidence_features.shape[-1]) < STEP5_EVIDENCE_FEATURE_DIM:
        raise RuntimeError(
            f"Step5 evidence_features must have {STEP5_EVIDENCE_FEATURE_DIM} fields; "
            f"got {int(batch.evidence_features.shape[-1])}."
        )
    return batch.evidence_features[:, index].view(-1)


def _required_vec(batch: GatheredBatch, attr: str) -> torch.Tensor:
    val = getattr(batch, attr)
    if val is None:
        raise RuntimeError(
            f"Step5 batch missing required posterior/control tensor: {attr}. "
            "This field must be produced by Step4 RCR export, converted by Step5 Processor, "
            "and preserved by collate; do not bypass the Step4 resolver or use a hand-written CSV."
        )
    return val.view(-1)


def _bucket_scale(confidence_bucket: torch.Tensor, weights: Mapping[str, float]) -> torch.Tensor:
    high = confidence_bucket >= 2.0
    medium = (confidence_bucket >= 1.0) & (~high)
    low_val = float(weights.get("low", 0.0))
    out = torch.full_like(confidence_bucket, low_val)
    out = torch.where(medium, out.new_tensor(float(weights.get("medium", 0.75))), out)
    out = torch.where(high, out.new_tensor(float(weights.get("high", 1.25))), out)
    return out


def build_rating_stability_control_gate(batch: GatheredBatch, cfg: Step5InnovationConfig) -> RatingStabilityControlGate:
    sample_weight = _required_vec(batch, "exp_sample_weight")
    route = _required_vec(batch, "route_scorer_mask").clamp(0.0, 1.0)
    uncertainty = _required_vec(batch, "uncertainty_score").clamp(0.0, 1.0)
    confidence = _required_vec(batch, "confidence_bucket")
    reliability = _feature(batch, CF_RELIABILITY, "cf_reliability_score").clamp(0.0, 1.0)
    rating_stability = _feature(batch, RATING_STABILITY, "rating_stability_score").clamp(0.0, 1.0)
    content_retention = _feature(batch, CONTENT_RETENTION, "content_retention_score").clamp(0.0, 1.0)
    uci_bucket = _bucket_scale(confidence.to(dtype=sample_weight.dtype), cfg.uci.bucket_weights)
    if not cfg.uci.enabled:
        uci_bucket = torch.ones_like(uci_bucket)
    uncertainty_scale = (1.0 - uncertainty).clamp(0.0, 1.0).pow(float(cfg.uci.uncertainty_temperature))
    posterior_strength = reliability * rating_stability * content_retention * uncertainty_scale
    scorer_weight = sample_weight * route * posterior_strength * uci_bucket
    eligible = (
        (route > 0.0)
        & (reliability >= float(cfg.lci.min_reliability))
        & (uncertainty <= float(cfg.lci.max_uncertainty))
    ).to(dtype=scorer_weight.dtype)
    lci_conf = _bucket_scale(confidence.to(dtype=sample_weight.dtype), cfg.lci.confidence_schedule)
    if not cfg.lci.enabled:
        lci_conf = torch.zeros_like(lci_conf)
    uci_weight = eligible * posterior_strength * lci_conf
    if float(cfg.uci.low_confidence_floor) > 0.0:
        floor = scorer_weight.new_tensor(float(cfg.uci.low_confidence_floor))
        uci_weight = torch.where((route > 0.0) & (confidence >= 1.0), uci_weight.clamp_min(floor), uci_weight)
    lci_weight = sample_weight * uci_weight
    return RatingStabilityControlGate(
        scorer_weight=scorer_weight,
        lci_weight=lci_weight,
        uci_weight=uci_weight,
        route_mask=route,
        reliability=reliability,
        uncertainty=uncertainty,
        confidence_bucket=confidence,
        rating_stability=rating_stability,
    )


def build_step5_explanation_gate(batch: GatheredBatch, cfg: Step5InnovationConfig) -> Step5ExplanationGate:
    sample_weight = _required_vec(batch, "exp_sample_weight")
    route = _required_vec(batch, "route_explainer_mask").clamp(0.0, 1.0)
    uncertainty = _required_vec(batch, "uncertainty_score").clamp(0.0, 1.0)
    confidence = _required_vec(batch, "confidence_bucket")
    reliability = _feature(batch, CF_RELIABILITY, "cf_reliability_score").clamp(0.0, 1.0)
    style_shift = _feature(batch, STYLE_SHIFT, "style_shift_score").clamp(0.0, 1.0)
    content_retention = _feature(batch, CONTENT_RETENTION, "content_retention_score").clamp(0.0, 1.0)
    gate_cfg = cfg.explainer_gate
    bucket = _bucket_scale(confidence.to(dtype=sample_weight.dtype), gate_cfg.bucket_weights)
    uncertainty_scale = (1.0 - uncertainty).clamp(0.0, 1.0).pow(float(gate_cfg.uncertainty_exponent))
    diversity_boost = 1.0 + float(gate_cfg.style_shift_diversity_boost) * style_shift
    unclamped_weight = sample_weight * reliability * uncertainty_scale * bucket * diversity_boost
    explainer_weight = torch.where(
        route > 0.0,
        unclamped_weight.clamp(float(gate_cfg.min_weight), float(gate_cfg.max_weight)),
        torch.zeros_like(unclamped_weight),
    )
    fca_route = (
        (route > 0.0)
        & (reliability >= float(cfg.fca.min_reliability))
        & (uncertainty <= float(cfg.fca.max_uncertainty))
    ).to(dtype=sample_weight.dtype)
    fca_weight = sample_weight * fca_route * reliability * content_retention * uncertainty_scale
    if not cfg.fca.enabled:
        fca_weight = torch.zeros_like(fca_weight)
    return Step5ExplanationGate(
        explainer_weight=explainer_weight,
        fca_weight=fca_weight,
        route_mask=route,
        reliability=reliability,
        uncertainty=uncertainty,
        confidence_bucket=confidence,
        style_shift=style_shift,
    )


def build_ccv_control_packet(
    batch: GatheredBatch,
    cfg: Step5InnovationConfig,
    *,
    producer: str = "build_ccv_control_packet",
    head: str = "unknown",
) -> CCVControlPacket:
    if not cfg.ccv.enabled:
        raise RuntimeError("CCV control packet requested while step5.ccv.enabled=false.")
    if cfg.ccv.control_packet_field_policy != "strict_required":
        raise RuntimeError(
            "Unsupported Step5 explanation CCV control packet policy "
            f"{cfg.ccv.control_packet_field_policy!r}; active path requires strict_required."
        )
    required = (
        "content_evidence_ids",
        "style_evidence_ids",
        "domain_style_anchor_ids",
        "local_style_hint_ids",
        "polarity_ids",
    )
    missing = [name for name in required if getattr(batch, name) is None]
    if missing:
        raise RuntimeError("Step5 explanation CCV control packet missing tensor fields: " + ", ".join(missing))
    packet = CCVControlPacket(
        content_evidence_ids=batch.content_evidence_ids,  # type: ignore[arg-type]
        style_evidence_ids=batch.style_evidence_ids,  # type: ignore[arg-type]
        domain_style_anchor_ids=batch.domain_style_anchor_ids,  # type: ignore[arg-type]
        local_style_hint_ids=batch.local_style_hint_ids,  # type: ignore[arg-type]
        polarity_ids=batch.polarity_ids,  # type: ignore[arg-type]
        route_scorer_mask=_required_vec(batch, "route_scorer_mask"),
        route_explainer_mask=_required_vec(batch, "route_explainer_mask"),
        sample_weight_hint=_required_vec(batch, "exp_sample_weight"),
        cf_reliability_score=_feature(batch, CF_RELIABILITY, "cf_reliability_score"),
        content_retention_score=_feature(batch, CONTENT_RETENTION, "content_retention_score"),
        style_shift_score=_feature(batch, STYLE_SHIFT, "style_shift_score"),
        rating_stability_score=_feature(batch, RATING_STABILITY, "rating_stability_score"),
        uncertainty_score=_required_vec(batch, "uncertainty_score"),
        confidence_bucket=_required_vec(batch, "confidence_bucket"),
        evidence_quality_prior=_feature(batch, EVIDENCE_QUALITY_PRIOR, "evidence_quality_prior"),
        content_anchor_score=_required_vec(batch, "content_anchor_score"),
        style_anchor_score=_required_vec(batch, "style_anchor_score"),
    )
    validate_ccv_control_packet_shapes(packet, producer=producer, head=head, strict=True)
    return packet


def lci_score_invariance_loss(
    *,
    factual_score: torch.Tensor,
    cf_score: torch.Tensor,
    robust_score: torch.Tensor,
    target_rating: torch.Tensor,
    gate: RatingStabilityControlGate,
    cfg: Step5InnovationConfig,
) -> RatingStabilityControlLCILoss:
    if not cfg.lci.enabled or cfg.lci.weight <= 0.0:
        z = factual_score.sum() * 0.0
        return RatingStabilityControlLCILoss(z, z, z, z, z, z, gate.scorer_weight.mean())
    w = gate.lci_weight.to(dtype=factual_score.dtype)
    consistency_ps = (factual_score - cf_score).pow(2)
    cf_score_ps = (cf_score - target_rating.to(dtype=cf_score.dtype)).pow(2)
    robust_ps = (cf_score - robust_score).pow(2)
    l_cons = route_weighted_mean(consistency_ps, w, gate.route_mask)
    l_cf = route_weighted_mean(cf_score_ps, w, gate.route_mask)
    l_rob = route_weighted_mean(robust_ps, w, gate.route_mask)
    raw = l_cons + float(cfg.lci.counterfactual_label_weight) * l_cf + float(cfg.lci.robustness_weight) * l_rob
    weighted = float(cfg.lci.weight) * raw
    return RatingStabilityControlLCILoss(
        lci_loss=raw,
        lci_weighted_loss=weighted,
        lci_consistency_loss=l_cons,
        lci_cf_score_loss=l_cf,
        lci_robustness_loss=l_rob,
        uci_weight_mean=gate.uci_weight.detach().mean(),
        scorer_weight_mean=gate.scorer_weight.detach().mean(),
    )


def evidence_basis_fca_loss(
    *,
    scorer_hidden: torch.Tensor,
    explainer_hidden: torch.Tensor,
    shared_latent: torch.Tensor,
    content_profile: torch.Tensor,
    content_evidence_latent: torch.Tensor,
    packet: CCVControlPacket,
    gate: Step5ExplanationGate,
    cfg: Step5InnovationConfig,
) -> Step5ExplanationFCALoss:
    if not cfg.fca.enabled or cfg.fca.weight <= 0.0:
        z = scorer_hidden.sum() * 0.0
        return Step5ExplanationFCALoss(z, z, z, scorer_hidden, explainer_hidden)
    eps = 1e-8
    c_ret = packet.content_retention_score.to(dtype=scorer_hidden.dtype).view(-1, 1).clamp(0.0, 1.0)
    c_anchor = packet.content_anchor_score.to(dtype=scorer_hidden.dtype).view(-1, 1).clamp(0.0, 1.0)
    r_stab = packet.rating_stability_score.to(dtype=scorer_hidden.dtype).view(-1, 1).clamp(0.0, 1.0)
    scorer_basis = (
        F.normalize(scorer_hidden, dim=-1, eps=eps) * c_ret
        + F.normalize(shared_latent, dim=-1, eps=eps) * r_stab
        + F.normalize(content_profile, dim=-1, eps=eps) * c_anchor
    )
    explainer_basis = (
        F.normalize(explainer_hidden, dim=-1, eps=eps) * c_ret
        + F.normalize(content_evidence_latent, dim=-1, eps=eps) * c_anchor
    )
    score_n = F.normalize(scorer_basis, dim=-1, eps=eps)
    explain_n = F.normalize(explainer_basis, dim=-1, eps=eps)
    one_minus = 1.0 - (score_n * explain_n).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    raw = route_weighted_mean(one_minus, gate.fca_weight.to(dtype=one_minus.dtype), gate.route_mask)
    weighted = float(cfg.fca.weight) * raw
    return Step5ExplanationFCALoss(
        fca_loss=raw,
        fca_weighted_loss=weighted,
        fca_weight_mean=gate.fca_weight.detach().mean(),
        scorer_evidence_basis=scorer_basis,
        explainer_evidence_basis=explainer_basis,
    )


__all__ = [
    "CCVControlPacket",
    "CF_RELIABILITY",
    "CONTENT_RETENTION",
    "EVIDENCE_QUALITY_PRIOR",
    "RATING_POLARITY",
    "RATING_STABILITY",
    "STEP5_EVIDENCE_FEATURE_DIM",
    "STYLE_SHIFT",
    "TEXT_QUALITY",
    "UNCERTAINTY",
    "Step5InnovationConfig",
    "Step5ExplainerGateConfig",
    "Step5NativeLoRAConfig",
    "build_ccv_control_packet",
    "build_rating_stability_control_gate",
    "build_step5_explanation_gate",
    "evidence_basis_fca_loss",
    "for_test_default_step5_innovation_config",
    "lci_score_invariance_loss",
    "parse_step5_innovation_config_json",
    "validate_ccv_control_packet_shapes",
]
