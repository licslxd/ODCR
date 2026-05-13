from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


CSB_ODCR_METHOD_NAME = "CSB-ODCR"
CSB_ODCR_METHOD_FULL_NAME = (
    "CSB-ODCR: Causal Structure Bottleneck for Orthogonal Disentangled "
    "Counterfactual Recommendation"
)
CSB_ODCR_METHOD_FAMILY = "csb_odcr"
CSB_METHOD_SCHEMA_VERSION = "csb_odcr_method/1"
CSB_PACKET_SCHEMA_VERSION = "csb_odcr_csb_packet/1"
CSB_FORWARD_OUTPUT_SCHEMA_VERSION = "csb_odcr_step3_forward_output/1"
CSB_CROSS_STAGE_CONTRACT_VERSION = "csb_odcr_cross_stage_contract/1"
CSB_CONFLICT_ROUTING_SCHEMA_VERSION = "csb_odcr_conflict_routing/1"

CSB_REQUIRED_TENSOR_FIELDS = ("z_content", "z_style", "z_domain", "z_uncertainty")
CSB_PACKET_REQUIRED_FIELDS = (
    "schema_version",
    "method_name",
    "tensor_fields",
    "contract_hash",
)
CSB_AUXILIARY_LOSS_GROUPS = ("easd_content", "hss_style", "disentangle_geometry")
CSB_AUXILIARY_COMPONENTS = frozenset(
    {
        "L_orthogonal",
        "L_variance",
        "L_shared_invariance",
        "L_specific_separation",
        "L_anchor_content",
        "L_anchor_style",
        "L_content_alignment",
        "L_style_alignment",
        "L_shared_proto",
        "L_domain_style_alignment",
        "L_local_style_alignment",
        "L_polarity_alignment",
        "L_residual_specific",
        "L_prototype_separation",
    }
)


def stable_csb_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def default_csb_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": CSB_PACKET_SCHEMA_VERSION,
        "cross_stage_contract_version": CSB_CROSS_STAGE_CONTRACT_VERSION,
        "method_name": CSB_ODCR_METHOD_NAME,
        "method_family": CSB_ODCR_METHOD_FAMILY,
        "required_tensor_fields": list(CSB_REQUIRED_TENSOR_FIELDS),
        "semantics": {
            "z_content": "rating-safe content basis and scorer-clean content signal",
            "z_style": "expression and explanation style basis",
            "z_domain": "domain-shift and counterfactual-domain variation",
            "z_uncertainty": "reliability, uncertainty, and routing confidence basis",
        },
        "training_signal_roles": {
            "EASD": "CSB z_content training signal",
            "HSS": "CSB z_style/z_domain training signal",
            "geometry": "CSB separation and stability regularizer",
        },
        "primary_path_boundary": {
            "rating_path": ["shared", "z_content", "z_uncertainty"],
            "style_domain_default_excluded_from_rating": ["z_style", "z_domain"],
            "auxiliary_losses_scope": "csb_branch_first",
            "primary_influence": "controlled_injection_or_rating_anchor_routing_only",
        },
    }


def csb_contract_hash(payload: Mapping[str, Any] | None = None) -> str:
    clean = dict(payload or default_csb_contract_payload())
    clean.pop("contract_hash", None)
    return stable_csb_hash(clean)


def method_payload() -> dict[str, Any]:
    payload = {
        "schema_version": CSB_METHOD_SCHEMA_VERSION,
        "method_name": CSB_ODCR_METHOD_NAME,
        "method_full_name": CSB_ODCR_METHOD_FULL_NAME,
        "method_family": CSB_ODCR_METHOD_FAMILY,
        "csb_packet_schema_version": CSB_PACKET_SCHEMA_VERSION,
        "cross_stage_contract_version": CSB_CROSS_STAGE_CONTRACT_VERSION,
    }
    payload["method_schema_hash"] = stable_csb_hash(payload)
    return payload


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(x) for x in shape]
    except Exception:
        return None


def build_csb_packet_summary(
    *,
    z_content: Any,
    z_style: Any,
    z_domain: Any,
    z_uncertainty: Any,
    diagnostics: Mapping[str, Any] | None = None,
    contract_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(contract_payload or default_csb_contract_payload())
    contract = csb_contract_hash(payload)
    tensor_shapes = {
        "z_content": _shape(z_content),
        "z_style": _shape(z_style),
        "z_domain": _shape(z_domain),
        "z_uncertainty": _shape(z_uncertainty),
    }
    return {
        "schema_version": CSB_PACKET_SCHEMA_VERSION,
        "method_name": CSB_ODCR_METHOD_NAME,
        "method_family": CSB_ODCR_METHOD_FAMILY,
        "tensor_fields": list(CSB_REQUIRED_TENSOR_FIELDS),
        "tensor_shapes": tensor_shapes,
        "contract_hash": contract,
        "cross_stage_contract_version": CSB_CROSS_STAGE_CONTRACT_VERSION,
        "diagnostics": dict(diagnostics or {}),
    }


def validate_csb_packet(packet: Mapping[str, Any], *, require_hash: bool = True) -> dict[str, Any]:
    missing = [key for key in CSB_PACKET_REQUIRED_FIELDS if key not in packet]
    tensors = packet.get("tensor_fields") if isinstance(packet.get("tensor_fields"), list) else []
    missing_tensors = [key for key in CSB_REQUIRED_TENSOR_FIELDS if key not in tensors]
    if require_hash and not str(packet.get("contract_hash") or "").strip():
        missing.append("contract_hash")
    status = "pass" if not missing and not missing_tensors else "fail"
    out = {
        "schema_version": "csb_odcr_packet_validation/1",
        "status": status,
        "method_name": str(packet.get("method_name") or ""),
        "missing_packet_fields": missing,
        "missing_tensor_fields": missing_tensors,
        "contract_hash": str(packet.get("contract_hash") or ""),
    }
    if status != "pass":
        raise ValueError(f"invalid CSB packet: {out}")
    return out


def validate_csb_forward_output_schema(output: Any) -> dict[str, Any]:
    missing = [name for name in CSB_REQUIRED_TENSOR_FIELDS if not hasattr(output, name)]
    packet = getattr(output, "csb_packet", None)
    if not isinstance(packet, Mapping):
        missing.append("csb_packet")
    diagnostics = getattr(output, "csb_diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        missing.append("csb_diagnostics")
    version = str(getattr(output, "csb_schema_version", "") or "")
    contract = str(getattr(output, "csb_contract_hash", "") or "")
    if version != CSB_FORWARD_OUTPUT_SCHEMA_VERSION:
        missing.append("csb_schema_version")
    if not contract:
        missing.append("csb_contract_hash")
    if packet is not None and isinstance(packet, Mapping):
        validate_csb_packet(packet)
    status = "pass" if not missing else "fail"
    out = {
        "schema_version": "csb_odcr_forward_output_validation/1",
        "status": status,
        "required_tensor_fields": list(CSB_REQUIRED_TENSOR_FIELDS),
        "missing": missing,
        "csb_schema_version": version,
        "csb_contract_hash": contract,
    }
    if status != "pass":
        raise ValueError(f"invalid CSB forward output: {out}")
    return out


def apply_csb_conflict_routing_weights(
    component_weights: Mapping[str, float],
    config: Mapping[str, Any] | None,
    *,
    csb_enabled: bool = True,
    controlled_injection_enabled: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    cfg = dict(config or {})
    routing = cfg.get("conflict_routing") if isinstance(cfg.get("conflict_routing"), Mapping) else cfg
    enabled = bool(routing.get("enabled", False)) and bool(csb_enabled)
    aux_cap = float(routing.get("aux_soft_cap", 1.0) or 1.0)
    dynamic = bool(routing.get("dynamic_downweight", False))
    out = {str(key): float(value) for key, value in component_weights.items()}
    routed: dict[str, dict[str, Any]] = {}
    if not csb_enabled:
        for key in list(out):
            if key in CSB_AUXILIARY_COMPONENTS:
                routed[key] = {"before": out[key], "after": 0.0, "reason": "csb_disabled_ablation"}
                out[key] = 0.0
    elif enabled and dynamic:
        cap = max(0.0, min(1.0, aux_cap))
        for key in list(out):
            if key in CSB_AUXILIARY_COMPONENTS:
                before = out[key]
                out[key] = min(before, before * cap)
                routed[key] = {
                    "before": before,
                    "after": out[key],
                    "reason": "rating_anchor_aux_soft_cap",
                }
    if not controlled_injection_enabled:
        routed["controlled_injection"] = {
            "before": 1.0,
            "after": 0.0,
            "reason": "controlled_injection_disabled_ablation",
        }
    summary = {
        "schema_version": CSB_CONFLICT_ROUTING_SCHEMA_VERSION,
        "enabled": enabled,
        "mode": str(routing.get("mode") or "off"),
        "rating_anchor": str(routing.get("rating_anchor") or "L_rating_shared"),
        "explanation_anchor": str(routing.get("explanation_anchor") or "L_light_explainer"),
        "diversity_guard": routing.get("diversity_guard") or ["DIST-1", "DIST-2"],
        "aux_soft_cap": aux_cap,
        "dynamic_downweight": dynamic,
        "csb_enabled": bool(csb_enabled),
        "controlled_injection_enabled": bool(controlled_injection_enabled),
        "routed_components": routed,
        "primary_path_policy": "rating_anchor_protected_csb_branch_first",
    }
    return out, summary


def require_csb_contract_for_stage(payload: Mapping[str, Any], *, consumer_stage: str) -> dict[str, Any]:
    contract = payload.get("csb_contract") if isinstance(payload.get("csb_contract"), Mapping) else {}
    packet = payload.get("csb_packet") if isinstance(payload.get("csb_packet"), Mapping) else {}
    packet_hash = str(packet.get("contract_hash") or contract.get("contract_hash") or payload.get("csb_contract_hash") or "")
    if not packet_hash:
        raise ValueError(f"{consumer_stage} refused upstream without CSB contract hash.")
    fields = contract.get("required_tensor_fields") or packet.get("tensor_fields") or []
    missing = [field for field in CSB_REQUIRED_TENSOR_FIELDS if field not in fields]
    if missing:
        raise ValueError(f"{consumer_stage} refused upstream with incomplete CSB fields: {missing}")
    return {
        "schema_version": "csb_odcr_cross_stage_gate/1",
        "consumer_stage": str(consumer_stage),
        "status": "pass",
        "contract_hash": packet_hash,
        "required_tensor_fields": list(CSB_REQUIRED_TENSOR_FIELDS),
    }
