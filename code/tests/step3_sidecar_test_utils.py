from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent

from executors.step3_train_core import Model  # noqa: E402
from odcr_core.csb_contract import csb_contract_hash, default_csb_contract_payload  # noqa: E402


class Cfg:
    contract = default_csb_contract_payload()
    contract["contract_hash"] = csb_contract_hash(contract)
    step3_structured_loss_weights_json = json.dumps(
        {
            "orthogonal": {"weight": 0.20, "xcov_weight": 1.0, "cosine_weight": 0.25},
            "variance_weight": 0.10,
            "shared_invariance_weight": 0.18,
            "specific_separation_weight": 0.16,
            "anchor_alignment_weight": 0.08,
            "content_alignment_weight": 0.12,
            "style_alignment_weight": 0.12,
            "shared_prototype_weight": 0.08,
            "domain_style_alignment_weight": 0.06,
            "local_style_alignment_weight": 0.06,
            "polarity_alignment_weight": 0.05,
            "residual_specific_weight": 0.025,
            "prototype_separation_weight": 0.04,
            "light_explainer_weight": 0.0,
        },
        sort_keys=True,
    )
    step3_loss_semantics_json = json.dumps(
        {
            "specific_separation_margin": 0.6,
            "variance_target_std": 0.7,
            "variance_eps": 1e-4,
            "orthogonal_eps": 1e-8,
            "cosine_eps": 1e-8,
            "sample_weight_eps": 1e-6,
            "prototype_separation_eps": 1e-8,
            "quality_weight": {
                "evidence_base": 0.55,
                "evidence_scale": 0.45,
                "anchor_base": 0.40,
                "anchor_scale": 0.60,
            },
        },
        sort_keys=True,
    )
    csb_odcr_config_json = json.dumps(
        {
            "enabled": True,
            "primary_training": "rating_only",
            "csb_mode": "sidecar",
            "gradient_firewall": True,
            "controlled_injection_formal_train": False,
            "light_explainer_step3_loss": False,
            "conflict_routing_step3_primary": False,
            "csb_rating_safe_adapter_train": False,
            "paper_metric_gate": False,
            "contract": contract,
            "controlled_injection": {"enabled": False, "gate_init": 0.35, "rating_safe_injection": False},
            "conflict_routing": {"enabled": False, "mode": "csb_branch_only"},
        },
        sort_keys=True,
    )
    cross_rank_structured_gather_config_json = "{}"


def build_model(*, apply_runtime: bool = False) -> Model:
    torch.manual_seed(19)
    nuser, nitem, ntoken, d = 16, 20, 100, 32
    model = Model(
        nuser=nuser,
        nitem=nitem,
        ntoken=ntoken,
        emsize=d,
        nhead=2,
        nhid=64,
        nlayers=1,
        dropout=0.0,
        user_content_profiles=torch.randn(nuser, d),
        user_style_profiles=torch.randn(nuser, d),
        item_content_profiles=torch.randn(nitem, d),
        item_style_profiles=torch.randn(nitem, d),
        domain_content_profiles=torch.randn(2, d),
        domain_style_profiles=torch.randn(2, d),
    )
    contract = default_csb_contract_payload()
    contract["contract_hash"] = csb_contract_hash(contract)
    model.csb_odcr_bottleneck.set_csb_contract_payload(contract)
    if apply_runtime:
        runtime_cfg = SimpleNamespace(
            decode_strategy="greedy",
            generate_temperature=1.0,
            generate_top_p=1.0,
            repetition_penalty=1.0,
            max_explanation_length=8,
            evidence_max_length=24,
            csb_odcr_config_json=Cfg.csb_odcr_config_json,
        )
        model.apply_runtime_config(runtime_cfg, SimpleNamespace(eos_token_id=2))
    return model


def forward_batch(model: Model) -> tuple[Any, Any]:
    torch.manual_seed(23)
    bsz, seq_len = 8, 12
    out = model(
        torch.randint(0, 16, (bsz,)),
        torch.randint(0, 20, (bsz,)),
        torch.randint(1, 100, (bsz, seq_len)),
        torch.randint(0, 2, (bsz,)),
        content_anchor=torch.rand(bsz),
        style_anchor=torch.rand(bsz),
        content_evidence_ids=torch.randint(0, 100, (bsz, 24)),
        style_evidence_ids=torch.randint(0, 100, (bsz, 24)),
        domain_style_anchor_ids=torch.randint(0, 100, (bsz, 24)),
        local_style_hint_ids=torch.randint(0, 100, (bsz, 24)),
        polarity_ids=torch.randint(0, 3, (bsz,)),
        evidence_quality_prior=torch.rand(bsz),
    )
    batch = SimpleNamespace(
        rating=torch.randn(bsz),
        tgt_output=torch.randint(1, 100, (bsz, seq_len)),
        domain_idx=torch.randint(0, 2, (bsz,)),
        content_anchor_score=torch.rand(bsz),
        style_anchor_score=torch.rand(bsz),
        evidence_quality_prior=torch.rand(bsz),
    )
    return out, batch

