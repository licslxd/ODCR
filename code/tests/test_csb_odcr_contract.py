from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors.step3_train_core import Model  # noqa: E402
from odcr_core.config_resolver import (  # noqa: E402
    _resolve_step5_innovation_config,
    load_yaml_config,
    resolve_config,
)
import odcr_core.config_resolver as config_resolver_module  # noqa: E402
from odcr_core.csb_contract import (  # noqa: E402
    CSB_REQUIRED_TENSOR_FIELDS,
    apply_csb_conflict_routing_weights,
    csb_contract_hash,
    default_csb_contract_payload,
    validate_csb_forward_output_schema,
    validate_csb_packet,
)
from odcr_core.index_contract import (  # noqa: E402
    build_step4_export_lineage,
    validate_csb_step4_gate,
)
from odcr_core.stage_truth_antiforgery import write_step3_fixture  # noqa: E402


def _build_model() -> Model:
    torch.manual_seed(11)
    nuser, nitem, ntoken, d = 8, 10, 40, 16
    model = Model(
        nuser=nuser,
        nitem=nitem,
        ntoken=ntoken,
        emsize=d,
        nhead=2,
        nhid=32,
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
    return model


class CSBODCRContractTest(unittest.TestCase):
    def test_resolver_exposes_csb_method_profile_and_contract(self) -> None:
        cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=["experiment_profile=csb_odcr_sidecar_stable"],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        payload = json.loads(cfg.effective_training_payload_json)
        self.assertEqual(snapshot["method"]["method_name"], "CSB-ODCR")
        self.assertEqual(payload["method"]["method_family"], "csb_odcr")
        self.assertEqual(payload["experiment_profile"]["name"], "csb_odcr_sidecar_stable")
        self.assertEqual(
            tuple(payload["step3_csb_odcr"]["contract"]["required_tensor_fields"]),
            CSB_REQUIRED_TENSOR_FIELDS,
        )
        self.assertFalse(payload["step3_csb_odcr"]["controlled_injection"]["enabled"])
        self.assertFalse(payload["step3_csb_odcr"]["conflict_routing"]["enabled"])

    def test_ablation_profiles_disable_only_their_owned_surface(self) -> None:
        _cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=["experiment_profile=csb_odcr_sidecar_stable"],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        csb = snapshot["step3_csb_odcr"]
        self.assertTrue(csb["enabled"])
        self.assertFalse(csb["controlled_injection"]["enabled"])
        self.assertFalse(csb["conflict_routing"]["enabled"])

    def test_forward_output_carries_stable_csb_packet(self) -> None:
        model = _build_model()
        out = model(
            torch.randint(0, 8, (3,)),
            torch.randint(0, 10, (3,)),
            torch.randint(1, 40, (3, 5)),
            torch.randint(0, 2, (3,)),
            content_anchor=torch.rand(3),
            style_anchor=torch.rand(3),
            content_evidence_ids=torch.randint(0, 40, (3, 6)),
            style_evidence_ids=torch.randint(0, 40, (3, 6)),
            domain_style_anchor_ids=torch.randint(0, 40, (3, 6)),
            local_style_hint_ids=torch.randint(0, 40, (3, 6)),
            polarity_ids=torch.randint(0, 3, (3,)),
            evidence_quality_prior=torch.rand(3),
        )
        validate_csb_forward_output_schema(out)
        validate_csb_packet(out.csb_packet)
        self.assertEqual(tuple(out.z_content.shape), (3, 16))
        self.assertEqual(out.diagnostics["primary_path"], "rating_only_primary_scorer")
        self.assertEqual(out.diagnostics["structure_branch"], "detached_csb_sidecar_z_content_z_style_z_domain_z_uncertainty")

    def test_conflict_routing_caps_auxiliary_losses_without_touching_anchors(self) -> None:
        weights = {"L_rating_shared": 1.0, "L_light_explainer": 1.0, "L_content_alignment": 0.8}
        routed, summary = apply_csb_conflict_routing_weights(
            weights,
            {
                "enabled": True,
                "mode": "rating_anchor_projection",
                "aux_soft_cap": 0.5,
                "dynamic_downweight": True,
            },
        )
        self.assertEqual(routed["L_rating_shared"], 1.0)
        self.assertEqual(routed["L_light_explainer"], 1.0)
        self.assertEqual(routed["L_content_alignment"], 0.4)
        self.assertEqual(summary["rating_anchor"], "L_rating_shared")
        self.assertEqual(summary["explanation_anchor"], "L_light_explainer")

    def test_step4_and_step5_csb_contract_gates_are_resolvable(self) -> None:
        contract = default_csb_contract_payload()
        lineage = build_step4_export_lineage(
            task_id=2,
            auxiliary_domain="AM_Movies",
            target_domain="AM_CDs",
            step3_checkpoint_lineage_hash="lineage",
            step4_rcr_config={"route": "rcr"},
            step4_run="1_1",
            csb_contract=contract,
        )
        gate = validate_csb_step4_gate({"step4_export_lineage": lineage})
        self.assertEqual(gate["status"], "pass")
        cfg = load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        step5 = _resolve_step5_innovation_config(cfg)
        packet = step5["ccv"]["csb_control_packet"]
        self.assertTrue(packet["required"])
        self.assertEqual(tuple(packet["scorer_clean_fields"]), ("z_content", "z_uncertainty"))
        self.assertEqual(tuple(packet["explainer_rich_fields"]), CSB_REQUIRED_TENSOR_FIELDS)

    def test_step4_resolver_accepts_csb_step3_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="1", active=True, eligible=True)
            old_root = config_resolver_module._REPO_ROOT
            config_resolver_module._REPO_ROOT = repo
            try:
                cfg, _sources, snapshot = resolve_config(
                    config_path=REPO_ROOT / "configs" / "odcr.yaml",
                    command="step4",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    run_id="auto",
                    mode="full",
                )
            finally:
                config_resolver_module._REPO_ROOT = old_root
        upstream = json.loads(cfg.upstream_resolution_json)
        self.assertEqual(upstream["stage_status"]["method_name"], "CSB-ODCR")
        self.assertTrue(upstream["stage_status"]["csb_contract_hash"])
        self.assertTrue(upstream["eligible"])
        self.assertEqual(snapshot["method"]["method_name"], "CSB-ODCR")


if __name__ == "__main__":
    unittest.main()
