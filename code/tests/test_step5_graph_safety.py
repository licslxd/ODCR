from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = Path(_CODE_DIR).parent
sys.path.insert(0, _CODE_DIR)

import odcr_core.config_resolver as config_resolver  # noqa: E402
from odcr_core.config_resolver import OneControlConfigError, resolve_config  # noqa: E402
from odcr_core.gather_schema import GatheredBatch  # noqa: E402
from odcr_core.index_contract import (  # noqa: E402
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    ODCR_ROUTING_TRAIN_CSV,
    STEP4_RCR_REQUIRED_COLUMNS,
    build_step4_export_lineage,
)
from odcr_core.csb_contract import default_csb_contract_payload  # noqa: E402
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST  # noqa: E402
from odcr_core.step5_innovation import (  # noqa: E402
    STEP5_EVIDENCE_FEATURE_DIM,
    build_ccv_control_packet,
    build_step5a_scorer_gate,
    build_step5b_explainer_gate,
    evidence_basis_fca_loss,
    for_test_default_step5_innovation_config,
    lci_score_invariance_loss,
)
from odcr_core.step5_word_losses import route_weighted_mean  # noqa: E402
from executors.step5_engine import compose_step5_total_loss  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_step4_upstream_fixture(repo: Path, *, task_id: int = 4, run_id: str = "1") -> None:
    run = repo / "runs" / "step4" / f"task{task_id}" / run_id
    meta = run / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    export_name = ODCR_ROUTING_TRAIN_CSV
    export = run / export_name
    row = {col: 1 for col in STEP4_RCR_REQUIRED_COLUMNS}
    row.update(
        {
            "route_reason_scorer": "rcr_scorer_clean",
            "route_reason_explainer": "rcr_explainer_rich",
            "confidence_bucket": 2,
            "preprocess_route_scorer_prior": 0,
            "preprocess_route_explainer_prior": 0,
        }
    )
    headers = list(STEP4_RCR_REQUIRED_COLUMNS)
    export.write_text(
        ",".join(headers) + "\n" + ",".join(str(row[col]) for col in headers) + "\n",
        encoding="utf-8",
    )
    lineage = build_step4_export_lineage(
        task_id=task_id,
        auxiliary_domain="A",
        target_domain="T",
        step3_checkpoint_lineage_hash="lineage",
        step4_rcr_config={"fixture": True},
        step4_run=run_id,
        frozen_step3_lineage={
            "upstream_step3_run_id": "2",
            "step3_checkpoint_path": f"runs/step3/task{task_id}/2/model/best_observed.pth",
            "step3_checkpoint_hash": "fixture_checkpoint_hash",
            "step3_stage_status_hash": "fixture_stage_status_hash",
            "step3_readiness_audit_hash": "fixture_readiness_audit_hash",
        },
        csb_contract=default_csb_contract_payload(),
    )
    _write_json(
        run / INDEX_CONTRACT_FILENAME,
        {
            "schema_version": INDEX_CONTRACT_SCHEMA_VERSION,
            "embed_dim": 1024,
            "backbones": {
                "sentence_embed": {
                    "model_id": "fixture",
                    "local_dir": "/tmp/fixture",
                    "family": "bge_large_en",
                    "hidden_size": 1024,
                    "dual_channel": True,
                }
            },
            "step4_export_lineage": lineage,
        },
    )
    _write_json(run / STEP4_EXPORT_MANIFEST, {"schema_version": "odcr_step4_train_table/1.2", "step4_export_lineage": lineage})
    _write_json(meta / "source_table.json", {"records": []})
    _write_json(meta / "resolved_config.json", {"task": {"id": task_id}})
    _write_json(meta / "run_summary.json", {"run_id": run_id, "stage": "step4", "task_id": task_id, "status": "ok"})
    _write_json(
        meta / "stage_status.json",
        {
            "schema_version": "odcr_stage_status/1",
            "stage": "step4",
            "task": task_id,
            "task_id": task_id,
            "run_id": run_id,
            "run_dir": f"runs/step4/task{task_id}/{run_id}",
            "final_status": "completed",
            "downstream_ready": True,
            "ready_for": ["step5"],
            "status_source": "test_fixture",
            "rejection_reasons": [],
            "selected_export": f"runs/step4/task{task_id}/{run_id}/{export_name}",
            "export_manifest": f"runs/step4/task{task_id}/{run_id}/{STEP4_EXPORT_MANIFEST}",
            "index_contract": f"runs/step4/task{task_id}/{run_id}/{INDEX_CONTRACT_FILENAME}",
            "artifacts": {
                "run_summary": {
                    "path": f"runs/step4/task{task_id}/{run_id}/meta/run_summary.json",
                    "exists": True,
                    "is_file": True,
                },
                "selected_export": {
                    "path": f"runs/step4/task{task_id}/{run_id}/{export_name}",
                    "exists": True,
                    "is_file": True,
                },
                "export_manifest": {
                    "path": f"runs/step4/task{task_id}/{run_id}/{STEP4_EXPORT_MANIFEST}",
                    "exists": True,
                    "is_file": True,
                },
                "index_contract": {
                    "path": f"runs/step4/task{task_id}/{run_id}/{INDEX_CONTRACT_FILENAME}",
                    "exists": True,
                    "is_file": True,
                },
            },
        },
    )
    _write_json(
        repo / "runs" / "step4" / f"task{task_id}" / "latest.json",
        {
            "latest_run_id": run_id,
            "latest_run_dir": f"runs/step4/task{task_id}/{run_id}",
            "latest_summary_path": f"runs/step4/task{task_id}/{run_id}/meta/run_summary.json",
            "latest_status": "ok",
        },
    )


def _batch(route_scorer: torch.Tensor, route_explainer: torch.Tensor, domain: torch.Tensor) -> GatheredBatch:
    bsz = int(route_scorer.numel())
    ev = torch.zeros(bsz, STEP5_EVIDENCE_FEATURE_DIM)
    ev[:, 0] = torch.tensor([0.7, 0.8, 0.6, 0.9])[:bsz]
    ev[:, 1] = torch.tensor([0.95, 0.85, 0.75, 0.90])[:bsz]
    ev[:, 2] = torch.tensor([0.2, 0.7, 0.3, 0.8])[:bsz]
    ev[:, 3] = torch.tensor([0.95, 0.85, 0.70, 0.90])[:bsz]
    ev[:, 4] = torch.tensor([0.95, 0.80, 0.70, 0.90])[:bsz]
    ev[:, 5] = torch.ones(bsz)
    ev[:, 6] = torch.tensor([0.05, 0.15, 0.35, 0.10])[:bsz]
    ev[:, 7] = torch.ones(bsz)
    ids = torch.arange(1, bsz * 3 + 1, dtype=torch.long).view(bsz, 3)
    return GatheredBatch(
        user_idx=torch.arange(bsz),
        item_idx=torch.arange(bsz),
        rating=torch.tensor([4.0, 3.0, 2.0, 5.0])[:bsz],
        tgt_input=ids,
        tgt_output=ids,
        domain_idx=domain.long(),
        sample_id=torch.arange(bsz),
        exp_sample_weight=torch.ones(bsz),
        route_scorer_mask=route_scorer.float(),
        route_explainer_mask=route_explainer.float(),
        uncertainty_score=ev[:, 6],
        confidence_bucket=torch.tensor([2.0, 1.0, 0.0, 2.0])[:bsz],
        content_anchor_score=torch.tensor([0.9, 0.8, 0.6, 0.85])[:bsz],
        style_anchor_score=torch.tensor([0.2, 0.7, 0.3, 0.8])[:bsz],
        evidence_features=ev,
        content_evidence_ids=ids,
        style_evidence_ids=ids,
        domain_style_anchor_ids=ids,
        local_style_hint_ids=ids,
        polarity_ids=torch.tensor([2, 1, 0, 2])[:bsz],
    )


class TestStep5GraphSafety(unittest.TestCase):
    def _run_route_case(self, route_scorer: list[float], route_explainer: list[float], domain: list[int]) -> None:
        cfg = for_test_default_step5_innovation_config()
        batch = _batch(torch.tensor(route_scorer), torch.tensor(route_explainer), torch.tensor(domain))
        gate_a = build_step5a_scorer_gate(batch, cfg)
        gate_b = build_step5b_explainer_gate(batch, cfg)
        packet = build_ccv_control_packet(batch, cfg)
        bsz = len(route_scorer)
        factual = torch.linspace(2.0, 5.0, bsz, requires_grad=True)
        cf_score = factual + 0.1
        robust = factual - 0.1
        scorer_h = torch.randn(bsz, 5, requires_grad=True)
        explainer_h = torch.randn(bsz, 5, requires_grad=True)
        shared = torch.randn(bsz, 5, requires_grad=True)
        content_profile = torch.randn(bsz, 5, requires_grad=True)
        content_evidence = torch.randn(bsz, 5, requires_grad=True)

        lci = lci_score_invariance_loss(
            factual_score=factual,
            cf_score=cf_score,
            robust_score=robust,
            target_rating=batch.rating,
            gate=gate_a,
            cfg=cfg,
        )
        fca = evidence_basis_fca_loss(
            scorer_hidden=scorer_h,
            explainer_hidden=explainer_h,
            shared_latent=shared,
            content_profile=content_profile,
            content_evidence_latent=content_evidence,
            packet=packet,
            gate=gate_b,
            cfg=cfg,
        )
        dom = batch.domain_idx.view(-1)
        loss_factual = route_weighted_mean((factual - batch.rating).pow(2), gate_a.scorer_weight, dom == 1)
        loss_counterfactual = route_weighted_mean(factual.pow(2), gate_b.explainer_weight, dom == 0)
        total = compose_step5_total_loss(
            loss_factual=loss_factual,
            loss_counterfactual=loss_counterfactual,
            loss_repeat_ul=factual.sum() * 0.0,
            loss_terminal_clean=factual.sum() * 0.0,
            loss_batch_diversity=factual.sum() * 0.0,
            repeat_ul_weight=0.0,
            terminal_clean_weight=0.0,
            batch_diversity_weight=0.0,
            lci_weighted_loss=lci.lci_weighted_loss,
            fca_weighted_loss=fca.fca_weighted_loss,
            ortho_keep_loss=factual.sum() * 0.0,
            ortho_keep_weight=0.0,
        )
        self.assertTrue(total.requires_grad)
        total.backward()
        self.assertIsNotNone(factual.grad)
        self.assertIsNotNone(scorer_h.grad)
        self.assertIsNotNone(explainer_h.grad)

    def test_empty_route_variants_keep_graph(self) -> None:
        cases = {
            "all_scorer": ([1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]),
            "all_explainer": ([0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]),
            "neither_route": ([0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0]),
            "mixed_route": ([1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0]),
        }
        for name, (rs, re_, dom) in cases.items():
            with self.subTest(name=name):
                self._run_route_case(rs, re_, dom)

    def test_route_weighted_mean_zero_denominator_is_safe(self) -> None:
        values = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        out = route_weighted_mean(values, torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.float32))
        self.assertEqual(out.dtype, values.dtype)
        self.assertEqual(out.device, values.device)
        self.assertTrue(out.requires_grad)
        self.assertEqual(float(out.detach()), 0.0)
        out.backward()
        self.assertIsNotNone(values.grad)

    def test_find_unused_false_requires_synthetic_preflight_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_step4_upstream_fixture(repo, task_id=4, run_id="1")
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                cfg, _, snapshot = resolve_config(
                    config_path=_REPO_ROOT / "configs" / "odcr.yaml",
                    command="step5",
                    task_id=4,
                    set_overrides=["step5.ddp.find_unused_parameters=false"],
                    dry_run=True,
                    from_step4="1",
                    eval_profile="balanced_2gpu",
                    mode="full",
                )
            finally:
                config_resolver._REPO_ROOT = old_root
        self.assertFalse(cfg.ddp_find_unused_parameters)
        self.assertEqual(cfg.ddp_find_unused_false_preflight, "synthetic_one_batch")
        self.assertFalse(snapshot["step5_ddp"]["ddp_find_unused_parameters"])
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_step4_upstream_fixture(repo, task_id=4, run_id="1")
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                with self.assertRaises(OneControlConfigError):
                    resolve_config(
                        config_path=_REPO_ROOT / "configs" / "odcr.yaml",
                        command="step5",
                        task_id=4,
                        set_overrides=[
                            "step5.ddp.find_unused_parameters=false",
                            "step5.ddp.find_unused_false_preflight=fail_fast",
                        ],
                        dry_run=True,
                        from_step4="1",
                        eval_profile="balanced_2gpu",
                        mode="full",
                    )
            finally:
                config_resolver._REPO_ROOT = old_root

    def test_flan_forward_has_no_hf_labels_and_lci_fca_are_not_repeated(self) -> None:
        source = (_REPO_ROOT / "code" / "executors" / "step5_engine.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\blabels\s*=", source))
        total = compose_step5_total_loss(
            loss_factual=torch.tensor(1.0),
            loss_counterfactual=torch.tensor(2.0),
            loss_repeat_ul=torch.tensor(3.0),
            loss_terminal_clean=torch.tensor(5.0),
            loss_batch_diversity=torch.tensor(7.0),
            repeat_ul_weight=0.1,
            terminal_clean_weight=0.2,
            batch_diversity_weight=0.3,
            lci_weighted_loss=torch.tensor(11.0),
            fca_weighted_loss=torch.tensor(13.0),
            ortho_keep_loss=torch.tensor(17.0),
            ortho_keep_weight=0.4,
        )
        expected = 1.0 + 2.0 + 0.3 + 1.0 + 2.1 + 11.0 + 13.0 + 6.8
        self.assertAlmostEqual(float(total), expected, places=5)


if __name__ == "__main__":
    unittest.main()
