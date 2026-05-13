"""Step3 quality/evidence/performance rebuild regression tests.

this test proves: Level 1/2 code contracts for global-best checkpoint policy, quality gates, finite-gradient policy, timing/memory/prefetch/gather schemas, and diagnostic sample protocols.
this test does not prove: controlled CUDA runtime behavior, H2D overlap, performance improvement, or full formal Step3 quality.
whether formal hot path is covered: partially; helper wiring used by the train loop is imported, but no formal train command runs.
whether runtime evidence is required: yes, Level 3/4 claims require Stage2 controlled probe or a future formal run.
regression bug it prevents: rolling previous-epoch best checkpoints, blocked latest consumption, nonfinite-gradient optimizer stepping, schema-free timing/memory evidence, and diagnostic eval being mislabeled as final metrics.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_quality import (  # noqa: E402
    DIAGNOSTIC_PROTOCOLS,
    MEMORY_REQUIRED_FIELDS,
    PREFETCH_EVIDENCE_FIELDS,
    TIMING_REQUIRED_FIELDS,
    checkpoint_filename_for_metric,
    checkpoint_sidecar_payload,
    collapse_stats_from_predictions,
    diagnostic_sample_record,
    inspect_gradients,
    metric_improved,
    timing_row_with_closure,
    validate_step3_downstream_quality_gate,
    Step3QualityGateError,
)
from executors.step3_train_core import (  # noqa: E402
    Step3CUDAPrefetcher,
    _step3_gpu_profile_row,
    gather_step3_structured_context_local_gradient,
    step3_target_only_diagnostic_protocol,
)


class Step3QualityEvidencePerformanceRebuildTests(unittest.TestCase):
    def test_checkpoint_global_best_not_previous_epoch_best(self) -> None:
        best = None
        events = []
        for epoch, valid_loss in ((1, 5.2), (2, 4.8), (6, 8.3), (7, 8.1)):
            if metric_improved(valid_loss, best, direction="min"):
                best = valid_loss
                events.append((epoch, valid_loss))
        self.assertEqual(events[-1], (2, 4.8))
        self.assertFalse(metric_improved(8.1, best, direction="min"))
        self.assertEqual(checkpoint_filename_for_metric(2, "valid_loss", 4.857804), "epoch_002_valid_loss_4.8578.pth")

    def test_checkpoint_sidecar_payload_records_epoch_metric_scope_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "model" / "best_observed.pth"
            ckpt.parent.mkdir()
            ckpt.write_bytes(b"checkpoint")
            sidecar = checkpoint_sidecar_payload(
                checkpoint_file=ckpt,
                checkpoint_epoch=2,
                selection_metric="valid_loss",
                selection_metric_value=4.8,
                selection_scope="best_observed",
                global_best_epoch=2,
                global_best_metric=4.8,
                after_min_epochs_best_epoch=7,
                after_min_epochs_best_metric=8.1,
                epoch_summary_path=None,
                metrics_jsonl_path=None,
                resolved_config_hash="resolved",
                training_runtime_config_hash="runtime",
                quality_status_at_save="not_evaluated",
                grad_inf_count_until_epoch=0,
            )
        self.assertEqual(sidecar["selection_scope"], "best_observed")
        self.assertEqual(sidecar["checkpoint_epoch"], 2)
        self.assertTrue(sidecar["checkpoint_file_hash"])
        self.assertEqual(sidecar["selection_direction"], "min")

    def test_readiness_gate_blocks_blocked_latest_before_checkpoint_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            meta = run / "meta"
            meta.mkdir(parents=True)
            (meta / "readiness_audit.json").write_text(
                json.dumps(
                    {
                        "schema_version": "odcr_step3_readiness_audit/1",
                        "readiness_gate": "step3_upstream_readiness_gate",
                        "readiness_status": "blocked",
                        "quality_status": "blocked",
                        "downstream_ready": False,
                        "readiness_block_reasons": ["best_checkpoint_not_global_best"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Step3QualityGateError, "upstream readiness gate blocked"):
                validate_step3_downstream_quality_gate(run)

    def test_nonfinite_grad_is_detected_for_optimizer_skip_policy(self) -> None:
        param = torch.nn.Parameter(torch.ones(2, 2))
        param.grad = torch.tensor([[math.inf, 1.0], [2.0, 3.0]])
        inspection = inspect_gradients([("layer.weight", param)], topk=3)
        self.assertFalse(inspection.grad_finite)
        self.assertGreater(inspection.nonfinite_param_count, 0)
        scheduler_step_on_skipped_optimizer = False
        optimizer_step_executed = bool(inspection.grad_finite)
        scheduler_step_executed = optimizer_step_executed or scheduler_step_on_skipped_optimizer
        self.assertFalse(optimizer_step_executed)
        self.assertFalse(scheduler_step_executed)

    def test_timing_closure_schema_and_unknown_gate(self) -> None:
        row = timing_row_with_closure(
            {
                "total_step_time": 1.0,
                "forward_time": 0.01,
                "loss_time": 0.02,
                "backward_time": 0.03,
                "optimizer_time": 0.04,
                "grad_check_ms": 10.0,
            },
            base={"epoch": 1, "global_step": 1, "rank": 0},
        )
        for field in TIMING_REQUIRED_FIELDS:
            self.assertIn(field, row)
        self.assertEqual(row["performance_timing_status"], "performance_not_closed")
        self.assertGreater(row["unknown_ms"], 0.0)

    def test_memory_phase_schema_fields_exist(self) -> None:
        cfg = SimpleNamespace(run_id="dry", task_idx=2, task_profile_id="profile")
        row = _step3_gpu_profile_row(final_cfg=cfg, rank=1, device="cpu", global_step=3, epoch=1, phase="after_grad_norm")
        for field in MEMORY_REQUIRED_FIELDS:
            self.assertIn(field, row)
        self.assertEqual(row["rank"], 1)
        self.assertEqual(row["phase"], "after_grad_norm")

    def test_prefetch_evidence_fields_and_fallback_are_explicit(self) -> None:
        prefetcher = Step3CUDAPrefetcher([], device="cpu", enabled=False, double_buffer=True, diagnostic_cpu_mode=True)
        evidence = prefetcher.last_evidence
        for field in PREFETCH_EVIDENCE_FIELDS:
            self.assertIn(field, evidence)
        self.assertTrue(evidence["double_buffer_configured"])
        self.assertFalse(evidence["double_buffer_active"])
        self.assertTrue(evidence["fallback_used"])

    def test_gather_whitelist_rejects_raw_and_logs_bytes(self) -> None:
        shared = torch.ones(2, 4)
        specific = torch.zeros(2, 4)
        domain = torch.tensor([0, 1])
        with self.assertRaisesRegex(RuntimeError, "forbids"):
            gather_step3_structured_context_local_gradient(
                shared_repr=shared,
                specific_repr=specific,
                domain_ids=domain,
                world_size=1,
                rank=0,
                requested_keys=["shared_repr", "specific_repr", "domain_ids", "raw_text"],
            )
        _, summary = gather_step3_structured_context_local_gradient(
            shared_repr=shared,
            specific_repr=specific,
            domain_ids=domain,
            world_size=1,
            rank=0,
        )
        self.assertTrue(summary["compact_gather_only"])
        self.assertGreater(summary["structured_gather_total_bytes"], 0)
        self.assertIn("structured_gather_ms", summary)

    def test_samples_collapse_stats_and_protocol_are_diagnostic_only(self) -> None:
        sample = diagnostic_sample_record(
            run_id="dry",
            epoch=1,
            split="valid",
            sample_id="s1",
            source_domain="AM_Movies",
            target_domain="AM_CDs",
            rating_gold=4.0,
            rating_pred=3.0,
            target_text="good disc",
            pred_text="",
            evaluator_protocol="odcr_step3_diagnostic",
        )
        self.assertTrue(sample["diagnostic_only"])
        self.assertTrue(sample["not_final_paper_metric"])
        stats = collapse_stats_from_predictions(["", ""], ["good disc", "nice disc"])
        self.assertEqual(stats["empty_rate"], 1.0)
        self.assertEqual(stats["distinct1"], 0.0)
        self.assertTrue(DIAGNOSTIC_PROTOCOLS["code1_target_only_comparable"]["not_final_paper_metric"])
        self.assertTrue(step3_target_only_diagnostic_protocol()["target_only"])

    def test_test_file_declares_evidence_limits(self) -> None:
        doc = __doc__ or ""
        for phrase in (
            "this test proves",
            "this test does not prove",
            "whether formal hot path is covered",
            "whether runtime evidence is required",
            "regression bug it prevents",
        ):
            self.assertIn(phrase, doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
