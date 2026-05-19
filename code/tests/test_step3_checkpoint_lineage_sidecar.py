"""Step3 checkpoint lineage sidecar tests.

this test proves: checkpoint sidecar lineage fields and hash gates reject stale or incompatible synthetic checkpoints.
this test does not prove: formal Step3 training quality or runtime checkpoint save behavior.
whether formal hot path is covered: no, this uses synthetic checkpoint bytes.
whether runtime evidence is required: yes for Level 3/4 checkpoint policy claims.
regression bug it prevents: missing sidecar fields or stale hashes being accepted downstream.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.training_checkpoint import (  # noqa: E402
    CheckpointLineageError,
    STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
    checkpoint_file_sha256,
    file_fingerprint,
    stable_hash,
    step3_source_table_compatibility_payload,
    validate_step3_checkpoint_lineage,
    write_checkpoint_lineage,
)


class Step3CheckpointLineageSidecarTest(unittest.TestCase):
    def _checkpoint(self, root: Path) -> Path:
        ckpt = root / "runs" / "step3" / "task4" / "unit" / "model" / "best.pth"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"checkpoint-v1")
        return ckpt

    def _tokenizer_manifest(self, root: Path) -> dict:
        manifest = root / "cache" / "task4" / "hf" / "cache" / "cache_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('{"schema":"odcr_step3_tokenizer_cache/2"}\n', encoding="utf-8")
        return {
            "schema_version": "odcr_step3_tokenizer_cache/2",
            "manifest_path": str(manifest),
            "manifest_fingerprint": file_fingerprint(manifest, sample_only=True),
            "cache_dir": str(manifest.parent),
            "cache_version": "v-test",
            "tokenizer_cache_compat_hash": "tokenizer-cache-compatible",
            "data_contract_hash": "tokenizer-data-contract",
            "preprocessing_artifact_hash": "tokenizer-preprocess-artifacts",
            "full_run_config_hash": "tokenizer-full-run-record",
            "train_runtime_config_hash": "tokenizer-runtime-record",
            "optimizer_config_hash": "tokenizer-optimizer-record",
            "performance_profile_hash": "tokenizer-performance-record",
            "compatibility_key": "cache-key",
            "fingerprint_hash": "cache-key",
            "cache_content_hash": "content-hash",
            "manifest_hash": stable_hash(file_fingerprint(manifest, sample_only=True)),
        }

    def _payload(self, root: Path, ckpt: Path, *, schema: str = STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION) -> dict:
        checkpoint_hash = checkpoint_file_sha256(ckpt)
        tokenizer_manifest = self._tokenizer_manifest(root)
        batch_semantics = {
            "batch_size": 8,
            "per_gpu_batch_size": 4,
            "ddp_world_size": 2,
            "batch_semantics_version": "odcr_no_accum/1",
            "formula": "global_batch_size = per_gpu_batch_size * ddp_world_size",
            "formula_proof": {"lhs": 8, "rhs": 8, "matches": True},
        }
        ddp_config = {"ddp_world_size": 2, "ddp_find_unused_parameters": True}
        precision_config = {"train_precision": "fp32"}
        loss_semantics_config = {"specific_separation_margin": 0.6, "variance_target_std": 0.7}
        profile = {"AM_Movies": {"user_content_profiles": {"sample_sha256": "profile-a"}}}
        domain = {"AM_Movies": {"domain_content": {"sample_sha256": "domain-a"}}}
        source_csv = {"AM_Movies": {"sample_sha256": "source-a"}}
        merged_csv = {"aug_train_csv": {"sample_sha256": "merged-train"}}
        preprocess_latest = {"a": "1", "b": "1", "c": "1"}
        run_summary = {"a": {"sample_sha256": "rs-a"}, "b": {"sample_sha256": "rs-b"}, "c": {"sample_sha256": "rs-c"}}
        stage_status = {"a": {"sample_sha256": "ss-a"}, "b": {"sample_sha256": "ss-b"}, "c": {"sample_sha256": "ss-c"}}
        stage_manifest = {"a": {"sample_sha256": "sm-a"}, "b": {"sample_sha256": "sm-b"}, "c": {"sample_sha256": "sm-c"}}
        pp_source_table = {"a": {"sample_sha256": "st-a"}, "b": {"sample_sha256": "st-b"}, "c": {"sample_sha256": "st-c"}}
        metrics = {"a": None, "b": {"sample_sha256": "mt-b"}, "c": {"sample_sha256": "mt-c"}}
        verify = {"a": None, "b": {"sample_sha256": "vr-b"}, "c": {"sample_sha256": "vr-c"}}
        data_contract = {
            "preprocess_contract_version": "test-preprocess/1",
            "source_task": {"task_id": 4, "auxiliary": "AM_Movies", "target": "AM_Electronics"},
            "source_csv_fingerprints": source_csv,
            "merged_csv_fingerprints": merged_csv,
        }
        artifact_lineage_contract = {
            "data_merged_artifact_fingerprint": stable_hash({"aug_train_csv": file_fingerprint(ckpt)}),
            "preprocess": {
                "preprocess_latest_run_ids": preprocess_latest,
                "preprocess_run_summary_fingerprints_hash": stable_hash(run_summary),
                "preprocess_stage_status_fingerprints_hash": stable_hash(stage_status),
                "preprocess_stage_manifest_fingerprints_hash": stable_hash(stage_manifest),
                "preprocess_source_table_fingerprints_hash": stable_hash(pp_source_table),
                "preprocess_metrics_fingerprints_hash": stable_hash(metrics),
                "preprocess_verify_report_fingerprints_hash": stable_hash(verify),
            },
            "profile_artifact_fingerprints": profile,
            "domain_artifact_fingerprints": domain,
            "sentence_embed_model_identity": {"identity": str(root / "models" / "bge"), "model_artifact_fingerprint": {"kind": "missing"}},
        }
        semantic_model = {
            "resolved_config_compatibility": {"task_id": 4, "model_architecture_config_hash": "arch-hash"},
            "source_table_compatibility": {"schema_version": schema},
            "embed_dim": 8,
            "model_architecture_config_hash": "arch-hash",
            "representation_output_contract_hash": "representation-contract",
            "structured_losses_hash": "loss-hash",
            "loss_semantics_hash": "loss-semantics-hash",
            "profile_artifact_fingerprints_hash": stable_hash(profile),
            "domain_artifact_fingerprints_hash": stable_hash(domain),
        }
        train_runtime = {"batch_semantics": batch_semantics, "ddp_config": ddp_config, "precision_config": precision_config}
        optimizer_runtime = {"step3_optimizer_config": {"name": "adamw"}, "learning_rate": 0.001}
        task_profile = {
            "profile_id": "task4_legacy_profile",
            "profile_isolation_hash": "profile-hash",
            "cross_rank_structured_gather": {"enabled": True, "mode": "local_gradient_context"},
        }
        performance_profile = {"candidate_role": "performance_candidate:G1", "step3_task_profile": task_profile}
        loss_config = {
            "step3_structured_losses": {"hash": "loss"},
            "step3_loss_semantics": loss_semantics_config,
            "cross_rank_structured_gather": {"enabled": True, "mode": "local_gradient_context"},
        }
        source_table_lineage = {
            "field_sources": {
                "task": "tasks.4",
                "step3_structured_losses": "step3.structured_losses",
                "step3_loss_semantics": "step3.loss_semantics",
                "step3_tokenizer": "step3.tokenizer",
                "step3_evidence": "step3.evidence",
                "step3_scenario_profile": "step3.scenario_profiles.legacy",
                "step3_task_profile": "step3.task_profiles.task4",
                "profile_isolation_hash": "resolver-derived",
                "step3_cross_rank_structured_gather": "step3.cross_rank_structured_gather",
                "embed_dim": "env.embed_dim",
            }
        }
        source_table_compatibility_hash = stable_hash(step3_source_table_compatibility_payload(source_table_lineage))
        payload = {
            "sidecar_schema_version": schema,
            "stage": "step3",
            "run_id": "unit",
            "task_id": 4,
            "source_domain": "AM_Movies",
            "target_domain": "AM_Electronics",
            "task_profile_id": "task4_legacy_profile",
            "task_profile_key": "task4_legacy_profile_key",
            "profile_isolation_hash": "profile-hash",
            "scenario": "legacy_scenario",
            "direction": "legacy",
            "checkpoint_path": str(ckpt.resolve()),
            "checkpoint_file_hash": checkpoint_hash,
            "checkpoint_file": file_fingerprint(ckpt),
            "checkpoint_epoch": 2,
            "selection_metric": "valid_loss",
            "selection_metric_value": 4.8,
            "selection_direction": "min",
            "selection_scope": "best_observed",
            "reason": "global_best_improved",
            "replaced_previous": False,
            "global_best_epoch": 2,
            "global_best_metric": 4.8,
            "after_min_epochs_best_epoch": 7,
            "after_min_epochs_best_metric": 8.1,
            "epoch_summary_hash": "epoch-summary-hash",
            "metrics_jsonl_hash": "metrics-jsonl-hash",
            "quality_status_at_save": "not_evaluated",
            "quality_status": "not_evaluated",
            "downstream_ready": False,
            "grad_inf_count_until_epoch": 0,
            "model_file_hash": checkpoint_hash,
            "optimizer_state_hash": "optimizer-state-hash",
            "code_commit": "abc",
            "created_at": "2026-05-02T00:00:00Z",
            "git_code_fingerprint": {"git_commit": "abc", "git_dirty_status": "", "critical_code_files": [], "fingerprint_hash": "code"},
            "one_control_resolved_config_path": str(root / "runs" / "step3" / "task4" / "unit" / "meta" / "resolved_config.json"),
            "one_control_resolved_config_hash": "resolved-hash",
            "resolved_config_hash": "resolved-hash",
            "resolved_config": {"hash": "resolved-hash"},
            "resolved_config_compatibility_hash": "resolved-compatible",
            "source_table_path": str(root / "runs" / "step3" / "task4" / "unit" / "meta" / "source_table.json"),
            "source_table_hash": "source-table-hash",
            "source_table": source_table_lineage,
            "source_table_compatibility_hash": source_table_compatibility_hash,
            "source_table_payload_summary": {"field_source_count": 3},
            "full_run_config_hash": "resolved-hash",
            "train_runtime_config_hash": stable_hash(train_runtime),
            "training_runtime_config_path": str(root / "runs" / "step3" / "task4" / "unit" / "meta" / "training_runtime_config.json"),
            "training_runtime_config_hash": "training-runtime-file-hash",
            "training_runtime_config": {"hash": "training-runtime-file-hash"},
            "optimizer_config_hash": stable_hash(optimizer_runtime),
            "performance_profile_hash": stable_hash(performance_profile),
            "loss_config_hash": stable_hash(loss_config),
            "loss_config": loss_config,
            "semantic_model_compat_hash": stable_hash(semantic_model),
            "data_contract_hash": stable_hash(data_contract),
            "artifact_lineage_hash": stable_hash(artifact_lineage_contract),
            "tokenizer_cache_compat_hash": tokenizer_manifest["tokenizer_cache_compat_hash"],
            "semantic_model_compatibility": semantic_model,
            "data_contract": data_contract,
            "artifact_lineage_contract": artifact_lineage_contract,
            "train_runtime_config": train_runtime,
            "optimizer_runtime_config": optimizer_runtime,
            "performance_profile": performance_profile,
            "training_semantic_fingerprint": "train-fp",
            "preprocess_contract_version": "test-preprocess/1",
            "artifact_lineage": {"aug_train_csv": file_fingerprint(ckpt)},
            "data_merged_artifact_fingerprint": stable_hash({"aug_train_csv": file_fingerprint(ckpt)}),
            "step3_upstream_preprocess_gate": {"status": "ok"},
            "step3_upstream_preprocess_gate_hash": "upstream-hash",
            "env": {"embed_dim": 8},
            "embed_dim": 8,
            "model_architecture_config_hash": "arch-hash",
            "step3_structured_losses_config_hash": "loss-hash",
            "step3_loss_semantics_config_hash": "loss-semantics-hash",
            "step3_loss_semantics_config": loss_semantics_config,
            "step3_optimizer_config": {"name": "adamw"},
            "step3_optimizer_config_hash": stable_hash({"name": "adamw"}),
            "step3_tokenizer_config": {"max_length": 48},
            "step3_tokenizer_config_hash": stable_hash({"max_length": 48}),
            "step3_evidence_config": {"max_evidence_length": 48},
            "step3_evidence_config_hash": stable_hash({"max_evidence_length": 48}),
            "step3_scheduler_config": {"name": "warmup_cosine", "warmup_ratio": 0.06, "min_lr_ratio": 0.05},
            "step3_scheduler_config_hash": stable_hash({"name": "warmup_cosine", "warmup_ratio": 0.06, "min_lr_ratio": 0.05}),
            "step3_valid_batch_config": {"derive_from_train": True, "valid_batch_size": 256, "valid_micro_batch_size": 128},
            "step3_valid_batch_config_hash": stable_hash({"derive_from_train": True, "valid_batch_size": 256, "valid_micro_batch_size": 128}),
            "step3_scenario_profile": {"name": "legacy_scenario"},
            "step3_scenario_profile_hash": stable_hash({"name": "legacy_scenario"}),
            "step3_task_profile": task_profile,
            "step3_task_profile_hash": stable_hash(task_profile),
            "step3_cross_rank_structured_gather": {"enabled": True, "mode": "local_gradient_context"},
            "ddp_config": ddp_config,
            "ddp_config_hash": stable_hash(ddp_config),
            "precision_config": precision_config,
            "precision_config_hash": stable_hash(precision_config),
            "batch_semantics": batch_semantics,
            "batch_semantics_hash": stable_hash(batch_semantics),
            "preprocess_latest_run_ids": preprocess_latest,
            "preprocess_run_summary_fingerprints": run_summary,
            "preprocess_run_summary_fingerprints_hash": stable_hash(run_summary),
            "preprocess_stage_status_fingerprints": stage_status,
            "preprocess_stage_status_fingerprints_hash": stable_hash(stage_status),
            "preprocess_stage_manifest_fingerprints": stage_manifest,
            "preprocess_stage_manifest_fingerprints_hash": stable_hash(stage_manifest),
            "preprocess_source_table_fingerprints": pp_source_table,
            "preprocess_source_table_fingerprints_hash": stable_hash(pp_source_table),
            "preprocess_metrics_fingerprints": metrics,
            "preprocess_metrics_fingerprints_hash": stable_hash(metrics),
            "preprocess_verify_report_fingerprints": verify,
            "preprocess_verify_report_fingerprints_hash": stable_hash(verify),
            "profile_artifact_fingerprints": profile,
            "profile_artifact_fingerprints_hash": stable_hash(profile),
            "domain_artifact_fingerprints": domain,
            "domain_artifact_fingerprints_hash": stable_hash(domain),
            "source_csv_fingerprints": source_csv,
            "source_csv_fingerprints_hash": stable_hash(source_csv),
            "merged_csv_fingerprints": merged_csv,
            "merged_csv_fingerprints_hash": stable_hash(merged_csv),
            "sentence_embed_model_identity": {"identity": str(root / "models" / "bge"), "model_artifact_fingerprint": {"kind": "missing"}},
            "step3_tokenizer_cache_manifest": tokenizer_manifest,
            "step3_tokenizer_cache_manifest_hash": stable_hash(tokenizer_manifest),
            "schema_contract_versions": {
                "preprocess_contract_version": "test-preprocess/1",
                "step3_upstream_gate_schema_version": "odcr_step3_upstream_gate/1",
                "step3_tokenizer_cache_schema_version": "odcr_step3_tokenizer_cache/2",
            },
            "compatibility_metadata": {
                "minimum_accepted_schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
                "downstream_compare_fields": ["checkpoint_file_hash"],
                "record_only_fields": ["step3_optimizer_config_hash", "batch_semantics_hash"],
            },
            "metrics_summary": {"status": "placeholder"},
            "source_task": {
                "task_id": 4,
                "auxiliary": "AM_Movies",
                "target": "AM_Electronics",
                "scenario": "legacy_scenario",
                "direction": "legacy",
                "task_profile_id": "task4_legacy_profile",
            },
        }
        payload["checkpoint_compatibility_hash"] = stable_hash(payload)
        return payload

    def _write_sidecar(self, root: Path, *, schema: str = STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION) -> tuple[Path, dict]:
        ckpt = self._checkpoint(root)
        payload = self._payload(root, ckpt, schema=schema)
        write_checkpoint_lineage(ckpt, payload)
        return ckpt, payload

    def _expected(self, payload: dict) -> dict:
        keys = (
            "sidecar_schema_version",
            "task_id",
            "source_domain",
            "target_domain",
            "task_profile_id",
            "profile_isolation_hash",
            "resolved_config_compatibility_hash",
            "source_table_compatibility_hash",
            "semantic_model_compat_hash",
            "data_contract_hash",
            "artifact_lineage_hash",
            "tokenizer_cache_compat_hash",
            "loss_config_hash",
            "preprocess_latest_run_ids",
            "preprocess_run_summary_fingerprints_hash",
            "preprocess_stage_status_fingerprints_hash",
            "preprocess_stage_manifest_fingerprints_hash",
            "preprocess_source_table_fingerprints_hash",
            "preprocess_metrics_fingerprints_hash",
            "preprocess_verify_report_fingerprints_hash",
            "profile_artifact_fingerprints_hash",
            "domain_artifact_fingerprints_hash",
            "source_csv_fingerprints_hash",
            "merged_csv_fingerprints_hash",
            "embed_dim",
            "model_architecture_config_hash",
            "step3_structured_losses_config_hash",
            "step3_loss_semantics_config_hash",
            "step3_tokenizer_config_hash",
            "step3_evidence_config_hash",
            "step3_scenario_profile_hash",
            "step3_task_profile_hash",
            "step3_tokenizer_cache_manifest_hash",
            "source_task",
        )
        return {key: payload[key] for key in keys}

    def test_complete_sidecar_with_matching_hash_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, payload = self._write_sidecar(Path(tmp))
            loaded = validate_step3_checkpoint_lineage(ckpt, expected=self._expected(payload))
            self.assertEqual(loaded["sidecar_schema_version"], STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION)

    def test_missing_sidecar_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._checkpoint(Path(tmp))
            with self.assertRaises(CheckpointLineageError):
                validate_step3_checkpoint_lineage(ckpt, expected={})

    def test_checkpoint_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, payload = self._write_sidecar(Path(tmp))
            ckpt.write_bytes(b"changed")
            with self.assertRaises(CheckpointLineageError):
                validate_step3_checkpoint_lineage(ckpt, expected=self._expected(payload))

    def test_old_sidecar_schema_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, payload = self._write_sidecar(Path(tmp), schema="odcr_step3_checkpoint_compat/0")
            with self.assertRaises(CheckpointLineageError):
                validate_step3_checkpoint_lineage(ckpt, expected=self._expected(payload))

    def test_preprocess_latest_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, payload = self._write_sidecar(Path(tmp))
            expected = self._expected(payload)
            expected["preprocess_latest_run_ids"] = {"a": "1", "b": "2", "c": "1"}
            with self.assertRaises(CheckpointLineageError):
                validate_step3_checkpoint_lineage(ckpt, expected=expected)

    def test_profile_domain_fingerprint_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, payload = self._write_sidecar(Path(tmp))
            expected = self._expected(payload)
            expected["profile_artifact_fingerprints_hash"] = "bad-profile"
            with self.assertRaises(CheckpointLineageError):
                validate_step3_checkpoint_lineage(ckpt, expected=expected)
            expected = self._expected(payload)
            expected["domain_artifact_fingerprints_hash"] = "bad-domain"
            with self.assertRaises(CheckpointLineageError):
                validate_step3_checkpoint_lineage(ckpt, expected=expected)

    def test_semantic_config_mismatch_fails_but_record_only_fields_do_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, payload = self._write_sidecar(Path(tmp))
            expected = self._expected(payload)
            expected["resolved_config_compatibility_hash"] = "bad-resolved"
            with self.assertRaises(CheckpointLineageError):
                validate_step3_checkpoint_lineage(ckpt, expected=expected)
            expected = self._expected(payload)
            expected["source_table_compatibility_hash"] = "bad-source-table"
            with self.assertRaises(CheckpointLineageError):
                validate_step3_checkpoint_lineage(ckpt, expected=expected)
            record_only_expected = self._expected(payload)
            loaded = validate_step3_checkpoint_lineage(ckpt, expected=record_only_expected)
            loaded["step3_optimizer_config_hash"] = "changed-record-only"
            self.assertEqual(loaded["sidecar_schema_version"], STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION)

    def test_embed_dim_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, payload = self._write_sidecar(Path(tmp))
            expected = self._expected(payload)
            expected["embed_dim"] = 16
            with self.assertRaises(CheckpointLineageError):
                validate_step3_checkpoint_lineage(ckpt, expected=expected)


if __name__ == "__main__":
    unittest.main()
