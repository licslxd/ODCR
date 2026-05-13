"""Step3 One-Control control-plane regression tests."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

import config as train_config  # noqa: E402
import yaml  # noqa: E402
from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.config_resolver import load_yaml_config  # noqa: E402
from odcr_core.config_schema import OneControlConfigError  # noqa: E402
from odcr_core.logging_meta import _stage_label  # noqa: E402
from odcr_core.manifests import build_formal_source_table_snapshot, build_run_manifest, formal_snapshot_view  # noqa: E402
from odcr_core.runners import _torchrun_hardware_env  # noqa: E402

_REPO_ROOT = Path(_CODE_DIR).resolve().parent


def _resolve_step3_task(task_id: int | None = 2, set_overrides: list[str] | None = None):
    return resolve_config(
        config_path=_REPO_ROOT / "configs" / "odcr.yaml",
        command="step3",
        task_id=task_id,
        set_overrides=set_overrides or [],
        dry_run=True,
        run_id="auto",
        mode="full",
    )


class TestStep3ControlPlane(unittest.TestCase):
    def test_step3_resolves_from_one_control_without_legacy_payload(self) -> None:
        cfg, sources, snapshot = _resolve_step3_task()
        payload = json.loads(cfg.effective_training_payload_json)

        self.assertEqual(cfg.command, "step3")
        self.assertEqual(snapshot["train"]["stage"], "step3")
        self.assertEqual(snapshot["field_sources"]["config"], str(_REPO_ROOT / "configs" / "odcr.yaml"))
        self.assertTrue(all("presets/" not in str(record.source) for record in sources))

        self.assertFalse(hasattr(cfg, "adv"))
        self.assertFalse(hasattr(cfg, "eta"))
        self.assertEqual(payload["preset_name"], "step3")
        self.assertEqual(payload["explainer_loss_weight"], 0.0)
        self.assertIn("step3_structured_losses", payload)
        self.assertIn("step3_loss_semantics", payload)
        self.assertIn("step3_ddp", payload)
        self.assertIn("step3_optimizer", payload)
        self.assertIn("step3_precision", payload)
        self.assertIn("step3_tokenizer", payload)
        self.assertIn("step3_evidence", payload)
        self.assertEqual(payload["scenario"], "strong_related")
        self.assertEqual(payload["direction"], "forward")
        self.assertEqual(payload["task_profile_id"], "task2_strong_forward_g1s")
        self.assertEqual(payload["step3_task_profile"]["profile_id"], "task2_strong_forward_g1s")
        self.assertTrue(payload["step3_task_profile"]["cross_rank_structured_gather"]["enabled"])
        self.assertFalse(payload["training_row"]["ddp_find_unused_parameters"])
        self.assertFalse(payload["training_row"]["ddp_static_graph"])
        self.assertTrue(payload["training_row"]["ddp_graph_safety_preflight"])
        self.assertFalse(cfg.ddp_find_unused_parameters)
        self.assertFalse(cfg.ddp_static_graph)
        self.assertTrue(cfg.ddp_graph_safety_preflight)
        self.assertNotIn("adv", payload["training_row"])
        self.assertNotIn("eta", payload["training_row"])
        self.assertNotIn("coef", payload["training_row"])

    def test_step3_child_hardware_payload_and_precision_are_one_control(self) -> None:
        cfg, sources, snapshot = _resolve_step3_task()
        payload = json.loads(cfg.effective_training_payload_json)
        hardware = json.loads(cfg.hardware_profile_json)
        source_map = {record.key: record.source for record in sources}

        for key in (
            "ddp_world_size",
            "num_proc",
            "max_parallel_cpu",
            "dataloader_num_workers_train",
            "dataloader_num_workers_valid",
            "dataloader_num_workers_test",
            "dataloader_prefetch_factor_train",
            "dataloader_prefetch_factor_valid",
            "dataloader_prefetch_factor_test",
            "pin_memory",
            "persistent_workers",
            "non_blocking_h2d",
        ):
            self.assertIn(key, hardware)

        self.assertEqual(hardware["max_parallel_cpu"], 12)
        self.assertEqual(hardware["dataloader_num_workers_train"], 4)
        self.assertEqual(hardware["dataloader_num_workers_valid"], 2)
        self.assertEqual(hardware["dataloader_num_workers_test"], 2)
        self.assertEqual(snapshot["hardware"]["max_parallel_cpu"], 12)
        self.assertEqual(snapshot["field_sources"]["hardware.max_parallel_cpu"], "hardware.profiles.default.max_parallel_cpu")
        worker_budget = snapshot["hardware"]["worker_budget_formula"]
        self.assertEqual(worker_budget["semantics"], "Step3 dataloader_num_workers_* are per rank; num_proc is pre-DDP datasets.map/tokenizer process count.")
        self.assertEqual(worker_budget["train_active_processes"], 10)
        self.assertEqual(worker_budget["tokenization_active_processes"], 10)
        self.assertEqual(worker_budget["tokenization_num_proc"], 8)
        self.assertLessEqual(worker_budget["train_active_processes"], 12)
        self.assertLessEqual(worker_budget["tokenization_active_processes"], 12)
        self.assertTrue(hardware["pin_memory"])
        self.assertTrue(hardware["persistent_workers"])
        self.assertTrue(hardware["non_blocking_h2d"])
        self.assertEqual(payload["training_row"]["train_precision"], "bf16")
        self.assertEqual(payload["runtime_precision"]["train_precision"], "bf16")
        self.assertTrue(payload["runtime_precision"]["allow_tf32"])
        self.assertEqual(snapshot["train"]["precision"], "bf16")
        self.assertEqual(snapshot["step3_optimizer"]["name"], "adamw")
        self.assertEqual(snapshot["step3_tokenizer"]["max_length"], 48)
        self.assertEqual(snapshot["step3_evidence"]["max_evidence_length"], 48)
        self.assertEqual(snapshot["step3_scheduler"]["warmup_ratio"], 0.10)
        self.assertEqual(snapshot["train"]["max_grad_norm"], 0.5)
        self.assertEqual(snapshot["task"]["task_profile_id"], "task2_strong_forward_g1s")
        self.assertEqual(snapshot["step3_task_profile"]["profile_id"], "task2_strong_forward_g1s")
        self.assertEqual(snapshot["train"]["batch_size"], 1408)
        self.assertEqual(snapshot["train"]["per_gpu_batch_size"], 704)
        self.assertEqual(snapshot["train"]["batch_semantics_version"], "odcr_no_accum/1")
        self.assertTrue(snapshot["train"]["grad_accum_removed"])
        self.assertNotIn("grad_accum", snapshot["train"])
        self.assertEqual(snapshot["train"]["step3_batch_semantics"], "odcr_no_accum/1")
        self.assertEqual(snapshot["train"]["step3_batch_formula"], "global_batch_size = per_gpu_batch_size * ddp_world_size")
        self.assertEqual(snapshot["step3_eval"]["valid_micro_batch_size"], 704)
        self.assertNotIn("step3_performance_ladder", snapshot)
        self.assertEqual(snapshot["step3_exploration_profiles"]["task2_g2_effective_pool_2048"]["probe_only"], True)
        self.assertEqual(snapshot["step3_exploration_profiles"]["task2_g2_effective_pool_2048"]["formal_allowed"], False)
        self.assertEqual(snapshot["step3_worker_profiles"]["W2"]["cpu_budget"], 12)
        self.assertTrue(snapshot["step3_prefetcher"]["enabled"])
        self.assertTrue(snapshot["step3_cross_rank_structured_gather"]["enabled"])
        self.assertEqual(snapshot["step3_cross_rank_structured_gather"]["mode"], "local_gradient_context")
        self.assertFalse(snapshot["step3_memory"]["activation_checkpointing"]["enabled"])
        self.assertEqual(
            snapshot["field_sources"]["train_precision"],
            "step3.experiment_profiles.csb_odcr_sidecar_stable.train.backend.train_precision",
        )
        self.assertEqual(source_map["runtime_precision_mode"], "ResolvedConfig.train_precision -> ODCR_RUNTIME_PRECISION_MODE transport")

        child_env = _torchrun_hardware_env(cfg)
        self.assertEqual(child_env["ODCR_RUNTIME_PRECISION_MODE"], "bf16")
        self.assertEqual(child_env["ODCR_RUNTIME_ALLOW_TF32"], "1")
        self.assertEqual(json.loads(child_env["ODCR_HARDWARE_PROFILE_JSON"])["max_parallel_cpu"], 12)

    def test_step3_child_config_build_smoke_uses_resolver_payload(self) -> None:
        cfg, _, _ = _resolve_step3_task()
        child_env = {
            "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON": cfg.effective_training_payload_json,
            "ODCR_HARDWARE_PROFILE_JSON": cfg.hardware_profile_json,
            "ODCR_HARDWARE_PRESET": cfg.hardware_preset_id,
            "ODCR_RUNTIME_PRECISION_MODE": cfg.train_precision,
            "ODCR_RUNTIME_ALLOW_TF32": "1" if cfg.allow_tf32 else "0",
            "ODCR_RUNTIME_AMP_AUTOCAST": "1" if cfg.amp_autocast else "0",
            "ODCR_RUNTIME_GRAD_SCALER": "1" if cfg.grad_scaler else "0",
            "ODCR_RESOLVED_DATA_DIR": cfg.data_dir,
            "ODCR_RESOLVED_MERGED_DIR": cfg.merged_dir,
            "ODCR_RESOLVED_MODELS_DIR": cfg.models_dir,
            "ODCR_RESOLVED_STEP5_TEXT_MODEL": cfg.step5_text_model,
            "ODCR_RESOLVED_SENTENCE_EMBED_MODEL": cfg.sentence_embed_model,
            "ODCR_RESOLVED_EMBED_DIM": str(cfg.embed_dim),
        }
        args = SimpleNamespace(
            epochs=None,
            learning_rate=None,
            emsize=None,
            nlayers=None,
            nhead=None,
            nhid=None,
            dropout=None,
        )
        with patch.dict(os.environ, child_env, clear=False):
            final_cfg = train_config.build_resolved_training_config(
                args,
                task_idx=cfg.task_id,
                world_size=cfg.ddp_world_size,
            )
        self.assertEqual(final_cfg.max_parallel_cpu, 12)
        self.assertEqual(final_cfg.num_proc, 8)
        self.assertEqual(final_cfg.dataloader_num_workers_train, 4)
        self.assertEqual(final_cfg.dataloader_num_workers_valid, 2)
        self.assertEqual(final_cfg.train_precision, "bf16")
        self.assertTrue(final_cfg.allow_tf32)
        self.assertTrue(final_cfg.amp_autocast)
        self.assertFalse(final_cfg.grad_scaler)
        self.assertEqual(final_cfg.tokenizer_max_length, 48)
        self.assertEqual(final_cfg.evidence_max_length, 48)
        self.assertEqual(final_cfg.max_grad_norm, 0.5)
        self.assertTrue(final_cfg.pin_memory)
        self.assertTrue(final_cfg.persistent_workers)
        self.assertTrue(final_cfg.non_blocking_h2d)
        self.assertEqual(json.loads(final_cfg.optimizer_config_json)["name"], "adamw")
        self.assertEqual(final_cfg.task_profile_id, "task2_strong_forward_g1s")
        self.assertFalse(hasattr(final_cfg, "performance_ladder_config_json"))
        self.assertTrue(json.loads(final_cfg.prefetcher_config_json)["enabled"])
        self.assertTrue(json.loads(final_cfg.cross_rank_structured_gather_config_json)["enabled"])
        self.assertEqual(json.loads(final_cfg.memory_config_json)["profile_buffer_policy"], "gpu_resident")
        self.assertEqual(dict(final_cfg.sources)["max_parallel_cpu"], "hardware_profile_json")
        self.assertEqual(dict(final_cfg.sources)["train_precision"], "effective_payload")

    def test_step3_default_task_and_scenarios_resolve_v0_profiles(self) -> None:
        cfg_default, _, snapshot_default = _resolve_step3_task(None)
        self.assertEqual(cfg_default.task_id, 2)
        self.assertEqual(snapshot_default["task"]["source"], "AM_Movies")
        self.assertEqual(snapshot_default["task"]["target"], "AM_CDs")
        self.assertEqual(snapshot_default["task"]["scenario"], "strong_related")
        self.assertEqual(snapshot_default["task"]["task_profile_id"], "task2_strong_forward_g1s")
        self.assertEqual(snapshot_default["train"]["lr"], 7.0e-4)
        self.assertEqual(snapshot_default["step3_tokenizer"]["max_length"], 48)
        self.assertEqual(snapshot_default["step3_evidence"]["max_evidence_length"], 48)

        cfg5, _, snap5 = _resolve_step3_task(5)
        self.assertEqual((cfg5.scenario, cfg5.direction), ("strong_related", "reverse"))
        self.assertEqual(cfg5.task_profile_id, "task5_strong_reverse_g1_init")
        self.assertEqual(snap5["train"]["lr"], 7.0e-4)
        self.assertEqual(snap5["train"]["batch_size"], 1408)
        self.assertEqual(snap5["train"]["per_gpu_batch_size"], 704)
        self.assertEqual(snap5["step3_tokenizer"]["max_length"], 48)
        self.assertNotIn("task2_", snap5["train"]["step3_batch_candidate_role"])

        for task_id, direction, profile_id in ((8, "forward", "task8_weak_forward_init"), (7, "reverse", "task7_weak_reverse_init")):
            with self.subTest(task_id=task_id):
                cfg, _, snap = _resolve_step3_task(task_id)
                self.assertEqual(cfg.scenario, "weak_cross_platform")
                self.assertEqual(cfg.direction, direction)
                self.assertEqual(cfg.task_profile_id, profile_id)
                self.assertEqual(snap["train"]["lr"], 7.0e-4)
                self.assertEqual(snap["train"]["batch_size"], 1408)
                self.assertEqual(snap["train"]["per_gpu_batch_size"], 704)
                self.assertEqual(snap["step3_tokenizer"]["max_length"], 64)
                self.assertEqual(snap["step3_evidence"]["max_evidence_length"], 64)
                self.assertNotIn("task2_", snap["train"]["step3_batch_candidate_role"])
                self.assertIn(profile_id, snap["train"]["step3_batch_candidate_role"])

    def test_step3_formal_default_view_hides_backup_probe_and_history(self) -> None:
        _cfg, _sources, snapshot = _resolve_step3_task(2)
        formal = formal_snapshot_view(snapshot)
        source_table = build_formal_source_table_snapshot(snapshot)
        self.assertEqual(formal["view"], "formal")
        self.assertEqual(formal["task"]["task_profile_id"], "task2_strong_forward_g1s")
        self.assertEqual(formal["train"]["batch_size"], 1408)
        self.assertNotIn("step3_backup_profiles", formal)
        self.assertNotIn("step3_exploration_profiles", formal)
        self.assertNotIn("step3_performance_ladder", formal)
        self.assertNotIn("step3_performance_probe", formal)
        self.assertNotIn("step3_short_pilot", formal)
        self.assertIn("step3_backup_profiles", snapshot)
        self.assertIn("step3_exploration_profiles", snapshot)
        text = json.dumps(formal, ensure_ascii=False)
        self.assertNotIn("task2_g2_effective_pool_2048", text)
        self.assertNotIn("probe_only", text)
        self.assertNotIn("performance_probe", text)
        self.assertEqual(source_table["view"], "formal")
        table_text = json.dumps(source_table, ensure_ascii=False)
        self.assertNotIn("backup", table_text)
        self.assertNotIn("exploration", table_text)
        self.assertNotIn("performance_probe", table_text)
        self.assertNotIn("short_pilot", table_text)
        self.assertNotIn("step5", table_text)

    def test_step3_verbose_view_exposes_backup_and_g2_profile_only(self) -> None:
        _cfg, _sources, snapshot = _resolve_step3_task(2)
        self.assertIn("step3_backup_profiles", snapshot)
        self.assertIn("step3_exploration_profiles", snapshot)
        self.assertNotIn("step3_performance_probe", snapshot)
        self.assertNotIn("step3_short_pilot", snapshot)
        self.assertTrue(snapshot["step3_backup_profiles"]["task2_g0_backup"]["backup_only"])
        g2 = snapshot["step3_exploration_profiles"]["task2_g2_effective_pool_2048"]
        self.assertTrue(g2["probe_only"])
        self.assertFalse(g2["formal_allowed"])

    def test_step3_no_accum_profile_and_batch_formula(self) -> None:
        cfg, _, snapshot = _resolve_step3_task()
        self.assertEqual(cfg.train_batch_size, 1408)
        self.assertEqual(cfg.per_device_train_batch_size, 704)
        self.assertEqual(snapshot["train"]["step3_batch_semantics"], "odcr_no_accum/1")
        self.assertEqual(snapshot["train"]["candidate"], "G1S-sidecar-stable")
        self.assertEqual(snapshot["step3_backup_profiles"]["task2_g1_backup"]["candidate"], "G1")
        self.assertNotIn("step3_performance_ladder", snapshot)
        g0 = snapshot["step3_backup_profiles"]["task2_g0_backup"]
        self.assertEqual((g0["batch_size"], g0["per_gpu_batch_size"]), (1024, 512))
        self.assertTrue(g0["backup_only"])
        self.assertTrue(g0["manual_selection_required"])
        self.assertFalse(g0["formal_allowed"])
        g2 = snapshot["step3_exploration_profiles"]["task2_g2_effective_pool_2048"]
        self.assertTrue(g2["probe_only"])
        self.assertFalse(g2["formal_allowed"])
        self.assertTrue(g2["exploration_only"])
        self.assertEqual((g2["batch_size"], g2["per_gpu_batch_size"]), (2048, 1024))
        source = (_REPO_ROOT / "code" / "executors" / "step3_train_core.py").read_text(encoding="utf-8")
        self.assertNotIn("_ddp_no_sync_model", source)
        self.assertNotIn(".no_sync(", source)

    def test_step3_no_accum_formula_cases_and_retired_keys_fail_fast(self) -> None:
        for batch, per_gpu in ((1024, 512), (1536, 768), (2048, 1024), (3072, 1536), (4096, 2048)):
            with self.subTest(batch=batch, per_gpu=per_gpu):
                cfg, _sources, snapshot = _resolve_step3_task(
                    set_overrides=[f"step3.train.batch_size={batch}", f"step3.train.per_gpu_batch_size={per_gpu}"]
                )
                self.assertEqual(cfg.train_batch_size, batch)
                self.assertEqual(snapshot["train"]["step3_batch_formula"], "global_batch_size = per_gpu_batch_size * ddp_world_size")
        with self.assertRaisesRegex(OneControlConfigError, "removed in ODCR no-accum"):
            _resolve_step3_task(set_overrides=["step3.train.batch_size=1025", "step3.train.per_gpu_batch_size=512", "step3.train.grad_accum=1"])
        with self.assertRaisesRegex(OneControlConfigError, "batch formula failed"):
            _resolve_step3_task(set_overrides=["step3.train.batch_size=1025", "step3.train.per_gpu_batch_size=512"])
        with self.assertRaisesRegex(OneControlConfigError, "removed in ODCR no-accum"):
            _resolve_step3_task(set_overrides=["step3.train.gradient_accumulation_steps=1"])
        with self.assertRaisesRegex(OneControlConfigError, "removed in ODCR no-accum"):
            _resolve_step3_task(set_overrides=["step3.train.accumulate_grad_batches=1"])
        with self.assertRaisesRegex(OneControlConfigError, "requires step3.cross_rank_structured_gather.enabled=true"):
            _resolve_step3_task(set_overrides=["step3.cross_rank_structured_gather.enabled=false"])
        cfg, _sources, snap = _resolve_step3_task(
            set_overrides=[
                "step3.cross_rank_structured_gather.enabled=false",
                "step3.cross_rank_structured_gather.diagnostic_allow_disabled=true",
            ]
        )
        self.assertFalse(snap["step3_cross_rank_structured_gather"]["enabled"])
        self.assertEqual(cfg.task_profile_id, "task2_strong_forward_g1s")

    def test_step3_hardware_profile_missing_max_parallel_cpu_fails_fast(self) -> None:
        raw = load_yaml_config(_REPO_ROOT / "configs" / "odcr.yaml")
        del raw["hardware"]["profiles"]["default"]["max_parallel_cpu"]
        with tempfile.TemporaryDirectory() as tmp_raw:
            cfg_path = Path(tmp_raw) / "odcr_missing_max_parallel.yaml"
            cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(OneControlConfigError, "max_parallel_cpu"):
                resolve_config(
                    config_path=cfg_path,
                    command="step3",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    run_id="auto",
                    mode="full",
                )

    def test_step3_hardware_profile_rejects_cpu_worker_oversubscription(self) -> None:
        raw = load_yaml_config(_REPO_ROOT / "configs" / "odcr.yaml")
        raw["hardware"]["profiles"]["default"]["max_parallel_cpu"] = 12
        raw["hardware"]["profiles"]["default"]["dataloader_num_workers_train"] = 8
        with tempfile.TemporaryDirectory() as tmp_raw:
            cfg_path = Path(tmp_raw) / "odcr_oversubscribe_workers.yaml"
            cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(OneControlConfigError, "dataloader_num_workers_train"):
                resolve_config(
                    config_path=cfg_path,
                    command="step3",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    run_id="auto",
                    mode="full",
                )

    def test_step3_hardware_profile_rejects_tokenizer_num_proc_oversubscription(self) -> None:
        raw = load_yaml_config(_REPO_ROOT / "configs" / "odcr.yaml")
        raw["hardware"]["profiles"]["default"]["max_parallel_cpu"] = 12
        raw["hardware"]["profiles"]["default"]["num_proc"] = 11
        with tempfile.TemporaryDirectory() as tmp_raw:
            cfg_path = Path(tmp_raw) / "odcr_oversubscribe_num_proc.yaml"
            cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(OneControlConfigError, "num_proc"):
                resolve_config(
                    config_path=cfg_path,
                    command="step3",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    run_id="auto",
                    mode="full",
                )

    def test_step3_precision_cannot_fall_back_to_helper_default(self) -> None:
        raw = load_yaml_config(_REPO_ROOT / "configs" / "odcr.yaml")
        del raw["step3"]["train"]["backend"]["train_precision"]
        with tempfile.TemporaryDirectory() as tmp_raw:
            cfg_path = Path(tmp_raw) / "odcr_missing_step3_precision.yaml"
            cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(OneControlConfigError, "step3.train.backend.train_precision"):
                resolve_config(
                    config_path=cfg_path,
                    command="step3",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    run_id="auto",
                    mode="full",
                )

    def test_step3_rejects_unknown_legacy_control(self) -> None:
        with self.assertRaisesRegex(OneControlConfigError, "unsupported key"):
            _resolve_step3_task(set_overrides=["step3.train.adv=0.9"])

    def test_step3_manifest_and_banner_use_structured_semantics(self) -> None:
        cfg, _, _ = _resolve_step3_task()
        manifest = build_run_manifest(cfg)

        self.assertEqual(_stage_label("step3"), "step3（结构化 shared/specific 解耦）")
        self.assertEqual(manifest["stage"], "step3_structured_disentanglement")
        self.assertNotIn("eta", manifest["hyperparameters"])
        self.assertNotIn("adv", manifest["hyperparameters"])
        self.assertNotIn("coef", manifest["hyperparameters"])
        self.assertEqual(manifest["hyperparameters"]["optimizer"]["name"], "adamw")

    def test_retired_typed_bridge_files_are_deleted(self) -> None:
        for name in ("step3_runtime.py", "step3_registry.py"):
            with self.subTest(name=name):
                self.assertFalse((_REPO_ROOT / "code" / "odcr_core" / name).exists())

        runners_source = (_REPO_ROOT / "code" / "odcr_core" / "runners.py").read_text(encoding="utf-8")
        self.assertNotIn("step3_runtime", runners_source)
        self.assertNotIn("step3_registry", runners_source)
        self.assertNotIn("instantiate_" + "step3_preset", runners_source)


if __name__ == "__main__":
    unittest.main()
