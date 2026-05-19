from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

_CODE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

import odcr_core.config_resolver as config_resolver  # noqa: E402
from executors import step5_engine  # noqa: E402
from odcr_core import step5_runtime_probe  # noqa: E402
from odcr_core.index_contract import (  # noqa: E402
    GLOBAL_COL_ITEM,
    GLOBAL_COL_USER,
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    ODCR_ROUTING_TRAIN_CSV,
    STEP4_RCR_REQUIRED_COLUMNS,
    refresh_index_contract_train_csv_fingerprint,
)
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST  # noqa: E402
from odcr_core.step5_export_loader import (  # noqa: E402
    STEP5_TRAIN_LOADER_COLUMNS,
    STEP5_TRAIN_VALIDATION_COLUMNS,
    Step5ExportLoaderError,
    load_step5_train_table,
    validate_step5_export_source,
)
from helpers.fixtures import write_json, write_step4_upstream_fixture  # noqa: E402


class _Cfg:
    def __init__(self, *, pin_memory: bool | None = None, non_blocking_h2d: bool | None = None) -> None:
        if pin_memory is not None:
            self.pin_memory = pin_memory
        if non_blocking_h2d is not None:
            self.non_blocking_h2d = non_blocking_h2d


class _FakeTensor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def to(self, device, non_blocking=False):  # noqa: ANN001
        self.calls.append({"device": device, "non_blocking": non_blocking})
        return self


def _row(i: int = 0) -> dict:
    out = {col: 1 for col in STEP5_TRAIN_LOADER_COLUMNS}
    out.update(
        {
            GLOBAL_COL_USER: i,
            GLOBAL_COL_ITEM: i + 10,
            "rating": 4.0,
            "domain": "auxiliary" if i % 2 else "target",
            "clean_text": f"clean {i}",
            "sample_id": i,
            "sample_origin": "aux_cf",
            "entropy_score": 0.1,
            "uncertainty_score": 0.2,
            "confidence_bucket": 2,
            "route_scorer": 1 if i % 2 == 0 else 0,
            "route_explainer": 1,
            "sample_weight_hint": 1.0,
            "content_evidence": "content",
            "style_evidence": "style",
            "domain_style_anchor": "anchor",
            "local_style_residual_hint": "hint",
            "polarity_anchor": "positive",
            "content_anchor_score": 0.8,
            "style_anchor_score": 0.7,
            "evidence_quality_prior": 0.9,
            "cf_reliability_score": 0.95,
            "content_retention_score": 0.9,
            "style_shift_score": 0.5,
            "rating_stability_score": 0.85,
            "text_quality_score": 1.0,
            "route_reason_scorer": "rcr_scorer_clean",
            "route_reason_explainer": "rcr_explainer_rich",
            "preprocess_route_scorer_prior": 0,
            "preprocess_route_explainer_prior": 0,
            "train_keep": 1,
        }
    )
    return out


def _write_export(root: Path, *, drop: str | None = None, bad_sha: bool = False) -> tuple[Path, Path, Path, dict]:
    run = root / "runs" / "step4" / "task2" / "1"
    run.mkdir(parents=True)
    export = run / ODCR_ROUTING_TRAIN_CSV
    rows = [_row(i) for i in range(4)]
    df = pd.DataFrame(rows)
    if drop:
        df = df.drop(columns=[drop])
    df.to_csv(export, index=False)
    contract = refresh_index_contract_train_csv_fingerprint(
        {
            "schema_version": INDEX_CONTRACT_SCHEMA_VERSION,
            "embed_dim": 1024,
            "nuser_global": 100,
            "nitem_global": 100,
            "backbones": {
                "sentence_embed": {
                    "model_id": "fixture",
                    "local_dir": "/tmp/fixture",
                    "family": "bge_large_en",
                    "hidden_size": 1024,
                    "dual_channel": True,
                }
            },
            "step4_export_lineage": {"lineage_hash": "abc"},
        },
        str(export),
    )
    if bad_sha:
        contract["fingerprints"]["train_csv"]["sha256"] = "0" * 64
    contract_path = run / INDEX_CONTRACT_FILENAME
    write_json(contract_path, contract)
    manifest_path = run / STEP4_EXPORT_MANIFEST
    write_json(
        manifest_path,
        {
            "schema_version": "odcr_step4_train_table/1.2",
            "row_counts": {"total_rows": len(df), "by_sample_origin": {"aux_cf": len(df)}},
            "step4_export_lineage": {"lineage_hash": "abc"},
        },
    )
    return export, contract_path, manifest_path, contract


class TestStep5NonGpuFix(unittest.TestCase):
    def test_step5_data_pipeline_one_control_and_cpu_budget_guard(self) -> None:
        cfg = config_resolver.load_yaml_config(_REPO_ROOT / "configs" / "odcr.yaml")
        resolved = config_resolver._resolve_step5_data_pipeline_config(cfg, ddp_world_size=2)
        self.assertTrue(resolved["sample_plan_enabled"])
        self.assertTrue(resolved["bounded_token_cache_enabled"])
        self.assertEqual(resolved["workers_per_rank_candidates"], [4, 5])
        self.assertEqual(resolved["prefetch_factor_candidates"], [4, 6, 8])
        self.assertEqual(cfg["hardware"]["profiles"]["default"]["dataloader_prefetch_factor_train"], 4)
        self.assertEqual(cfg["step5"]["batch_candidates"]["selected_default"], "B224")
        batch_cfg = config_resolver._resolve_step5_batch_candidates_config(cfg)
        self.assertEqual(batch_cfg["selected_default"], "B224")
        required_batches = {
            item["id"]: (item["per_gpu_batch_size"], item["global_batch_size"])
            for item in batch_cfg["candidates"]
        }
        self.assertEqual(
            {key: required_batches[key] for key in ("B96", "B128", "B160", "B192")},
            {"B96": (96, 192), "B128": (128, 256), "B160": (160, 320), "B192": (192, 384)},
        )
        e4_cfg = config_resolver._resolve_step5_e4_bounded_config(cfg)
        e4_batches = {item["id"]: (item["per_gpu_batch_size"], item["global_batch_size"]) for item in e4_cfg["batch_candidates"]}
        self.assertEqual(
            {key: e4_batches[key] for key in ("B96", "B128", "B160", "B192")},
            {"B96": (96, 192), "B128": (128, 256), "B160": (160, 320), "B192": (192, 384)},
        )
        sampler_cfg = config_resolver._resolve_step5_sampler_config(cfg)
        self.assertTrue(sampler_cfg["auto_budget"]["enabled"])
        self.assertNotIn("effective_samples_per_epoch_candidates", sampler_cfg["step5A"])
        self.assertEqual(sampler_cfg["epochs"]["pilot_fraction_candidates"], [0.1, 0.2, 0.5])
        tuning_cfg = config_resolver._resolve_step5_tuning_config(cfg, batch_cfg)
        self.assertEqual(tuning_cfg["batch_candidate"], "B224")
        self.assertEqual(tuning_cfg["fallback_batch_candidate"], "B192")
        self.assertIn(0.0005, tuning_cfg["lr_candidates"])
        self.assertTrue(all(item["ok"] for item in resolved["cpu_budget_formulas"]))
        bad = json.loads(json.dumps(cfg))
        bad["step5"]["data_pipeline"]["workers_per_rank_candidates"] = [6]
        with self.assertRaisesRegex(config_resolver.OneControlConfigError, "CPU budget"):
            config_resolver._resolve_step5_data_pipeline_config(bad, ddp_world_size=2)

    def test_pipeline_timing_and_gpu_util_contract_is_explicit(self) -> None:
        source = (_REPO_ROOT / "code" / "odcr_core" / "step5_runtime_probe.py").read_text(encoding="utf-8")
        for field in (
            "sampler_plan_time",
            "parquet_read_time",
            "prompt_build_time",
            "tokenize_time",
            "collate_time",
            "dataloader_wait_time",
            "dataloader_queue_wait",
            "h2d_time",
            "forward_time",
            "backward_time",
            "optimizer_time",
            "total_step_time",
            "data_wait_ratio",
            "gpu_util_available",
            "per_rank_worker_count",
        ):
            self.assertIn(f'"{field}"', source)
        unavailable = step5_runtime_probe._gpu_util_summary(
            [{"available": False, "error": "nvidia-smi unavailable", "gpus": []}]
        )
        self.assertFalse(unavailable["available"])
        self.assertIsNone(unavailable["mean"])
        self.assertIsNone(unavailable["p50"])
        self.assertIsNone(unavailable["p95"])

    def test_bounded_token_cache_manifest_stays_outside_formal_namespace(self) -> None:
        class FakeProcessor:
            def __call__(self, sample):  # noqa: ANN001
                idx = int(sample["sample_id"])
                return {
                    "user_idx": torch.tensor(idx),
                    "item_idx": torch.tensor(idx + 1),
                    "rating": torch.tensor(4.0),
                    "explanation_idx": torch.tensor([1, 2, 3]),
                    "domain_idx": torch.tensor(1),
                    "sample_id": torch.tensor(idx),
                    "exp_sample_weight": torch.tensor(1.0),
                    "route_scorer_mask": torch.tensor(1.0),
                    "route_explainer_mask": torch.tensor(1.0),
                    "entropy_score": torch.tensor(0.1),
                    "uncertainty_score": torch.tensor(0.2),
                    "confidence_bucket": torch.tensor(2.0),
                    "content_anchor_score": torch.tensor(0.8),
                    "style_anchor_score": torch.tensor(0.7),
                    "evidence_features": torch.zeros(8),
                    "content_evidence_ids": torch.tensor([1]),
                    "style_evidence_ids": torch.tensor([1]),
                    "domain_style_anchor_ids": torch.tensor([1]),
                    "local_style_hint_ids": torch.tensor([1]),
                    "polarity_ids": torch.tensor([1]),
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = pd.DataFrame([_row(0), _row(1)])
            rows, manifest, _elapsed = step5_runtime_probe._write_bounded_token_cache(
                output_dir=root / "AI_analysis" / "test_artifacts" / "bounded_probe",
                rank=0,
                train_df=df,
                processor=FakeProcessor(),
                source_table={"scope": "unit"},
                sample_plan_manifest={"plan_hash": "abc"},
            )
            self.assertEqual(len(rows), 2)
            self.assertTrue(manifest["hot_path_tokenize_removed"])
            self.assertTrue(Path(manifest["cache_file"]).is_file())
            self.assertFalse(manifest["formal_namespace_write"])
            self.assertFalse((root / "runs" / "step5" / "task2" / "latest.json").exists())

    def test_runtime_transfer_knobs_follow_final_cfg(self) -> None:
        self.assertFalse(step5_engine._step5_pin_memory_from_final_cfg(_Cfg(pin_memory=False)))
        self.assertTrue(step5_engine._step5_pin_memory_from_final_cfg(_Cfg(pin_memory=True)))
        self.assertFalse(step5_engine._step5_non_blocking_h2d_from_final_cfg(_Cfg(non_blocking_h2d=False)))
        self.assertTrue(step5_engine._step5_non_blocking_h2d_from_final_cfg(_Cfg(non_blocking_h2d=True)))
        with self.assertRaisesRegex(RuntimeError, "FinalTrainingConfig.pin_memory"):
            step5_engine._step5_pin_memory_from_final_cfg(_Cfg())
        with self.assertRaisesRegex(RuntimeError, "FinalTrainingConfig.non_blocking_h2d"):
            step5_engine._step5_non_blocking_h2d_from_final_cfg(_Cfg())

    def test_h2d_helper_uses_explicit_non_blocking_value(self) -> None:
        false_tensor = _FakeTensor()
        self.assertIs(step5_engine._move_to_device(false_tensor, "cuda:0", non_blocking=False), false_tensor)
        self.assertEqual(false_tensor.calls, [{"device": "cuda:0", "non_blocking": False}])

        true_tensor = _FakeTensor()
        self.assertIs(step5_engine._move_to_device(true_tensor, "cuda:0", non_blocking=True), true_tensor)
        self.assertEqual(true_tensor.calls, [{"device": "cuda:0", "non_blocking": True}])

    def test_step5_dataloaders_use_resolved_pin_memory_symbol(self) -> None:
        source = (_REPO_ROOT / "code" / "executors" / "step5_engine.py").read_text(encoding="utf-8")
        self.assertIn("pin_memory = _step5_pin_memory_from_final_cfg(final_cfg)", source)
        self.assertGreaterEqual(len(re.findall(r"pin_memory=pin_memory", source)), 3)
        self.assertIsNone(re.search(r"pin_memory\\s*=\\s*torch\\.cuda\\.is_available", source))
        self.assertIsNone(re.search(r"pin_memory=torch\\.cuda\\.is_available", source))
        self.assertIsNone(re.search(r"non_blocking\\s*=\\s*True", source))

    def test_step5_optimizer_excludes_frozen_params(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        for param in model[0].parameters():
            param.requires_grad_(False)
        params = step5_engine._step5_trainable_parameters(model)
        optimizer = torch.optim.Adam(params, lr=1e-3)
        self.assertTrue(step5_engine._step5_optimizer_excludes_frozen(model, optimizer))
        self.assertEqual(sum(p.numel() for p in params), sum(p.numel() for p in model[1].parameters()))

    def test_gradient_checkpointing_and_use_cache_training_policy(self) -> None:
        class FakeFlan(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = type("Cfg", (), {"use_cache": True})()
                self.is_gradient_checkpointing = False

            def gradient_checkpointing_enable(self, **kwargs):
                self.gradient_checkpointing_kwargs = kwargs
                self.is_gradient_checkpointing = True

            def enable_input_require_grads(self):
                self.input_require_grads_enabled = True

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.flan_explainer = FakeFlan()

        cfg = _Cfg()
        cfg.step5_memory_truth_config_json = json.dumps(
            {
                "reserved_diagnostic_only": True,
                "reject_on_reserved": False,
                "gradient_checkpointing_enabled": True,
                "gradient_checkpointing_reentrant_policy": "non_reentrant",
                "disable_use_cache_during_training": True,
            }
        )
        model = Wrapper()
        meta = step5_engine._configure_step5_training_memory_policy(model, cfg)
        self.assertTrue(meta["gradient_checkpointing_enabled"])
        self.assertEqual(meta["gradient_checkpointing_reentrant_policy"], "non_reentrant")
        self.assertEqual(meta["requested_policy"], "non_reentrant")
        self.assertEqual(meta["actual_policy"], "non_reentrant")
        self.assertTrue(meta["api_supports_gradient_checkpointing_kwargs"])
        self.assertTrue(meta["official_api_used"])
        self.assertFalse(meta["manual_non_reentrant_patch_used"])
        self.assertIs(meta["use_reentrant_effective"], False)
        self.assertEqual(
            model.flan_explainer.gradient_checkpointing_kwargs,
            {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        )
        self.assertTrue(meta["use_cache_training_disabled"])
        self.assertFalse(model.flan_explainer.config.use_cache)

    def test_non_reentrant_checkpointing_manual_patch_when_transformers_lacks_kwargs(self) -> None:
        class FakeCheckpointBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gradient_checkpointing = False
                self._gradient_checkpointing_func = lambda *args, **kwargs: None

        class FakeOldFlan(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = type("Cfg", (), {"use_cache": True})()
                self.is_gradient_checkpointing = False
                self.block = FakeCheckpointBlock()

            def gradient_checkpointing_enable(self):
                self.is_gradient_checkpointing = True

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.flan_explainer = FakeOldFlan()

        cfg = _Cfg()
        cfg.step5_memory_truth_config_json = json.dumps(
            {
                "reserved_diagnostic_only": True,
                "reject_on_reserved": False,
                "gradient_checkpointing_enabled": True,
                "gradient_checkpointing_reentrant_policy": "non_reentrant",
                "disable_use_cache_during_training": True,
            }
        )
        model = Wrapper()
        meta = step5_engine._configure_step5_training_memory_policy(model, cfg)
        self.assertTrue(meta["gradient_checkpointing_enabled"])
        self.assertEqual(meta["actual_policy"], "non_reentrant")
        self.assertFalse(meta["api_supports_gradient_checkpointing_kwargs"])
        self.assertFalse(meta["official_api_used"])
        self.assertTrue(meta["manual_non_reentrant_patch_used"])
        self.assertEqual(meta["patched_module_count"], 1)
        self.assertIs(meta["use_reentrant_effective"], False)
        patched = model.flan_explainer.block._gradient_checkpointing_func
        self.assertEqual(getattr(patched, "keywords", {}).get("use_reentrant"), False)
        self.assertTrue(model.flan_explainer.block.gradient_checkpointing)

    def test_non_reentrant_checkpointing_disables_when_no_safe_patch_exists(self) -> None:
        class FakeUnsupportedFlan(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = type("Cfg", (), {"use_cache": True})()
                self.is_gradient_checkpointing = False

            def gradient_checkpointing_enable(self):
                self.is_gradient_checkpointing = True

            def gradient_checkpointing_disable(self):
                self.is_gradient_checkpointing = False

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.flan_explainer = FakeUnsupportedFlan()

        cfg = _Cfg()
        cfg.step5_memory_truth_config_json = json.dumps(
            {
                "reserved_diagnostic_only": True,
                "reject_on_reserved": False,
                "gradient_checkpointing_enabled": True,
                "gradient_checkpointing_reentrant_policy": "non_reentrant",
                "disable_use_cache_during_training": True,
            }
        )
        meta = step5_engine._configure_step5_training_memory_policy(Wrapper(), cfg)
        self.assertFalse(meta["gradient_checkpointing_enabled"])
        self.assertEqual(meta["actual_policy"], "disabled")
        self.assertEqual(meta["gradient_checkpointing_reentrant_policy"], "disabled")
        self.assertFalse(meta["manual_non_reentrant_patch_used"])
        self.assertEqual(meta["patched_module_count"], 0)
        self.assertIsNone(meta["use_reentrant_effective"])
        self.assertIn("could_not_be_verified", meta["fallback_reason"])

    def test_flan_training_forward_disables_cache_and_hidden_state_retention(self) -> None:
        source = (_REPO_ROOT / "code" / "executors" / "step5_engine.py").read_text(encoding="utf-8")
        self.assertIn("use_cache=False", source)
        self.assertIn("output_hidden_states=False", source)
        self.assertNotIn("output_hidden_states=True", source)

    def test_step5_profile_evidence_tensors_are_frozen(self) -> None:
        source = (_REPO_ROOT / "code" / "executors" / "step5_engine.py").read_text(encoding="utf-8")
        for name in (
            "domain_content_profiles",
            "domain_style_profiles",
            "user_content_profiles",
            "user_style_profiles",
            "item_content_profiles",
            "item_style_profiles",
        ):
            self.assertIn(f"self.{name} = nn.Parameter({name}, requires_grad=False)", source)
            self.assertNotIn(f"uniform_(self.{name}.data", source)

    def test_resolver_overrides_hardware_transfer_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step4_upstream_fixture(repo, task_id=4, run_id="1")
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                cfg, _, snapshot = config_resolver.resolve_config(
                    config_path=_REPO_ROOT / "configs" / "odcr.yaml",
                    command="step5",
                    task_id=4,
                    set_overrides=[
                        "hardware.profiles.default.pin_memory=false",
                        "hardware.profiles.default.non_blocking_h2d=false",
                    ],
                    dry_run=True,
                    from_step4="1",
                    eval_profile="balanced_2gpu",
                    mode="full",
                )
            finally:
                config_resolver._REPO_ROOT = old_root
        self.assertFalse(cfg.pin_memory)
        self.assertFalse(cfg.non_blocking_h2d)
        self.assertFalse(snapshot["hardware"]["pin_memory"])
        self.assertFalse(snapshot["hardware"]["non_blocking_h2d"])

    def test_route_mask_collate_missing_fields_fail_fast(self) -> None:
        item = {
            "user_idx": torch.tensor(0),
            "item_idx": torch.tensor(1),
            "rating": torch.tensor(4.0),
            "explanation_idx": torch.tensor([1, 2]),
            "domain_idx": torch.tensor(1),
            "sample_id": torch.tensor(0),
            "exp_sample_weight": torch.tensor(1.0),
            "route_scorer_mask": torch.tensor(1.0),
            "route_explainer_mask": torch.tensor(1.0),
            "entropy_score": torch.tensor(0.1),
            "uncertainty_score": torch.tensor(0.2),
            "confidence_bucket": torch.tensor(2.0),
            "content_anchor_score": torch.tensor(0.8),
            "style_anchor_score": torch.tensor(0.7),
            "evidence_features": torch.zeros(8),
            "content_evidence_ids": torch.tensor([1]),
            "style_evidence_ids": torch.tensor([1]),
            "domain_style_anchor_ids": torch.tensor([1]),
            "local_style_hint_ids": torch.tensor([1]),
            "polarity_ids": torch.tensor([1]),
        }
        for key in (
            "route_scorer_mask",
            "route_explainer_mask",
            "exp_sample_weight",
            "uncertainty_score",
            "confidence_bucket",
        ):
            bad = dict(item)
            bad.pop(key)
            with self.subTest(key=key):
                with self.assertRaisesRegex(KeyError, re.escape(key)):
                    step5_engine._step5_collate_dynamic([bad], dynamic_padding=True, fixed_max_length=8)

    def test_route_mask_default_get_removed_from_collate_source(self) -> None:
        source = (_REPO_ROOT / "code" / "executors" / "step5_engine.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r'get\("route_scorer_mask",\s*1\.0\)', source))
        self.assertIsNone(re.search(r'get\("route_explainer_mask",\s*1\.0\)', source))

    def test_validate_only_and_bounded_do_not_full_read_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export, contract_path, manifest_path, contract = _write_export(Path(tmp))
            real_read_csv = pd.read_csv
            calls: list[dict] = []

            def tracked(*args, **kwargs):
                calls.append(dict(kwargs))
                return real_read_csv(*args, **kwargs)

            with mock.patch("odcr_core.step5_export_loader.pd.read_csv", side_effect=tracked):
                validate_only = load_step5_train_table(
                    export,
                    cache_root=Path(tmp) / "cache",
                    index_contract_path=contract_path,
                    manifest_path=manifest_path,
                    index_contract=contract,
                    mode="validate_only",
                    verify_sha256=False,
                )
            self.assertFalse(validate_only.stats["full_csv_parse"])
            self.assertTrue(calls)
            self.assertTrue(all(("nrows" in call and call["nrows"] is not None) for call in calls))

            calls.clear()
            with mock.patch("odcr_core.step5_export_loader.pd.read_csv", side_effect=tracked):
                bounded = load_step5_train_table(
                    export,
                    cache_root=Path(tmp) / "cache",
                    index_contract_path=contract_path,
                    manifest_path=manifest_path,
                    index_contract=contract,
                    mode="bounded",
                    bounded_max_rows=2,
                    verify_sha256=False,
                    validation_ctx={"task_id": 2},
                )
            self.assertFalse(bounded.stats["full_csv_parse"])
            self.assertLessEqual(bounded.raw_row_count, 2)
            self.assertTrue(any(call.get("nrows") == 2 for call in calls))
            self.assertFalse((Path(tmp) / "runs" / "step5").exists())

    def test_missing_required_export_fields_and_sha_mismatch_fail_fast(self) -> None:
        for field in ("route_scorer", "route_explainer", "sample_weight_hint", "uncertainty_score", "confidence_bucket"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                export, contract_path, manifest_path, contract = _write_export(Path(tmp), drop=field)
                with self.assertRaisesRegex(Step5ExportLoaderError, field):
                    validate_step5_export_source(
                        export,
                        index_contract_path=contract_path,
                        manifest_path=manifest_path,
                        index_contract=contract,
                        required_columns=STEP5_TRAIN_VALIDATION_COLUMNS,
                    )
        with tempfile.TemporaryDirectory() as tmp:
            export, contract_path, manifest_path, contract = _write_export(Path(tmp), bad_sha=True)
            with self.assertRaisesRegex(Step5ExportLoaderError, "sha256 mismatch"):
                validate_step5_export_source(
                    export,
                    index_contract_path=contract_path,
                    manifest_path=manifest_path,
                    index_contract=contract,
                    required_columns=STEP5_TRAIN_VALIDATION_COLUMNS,
                    verify_sha256=True,
                )

    def test_formal_cache_manifest_mismatch_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export, contract_path, manifest_path, contract = _write_export(root)
            first = load_step5_train_table(
                export,
                cache_root=root / "cache" / "step5_export_loader",
                index_contract_path=contract_path,
                manifest_path=manifest_path,
                index_contract=contract,
                mode="formal_train",
                chunk_rows=2,
                verify_sha256=True,
                validation_ctx={"task_id": 2},
            )
            self.assertFalse(first.cache_hit)
            manifest = json.loads(first.cache_manifest_path.read_text(encoding="utf-8"))
            manifest["source"]["expected_sha256"] = "bad"
            first.cache_manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(Step5ExportLoaderError, "source fingerprint mismatch"):
                load_step5_train_table(
                    export,
                    cache_root=root / "cache" / "step5_export_loader",
                    index_contract_path=contract_path,
                    manifest_path=manifest_path,
                    index_contract=contract,
                    mode="formal_train",
                    chunk_rows=2,
                    verify_sha256=True,
                    stale_policy="fail_fast",
                    validation_ctx={"task_id": 2},
                )
            self.assertFalse((root / "runs" / "step5").exists())


if __name__ == "__main__":
    unittest.main()
