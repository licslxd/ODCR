"""preprocess_b: BF16 / TF32 config and runtime command regression."""
import os
import sys
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

import compute_embeddings as ce  # noqa: E402
from configs.preprocess.preprocess_b import build_preprocess_b_stage  # noqa: E402
from odcr_core.config_resolver import build_preprocess_config  # noqa: E402
from odcr_core.preprocess_registry import render_internal_preprocess_config  # noqa: E402
from odcr_core.preprocess_runtime import PreprocessRuntime  # noqa: E402
from odcr_core.preprocess_schema import apply_preprocess_cli_overrides  # noqa: E402


def _preprocess_b_cli_args(**overrides):
    base = {
        "datasets": None,
        "resume": None,
        "skip_completed": None,
        "verify_only": False,
        "dry_run": False,
        "workers": None,
        "force_datasets": None,
        "gpu_ids": None,
        "embed_batch_size": None,
        "read_chunk_rows": None,
        "group_shard_size": None,
        "grouped_text_cache_enabled": None,
        "grouped_text_cache_dir": None,
        "grouped_text_cache_version": None,
        "bf16_enabled": None,
        "tf32_enabled": None,
        "verify_sample_size": None,
        "verify_seed": None,
        "verify_user_indices": None,
        "verify_item_indices": None,
        "chunk_batch_size": None,
        "tokenizer_hotpath_enabled": None,
        "token_window_cache_enabled": None,
        "token_window_cache_dir": None,
        "token_window_cache_version": None,
        "token_window_cache_shard_size": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestPreprocessBPrecision(unittest.TestCase):
    def test_resolve_precision_config_cpu_disables_autocast(self) -> None:
        cfg = ce._resolve_precision_config(
            SimpleNamespace(bf16_enabled=True, tf32_enabled=True),
            torch.device("cpu"),
        )
        self.assertTrue(cfg.bf16_enabled)
        self.assertTrue(cfg.tf32_enabled)
        self.assertFalse(cfg.autocast_enabled)
        self.assertEqual(cfg.autocast_dtype, torch.bfloat16)

    def test_preprocess_b_cli_overrides_accept_precision_flags(self) -> None:
        preset = build_preprocess_b_stage(
            preset_name="unit_test_preprocess_b",
            description="unit test",
        )
        updated = apply_preprocess_cli_overrides(
            preset,
            _preprocess_b_cli_args(bf16_enabled=False, tf32_enabled=False),
        )
        self.assertFalse(updated.bf16_enabled)
        self.assertFalse(updated.tf32_enabled)

    def test_runtime_command_propagates_precision_flags(self) -> None:
        preset = build_preprocess_config(
            config_path=os.path.join(os.path.dirname(_CODE_DIR), "configs", "odcr.yaml"),
            stage_letter="b",
            set_overrides=[],
            dry_run=True,
        )
        cfg = replace(preset, bf16_enabled=False, tf32_enabled=True)
        cfg = replace(cfg, resolved=replace(cfg.resolved, bf16=False, tf32=True))
        runtime = PreprocessRuntime(cfg)
        cmd = runtime._gpu_dataset_command(cfg, dataset="TripAdvisor", gpu_id=1)
        self.assertIn("--no-bf16", cmd)
        self.assertIn("--tf32", cmd)
        self.assertNotIn("--bf16", cmd)
        self.assertNotIn("--no-tf32", cmd)
        self.assertIn("--data-dir", cmd)
        self.assertIn(str(runtime.data_root), cmd)
        self.assertIn("--sentence-embed-model", cmd)
        self.assertIn(str(runtime.sentence_embed_model_path), cmd)

    def test_preprocess_source_table_records_key_metadata(self) -> None:
        cfg_b = build_preprocess_config(
            config_path=os.path.join(os.path.dirname(_CODE_DIR), "configs", "odcr.yaml"),
            stage_letter="b",
            set_overrides=[],
            dry_run=True,
        )
        runtime_b = PreprocessRuntime(cfg_b)
        source_b = runtime_b._source_table_payload()
        keys_b = {record["key"] for record in source_b["records"]}
        for key in (
            "preprocess.b.embed_batch_size",
            "preprocess.b.read_chunk_rows",
            "preprocess.b.group_shard_size",
            "preprocess.b.cache_key_fields",
            "preprocess.b.cache_stale_policy",
            "sentence_embed_model_path",
            "embed_dim",
            "local_files_only",
        ):
            self.assertIn(key, keys_b)
        self.assertEqual(source_b["source_policy"], "One-Control resolved payload and runtime transport only; no child YAML/env fallback")
        self.assertIn("canonical_column_hash", runtime_b.stage_metadata)
        self.assertIn("cache_key_fields", runtime_b.stage_metadata["stage_specific"])

        cfg_c = build_preprocess_config(
            config_path=os.path.join(os.path.dirname(_CODE_DIR), "configs", "odcr.yaml"),
            stage_letter="c",
            set_overrides=[],
            dry_run=True,
        )
        runtime_c = PreprocessRuntime(cfg_c)
        keys_c = {record["key"] for record in runtime_c._source_table_payload()["records"]}
        for key in (
            "preprocess.c.chunk_batch_size",
            "preprocess.c.token_window_cache_version",
            "preprocess.c.token_window_cache_shard_size",
            "preprocess.c.cache_key_fields",
            "preprocess.c.cache_stale_policy",
        ):
            self.assertIn(key, keys_c)
        self.assertEqual(cfg_c.token_window_cache_shard_size, 4096)
        self.assertEqual(runtime_c.stage_metadata["stage_specific"]["token_window_cache_shard_size"], 4096)

    def test_preprocess_a_metadata_records_internal_chunk_size(self) -> None:
        cfg = build_preprocess_config(
            config_path=os.path.join(os.path.dirname(_CODE_DIR), "configs", "odcr.yaml"),
            stage_letter="a",
            set_overrides=[],
            dry_run=True,
        )
        runtime = PreprocessRuntime(cfg)
        self.assertEqual(runtime.stage_metadata["stage_specific"]["canonical_asset_chunk_size"], 50_000)
        source = runtime._source_table_payload()
        keys = {record["key"] for record in source["records"]}
        self.assertIn("preprocess.a.canonical_asset_chunk_size", keys)
        self.assertIn("preprocess.a.csv_header_hash", keys)

    def test_internal_config_includes_preprocess_b_precision_defaults(self) -> None:
        rendered = render_internal_preprocess_config("preprocess_b_a100_2gpu")
        config = rendered["config"]
        self.assertTrue(config["bf16_enabled"])
        self.assertTrue(config["tf32_enabled"])


if __name__ == "__main__":
    unittest.main()
