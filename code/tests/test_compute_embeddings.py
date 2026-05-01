"""compute_embeddings：spec-by-spec cold path / probe helpers / grouped-text semantics regression."""
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

import compute_embeddings as ce  # noqa: E402


class _DummyTokenizer:
    name_or_path = "dummy-tokenizer"
    is_fast = True
    model_max_length = 512

    def __len__(self) -> int:
        return 321

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2 if not pair else 3


class _DummyModel:
    config = SimpleNamespace(_name_or_path="dummy-model", model_type="dummy", hidden_size=1024)


class TestComputeEmbeddings(unittest.TestCase):
    def _cache_identity(self) -> tuple[_DummyTokenizer, _DummyModel]:
        return _DummyTokenizer(), _DummyModel()

    def _install_resolved_runtime_env(self, root: str) -> None:
        keys = (
            "ODCR_RESOLVED_DATA_DIR",
            "ODCR_RESOLVED_MODELS_DIR",
            "ODCR_RESOLVED_SENTENCE_EMBED_MODEL",
            "ODCR_RESOLVED_EMBED_DIM",
            "ODCR_RESOLVED_LOCAL_FILES_ONLY",
        )
        old = {key: os.environ.get(key) for key in keys}

        def restore() -> None:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore)
        model_dir = os.path.join(root, "model")
        os.makedirs(model_dir, exist_ok=True)
        os.environ["ODCR_RESOLVED_DATA_DIR"] = root
        os.environ["ODCR_RESOLVED_MODELS_DIR"] = os.path.join(root, "models")
        os.environ["ODCR_RESOLVED_SENTENCE_EMBED_MODEL"] = model_dir
        os.environ["ODCR_RESOLVED_EMBED_DIM"] = "1024"
        os.environ["ODCR_RESOLVED_LOCAL_FILES_ONLY"] = "1"

    def test_resolve_profile_specs_defaults_and_dedupes(self) -> None:
        self.assertEqual(ce._resolve_profile_specs(None), ce.PROFILE_SPECS)
        resolved = ce._resolve_profile_specs("item_style,user_content,item_style")
        self.assertEqual([spec.name for spec in resolved], ["item_style", "user_content"])

    def test_resolve_probe_config_requires_probe_only(self) -> None:
        with self.assertRaises(ValueError):
            ce._resolve_probe_config(
                SimpleNamespace(
                    probe_only=False,
                    probe_max_groups_per_spec=8,
                    probe_max_batches_per_spec=None,
                )
            )

        cfg = ce._resolve_probe_config(
            SimpleNamespace(
                probe_only=True,
                probe_max_groups_per_spec=8,
                probe_max_batches_per_spec=3,
            )
        )
        self.assertTrue(cfg.probe_only)
        self.assertEqual(cfg.max_groups_per_spec, 8)
        self.assertEqual(cfg.max_batches_per_spec, 3)

    def test_bare_embed_batch_env_does_not_bypass_required_transport(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._install_resolved_runtime_env(td)
            model_dir = os.environ["ODCR_RESOLVED_SENTENCE_EMBED_MODEL"]
            old_embed_batch = os.environ.get("EMBED_BATCH_SIZE")
            os.environ["EMBED_BATCH_SIZE"] = "77"
            argv = [
                "compute_embeddings.py",
                "--data-dir",
                td,
                "--models-dir",
                os.path.join(td, "models"),
                "--sentence-embed-model",
                model_dir,
                "--embed-dim",
                "1024",
                "--allow-cpu-debug",
                "--bf16",
                "--tf32",
                "--read-chunk-rows",
                "8",
                "--group-shard-size",
                "4",
                "--grouped-text-cache",
                "--grouped-text-cache-dir",
                os.path.join(td, "cache"),
                "--grouped-text-cache-version",
                "vtest",
                "--probe-only",
                "--specs",
                "user_content",
                "--datasets",
                "Yelp",
            ]
            try:
                with mock.patch.object(sys, "argv", argv):
                    with self.assertRaisesRegex(ValueError, "--embed-batch-size is required"):
                        ce.main()
            finally:
                if old_embed_batch is None:
                    os.environ.pop("EMBED_BATCH_SIZE", None)
                else:
                    os.environ["EMBED_BATCH_SIZE"] = old_embed_batch

    def test_resolved_runtime_transport_installs_child_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._install_resolved_runtime_env(td)
            model_dir = os.environ["ODCR_RESOLVED_SENTENCE_EMBED_MODEL"]
            args = SimpleNamespace(
                data_dir=os.path.join(td, "data"),
                models_dir=os.path.join(td, "models"),
                sentence_embed_model=model_dir,
                embed_dim=1024,
            )
            ce._install_resolved_preprocess_context(args)
            self.assertEqual(os.environ["ODCR_RESOLVED_DATA_DIR"], os.path.abspath(args.data_dir))
            self.assertEqual(os.environ["ODCR_RESOLVED_SENTENCE_EMBED_MODEL"], os.path.abspath(model_dir))
            self.assertEqual(os.environ["ODCR_RESOLVED_EMBED_DIM"], "1024")

    def test_chunked_group_accumulator_matches_legacy_content_and_style(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx": [2, 1, 2, 0, 1],
                "item_idx": [0, 1, 0, 2, 1],
                "review": ["r20", "r10", "r21", "r00", "r11"],
                "content_evidence": ["c20", "c10", "", "c00", "c11"],
                "explanation": ["e20", "e10", "e21", "e00", "e11"],
                "style_evidence": ["s20", "s10", "s21", "", "s11"],
                "domain_style_anchor": ["da20", "da10", "da21", "da00", "da11"],
                "polarity_anchor": ["p20", "p10", "p21", "p00", "p11"],
                "local_style_residual_hint": ["lh20", "lh10", "lh21", "lh00", "lh11"],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            self._install_resolved_runtime_env(td)
            source_path = os.path.join(td, "train.csv")
            df.to_csv(source_path, index=False)

            for spec_name, legacy_builder in (
                ("user_content", ce._build_content_profile_text),
                ("user_style", ce._build_style_profile_text),
            ):
                spec = ce.PROFILE_SPEC_BY_NAME[spec_name]
                phase = ce.PhaseTiming()
                acc = ce._read_spec_group_accumulator(
                    source_path=source_path,
                    spec=spec,
                    read_chunk_rows=2,
                    phase_timing=phase,
                )

                legacy = legacy_builder(df, spec.group_col)
                materialized: dict[int, str] = {}
                phase_for_shards = ce.PhaseTiming()
                for shard in ce._iter_group_text_shards(
                    acc,
                    shard_size=2,
                    max_groups=None,
                    phase_timing=phase_for_shards,
                ):
                    materialized.update(zip(shard.group_indices.tolist(), shard.texts))

                ordered_materialized = [materialized[int(group_idx)] for group_idx in legacy.index.tolist()]
                self.assertEqual(acc.order_keys, legacy.index.tolist())
                self.assertEqual(ordered_materialized, legacy.tolist())
                self.assertEqual(acc.target_size, int(df[spec.group_col].max()) + 1)

    def test_iter_group_text_shards_respects_group_limit(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx": [3, 1, 3, 2],
                "review": ["a0", "b0", "a1", "c0"],
                "content_evidence": ["ca0", "cb0", "ca1", "cc0"],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            self._install_resolved_runtime_env(td)
            source_path = os.path.join(td, "train.csv")
            df.to_csv(source_path, index=False)
            spec = ce.PROFILE_SPEC_BY_NAME["user_content"]
            acc = ce._read_spec_group_accumulator(
                source_path=source_path,
                spec=spec,
                read_chunk_rows=2,
                phase_timing=ce.PhaseTiming(),
            )
            phase = ce.PhaseTiming()
            shards = list(
                ce._iter_group_text_shards(
                    acc,
                    shard_size=4,
                    max_groups=2,
                    phase_timing=phase,
                )
            )
            self.assertEqual(len(shards), 1)
            self.assertEqual(shards[0].group_indices.tolist(), [3, 1])
            self.assertEqual(shards[0].texts, ["a0 a1 ca0 ca1", "b0 cb0"])
            self.assertEqual(phase.groups_finalized, 2)

    def test_group_accumulator_rejects_retired_detail_columns(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx": [0],
                "review": ["alpha"],
                "content_evidence": ["keywords alpha ; aspects none ; entities none"],
                "content_keywords": ["alpha"],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            self._install_resolved_runtime_env(td)
            source_path = os.path.join(td, "train.csv")
            df.to_csv(source_path, index=False)
            with self.assertRaisesRegex(ValueError, "retired preprocess detail columns"):
                ce._read_spec_group_accumulator(
                    source_path=source_path,
                    spec=ce.PROFILE_SPEC_BY_NAME["user_content"],
                    read_chunk_rows=2,
                    phase_timing=ce.PhaseTiming(),
                )

    def test_grouped_text_cache_roundtrip_hits_and_replays_shards(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx": [2, 1, 2, 0, 1],
                "review": ["r20", "r10", "r21", "r00", "r11"],
                "content_evidence": ["c20", "c10", "", "c00", "c11"],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            self._install_resolved_runtime_env(td)
            source_path = os.path.join(td, "train.csv")
            cache_dir = os.path.join(td, "cache")
            df.to_csv(source_path, index=False)
            spec = ce.PROFILE_SPEC_BY_NAME["user_content"]
            accumulator = ce._read_spec_group_accumulator(
                source_path=source_path,
                spec=spec,
                read_chunk_rows=2,
                phase_timing=ce.PhaseTiming(),
            )
            phase_for_request = ce.PhaseTiming()
            request = ce._build_grouped_text_cache_request(
                cache_config=ce.GroupedTextCacheConfig(
                    enabled=True,
                    cache_dir=cache_dir,
                    version="test_grouped_text_cache_v1",
                ),
                dataset="TripAdvisor",
                source_path=source_path,
                spec=spec,
                tokenizer=self._cache_identity()[0],
                model=self._cache_identity()[1],
                read_chunk_rows=2,
                group_shard_size=2,
                phase_timing=phase_for_request,
            )
            self.assertIsNotNone(request)
            self.assertIsNone(ce._load_grouped_text_cache(request, phase_timing=ce.PhaseTiming()))

            write_phase = ce.PhaseTiming()
            entry = ce._write_grouped_text_cache(
                request,
                accumulator,
                shard_size=2,
                phase_timing=write_phase,
            )
            self.assertEqual(write_phase.group_text_cache_status, "miss_written")
            self.assertEqual(write_phase.group_text_cache_shards_written, 2)
            self.assertEqual(write_phase.group_text_cache_groups_written, 3)

            hit_phase = ce.PhaseTiming()
            loaded = ce._load_grouped_text_cache(request, phase_timing=hit_phase)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(hit_phase.group_text_cache_status, "hit")
            self.assertEqual(loaded.group_count, 3)
            self.assertEqual(loaded.target_size, 3)

            shard_phase = ce.PhaseTiming()
            shards = list(
                ce._iter_cached_group_text_shards(
                    loaded,
                    max_groups=2,
                    phase_timing=shard_phase,
                )
            )
            self.assertEqual(len(shards), 1)
            self.assertTrue(np.array_equal(shards[0].group_indices, np.asarray([2, 1], dtype=np.int64)))
            self.assertEqual(shards[0].texts, ["r20 r21 c20", "r10 r11 c10 c11"])
            self.assertEqual(shard_phase.group_text_cache_shards_loaded, 1)
            self.assertEqual(shard_phase.groups_finalized, 2)

    def test_grouped_text_cache_key_changes_with_source_fingerprint(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx": [0, 1],
                "review": ["a", "b"],
                "content_evidence": ["ca", "cb"],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            self._install_resolved_runtime_env(td)
            source_path = os.path.join(td, "train.csv")
            cache_dir = os.path.join(td, "cache")
            df.to_csv(source_path, index=False)
            spec = ce.PROFILE_SPEC_BY_NAME["user_content"]

            req1 = ce._build_grouped_text_cache_request(
                cache_config=ce.GroupedTextCacheConfig(
                    enabled=True,
                    cache_dir=cache_dir,
                    version="test_grouped_text_cache_v1",
                ),
                dataset="Yelp",
                source_path=source_path,
                spec=spec,
                tokenizer=self._cache_identity()[0],
                model=self._cache_identity()[1],
                read_chunk_rows=8,
                group_shard_size=4,
                phase_timing=ce.PhaseTiming(),
            )
            self.assertIsNotNone(req1)
            self.assertIn("canonical_column_hash", req1.key_payload)
            self.assertTrue(req1.key_payload["canonical_column_hash"])
            df2 = df.copy()
            df2.loc[1, "content_evidence"] = "cb2"
            df2.to_csv(source_path, index=False)
            req2 = ce._build_grouped_text_cache_request(
                cache_config=ce.GroupedTextCacheConfig(
                    enabled=True,
                    cache_dir=cache_dir,
                    version="test_grouped_text_cache_v1",
                ),
                dataset="Yelp",
                source_path=source_path,
                spec=spec,
                tokenizer=self._cache_identity()[0],
                model=self._cache_identity()[1],
                read_chunk_rows=8,
                group_shard_size=4,
                phase_timing=ce.PhaseTiming(),
            )
            self.assertIsNotNone(req2)
            assert req1 is not None and req2 is not None
            self.assertNotEqual(req1.cache_key, req2.cache_key)


if __name__ == "__main__":
    unittest.main()
