"""infer_domain_semantics：token-window cache / fast path / manual padding 回归。"""
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

import infer_domain_semantics as ids  # noqa: E402


class _DummyTokenizer:
    name_or_path = "dummy_fast_tokenizer"
    is_fast = True
    model_max_length = 512
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0
    eos_token_id = None
    unk_token_id = 999

    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def __len__(self) -> int:
        return 30522

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        del pair
        return 2

    def build_inputs_with_special_tokens(self, token_ids):
        return [self.cls_token_id, *list(token_ids), self.sep_token_id]

    def pad(self, *args, **kwargs):  # pragma: no cover - failure path only
        raise AssertionError("manual batch builder should not call tokenizer.pad()")

    def _encode_one(self, text: str) -> list[int]:
        tokens = [tok for tok in str(text).replace("\n", " ").split(" ") if tok]
        if not tokens:
            return [self.unk_token_id]
        return [sum(ord(ch) for ch in tok) % 251 + 1 for tok in tokens]

    def __call__(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        self.call_sizes.append(len(texts))
        return {"input_ids": [self._encode_one(text) for text in texts]}


class TestInferDomainSemantics(unittest.TestCase):
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

    def test_resolve_domain_specs_defaults_and_dedupes(self) -> None:
        self.assertEqual(ids._resolve_domain_specs(None), ids.DOMAIN_SPECS)
        resolved = ids._resolve_domain_specs("style,content,style")
        self.assertEqual([spec.name for spec in resolved], ["style", "content"])

    def test_resolve_domain_specs_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            ids._resolve_domain_specs("content,unknown")

    def test_resolve_probe_config_requires_probe_only_for_chunk_cap(self) -> None:
        with self.assertRaises(ValueError):
            ids._resolve_probe_config(
                SimpleNamespace(
                    probe_only=False,
                    probe_max_chunks_per_domain=128,
                )
            )
        cfg = ids._resolve_probe_config(
            SimpleNamespace(
                probe_only=True,
                probe_max_chunks_per_domain=128,
            )
        )
        self.assertTrue(cfg.probe_only)
        self.assertEqual(cfg.max_chunks_per_domain, 128)

    def test_effective_chunk_count_caps_when_requested(self) -> None:
        self.assertEqual(ids._effective_chunk_count(100, max_chunks=None), 100)
        self.assertEqual(ids._effective_chunk_count(100, max_chunks=64), 64)
        self.assertEqual(ids._effective_chunk_count(32, max_chunks=64), 32)

    def test_bare_domain_chunk_env_does_not_bypass_required_transport(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._install_resolved_runtime_env(td)
            model_dir = os.environ["ODCR_RESOLVED_SENTENCE_EMBED_MODEL"]
            old_chunk = os.environ.get("DOMAIN_CHUNK_BATCH_SIZE")
            os.environ["DOMAIN_CHUNK_BATCH_SIZE"] = "77"
            argv = [
                "infer_domain_semantics.py",
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
                "--tokenizer-hotpath",
                "--token-window-cache",
                "--token-window-cache-dir",
                os.path.join(td, "cache"),
                "--token-window-cache-version",
                "vtest",
                "--token-window-cache-shard-size",
                "1",
                "--probe-only",
                "--domains",
                "content",
                "--datasets",
                "Yelp",
            ]
            try:
                with mock.patch.object(sys, "argv", argv):
                    with self.assertRaisesRegex(ValueError, "--chunk-batch-size is required"):
                        ids.main()
            finally:
                if old_chunk is None:
                    os.environ.pop("DOMAIN_CHUNK_BATCH_SIZE", None)
                else:
                    os.environ["DOMAIN_CHUNK_BATCH_SIZE"] = old_chunk

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
            ids._install_resolved_preprocess_context(args)
            self.assertEqual(os.environ["ODCR_RESOLVED_DATA_DIR"], os.path.abspath(args.data_dir))
            self.assertEqual(os.environ["ODCR_RESOLVED_SENTENCE_EMBED_MODEL"], os.path.abspath(model_dir))
            self.assertEqual(os.environ["ODCR_RESOLVED_EMBED_DIM"], "1024")

    def test_iter_piece_text_batches_keeps_single_leading_space_contract(self) -> None:
        batches = list(ids._iter_piece_text_batches_from_cells(["alpha", "beta", " ", "gamma"], batch_size=2))
        self.assertEqual(batches, [["alpha", " beta"], [" gamma"]])

    def test_iter_token_window_shards_streams_batched_tokenizer_calls(self) -> None:
        tok = _DummyTokenizer()
        shards = list(
            ids._iter_token_window_shards_from_cells(
                cell_iter=["a b c", "d e f", "g h i", "j k"],
                tokenizer=tok,
                max_total_tokens=8,
                shard_size=1,
                batch_size=2,
            )
        )
        self.assertEqual(tok.call_sizes, [2, 2])
        self.assertEqual([shard.window_count for shard in shards], [1, 1])
        self.assertEqual(shards[0].window_lengths.tolist(), [6])
        self.assertEqual(shards[1].window_lengths.tolist(), [5])

    def test_prepare_chunk_batch_uses_manual_padding(self) -> None:
        tok = _DummyTokenizer()
        window_ids = np.array([[11, 12, 0], [21, 0, 0]], dtype=np.int32)
        window_lengths = np.array([2, 1], dtype=np.int32)
        batch = ids._prepare_chunk_batch(
            window_ids,
            window_lengths,
            tok,
            device=ids.torch.device("cpu"),
            max_total_tokens=8,
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(batch["input_ids"][0].tolist(), [101, 11, 12, 102])
        self.assertEqual(batch["input_ids"][1].tolist(), [101, 21, 102, 0])
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 1, 0])

    def test_token_window_cache_hits_with_hard_fingerprint(self) -> None:
        tok = _DummyTokenizer()
        df = pd.DataFrame(
            {
                "review": ["alpha beta gamma", "delta epsilon", "zeta eta theta"],
                "content_evidence": [
                    "keywords x y ; aspects none ; entities none",
                    "keywords none ; aspects none ; entities none",
                    "keywords p q ; aspects none ; entities none",
                ],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            self._install_resolved_runtime_env(td)
            source_path = os.path.join(td, "train.csv")
            with open(source_path, "w", encoding="utf-8") as handle:
                handle.write("review,content_evidence\n")
                handle.write("alpha beta gamma,keywords x y ; aspects none ; entities none\n")
                handle.write("delta epsilon,keywords none ; aspects none ; entities none\n")
            cache_cfg = ids.TokenWindowCacheConfig(enabled=True, cache_dir=os.path.join(td, "cache"), version="vtest", shard_size=1)
            payload_budget = ids._token_window_budget(tok, max_total_tokens=8)
            fingerprint = ids._token_window_cache_fingerprint(
                dataset="AM_Movies",
                spec=ids.DOMAIN_SPECS[0],
                source_path=source_path,
                tokenizer=tok,
                max_total_tokens=8,
                payload_budget=payload_budget,
                cache_version=cache_cfg.version,
                probe_chunk_limit=None,
            )
            self.assertIn("canonical_column_hash", fingerprint)
            self.assertTrue(fingerprint["canonical_column_hash"])
            cache_path = ids._token_window_cache_path(
                cache_cfg,
                dataset="AM_Movies",
                spec=ids.DOMAIN_SPECS[0],
                fingerprint=fingerprint,
            )
            built_shards = list(
                ids._iter_and_maybe_cache_token_window_shards(
                    shard_iter=ids._iter_token_window_shards_from_cells(
                        cell_iter=ids._iter_domain_cells(df, ids.DOMAIN_SPECS[0].column_names),
                        tokenizer=tok,
                        max_total_tokens=8,
                        shard_size=cache_cfg.shard_size,
                    ),
                    cache_path=cache_path,
                    fingerprint=fingerprint,
                    payload_budget=payload_budget,
                    shard_size=cache_cfg.shard_size,
                    persist_cache=True,
                    dataset="AM_Movies",
                    spec=ids.DOMAIN_SPECS[0],
                )
            )
            calls_after_build = list(tok.call_sizes)
            manifest = ids._load_token_window_cache_manifest(
                cache_path,
                expected_fingerprint=fingerprint,
                payload_budget=payload_budget,
            )
            self.assertIsNotNone(manifest)
            cached_shards = list(
                ids._iter_cached_token_window_shards(
                    cache_path,
                    manifest=manifest,  # type: ignore[arg-type]
                )
            )
            self.assertEqual(calls_after_build, tok.call_sizes)
            self.assertTrue(os.path.isdir(cache_path))
            self.assertEqual(sum(shard.window_count for shard in built_shards), manifest.window_count)  # type: ignore[union-attr]
            built_ids = np.concatenate([shard.window_ids for shard in built_shards], axis=0)
            built_lengths = np.concatenate([shard.window_lengths for shard in built_shards], axis=0)
            cached_ids = np.concatenate([shard.window_ids for shard in cached_shards], axis=0)
            cached_lengths = np.concatenate([shard.window_lengths for shard in cached_shards], axis=0)
            self.assertTrue(np.array_equal(built_ids, cached_ids))
            self.assertTrue(np.array_equal(built_lengths, cached_lengths))

    def test_probe_chunk_limit_changes_cache_fingerprint(self) -> None:
        tok = _DummyTokenizer()
        with tempfile.TemporaryDirectory() as td:
            self._install_resolved_runtime_env(td)
            source_path = os.path.join(td, "train.csv")
            with open(source_path, "w", encoding="utf-8") as handle:
                handle.write("review,content_evidence\n")
                handle.write("alpha beta gamma,keywords x y ; aspects none ; entities none\n")
            payload_budget = ids._token_window_budget(tok, max_total_tokens=8)
            full_fingerprint = ids._token_window_cache_fingerprint(
                dataset="AM_Movies",
                spec=ids.DOMAIN_SPECS[0],
                source_path=source_path,
                tokenizer=tok,
                max_total_tokens=8,
                payload_budget=payload_budget,
                cache_version="vtest",
                probe_chunk_limit=None,
            )
            probe_fingerprint = ids._token_window_cache_fingerprint(
                dataset="AM_Movies",
                spec=ids.DOMAIN_SPECS[0],
                source_path=source_path,
                tokenizer=tok,
                max_total_tokens=8,
                payload_budget=payload_budget,
                cache_version="vtest",
                probe_chunk_limit=2,
            )
            self.assertNotEqual(full_fingerprint, probe_fingerprint)


if __name__ == "__main__":
    unittest.main()
