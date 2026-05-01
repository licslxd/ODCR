"""Step4/Step5 index_contract：契约构建、local→global、CPU fail-fast、profile 一致性。"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from odcr_core.index_contract import (  # noqa: E402
    GLOBAL_COL_ITEM,
    GLOBAL_COL_USER,
    IndexContractError,
    build_step4_export_lineage,
    build_index_contract,
    load_index_contract,
    normalize_split_indices_to_global,
    remap_step4_train_df_to_global_columns,
    validate_step4_export_lineage,
    validate_first_batch_indices,
    validate_index_contract_against_profiles,
    validate_split_indices,
    write_index_contract,
)
from odcr_core.training_checkpoint import CheckpointLineageError  # noqa: E402
from executors.step5_engine import _rerank_eval_cli_resolved, _step5_collate_dynamic  # noqa: E402
from functools import partial  # noqa: E402


def _write_odcr_dual_channel_files(
    data_root: str | Path,
    *,
    target_domain: str,
    aux_domain: str,
    target_user_count: int,
    aux_user_count: int,
    target_item_count: int,
    aux_item_count: int,
    hidden: int = 1024,
) -> None:
    root = Path(data_root)
    for dom, nu, ni in (
        (target_domain, target_user_count, target_item_count),
        (aux_domain, aux_user_count, aux_item_count),
    ):
        d = root / dom
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "user_content_profiles.npy", np.zeros((nu, hidden), dtype=np.float32))
        np.save(d / "user_style_profiles.npy", np.zeros((nu, hidden), dtype=np.float32))
        np.save(d / "item_content_profiles.npy", np.zeros((ni, hidden), dtype=np.float32))
        np.save(d / "item_style_profiles.npy", np.zeros((ni, hidden), dtype=np.float32))
        np.save(d / "domain_content.npy", np.zeros(hidden, dtype=np.float32))
        np.save(d / "domain_style.npy", np.zeros(hidden, dtype=np.float32))


class TestIndexContract(unittest.TestCase):
    def test_build_contract_counts_and_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_odcr_dual_channel_files(
                td,
                target_domain="T",
                aux_domain="A",
                target_user_count=3,
                aux_user_count=5,
                target_item_count=7,
                aux_item_count=11,
            )
            c = build_index_contract(
                task_id=4,
                iteration_id="v2",
                step4_run="2_1",
                auxiliary_domain="A",
                target_domain="T",
                data_root=td,
                train_csv_path="/tmp/out/odcr_routing_train.csv",
                valid_csv_path="/tmp/d/T/valid.csv",
                test_csv_path="/tmp/d/T/test.csv",
                target_user_count=3,
                aux_user_count=5,
                target_item_count=7,
                aux_item_count=11,
            )
        self.assertEqual(c["nuser_global"], 8)
        self.assertEqual(c["nitem_global"], 18)
        self.assertEqual(c["target_user_offset"], 0)
        self.assertEqual(c["aux_user_offset"], 3)
        self.assertEqual(c["train_index_space"], "global")
        self.assertEqual(c["valid_index_space"], "target_local")
        self.assertIn("target_user_content_profiles_path", c)
        self.assertEqual(c["profile_assets"]["kind"], "odcr_dual_channel")
        self.assertEqual(c["profile_assets"].get("consumption"), "physical_separate")
        self.assertEqual(int(c["embed_dim"]), 1024)
        self.assertEqual(c.get("schema_version"), "odcr_index_contract/2.2")
        bb = c.get("backbones") or {}
        se = bb.get("sentence_embed") or {}
        self.assertEqual(int(se.get("hidden_size")), int(c["embed_dim"]))
        self.assertTrue(se.get("dual_channel") is True)
        self.assertIn("local_dir", se)
        self.assertIn("model_id", se)
        s4 = c.get("step4_export_contract") or {}
        self.assertEqual(s4.get("schema_version"), "odcr_step4_rcr_export/1.0")
        self.assertIn("content_retention_score", s4.get("required_columns", []))
        self.assertIn("preprocess_route_scorer_prior", s4.get("prior_columns", []))
        self.assertIn("preprocess_route_explainer_prior", s4.get("prior_columns", []))
        defs = s4.get("field_definitions", {})
        self.assertIn("Step4 posterior", defs.get("route_scorer", ""))
        self.assertIn("Step4 posterior", defs.get("route_explainer", ""))
        boundary = s4.get("prior_posterior_boundary", {})
        self.assertIn("route_scorer", boundary.get("posterior_fields", []))
        profile_fp = c["fingerprints"]["target_user_content_profiles"]
        self.assertEqual(profile_fp.get("fingerprint_version"), profile_fp.get("schema_version"))
        self.assertTrue(profile_fp.get("exists"))
        self.assertGreater(int(profile_fp.get("size")), 0)
        self.assertIsInstance(profile_fp.get("mtime_ns"), int)
        self.assertRegex(str(profile_fp.get("sha256")), r"^[0-9a-f]{64}$")
        missing_train_fp = c["fingerprints"]["train_csv"]
        self.assertFalse(missing_train_fp.get("exists"))
        self.assertIn("fingerprint_version", missing_train_fp)
        self.assertIn("size", missing_train_fp)
        self.assertIn("mtime_ns", missing_train_fp)
        self.assertIn("sha256", missing_train_fp)

    def test_step4_export_lineage_hard_gate(self) -> None:
        rcr_cfg = {"required_fields": ["route_scorer", "route_explainer"], "thresholds": {"x": 0.7}}
        lineage = build_step4_export_lineage(
            task_id=4,
            auxiliary_domain="AM_Movies",
            target_domain="AM_Electronics",
            step3_checkpoint_lineage_hash="abc123",
            step4_rcr_config=rcr_cfg,
        )
        contract = {"step4_export_lineage": lineage}
        self.assertEqual(
            validate_step4_export_lineage(
                contract,
                current_step4_rcr_config=rcr_cfg,
                task_id=4,
                auxiliary_domain="AM_Movies",
                target_domain="AM_Electronics",
            )["lineage_hash"],
            lineage["lineage_hash"],
        )
        with self.assertRaises(CheckpointLineageError):
            validate_step4_export_lineage(
                contract,
                current_step4_rcr_config={**rcr_cfg, "thresholds": {"x": 0.8}},
                task_id=4,
                auxiliary_domain="AM_Movies",
                target_domain="AM_Electronics",
            )
        tampered = json.loads(json.dumps(contract))
        tampered["step4_export_lineage"]["route_posterior_contract_version"] = "old"
        with self.assertRaises(CheckpointLineageError):
            validate_step4_export_lineage(
                tampered,
                current_step4_rcr_config=rcr_cfg,
                task_id=4,
                auxiliary_domain="AM_Movies",
                target_domain="AM_Electronics",
            )

    def test_rerank_cli_requires_resolver_transport(self) -> None:
        args = SimpleNamespace(
            num_return_sequences=4,
            rerank_method="rule_v3",
            rerank_top_k=1,
            rerank_weight_logprob=0.45,
            rerank_weight_length=0.12,
            rerank_weight_repeat=0.18,
            rerank_weight_dirty=0.25,
            rerank_target_len_ratio=1.10,
            export_examples_mode="head50",
            rerank_malformed_tail_penalty=0.15,
            rerank_malformed_token_penalty=0.18,
        )
        old = os.environ.get("ODCR_RERANK_PROFILE_JSON")
        try:
            os.environ.pop("ODCR_RERANK_PROFILE_JSON", None)
            with self.assertRaises(RuntimeError):
                _rerank_eval_cli_resolved(args)
            os.environ["ODCR_RERANK_PROFILE_JSON"] = json.dumps({"method": "rule_v3"})
            resolved = _rerank_eval_cli_resolved(args)
            self.assertEqual(resolved["rerank_method"], "rule_v3")
            self.assertIn("rerank_source_table", resolved)
            self.assertIn("ODCR_RERANK_PROFILE_JSON", resolved["rerank_source_table"])
        finally:
            if old is None:
                os.environ.pop("ODCR_RERANK_PROFILE_JSON", None)
            else:
                os.environ["ODCR_RERANK_PROFILE_JSON"] = old

    def test_remap_csv_columns(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx": [0, 5],
                "item_idx": [1, 12],
                "domain": ["target", "auxiliary"],
            }
        )
        out = remap_step4_train_df_to_global_columns(df)
        self.assertIn(GLOBAL_COL_USER, out.columns)
        self.assertNotIn("user_idx", out.columns)
        self.assertEqual(list(out[GLOBAL_COL_USER]), [0, 5])

    def test_normalize_target_local_to_global(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_odcr_dual_channel_files(
                td,
                target_domain="Y",
                aux_domain="X",
                target_user_count=10,
                aux_user_count=1,
                target_item_count=20,
                aux_item_count=1,
            )
            c = build_index_contract(
                task_id=1,
                iteration_id="v1",
                step4_run="1",
                auxiliary_domain="X",
                target_domain="Y",
                data_root=td,
                train_csv_path="/t.csv",
                valid_csv_path="/v.csv",
                test_csv_path="/e.csv",
                target_user_count=10,
                aux_user_count=1,
                target_item_count=20,
                aux_item_count=1,
            )
        v = pd.DataFrame({"user_idx": [0, 9], "item_idx": [3, 19]})
        g = normalize_split_indices_to_global(v, c, "valid", ctx={"contract_path": "c.json"})
        self.assertEqual(list(g[GLOBAL_COL_USER]), [0, 9])
        self.assertEqual(list(g[GLOBAL_COL_ITEM]), [3, 19])

    def test_validate_split_oob_user_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_odcr_dual_channel_files(
                td,
                target_domain="Y",
                aux_domain="X",
                target_user_count=2,
                aux_user_count=1,
                target_item_count=2,
                aux_item_count=1,
            )
            c = build_index_contract(
                task_id=1,
                iteration_id="v1",
                step4_run="1",
                auxiliary_domain="X",
                target_domain="Y",
                data_root=td,
                train_csv_path="/t.csv",
                valid_csv_path="/v.csv",
                test_csv_path="/e.csv",
                target_user_count=2,
                aux_user_count=1,
                target_item_count=2,
                aux_item_count=1,
            )
        bad = pd.DataFrame({GLOBAL_COL_USER: [0, 99], GLOBAL_COL_ITEM: [0, 0]})
        with self.assertRaises(IndexContractError) as ar:
            validate_split_indices(bad, c, "train", ctx={"contract_path": "x"})
        self.assertIn("越界", str(ar.exception))

    def test_validate_domain_batch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_odcr_dual_channel_files(
                td,
                target_domain="Y",
                aux_domain="X",
                target_user_count=2,
                aux_user_count=1,
                target_item_count=2,
                aux_item_count=1,
            )
            c = build_index_contract(
                task_id=1,
                iteration_id="v1",
                step4_run="1",
                auxiliary_domain="X",
                target_domain="Y",
                data_root=td,
                train_csv_path="/t.csv",
                valid_csv_path="/v.csv",
                test_csv_path="/e.csv",
                target_user_count=2,
                aux_user_count=1,
                target_item_count=2,
                aux_item_count=1,
            )
        batch = (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([[1]], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0], dtype=torch.float32),
        )
        with self.assertRaises(IndexContractError):
            validate_first_batch_indices(batch, c, "train", ctx={})

    def test_profile_mismatch_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_odcr_dual_channel_files(
                td,
                target_domain="Y",
                aux_domain="X",
                target_user_count=2,
                aux_user_count=1,
                target_item_count=2,
                aux_item_count=1,
            )
            c = build_index_contract(
                task_id=1,
                iteration_id="v1",
                step4_run="1",
                auxiliary_domain="X",
                target_domain="Y",
                data_root=td,
                train_csv_path="/t.csv",
                valid_csv_path="/v.csv",
                test_csv_path="/e.csv",
                target_user_count=2,
                aux_user_count=1,
                target_item_count=2,
                aux_item_count=1,
            )
        wrong = torch.zeros(2, 4)
        with self.assertRaises(IndexContractError):
            validate_index_contract_against_profiles(c, wrong, wrong, ctx={})

    def test_roundtrip_write_load(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_odcr_dual_channel_files(
                td,
                target_domain="Y",
                aux_domain="X",
                target_user_count=1,
                aux_user_count=1,
                target_item_count=1,
                aux_item_count=1,
            )
            c = build_index_contract(
                task_id=1,
                iteration_id="v1",
                step4_run="1",
                auxiliary_domain="X",
                target_domain="Y",
                data_root=td,
                train_csv_path="/t.csv",
                valid_csv_path="/v.csv",
                test_csv_path="/e.csv",
                target_user_count=1,
                aux_user_count=1,
                target_item_count=1,
                aux_item_count=1,
            )
            p = os.path.join(td, "index_contract.json")
            write_index_contract(c, p)
            c2 = load_index_contract(p)
            self.assertEqual(c2["nuser_global"], 2)
            self.assertIn("backbones", c2)
            self.assertEqual(
                int((c2.get("backbones") or {}).get("sentence_embed", {}).get("hidden_size", -1)),
                int(c2["embed_dim"]),
            )

    def test_integration_collate_first_batch_ok(self) -> None:
        """dataset 行：全局列 → Processor 风格字段由 map 产生；此处直接模拟 map 后 dict。"""
        with tempfile.TemporaryDirectory() as td:
            _write_odcr_dual_channel_files(
                td,
                target_domain="Y",
                aux_domain="X",
                target_user_count=4,
                aux_user_count=1,
                target_item_count=4,
                aux_item_count=1,
            )
            c = build_index_contract(
                task_id=1,
                iteration_id="v1",
                step4_run="1",
                auxiliary_domain="X",
                target_domain="Y",
                data_root=td,
                train_csv_path="/t.csv",
                valid_csv_path="/v.csv",
                test_csv_path="/e.csv",
                target_user_count=4,
                aux_user_count=1,
                target_item_count=4,
                aux_item_count=1,
            )

        class _Ds:
            def __init__(self) -> None:
                self._rows = [
                    {
                        "user_idx": torch.tensor(0, dtype=torch.long),
                        "item_idx": torch.tensor(1, dtype=torch.long),
                        "rating": torch.tensor(3.0, dtype=torch.float32),
                        "explanation_idx": torch.tensor([1, 2], dtype=torch.long),
                        "domain_idx": torch.tensor(1, dtype=torch.long),
                        "sample_id": torch.tensor(0, dtype=torch.long),
                        "exp_sample_weight": torch.tensor(1.0, dtype=torch.float32),
                        "route_scorer_mask": torch.tensor(1.0, dtype=torch.float32),
                        "route_explainer_mask": torch.tensor(1.0, dtype=torch.float32),
                        "entropy_score": torch.tensor(0.0, dtype=torch.float32),
                        "uncertainty_score": torch.tensor(0.0, dtype=torch.float32),
                        "confidence_bucket": torch.tensor(2.0, dtype=torch.float32),
                        "content_anchor_score": torch.tensor(1.0, dtype=torch.float32),
                        "style_anchor_score": torch.tensor(1.0, dtype=torch.float32),
                        "evidence_features": torch.tensor(
                            [0.8, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32
                        ),
                        "content_evidence_ids": torch.tensor([1, 2], dtype=torch.long),
                        "style_evidence_ids": torch.tensor([3], dtype=torch.long),
                        "domain_style_anchor_ids": torch.tensor([4], dtype=torch.long),
                        "local_style_hint_ids": torch.tensor([5], dtype=torch.long),
                        "polarity_ids": torch.tensor([6], dtype=torch.long),
                    }
                ]

            def __len__(self) -> int:
                return 1

            def __getitem__(self, i: int):
                return self._rows[i]

        collate = partial(
            _step5_collate_dynamic, dynamic_padding=True, fixed_max_length=64
        )
        batch = collate([_Ds()[0]])
        validate_first_batch_indices(batch, c, "train", ctx={"contract_path": "c.json"})


class TestIndexContractFiles(unittest.TestCase):
    def test_contract_vs_real_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_odcr_dual_channel_files(
                root,
                target_domain="T",
                aux_domain="A",
                target_user_count=2,
                aux_user_count=1,
                target_item_count=2,
                aux_item_count=1,
                hidden=1024,
            )
            c = build_index_contract(
                task_id=1,
                iteration_id="v1",
                step4_run="1",
                auxiliary_domain="A",
                target_domain="T",
                data_root=str(root),
                train_csv_path=str(root / "f.csv"),
                valid_csv_path=str(root / "v.csv"),
                test_csv_path=str(root / "e.csv"),
                target_user_count=2,
                aux_user_count=1,
                target_item_count=2,
                aux_item_count=1,
            )
            icp = str(root / "index_contract.json")
            write_index_contract(c, icp)
            from odcr_core.index_contract import load_profile_tensors_from_contract  # noqa: E402

            dc, ds, uc, us, ic, ist = load_profile_tensors_from_contract(load_index_contract(icp), "cpu")
            _ = (dc, ds, us, ist)
            validate_index_contract_against_profiles(c, uc, ic, ctx={"contract_path": icp})


if __name__ == "__main__":
    unittest.main()
