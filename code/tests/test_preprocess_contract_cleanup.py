from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import pandas as pd

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

import data_contract as dc  # noqa: E402
import preprocess_data as pp  # noqa: E402
import split_data  # noqa: E402
from odcr_core.preprocess_metadata import csv_header_metadata  # noqa: E402
from odcr_core.preprocess_schema import (  # noqa: E402
    PREPROCESS_C_DOMAIN_CONTRACT_VERSION,
    preprocess_b_expected_shape_dtype,
    preprocess_c_expected_shape_dtype,
    render_preprocess_stage_contract,
)
from odcr_core.training_checkpoint import stable_hash  # noqa: E402


def _processed_row() -> dict[str, object]:
    return {
        "user": "u1",
        "item": "i1",
        "rating": 5.0,
        "review": "great camera focus",
        "explanation": "great camera focus",
        "content_evidence": "keywords camera focus ; aspects quality ; entities none",
        "content_anchor_score": 0.76,
        "polarity_anchor": "positive",
        "domain_style_anchor": "AM_Movies:plain_statement:short:positive",
        "local_style_residual_hint": "perspective=external;intensity=steady;discourse=direct;punctuation=flat",
        "style_evidence": (
            "markers none ; template_family plain_statement ; polarity positive ; length short ; "
            "domain_style_anchor AM_Movies:plain_statement:short:positive ; "
            "local_style_residual_hint perspective=external;intensity=steady;discourse=direct;punctuation=flat"
        ),
        "style_anchor_score": 0.42,
        "evidence_quality_prior": 0.61,
        "preprocess_route_scorer_prior": 1,
        "preprocess_route_explainer_prior": 1,
    }


def _processed_df() -> pd.DataFrame:
    return pd.DataFrame([_processed_row()]).loc[:, list(dc.PROCESSED_COLUMN_ORDER)]


class TestPreprocessContractCleanup(unittest.TestCase):
    def test_preprocess_b_profile_contract_remains_entity_matrix(self) -> None:
        expected = preprocess_b_expected_shape_dtype()
        contract = render_preprocess_stage_contract("preprocess_b")
        self.assertEqual(expected["shape"], "[entity_count, env.embed_dim]")
        self.assertEqual(expected["dtype"], "float32")
        for name in (
            "user_content_profiles.npy",
            "user_style_profiles.npy",
            "item_content_profiles.npy",
            "item_style_profiles.npy",
        ):
            self.assertEqual(
                contract["output_artifact_contract"][name]["shape"],
                ["entity_count", "env.embed_dim"],
            )

    def test_preprocess_c_domain_contract_is_embed_dim_vector(self) -> None:
        expected = preprocess_c_expected_shape_dtype()
        contract = render_preprocess_stage_contract("preprocess_c")
        self.assertEqual(expected["shape"], "[env.embed_dim]")
        self.assertEqual(expected["dtype"], "float32")
        self.assertEqual(expected["contract_version"], PREPROCESS_C_DOMAIN_CONTRACT_VERSION)
        self.assertEqual(contract["domain_shape_contract_version"], PREPROCESS_C_DOMAIN_CONTRACT_VERSION)
        for name in ("domain_content.npy", "domain_style.npy"):
            self.assertEqual(contract["output_artifact_contract"][name]["shape"], ["env.embed_dim"])
            self.assertEqual(contract["output_artifact_contract"][name]["dtype"], "float32")
        rendered = json.dumps(contract, sort_keys=True)
        self.assertNotIn("[row_count, env.embed_dim]", rendered)
        self.assertNotIn("[token_window_count, env.embed_dim]", rendered)
        self.assertNotIn("[nitems, env.embed_dim]", rendered)
        self.assertNotIn("[nusers, env.embed_dim]", rendered)

    def test_retired_detail_fields_are_not_contract_columns(self) -> None:
        retired = set(dc.DEPRECATED_PREPROCESS_DETAIL_COLUMNS)
        posterior = set(dc.STEP4_POSTERIOR_ROUTE_COLUMNS)
        self.assertFalse(retired & {spec.name for spec in dc.PREPROCESS_FIELD_SPECS})
        self.assertFalse(retired & set(dc.PROCESSED_COLUMN_ORDER))
        self.assertFalse(retired & set(dc.SPLIT_COLUMN_ORDER))
        self.assertFalse(retired & set(dc.MERGED_COLUMN_ORDER))
        self.assertFalse(posterior & set(dc.PROCESSED_COLUMN_ORDER))
        self.assertFalse(posterior & set(dc.SPLIT_COLUMN_ORDER))
        self.assertFalse(posterior & set(dc.MERGED_COLUMN_ORDER))
        self.assertIn("preprocess_route_scorer_prior", dc.PROCESSED_COLUMN_ORDER)
        self.assertIn("preprocess_route_explainer_prior", dc.PROCESSED_COLUMN_ORDER)

        for snapshot in (
            dc.render_preprocess_contract_snapshot(),
            dc.render_preprocess_contract_snapshot(require_split_indices=True),
            dc.render_preprocess_contract_snapshot(require_split_indices=True, require_domain=True),
        ):
            self.assertFalse(retired & set(snapshot["required_columns"]))
            self.assertFalse(retired & {field["name"] for field in snapshot["fields"]})

    def test_processed_split_merged_positive_contracts_do_not_emit_retired_fields(self) -> None:
        retired = set(dc.DEPRECATED_PREPROCESS_DETAIL_COLUMNS)
        processed = _processed_df()
        split = processed.assign(user_idx=0, item_idx=0).loc[:, list(dc.SPLIT_COLUMN_ORDER)]
        merged = split.assign(domain="target").loc[:, list(dc.MERGED_COLUMN_ORDER)]

        with tempfile.TemporaryDirectory() as td:
            paths = {
                "processed": os.path.join(td, "processed.csv"),
                "split": os.path.join(td, "split.csv"),
                "merged": os.path.join(td, "merged.csv"),
            }
            dc.write_preprocess_csv(processed, paths["processed"])
            dc.write_preprocess_csv(split, paths["split"], require_split_indices=True)
            dc.write_preprocess_csv(
                merged,
                paths["merged"],
                require_split_indices=True,
                require_domain=True,
            )
            for path in paths.values():
                header = pd.read_csv(path, nrows=0).columns.tolist()
                self.assertFalse(retired & set(header))
                self.assertFalse(set(dc.STEP4_POSTERIOR_ROUTE_COLUMNS) & set(header))
                self.assertIn("preprocess_route_scorer_prior", header)
                self.assertIn("preprocess_route_explainer_prior", header)

    def test_current_header_metadata_matches_existing_csv(self) -> None:
        processed = _processed_df()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "processed.csv")
            dc.write_preprocess_csv(processed, path)
            metadata = csv_header_metadata(path, contract_kind="processed")
            self.assertTrue(metadata["exists"])
            self.assertTrue(metadata["header_match"])
            self.assertEqual(metadata["header"], list(dc.PROCESSED_COLUMN_ORDER))
            self.assertEqual(metadata["header_hash"], stable_hash(list(dc.PROCESSED_COLUMN_ORDER)))
            self.assertGreater(metadata["file_size"], 0)
            self.assertIsInstance(metadata["mtime_ns"], int)
            self.assertEqual(metadata["contract_kind"], "processed")

    def test_split_func_with_stats_records_expected_policy(self) -> None:
        rows = []
        for idx in range(20):
            row = _processed_row()
            row["user"] = f"u{idx % 4}"
            row["item"] = f"i{idx % 5}"
            rows.append(row)
        df = pd.DataFrame(rows).loc[:, list(dc.PROCESSED_COLUMN_ORDER)]
        train_df, valid_df, test_df, stats = split_data.split_func_with_stats(df)
        self.assertEqual(stats["processed_rows"], 20)
        self.assertEqual(stats["train_rows"], len(train_df))
        self.assertEqual(stats["valid_rows_after_filter"], len(valid_df))
        self.assertEqual(stats["test_rows_after_filter"], len(test_df))
        self.assertEqual(stats["split_loss_policy"], "filter_valid_test_cold_user_item")
        self.assertTrue(stats["split_loss_expected"])
        self.assertIn("filtered_cold_user_item_rows", stats)

    def test_retired_detail_fields_fail_fast_on_contract_io(self) -> None:
        bad = _processed_df()
        bad["content_keywords"] = "camera|focus"
        with self.assertRaisesRegex(ValueError, "retired preprocess detail columns"):
            dc.normalize_preprocess_dataframe(bad)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "processed.csv")
            bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Rerun preprocess_data.py / preprocess_a"):
                dc.read_preprocess_csv(path)

    def test_step4_posterior_route_fields_fail_fast_in_preprocess_csv(self) -> None:
        bad = _processed_df().rename(
            columns={
                "preprocess_route_scorer_prior": "route_scorer",
                "preprocess_route_explainer_prior": "route_explainer",
            }
        )
        with self.assertRaisesRegex(ValueError, "Step4 posterior route columns"):
            dc.normalize_preprocess_dataframe(bad)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "processed.csv")
            bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Step4 posterior-only"):
                dc.read_preprocess_csv(path)

    def test_preprocess_asset_builder_keeps_detail_fields_internal(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "user": "u1",
                    "item": "i1",
                    "rating": 5.0,
                    "review": "Great camera focus.",
                    "explanation": "Great camera focus.",
                }
            ]
        )
        out = pp._build_canonical_preprocess_assets("AM_Movies", raw)
        self.assertFalse(set(dc.DEPRECATED_PREPROCESS_DETAIL_COLUMNS) & set(out.columns))
        self.assertFalse(set(dc.STEP4_POSTERIOR_ROUTE_COLUMNS) & set(out.columns))
        self.assertIn("preprocess_route_scorer_prior", out.columns)
        self.assertIn("preprocess_route_explainer_prior", out.columns)
        for column in dc.PROCESSED_COLUMN_ORDER:
            self.assertIn(column, out.columns)

    def test_split_func_rejects_retired_detail_input(self) -> None:
        bad = _processed_df()
        bad["style_markers"] = "none"
        with self.assertRaisesRegex(ValueError, "retired preprocess detail columns"):
            split_data.split_func(bad)


if __name__ == "__main__":
    unittest.main()
