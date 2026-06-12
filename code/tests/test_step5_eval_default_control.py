from __future__ import annotations

import os
import sys
import unittest

import pandas as pd

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from executors.step5_engine import (  # noqa: E402
    STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION,
    STEP5_CLEAN_MEMORY_EVAL_CONTROL_SCHEMA_VERSION,
    STEP5_CLEAN_MEMORY_MODE,
    STEP5_CLEAN_MEMORY_SOURCE,
    _apply_step5_clean_memory_eval_controls,
    _require_step5_rcr_posterior_controls,
    _validate_step5_warmup_schedule,
    step5_clean_memory_eval_control_contract,
)


class TestStep5CleanMemoryEvalControl(unittest.TestCase):
    def _clean_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_idx_global": [0],
                "item_idx_global": [1],
                "rating": [4.2],
                "explanation": ["target factual explanation"],
                "clean_text": ["target factual explanation"],
                "content_evidence": ["ITEM_HISTORY: clean train evidence"],
                "style_evidence": ["USER_STYLE: clean train style"],
                "domain_style_anchor": ["target:train_memory_style_prior:positive"],
                "local_style_residual_hint": ["source=train_only_memory"],
                "polarity_anchor": ["positive"],
                "content_anchor_score": [0.6],
                "style_anchor_score": [0.5],
                "evidence_quality_prior": [0.56],
                "sample_weight_hint": [0.56],
                "route_scorer": [1],
                "route_explainer": [1],
                "step5_clean_control_source": [STEP5_CLEAN_MEMORY_SOURCE],
                "step5_clean_control_contract_version": [STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION],
                "step5_control_mode": [STEP5_CLEAN_MEMORY_MODE],
                "step5_control_source": [STEP5_CLEAN_MEMORY_SOURCE],
                "step5_control_contract_version": [STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION],
                "step5_leave_one_out_memory": [False],
            }
        )

    def test_clean_memory_eval_controls_are_labeled_not_rcr_posterior(self) -> None:
        df = self._clean_df()
        out = _apply_step5_clean_memory_eval_controls(df, split_label="valid")
        self.assertEqual(out.loc[0, "step5_control_mode"], "train_memory_eval_controls")
        self.assertEqual(
            out.loc[0, "step5_control_contract_version"],
            STEP5_CLEAN_MEMORY_EVAL_CONTROL_SCHEMA_VERSION,
        )
        self.assertEqual(int(out.loc[0, "route_scorer"]), 1)
        self.assertEqual(int(out.loc[0, "route_explainer"]), 1)
        self.assertFalse(bool(out.loc[0, "step5_control_is_rcr_posterior"]))
        self.assertFalse(bool(out.loc[0, "step5_control_is_step4_export_posterior"]))
        contract = step5_clean_memory_eval_control_contract("valid")
        self.assertEqual(contract["mode"], "train_memory_eval_controls")
        self.assertEqual(contract["control_source"], "train_only_memory_controls")
        self.assertFalse(contract["is_rcr_posterior"])
        self.assertFalse(contract["is_train_route"])

    def test_clean_memory_eval_controls_are_rejected_on_train_export_path(self) -> None:
        df = self._clean_df()
        df["sample_origin"] = "target_gold"
        df["train_keep"] = 1
        out = _apply_step5_clean_memory_eval_controls(df, split_label="valid")
        with self.assertRaisesRegex(ValueError, "eval/control-mode rows"):
            _require_step5_rcr_posterior_controls(out, ctx="unit-test train path")

    def test_clean_memory_eval_rejects_current_row_controls_without_clean_marker(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx_global": [0],
                "item_idx_global": [1],
                "rating": [4.0],
                "explanation": ["target factual explanation"],
                "clean_text": ["target factual explanation"],
                "content_evidence": ["current row evidence"],
                "polarity_anchor": ["positive"],
            }
        )
        with self.assertRaisesRegex(ValueError, "missing CleanMemory control columns"):
            _apply_step5_clean_memory_eval_controls(df, split_label="valid")

    def test_train_export_path_missing_rcr_controls_fails_fast(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx_global": [0],
                "item_idx_global": [1],
                "rating": [4.0],
                "clean_text": ["missing controls"],
                "sample_origin": ["aux_cf"],
            }
        )
        with self.assertRaisesRegex(ValueError, "requires canonical Step4 RCR posterior/control columns"):
            _require_step5_rcr_posterior_controls(df, ctx="unit-test train path")

    def test_clean_memory_warmup_schedule_rejects_all_warmup(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "forbids all-warmup"):
            _validate_step5_warmup_schedule(warmup_steps=100, total_steps=100)

    def test_clean_memory_warmup_schedule_rejects_long_warmup(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "warmup is too long"):
            _validate_step5_warmup_schedule(warmup_steps=200, total_steps=1000)

    def test_clean_memory_warmup_schedule_accepts_short_warmup(self) -> None:
        _validate_step5_warmup_schedule(warmup_steps=50, total_steps=1000)


if __name__ == "__main__":
    unittest.main()
