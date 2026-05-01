from __future__ import annotations

import os
import sys
import unittest

import pandas as pd

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from executors.step5_engine import (  # noqa: E402
    STEP5_FACTUAL_EVAL_CONTROL_SCHEMA_VERSION,
    _apply_step5_factual_eval_default_controls,
    _require_step5_rcr_posterior_controls,
    step5_factual_eval_control_contract,
)


class TestStep5FactualEvalDefaultControl(unittest.TestCase):
    def test_factual_eval_defaults_are_labeled_not_rcr_posterior(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx_global": [0],
                "item_idx_global": [1],
                "rating": [4.0],
                "explanation": ["target factual explanation"],
                "clean_text": ["target factual explanation"],
            }
        )
        out = _apply_step5_factual_eval_default_controls(df, split_label="valid")
        self.assertEqual(out.loc[0, "step5_control_mode"], "factual_eval_default")
        self.assertEqual(
            out.loc[0, "step5_control_contract_version"],
            STEP5_FACTUAL_EVAL_CONTROL_SCHEMA_VERSION,
        )
        self.assertEqual(int(out.loc[0, "route_scorer"]), 1)
        self.assertEqual(int(out.loc[0, "route_explainer"]), 1)
        self.assertFalse(bool(out.loc[0, "step5_control_is_rcr_posterior"]))
        self.assertFalse(bool(out.loc[0, "step5_control_is_step4_export_posterior"]))
        contract = step5_factual_eval_control_contract("valid")
        self.assertEqual(contract["mode"], "factual_eval_default")
        self.assertFalse(contract["is_rcr_posterior"])
        self.assertFalse(contract["is_train_route"])

    def test_factual_eval_defaults_are_rejected_on_train_export_path(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx_global": [0],
                "item_idx_global": [1],
                "rating": [4.0],
                "explanation": ["target factual explanation"],
                "clean_text": ["target factual explanation"],
                "sample_origin": ["target_gold"],
                "train_keep": [1],
            }
        )
        out = _apply_step5_factual_eval_default_controls(df, split_label="valid")
        with self.assertRaisesRegex(ValueError, "factual_eval_default rows"):
            _require_step5_rcr_posterior_controls(out, ctx="unit-test train path")

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


if __name__ == "__main__":
    unittest.main()
