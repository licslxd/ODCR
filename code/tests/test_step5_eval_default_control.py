from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from executors.step5_engine import (  # noqa: E402
    STEP5_FACTUAL_EVAL_CONTROL_SCHEMA_VERSION,
    STEP5_CONTENT_EVIDENCE_POLICY_NEUTRAL_CONSTANT,
    STEP5_GENERATION_INPUT_POLICY_D4C_COMPAT,
    STEP5_GENERATION_INPUT_POLICY_NO_REF,
    STEP5_REFERENCE_USAGE_METRIC_ONLY,
    _apply_step5_factual_eval_default_controls,
    _apply_step5_paper_compatible_eval_default_controls,
    _build_step5_easd_hss_rcr_evidence_card,
    _require_step5_rcr_posterior_controls,
    step5_factual_eval_control_contract,
)


class TestStep5FactualEvalDefaultControl(unittest.TestCase):
    def _with_train_history(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        for dataset in ("Aux", "Target"):
            d = root / dataset
            d.mkdir(parents=True)
            (d / "train.csv").write_text(
                "user,item,explanation\n"
                "u1,i1,cozy service historic lobby\n"
                "u2,i2,fresh tacos quick lunch\n",
                encoding="utf-8",
            )
        patch = mock.patch("executors.step5_engine.get_data_dir", return_value=str(root))
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(tmp.cleanup)

    @staticmethod
    def _eval_cfg() -> dict:
        return {
            "generation_input_policy": STEP5_GENERATION_INPUT_POLICY_NO_REF,
            "content_evidence_policy": "train_only_history",
            "reference_usage": STEP5_REFERENCE_USAGE_METRIC_ONLY,
            "neutral_content_evidence": "keywords none ; aspects none ; entities none",
            "forbid_current_eval_fields_in_generation": [
                "explanation",
                "review",
                "clean_text",
                "raw_ref_text",
                "ref_text",
                "metric_ref_text",
                "content_evidence",
                "style_evidence",
            ],
        }

    def test_factual_eval_defaults_are_labeled_not_rcr_posterior(self) -> None:
        self._with_train_history()
        df = pd.DataFrame(
            {
                "user_idx_global": [0],
                "item_idx_global": [1],
                "user": ["u1"],
                "item": ["i1"],
                "rating": [4.0],
                "explanation": ["target factual explanation"],
                "clean_text": ["target factual explanation"],
                "content_evidence": ["oracle current row secret"],
                "style_evidence": ["oracle style"],
            }
        )
        out = _apply_step5_factual_eval_default_controls(
            df,
            split_label="valid",
            auxiliary="Aux",
            target="Target",
            final_eval_config=self._eval_cfg(),
        )
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
        self.assertEqual(out.loc[0, "generation_input_policy"], STEP5_GENERATION_INPUT_POLICY_NO_REF)
        self.assertEqual(out.loc[0, "step5_prompt_input_role"], "easd_hss_rcr_no_ref_control_card")
        self.assertEqual(out.loc[0, "hss_style_state_source"], "target_train_style_state_by_rating_polarity")
        self.assertFalse(bool(out.loc[0, "reference_before_generation"]))
        self.assertTrue(bool(out.loc[0, "reference_after_generation_metric_only"]))
        self.assertFalse(bool(out.loc[0, "current_eval_row_content_evidence_used_in_generation"]))
        self.assertNotIn("oracle current row secret", str(out.loc[0, "content_evidence"]))

    def test_paper_compatible_eval_uses_neutral_constant_not_history_evidence(self) -> None:
        df = pd.DataFrame(
            {
                "user_idx_global": [0],
                "item_idx_global": [1],
                "user": ["u1"],
                "item": ["i1"],
                "rating": [4.0],
                "explanation": ["target factual explanation"],
                "review": ["current review must not enter generation"],
                "clean_text": ["target factual explanation"],
                "content_evidence": ["oracle current row secret"],
                "style_evidence": ["oracle style"],
            }
        )
        out = _apply_step5_paper_compatible_eval_default_controls(
            df,
            split_label="valid",
            auxiliary="Aux",
            target="Target",
            final_eval_config={
                "generation_input_policy": STEP5_GENERATION_INPUT_POLICY_D4C_COMPAT,
                "content_evidence_policy": STEP5_CONTENT_EVIDENCE_POLICY_NEUTRAL_CONSTANT,
                "reference_usage": STEP5_REFERENCE_USAGE_METRIC_ONLY,
                "neutral_content_evidence": "keywords none ; aspects none ; entities none",
                "forbid_current_eval_fields_in_generation": [
                    "explanation",
                    "review",
                    "clean_text",
                    "raw_ref_text",
                    "ref_text",
                    "metric_ref_text",
                    "content_evidence",
                    "style_evidence",
                ],
            },
        )
        self.assertEqual(out.loc[0, "generation_input_policy"], STEP5_GENERATION_INPUT_POLICY_D4C_COMPAT)
        self.assertEqual(out.loc[0, "content_evidence_policy"], STEP5_CONTENT_EVIDENCE_POLICY_NEUTRAL_CONSTANT)
        self.assertEqual(out.loc[0, "content_evidence_source"], "neutral_constant_paper_compatible")
        self.assertEqual(out.loc[0, "content_evidence"], "keywords none ; aspects none ; entities none")
        self.assertNotIn("oracle current row secret", str(out.loc[0, "content_evidence"]))

    def test_factual_eval_defaults_are_rejected_on_train_export_path(self) -> None:
        self._with_train_history()
        df = pd.DataFrame(
            {
                "user_idx_global": [0],
                "item_idx_global": [1],
                "user": ["u1"],
                "item": ["i1"],
                "rating": [4.0],
                "explanation": ["target factual explanation"],
                "clean_text": ["target factual explanation"],
                "sample_origin": ["target_gold"],
                "train_keep": [1],
            }
        )
        out = _apply_step5_factual_eval_default_controls(
            df,
            split_label="valid",
            auxiliary="Aux",
            target="Target",
            final_eval_config=self._eval_cfg(),
        )
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

    def test_compact_anchor_card_uses_copyable_fields_without_generic_fillers(self) -> None:
        card = _build_step5_easd_hss_rcr_evidence_card(
            {
                "content_evidence": "Item evidence: good food; smoky brisket; patio seating; quick lunch.",
                "domain_style_anchor": "short-positive-service",
                "polarity_anchor": "positive",
                "confidence_bucket": 2,
                "cf_reliability_score": 0.82,
                "uncertainty_score": 0.17,
                "route_explainer": 1,
                "step5_control_mode": "rcr_posterior",
                "sample_origin": "target_gold",
            }
        )
        self.assertIn("C: smoky brisket | patio seating | quick lunch", card)
        self.assertIn("S: short-positive-service positive high plain", card)
        self.assertIn("R: rel=0.82 unc=0.17 route=exp", card)
        self.assertIn("Y: write 10-14 words", card)
        self.assertNotIn("good food |", card)


if __name__ == "__main__":
    unittest.main()
