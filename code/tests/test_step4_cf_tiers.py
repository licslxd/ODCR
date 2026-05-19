from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.gold_quality import assign_cf_tiers  # noqa: E402


def _cf_row(**kw) -> dict:
    row = {
        "sample_origin": "aux_cf",
        "clean_text": "counterfactual style shifted review with meaningful evidence",
        "rating": 4.0,
        "route_scorer": 1,
        "route_explainer": 1,
        "confidence_bucket": 2,
        "sample_weight_hint": 0.0,
        "cf_reliability_score": 0.82,
        "content_retention_score": 0.945,
        "style_shift_score": 0.32,
        "rating_stability_score": 0.98,
        "uncertainty_score": 0.03,
        "text_quality_score": 0.95,
        "bad_tail_hit": 0,
        "template_hard_drop_hit": 0,
        "short_fragment_hit": 0,
        "repeat_tail_hit": 0,
    }
    row.update(kw)
    return row


class CFTierTest(unittest.TestCase):
    def test_cf_tiers_do_not_use_sample_weight_as_sole_gate(self) -> None:
        out = assign_cf_tiers(pd.DataFrame([_cf_row(sample_weight_hint=0.0)]))
        self.assertEqual(out.loc[0, "cf_tier_step5A"], "high")
        self.assertEqual(out.loc[0, "cf_tier_step5B"], "high")

    def test_low_weighted_and_hard_reject_are_distinct(self) -> None:
        out = assign_cf_tiers(
            pd.DataFrame(
                [
                    _cf_row(route_scorer=0, route_explainer=0, confidence_bucket=0, style_shift_score=0.12),
                    _cf_row(clean_text="", uncertainty_score=0.99, text_quality_score=0.0),
                ]
            )
        )
        self.assertEqual(out.loc[0, "cf_tier_step5B"], "low_weighted")
        self.assertEqual(out.loc[1, "cf_tier_step5A"], "reject")
        self.assertIn("recommended_sampling_weight_step5B", out.columns)


if __name__ == "__main__":
    unittest.main()
