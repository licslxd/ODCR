from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.gold_quality import score_gold_quality  # noqa: E402


def _gold_row(i: int, *, origin: str = "target_gold", text: str = "thoughtful clean review text with enough evidence", **kw) -> dict:
    row = {
        "sample_origin": origin,
        "domain": "target" if origin == "target_gold" else "auxiliary",
        "clean_text": text,
        "rating": 4.0,
        "sample_weight_hint": 1.0 if origin == "target_gold" else 0.9,
        "text_quality_score": 0.95,
        "uncertainty_score": 0.05,
        "confidence_bucket": 2,
        "user_idx_global": i,
        "item_idx_global": i + 100,
        "content_evidence": "content",
        "style_evidence": "style",
        "domain_style_anchor": "anchor",
        "local_style_residual_hint": "hint",
        "polarity_anchor": "positive",
        "content_anchor_score": 0.9,
        "style_anchor_score": 0.85,
        "evidence_quality_prior": 0.9,
        "bad_tail_hit": 0,
        "template_hit": 0,
        "template_hard_drop_hit": 0,
        "short_fragment_hit": 0,
        "repeat_tail_hit": 0,
        "template_downweighted": 0,
        "noisy_tail_downweighted": 0,
    }
    row.update(kw)
    return row


class GoldQualityScoreTest(unittest.TestCase):
    def test_high_is_not_not_reject_and_medium_has_real_rule(self) -> None:
        df = pd.DataFrame(
            [
                _gold_row(200000, text="excellent detailed evidence with balanced content style and strong user fit"),
                _gold_row(1, text="short but usable", text_quality_score=0.75, sample_weight_hint=0.55),
                _gold_row(2, text="", bad_tail_hit=1),
            ]
        )
        out = score_gold_quality(df, {"high_min_score": 0.90, "medium_min_score": 0.45})
        self.assertEqual(out.loc[0, "gold_quality_tier"], "high")
        self.assertEqual(out.loc[1, "gold_quality_tier"], "medium")
        self.assertEqual(out.loc[2, "gold_quality_tier"], "reject")
        self.assertLess(out.loc[1, "gold_quality_score"], out.loc[0, "gold_quality_score"])
        self.assertIn("empty_text", out.loc[2, "hard_reject_flags"])

    def test_contract_columns_and_no_quota_randomness(self) -> None:
        rows = [_gold_row(i, origin="aux_gold" if i % 2 else "target_gold") for i in range(20, 30)]
        out1 = score_gold_quality(pd.DataFrame(rows))
        out2 = score_gold_quality(pd.DataFrame(rows))
        for col in (
            "gold_quality_score",
            "gold_quality_tier",
            "gold_quality_reasons",
            "hard_reject_flags",
            "coverage_bucket",
            "rating_bucket",
            "length_bucket",
            "domain_role",
            "recommended_sampling_weight",
        ):
            self.assertIn(col, out1.columns)
        self.assertEqual(out1["gold_quality_tier"].tolist(), out2["gold_quality_tier"].tolist())


if __name__ == "__main__":
    unittest.main()
