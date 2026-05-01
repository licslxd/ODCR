from __future__ import annotations

import os
import sys
import unittest

import pandas as pd

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from odcr_core.odcr_cf_routing import ODCFRoutingConfig, attach_odcr_cf_routing  # noqa: E402
from odcr_core.step4_training_export import assemble_step4_training_table  # noqa: E402


def _target_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": ["u1", "u2"],
            "item": ["i1", "i2"],
            "user_idx": [0, 1],
            "item_idx": [0, 1],
            "rating": [5.0, 4.0],
            "review": [
                "camera lens battery and focus are strong",
                "camera lens battery and focus are strong",
            ],
            "explanation": [
                "The camera keeps sharp focus and battery life.",
                "The camera keeps sharp focus and battery life.",
            ],
            "domain": ["target", "target"],
            "content_evidence": [
                "keywords camera lens battery focus ; aspects image battery focus ; entities camera",
                "keywords camera lens battery focus ; aspects image battery focus ; entities camera",
            ],
            "content_anchor_score": [0.90, 0.90],
            "polarity_anchor": ["positive", "positive"],
            "domain_style_anchor": [
                "target:causal_statement:medium:positive",
                "target:causal_statement:medium:positive",
            ],
            "local_style_residual_hint": [
                "perspective=external;intensity=strong;discourse=causal;punctuation=flat",
                "perspective=external;intensity=strong;discourse=causal;punctuation=flat",
            ],
            "style_evidence": [
                "markers warm concise ; template_family causal_statement ; polarity positive",
                "markers warm concise ; template_family causal_statement ; polarity positive",
            ],
            "style_anchor_score": [0.85, 0.85],
            "evidence_quality_prior": [0.20, 0.80],
            "preprocess_route_scorer_prior": [0, 0],
            "preprocess_route_explainer_prior": [0, 0],
        }
    )


class TestStep4RCRRouting(unittest.TestCase):
    def _cfg(self) -> ODCFRoutingConfig:
        return ODCFRoutingConfig.for_test_default()

    def test_latent_rcr_routes_and_preserves_prior(self) -> None:
        target = _target_rows()
        cf = target.copy()
        cf["domain"] = "auxiliary"
        cf["explanation"] = [
            "Sharp camera focus with battery life in a lively concise voice.",
            "Sharp camera focus with battery life in a lively concise voice.",
        ]
        cf["entropy"] = [0.10, 0.10]
        cf["rating_delta"] = [0.10, 0.80]
        cf["shared_latent_similarity"] = [0.92, 0.92]
        cf["specific_latent_shift"] = [0.72, 0.72]

        routed = attach_odcr_cf_routing(target, cf, cfg=self._cfg())

        self.assertIn("content_retention_score", routed.columns)
        self.assertNotIn("content_preserve_score", routed.columns)
        self.assertEqual(list(routed["evidence_quality_prior"]), [0.20, 0.80])
        self.assertEqual(list(routed["preprocess_route_scorer_prior"]), [0, 0])
        self.assertEqual(list(routed["preprocess_route_explainer_prior"]), [0, 0])
        self.assertEqual(int(routed.loc[0, "route_scorer"]), 1)
        self.assertEqual(int(routed.loc[0, "route_explainer"]), 1)
        self.assertEqual(int(routed.loc[1, "route_scorer"]), 0)
        self.assertEqual(str(routed.loc[1, "route_reason_scorer"]), "rating_delta_gt_0.5")
        self.assertGreaterEqual(float(routed.loc[1, "route_explainer"]), 0.0)
        self.assertNotEqual(
            round(float(routed.loc[0, "uncertainty_score"]), 4),
            round(1.0 - float(routed.loc[0, "cf_reliability_score"]), 4),
        )

    def test_assembly_uses_step4_posterior_not_preprocess_route_hints(self) -> None:
        target = _target_rows()
        cf = target.copy()
        cf["domain"] = "auxiliary"
        cf["explanation"] = [
            "Sharp camera focus with battery life in a lively concise voice.",
            "Sharp camera focus with battery life in a lively concise voice.",
        ]
        cf["entropy"] = [0.10, 0.10]
        cf["rating_delta"] = [0.10, 0.80]
        cf["shared_latent_similarity"] = [0.92, 0.92]
        cf["specific_latent_shift"] = [0.72, 0.72]
        cfg = self._cfg()
        routed = attach_odcr_cf_routing(target, cf, cfg=cfg)

        out = assemble_step4_training_table(target, routed, rcr_config=cfg, template_min_count=99)
        gold = out[out["sample_origin"] == "target_gold"].reset_index(drop=True)
        self.assertTrue((gold["route_scorer"].astype(int) == 1).all())
        self.assertTrue((gold["route_explainer"].astype(int) == 1).all())
        self.assertEqual(list(gold["evidence_quality_prior"].round(2)), [0.20, 0.80])
        self.assertEqual(list(gold["preprocess_route_scorer_prior"].astype(int)), [0, 0])
        self.assertEqual(list(gold["preprocess_route_explainer_prior"].astype(int)), [0, 0])
        self.assertTrue((out["content_retention_score"].astype(float) >= 0.0).all())
        self.assertNotIn("content_preserve_score", out.columns)

    def test_assembly_rejects_entropy_only_cf_rows(self) -> None:
        target = _target_rows()
        old_cf = target.copy()
        old_cf["domain"] = "auxiliary"
        old_cf["entropy"] = [0.1, 0.2]
        with self.assertRaisesRegex(ValueError, "RCR posterior fields"):
            assemble_step4_training_table(target, old_cf, rcr_config=self._cfg(), template_min_count=99)

    def test_routing_rejects_missing_live_latent_diagnostics(self) -> None:
        target = _target_rows()
        old_cf = target.copy()
        old_cf["domain"] = "auxiliary"
        old_cf["entropy"] = [0.1, 0.2]
        with self.assertRaisesRegex(ValueError, "requires live latent/rating diagnostics"):
            attach_odcr_cf_routing(target, old_cf, cfg=self._cfg())

    def test_routing_rejects_nonfinite_live_latent_diagnostics(self) -> None:
        target = _target_rows()
        cf = target.copy()
        cf["domain"] = "auxiliary"
        cf["entropy"] = [0.1, 0.2]
        cf["rating_delta"] = [0.10, float("nan")]
        cf["shared_latent_similarity"] = [0.92, 0.92]
        cf["specific_latent_shift"] = [0.72, 0.72]
        with self.assertRaisesRegex(ValueError, "requires finite live latent/rating diagnostics"):
            attach_odcr_cf_routing(target, cf, cfg=self._cfg())

    def test_one_control_rcr_config_changes_route_threshold(self) -> None:
        target = _target_rows()
        cf = target.copy()
        cf["domain"] = "auxiliary"
        cf["explanation"] = [
            "Sharp camera focus with battery life in a lively concise voice.",
            "Sharp camera focus with battery life in a lively concise voice.",
        ]
        cf["entropy"] = [0.10, 0.10]
        cf["rating_delta"] = [0.10, 0.10]
        cf["shared_latent_similarity"] = [0.92, 0.92]
        cf["specific_latent_shift"] = [0.72, 0.72]
        cfg = ODCFRoutingConfig.from_mapping(
            {"route_scorer": {"min_reliability": 0.99}},
            allow_test_defaults=True,
        )
        routed = attach_odcr_cf_routing(target, cf, cfg=cfg)
        self.assertTrue((routed["route_scorer"].astype(int) == 0).all())


if __name__ == "__main__":
    unittest.main()
