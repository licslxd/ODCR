from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402
from executors.step3_train_core import (  # noqa: E402
    Step3StructuredLossWeights,
    apply_step3_numerical_stability_warmup,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _weights(value: float = 1.0) -> Step3StructuredLossWeights:
    return Step3StructuredLossWeights(
        orthogonal_weight=value,
        orthogonal_xcov_weight=value,
        orthogonal_cosine_weight=value,
        variance_weight=value,
        shared_invariance_weight=value,
        specific_separation_weight=value,
        anchor_alignment_weight=value,
        content_alignment_weight=value,
        style_alignment_weight=value,
        shared_prototype_weight=value,
        domain_style_alignment_weight=value,
        local_style_alignment_weight=value,
        polarity_alignment_weight=value,
        residual_specific_weight=value,
        prototype_separation_weight=value,
        light_explainer_weight=value,
    )


class Step3CsbAuxiliaryWarmupTest(unittest.TestCase):
    def test_safe704_numerical_warmup_resolves_and_applies_epoch2(self) -> None:
        cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=["experiment_profile=csb_odcr_full_safe"],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        self.assertTrue(snapshot["step3_numerical_stability"]["enabled"])
        self.assertEqual(snapshot["step3_numerical_stability"]["auxiliary_warmup"]["schedule"]["epoch2"], 0.75)
        self.assertEqual(snapshot["step3_numerical_stability"]["light_explainer_warmup"]["schedule"]["epoch2"], 0.7)
        self.assertEqual(snapshot["step3_scheduler"]["warmup_ratio"], 0.10)

        final_cfg = SimpleNamespace(numerical_stability_config_json=cfg.numerical_stability_config_json)
        warmed, summary = apply_step3_numerical_stability_warmup(_weights(1.0), final_cfg=final_cfg, epoch=2)

        self.assertAlmostEqual(warmed.orthogonal_weight, 0.75)
        self.assertAlmostEqual(warmed.light_explainer_weight, 0.75 * 0.7)
        self.assertAlmostEqual(summary["conflict_routing_multiplier"], 0.6)
        self.assertAlmostEqual(summary["controlled_injection_multiplier"], 0.75)
        self.assertNotIn("L_rating_shared", summary["component_multipliers"])

    def test_disabled_warmup_is_identity(self) -> None:
        final_cfg = SimpleNamespace(numerical_stability_config_json=json.dumps({"enabled": False}))
        warmed, summary = apply_step3_numerical_stability_warmup(_weights(0.3), final_cfg=final_cfg, epoch=1)

        self.assertFalse(summary["enabled"])
        self.assertAlmostEqual(warmed.light_explainer_weight, 0.3)
        self.assertEqual(summary["component_multipliers"], {})


if __name__ == "__main__":
    unittest.main()
