from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step5_prompt_templates import default_prompt_registry  # noqa: E402


class Step5PromptTemplatesTest(unittest.TestCase):
    def test_task_decoupled_templates_and_fixed_eval_policy(self) -> None:
        registry = default_prompt_registry()
        manifest = registry.manifest()
        self.assertEqual(manifest["template_count"], 4)
        ids = {item["canonical_id"] for item in manifest["templates"]}
        self.assertEqual(
            ids,
            {
                "A_target_gold_scorer_v1",
                "B_target_gold_explainer_v1",
                "B_aux_gold_explainer_v1",
                "B_aux_cf_explainer_v1",
            },
        )
        with self.assertRaises(KeyError):
            registry.render(sample={"sample_id": 1}, task_head="step5A", sample_origin="aux_gold", seed=9)
        valid = registry.render(sample={"sample_id": 1}, task_head="step5B", sample_origin="target_gold", seed=9, split="test")
        self.assertEqual(valid["step5_prompt_mode"], "fixed_canonical")
        self.assertIn("CCV", manifest["does_not_replace"])

    def test_no_internal_rcr_field_names_in_prompt_text(self) -> None:
        for item in default_prompt_registry().manifest()["templates"]:
            for forbidden in ("route_scorer", "route_explainer", "confidence_bucket", "sample_weight_hint"):
                self.assertNotIn(forbidden, item["text"])


if __name__ == "__main__":
    unittest.main()
