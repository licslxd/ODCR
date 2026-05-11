"""Step3 clean-baseline static checks."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
_REPO_ROOT = Path(_CODE_DIR).resolve().parent


class TestStep3CleanBaselineNoLegacyParams(unittest.TestCase):
    def test_step3_active_config_has_no_ladder_probe_or_old_controls(self) -> None:
        text = (_REPO_ROOT / "configs" / "odcr.yaml").read_text(encoding="utf-8")
        step3 = text.split("\nstep4:", 1)[0].split("\nstep3:", 1)[1]
        for term in (
            "smoke_ladder:",
            "performance_ladder:",
            "performance_probe:",
            "short_pilot:",
            "grad_accum:",
            "gradient_accumulation_steps:",
            "adv:",
            "eta:",
            "max_length: 25",
            "max_evidence_length: 24",
        ):
            self.assertNotIn(term, step3)

    def test_step3_schema_and_parser_do_not_expose_removed_surfaces(self) -> None:
        schema = (_REPO_ROOT / "code" / "odcr_core" / "config_schema.py").read_text(encoding="utf-8")
        resolver = (_REPO_ROOT / "code" / "odcr_core" / "config_resolver.py").read_text(encoding="utf-8")
        core = (_REPO_ROOT / "code" / "executors" / "step3_train_core.py").read_text(encoding="utf-8")
        for term in (
            "smoke_ladder_config_json",
            "performance_ladder_config_json",
            "performance_probe_config_json",
            "short_pilot_config_json",
            "_resolve_step3_ladder_config",
            "_resolve_step3_performance_probe_config",
            "_resolve_step3_short_pilot_config",
            "STEP3_GRAD_ACCUM_REMOVED_MESSAGE",
        ):
            self.assertNotIn(term, schema + resolver + core)
        self.assertNotIn('"--coef"', core)
        self.assertNotIn('"--gradient-accumulation-steps"', core)


if __name__ == "__main__":
    unittest.main()
