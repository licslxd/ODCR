from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import load_yaml_config, resolve_config
from odcr_core.step4_runtime import reject_step4_formal_env_overrides, step4_runtime_env


class Step4PerfKnobsOneControlTest(unittest.TestCase):
    def test_runtime_knobs_are_in_yaml_and_resolved_transport(self) -> None:
        raw = load_yaml_config("configs/odcr.yaml")
        self.assertIn("runtime", raw["step4"])
        cfg, _, snapshot = resolve_config(
            config_path="configs/odcr.yaml",
            command="step4",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            eval_profile="balanced_2gpu",
        )
        runtime = json.loads(cfg.step4_runtime_config_json)
        self.assertEqual(runtime["partial_format"], raw["step4"]["runtime"]["partial_format"])
        self.assertEqual(snapshot["step4_runtime"]["decode_chunk"], raw["step4"]["runtime"]["decode_chunk"])
        env = step4_runtime_env(cfg, mode="formal")
        self.assertIn("ODCR_STEP4_RUNTIME_CONFIG_JSON", env)

    def test_formal_bare_env_knob_fails_fast(self) -> None:
        with self.assertRaises(Exception):
            reject_step4_formal_env_overrides(
                mode="formal",
                environ={"ODCR_STEP4_DECODE_THREADS": "8"},
            )


if __name__ == "__main__":
    unittest.main()
