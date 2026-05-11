from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.manifests import build_formal_source_table_snapshot  # noqa: E402


RETIRED = {"grad_accum", "gradient_accumulation_steps", "accumulate_grad_batches", "accum_steps", "accumulation_steps"}


def _assert_no_retired_keys(testcase: unittest.TestCase, value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_s = str(key)
            testcase.assertNotIn(key_s, RETIRED, f"retired key at {path}.{key_s}")
            _assert_no_retired_keys(testcase, child, f"{path}.{key_s}" if path else key_s)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _assert_no_retired_keys(testcase, child, f"{path}[{idx}]")


class NoAccumBatchSemanticsTest(unittest.TestCase):
    def test_active_yaml_has_no_accumulation_fields_and_formula_holds(self) -> None:
        raw = yaml.safe_load((REPO_ROOT / "configs" / "odcr.yaml").read_text(encoding="utf-8"))
        _assert_no_retired_keys(self, raw)
        ddp_world = int(raw["hardware"]["profiles"]["default"]["ddp_world_size"])
        for stage in ("step3", "step4", "step5"):
            train = raw[stage]["train"]
            self.assertNotIn("micro_batch_size", train)
            self.assertEqual(int(train["batch_size"]), int(train["per_gpu_batch_size"]) * ddp_world)

    def test_resolved_snapshot_and_source_table_are_no_accum_clean(self) -> None:
        _cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="1",
            mode="full",
        )
        train = snapshot["train"]
        self.assertEqual(train["batch_semantics_version"], "odcr_no_accum/1")
        self.assertTrue(train["grad_accum_removed"])
        self.assertEqual(train["batch_formula"], "global_batch_size = per_gpu_batch_size * ddp_world_size")
        self.assertEqual(train["global_batch_size"], train["per_gpu_batch_size"] * train["ddp_world_size"])
        for key in RETIRED:
            self.assertNotIn(key, train)
        source_table = build_formal_source_table_snapshot(snapshot)
        _assert_no_retired_keys(self, source_table)


if __name__ == "__main__":
    unittest.main()
