"""Legacy Step3 keys enter through generic strict-schema rejection."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
_REPO_ROOT = Path(_CODE_DIR).resolve().parent

import yaml  # noqa: E402
from odcr_core.config_resolver import load_yaml_config, resolve_config  # noqa: E402
from odcr_core.config_schema import OneControlConfigError  # noqa: E402


def _resolve_with_sets(sets: list[str]) -> None:
    resolve_config(
        config_path=_REPO_ROOT / "configs" / "odcr.yaml",
        command="step3",
        task_id=2,
        set_overrides=sets,
        dry_run=True,
        run_id="auto",
        mode="full",
    )


class TestStep3LegacyUnknownKeyRejected(unittest.TestCase):
    def test_cli_set_legacy_keys_are_unknown_schema_errors(self) -> None:
        for key in (
            "step3.train.adv",
            "step3.train.eta",
            "step3.train.coef",
            "step3.performance_ladder.G1.batch_size",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(OneControlConfigError, "unsupported key"):
                    _resolve_with_sets([f"{key}=1"])
        for key in (
            "step3.train.grad_accum",
            "step3.train.gradient_accumulation_steps",
            "step3.train.accumulate_grad_batches",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(OneControlConfigError, "removed in ODCR no-accum"):
                    _resolve_with_sets([f"{key}=1"])

    def test_yaml_legacy_keys_are_unknown_schema_errors(self) -> None:
        raw = load_yaml_config(_REPO_ROOT / "configs" / "odcr.yaml")
        raw["step3"]["train"]["grad_accum"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "odcr.yaml"
            cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(OneControlConfigError, "removed in ODCR no-accum"):
                resolve_config(
                    config_path=cfg_path,
                    command="step3",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    run_id="auto",
                    mode="full",
                )


if __name__ == "__main__":
    unittest.main()
