"""Step3 One-Control control-plane regression tests."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.config_schema import OneControlConfigError  # noqa: E402
from odcr_core.logging_meta import _stage_label  # noqa: E402
from odcr_core.manifests import build_run_manifest  # noqa: E402

_REPO_ROOT = Path(_CODE_DIR).resolve().parent


def _resolve_step3_task4(set_overrides: list[str] | None = None):
    return resolve_config(
        config_path=_REPO_ROOT / "configs" / "odcr.yaml",
        command="step3",
        task_id=4,
        set_overrides=set_overrides or [],
        dry_run=True,
        run_id="auto",
        mode="full",
    )


class TestStep3ControlPlane(unittest.TestCase):
    def test_step3_resolves_from_one_control_without_retired_adv_payload(self) -> None:
        cfg, sources, snapshot = _resolve_step3_task4()
        payload = json.loads(cfg.effective_training_payload_json)

        self.assertEqual(cfg.command, "step3")
        self.assertEqual(snapshot["train"]["stage"], "step3")
        self.assertEqual(snapshot["field_sources"]["config"], str(_REPO_ROOT / "configs" / "odcr.yaml"))
        self.assertTrue(all("presets/" not in str(record.source) for record in sources))

        self.assertEqual(cfg.adv, 0.0)
        self.assertEqual(cfg.eta, 0.0)
        self.assertEqual(payload["preset_name"], "step3")
        self.assertEqual(payload["explainer_loss_weight"], 0.0)
        self.assertIn("step3_structured_losses", payload)
        self.assertNotIn("adv", payload["training_row"])
        self.assertNotIn("eta", payload["training_row"])

    def test_step3_rejects_explicit_retired_adv_control(self) -> None:
        with self.assertRaisesRegex(OneControlConfigError, "retired"):
            _resolve_step3_task4(["step3.train.adv=0.9"])

    def test_step3_manifest_and_banner_use_structured_semantics(self) -> None:
        cfg, _, _ = _resolve_step3_task4()
        manifest = build_run_manifest(cfg)

        self.assertEqual(_stage_label("step3"), "step3（结构化 shared/specific 解耦）")
        self.assertEqual(manifest["stage"], "step3_structured_disentanglement")
        self.assertNotIn("eta", manifest["hyperparameters"])
        self.assertNotIn("adv", manifest["hyperparameters"])

    def test_retired_typed_bridge_files_are_deleted(self) -> None:
        for name in ("step3_runtime.py", "step3_registry.py"):
            with self.subTest(name=name):
                self.assertFalse((_REPO_ROOT / "code" / "odcr_core" / name).exists())

        runners_source = (_REPO_ROOT / "code" / "odcr_core" / "runners.py").read_text(encoding="utf-8")
        self.assertNotIn("step3_runtime", runners_source)
        self.assertNotIn("step3_registry", runners_source)
        self.assertNotIn("instantiate_" + "step3_preset", runners_source)


if __name__ == "__main__":
    unittest.main()
