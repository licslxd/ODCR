from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from odcr import build_parser
from odcr_core import config_resolver
from odcr_core.config_resolver import resolve_config
from odcr_core.run_naming import (
    allocate_step5_run_id,
    normalize_step5_run_id_for_step4,
    step4_slug_from_step5_slug,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class Step5HeadSplitCliTest(unittest.TestCase):
    def test_step5_head_choices_are_user_visible(self) -> None:
        parser = build_parser()
        for head in ("step5A", "step5B", "combined"):
            ns = parser.parse_args(["step5", "--task", "2", "--head", head, "--dry-run"])
            self.assertEqual(ns.command, "step5")
            self.assertEqual(ns.head, head)
        with self.assertRaises(SystemExit):
            parser.parse_args(["step5", "--task", "2", "--head", "probe", "--dry-run"])

    def test_head_aware_run_id_allocation_and_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            self.assertEqual(allocate_step5_run_id(parent, "1", head="step5A"), "1_1_step5A")
            (parent / "1_1_step5A").mkdir()
            self.assertEqual(allocate_step5_run_id(parent, "1", head="step5A"), "1_2_step5A")
            self.assertEqual(allocate_step5_run_id(parent, "1", head="step5B"), "1_1_step5B")
        self.assertEqual(step4_slug_from_step5_slug("1_1_step5A"), "1")
        self.assertEqual(step4_slug_from_step5_slug("1_1_step5B"), "1")
        with self.assertRaisesRegex(ValueError, "Step5 run-id must include consumed Step4 run prefix"):
            normalize_step5_run_id_for_step4("1", step4_run="1", head="step5A")
        self.assertEqual(
            normalize_step5_run_id_for_step4("1_1_step5A", step4_run="1", head="step5A"),
            "1_1_step5A",
        )

    def test_resolver_auto_generates_head_run_ids_and_candidate_contract(self) -> None:
        old_root = config_resolver._REPO_ROOT
        try:
            config_resolver._REPO_ROOT = REPO_ROOT
            for head, expected_run in (("step5A", "1_2_step5A"), ("step5B", "1_1_step5B")):
                cfg, _sources, snapshot = resolve_config(
                    config_path=REPO_ROOT / "configs" / "odcr.yaml",
                    command="step5",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    from_step4="1",
                    run_id="auto",
                    step5_head=head,
                )
                self.assertEqual(cfg.step5_run, expected_run)
                self.assertEqual(cfg.step5_head, head)
                self.assertEqual(snapshot["run"]["from_step4"], "1")
                self.assertEqual(snapshot["step5_head"], head)
                self.assertEqual(snapshot["train"]["batch_candidate"], "B224")
                self.assertEqual(snapshot["train"]["effective_samples"]["step5A"], 190646)
                self.assertEqual(snapshot["train"]["effective_samples"]["step5B"], 426541)
                self.assertEqual(snapshot["train"]["optimizer_steps"]["step5A"], 426)
                self.assertEqual(snapshot["train"]["optimizer_steps"]["step5B"], 953)
                self.assertEqual(
                    snapshot["selected_tuning_candidate"],
                    "A_RATIO_0+B_RATIO_0+A_CF_MIX_FORMAL_HIGH_ONLY+B_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_1e-3+W0",
                )
                self.assertEqual(
                    snapshot["step5_formal_active_candidate"]["step5A_cf_mix_id"],
                    "A_CF_MIX_FORMAL_HIGH_ONLY",
                )
                self.assertEqual(
                    snapshot["step5_formal_active_candidate"]["step5B_cf_mix_id"],
                    "B_CF_MIX_FORMAL_HIGH_MEDIUM",
                )
                self.assertFalse(snapshot["step5_sampler"]["full_audit_default_allowed"])
                self.assertFalse(snapshot["step5_sampler"]["legacy_gold_heavy_exports_allowed"])
        finally:
            config_resolver._REPO_ROOT = old_root


if __name__ == "__main__":
    unittest.main()
