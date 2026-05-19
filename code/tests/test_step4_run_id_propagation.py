from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core import config_resolver  # noqa: E402
from odcr_core.step4_runtime import next_available_step4_run_id  # noqa: E402
from odcr_core.stage_truth_antiforgery import write_step3_fixture  # noqa: E402


class Step4RunIdPropagationTest(unittest.TestCase):
    def test_step4_dry_run_auto_plans_next_integer_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            (repo / "runs" / "step4" / "task2" / "1" / "meta").mkdir(parents=True)
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                cfg, _, snapshot = config_resolver.resolve_config(
                    config_path=REPO_ROOT / "configs" / "odcr.yaml",
                    command="step4",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    from_step3="latest",
                )
            finally:
                config_resolver._REPO_ROOT = old_root
            self.assertEqual(cfg.step4_run, "2")
            self.assertTrue(str(cfg.checkpoint_dir).endswith("runs/step4/task2/2"))
            self.assertEqual(snapshot["run"]["from_step4"], "2")
            self.assertEqual(next_available_step4_run_id(cfg), 2)

    def test_explicit_existing_step4_run_id_is_not_overwritten_in_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            (repo / "runs" / "step4" / "task2" / "1").mkdir(parents=True)
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                with self.assertRaisesRegex(Exception, "already exists"):
                    config_resolver.resolve_config(
                        config_path=REPO_ROOT / "configs" / "odcr.yaml",
                        command="step4",
                        task_id=2,
                        set_overrides=[],
                        dry_run=True,
                        from_step3="latest",
                        run_id="1",
                    )
            finally:
                config_resolver._REPO_ROOT = old_root


if __name__ == "__main__":
    unittest.main()
