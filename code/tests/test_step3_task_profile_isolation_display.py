"""Task-local Step3 profile labels do not leak task2 ladder names."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
_REPO_ROOT = Path(_CODE_DIR).resolve().parent

from odcr_core.config_resolver import resolve_config  # noqa: E402


def _show(task_id: int) -> str:
    proc = subprocess.run(
        [sys.executable, "code/odcr.py", "show", "--stage", "step3", "--task", str(task_id)],
        cwd=_REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def _first_json_object(text: str) -> dict:
    obj, _idx = json.JSONDecoder().raw_decode(text.lstrip())
    return obj


class TestStep3TaskProfileIsolationDisplay(unittest.TestCase):
    def test_non_task2_show_outputs_are_profile_local(self) -> None:
        for task_id in (5, 8, 7):
            with self.subTest(task_id=task_id):
                out = _show(task_id)
                self.assertNotIn("task2_g1", out)
                self.assertNotIn("task2_g0", out)
                self.assertNotIn("task2 ladder", out.lower())
                payload = _first_json_object(out)
                role = payload["train"]["step3_batch_candidate_role"]
                self.assertIn(payload["task"]["task_profile_id"], role)

    def test_profile_isolation_hashes_are_task_bound(self) -> None:
        hashes: dict[int, str] = {}
        for task_id in (2, 5, 8, 7):
            cfg, _sources, snapshot = resolve_config(
                config_path=_REPO_ROOT / "configs" / "odcr.yaml",
                command="step3",
                task_id=task_id,
                set_overrides=[],
                dry_run=True,
                run_id="auto",
                mode="full",
            )
            self.assertEqual(cfg.profile_isolation_hash, snapshot["task"]["profile_isolation_hash"])
            hashes[task_id] = cfg.profile_isolation_hash
        self.assertEqual(len(set(hashes.values())), len(hashes))


if __name__ == "__main__":
    unittest.main()
