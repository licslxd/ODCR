"""Default Step3 show/dry-run views stay formal-only."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = Path(_CODE_DIR).resolve().parent


def _run_odcr(*args: str) -> str:
    cmd = [sys.executable, "code/odcr.py", *args]
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == -9:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.check_returncode()
    return proc.stdout


def _first_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    obj, _idx = decoder.raw_decode(text.lstrip())
    return obj


class TestStep3ShowFormalOnly(unittest.TestCase):
    def test_task2_default_show_is_g1s_formal_only(self) -> None:
        out = _run_odcr("show", "--stage", "step3", "--task", "2")
        payload = _first_json_object(out)
        text = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(payload["train"]["candidate"], "G1S-sidecar-stable")
        self.assertEqual(payload["task"]["task_profile_id"], "task2_strong_forward_g1s")
        self.assertNotIn("step3_backup_profiles", payload)
        self.assertNotIn("step3_exploration_profiles", payload)
        self.assertNotIn("task2_g0_backup", text)
        self.assertNotIn("task2_g2_effective_pool_2048", text)
        self.assertNotIn("performance_probe", text)
        self.assertNotIn("short_pilot", text)

    def test_task2_verbose_show_exposes_nondefault_profiles(self) -> None:
        out = _run_odcr("show", "--stage", "step3", "--task", "2", "--verbose")
        payload = _first_json_object(out)
        self.assertIn("step3_backup_profiles", payload)
        self.assertIn("step3_exploration_profiles", payload)
        self.assertTrue(payload["step3_backup_profiles"]["task2_g1_backup"]["backup_only"])
        self.assertTrue(payload["step3_backup_profiles"]["task2_g0_backup"]["backup_only"])
        self.assertTrue(payload["step3_exploration_profiles"]["task2_g2_effective_pool_2048"]["probe_only"])

    def test_task2_dry_run_uses_formal_payload(self) -> None:
        out = _run_odcr("step3", "--task", "2", "--dry-run")
        payload = _first_json_object(out)
        text = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(payload["train"]["candidate"], "G1S-sidecar-stable")
        self.assertNotIn("task2_g0_backup", text)
        self.assertNotIn("task2_g2_effective_pool_2048", text)


if __name__ == "__main__":
    unittest.main()
