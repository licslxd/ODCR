from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import OneControlConfigError, resolve_config  # noqa: E402


MESSAGE = (
    "grad_accum has been removed in ODCR no-accum architecture; use per_gpu_batch_size "
    "and global_batch_size = per_gpu_batch_size * ddp_world_size."
)


class GradAccumRemovedTest(unittest.TestCase):
    def _clean_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for name in ("ODCR_GRAD_ACCUM", "ODCR_GRADIENT_ACCUMULATION_STEPS", "ODCR_ACCUMULATE_GRAD_BATCHES"):
            env.pop(name, None)
        return env

    def test_resolver_rejects_retired_set_overrides(self) -> None:
        for override in (
            "step3.train.grad_accum=1",
            "step3.train.gradient_accumulation_steps=1",
            "step3.train.accumulate_grad_batches=1",
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(OneControlConfigError, "removed in ODCR no-accum"):
                    resolve_config(
                        config_path=REPO_ROOT / "configs" / "odcr.yaml",
                        command="step3",
                        task_id=2,
                        set_overrides=[override],
                        dry_run=True,
                        run_id="1",
                        mode="full",
                    )

    def test_retired_cli_options_fail_fast(self) -> None:
        for option in ("--grad-accum", "--gradient-accumulation-steps", "--accumulate-grad-batches"):
            with self.subTest(option=option):
                proc = subprocess.run(
                    ["./odcr", "step3", "--task", "2", "--dry-run", option, "2"],
                    cwd=REPO_ROOT,
                    env=self._clean_env(),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=60,
                )
                self.assertNotEqual(proc.returncode, 0, proc.stdout)
                self.assertIn(MESSAGE, proc.stdout)

    def test_retired_env_vars_fail_fast(self) -> None:
        for name in ("ODCR_GRAD_ACCUM", "ODCR_GRADIENT_ACCUMULATION_STEPS"):
            with self.subTest(name=name):
                env = self._clean_env()
                env[name] = "2"
                proc = subprocess.run(
                    ["./odcr", "step3", "--task", "2", "--dry-run"],
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=60,
                )
                self.assertNotEqual(proc.returncode, 0, proc.stdout)
                self.assertIn(MESSAGE, proc.stdout)


if __name__ == "__main__":
    unittest.main()
