from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_rating_seed_runner import run_step3_rating_seed_runner  # noqa: E402


class Step3RatingSeedRunnerGpuLauncherTest(unittest.TestCase):
    def test_gpu_launch_is_direct_odcr_nohup_without_shell_launcher(self) -> None:
        self.assertFalse((ROOT / "test" / "step3_rating_task2_multi_run10_5seed_gpu_launcher.sh").exists())
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "odcr").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            result = run_step3_rating_seed_runner(
                repo,
                task=2,
                mode="multi",
                config_path="configs/odcr.yaml",
                run_id="2",
                dry_run=True,
            )
        launch = result["direct_odcr_nohup"]
        text = "\n".join(launch["command"])
        self.assertIn("nohup ./odcr step3-rating --task 2 --mode multi --run-id 2", text)
        self.assertIn("step3_rating_task2_eval_5seed_run2.driver.nohup.log", text)
        self.assertIn("step3_rating_task2_eval_5seed_run2.driver.pid", text)
        self.assertIn("runs/step3/task2/eval/5", result["eval_namespace"])
        self.assertNotIn("missing4seeds", text)
        self.assertNotIn("reuse_existing", text)
        self.assertNotIn(".sh", text)


if __name__ == "__main__":
    unittest.main()
