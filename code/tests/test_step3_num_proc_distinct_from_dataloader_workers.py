from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402


class TestStep3NumProcDistinctFromDataloaderWorkers(unittest.TestCase):
    def test_tokenization_num_proc_is_not_train_workers_per_rank(self) -> None:
        _cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        self.assertEqual(snapshot["hardware"]["tokenization_num_proc"], 8)
        self.assertEqual(snapshot["hardware"]["dataloader_num_workers_train"], 4)
        self.assertNotEqual(snapshot["hardware"]["tokenization_num_proc"], snapshot["hardware"]["dataloader_num_workers_train"])


if __name__ == "__main__":
    unittest.main()
