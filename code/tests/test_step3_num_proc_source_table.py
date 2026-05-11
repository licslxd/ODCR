from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.manifests import build_formal_source_table_snapshot  # noqa: E402


class TestStep3NumProcSourceTable(unittest.TestCase):
    def test_tokenization_num_proc_source_table_records_auto_resolver(self) -> None:
        _cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        table = build_formal_source_table_snapshot(snapshot)
        records = {row["key"]: row["source"] for row in table["records"]}
        self.assertEqual(records["hardware.max_num_proc"], "hardware.profiles.default.max_num_proc")
        self.assertEqual(records["hardware.reserved_cpu"], "hardware.profiles.default.reserved_cpu")
        self.assertEqual(records["hardware.tokenization_num_proc"], "hardware.profiles.default.num_proc auto resolver")


if __name__ == "__main__":
    unittest.main()
