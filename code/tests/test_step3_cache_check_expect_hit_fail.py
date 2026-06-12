from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402
from tools.odcr_step3_cache_check import run_cache_check  # noqa: E402

REPO_ROOT = CODE_DIR.parent


class TestStep3CacheCheckExpectHitFail(unittest.TestCase):
    def test_expect_cache_hit_fails_for_missing_namespace(self) -> None:
        _cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=8,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("tools.odcr_step3_cache_check.REPO_ROOT", Path(tmpdir)):
                with self.assertRaises(SystemExit):
                    run_cache_check(
                        task_id=8,
                        expected_profile="task8_weak_forward_init",
                        expect_cache_hit=True,
                        resolved_snapshot=snapshot,
                    )


if __name__ == "__main__":
    unittest.main()
