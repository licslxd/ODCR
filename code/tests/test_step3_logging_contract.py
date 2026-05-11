"""Step3 run-meta logging contract tests."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
_REPO_ROOT = Path(_CODE_DIR).resolve().parent

from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.logging_meta import (  # noqa: E402
    append_error_log,
    console_summary_lines,
    initialize_run_log_files,
    run_log_paths,
)


class TestStep3LoggingContract(unittest.TestCase):
    def test_console_is_compact_and_full_is_authoritative(self) -> None:
        cfg, _sources, snapshot = resolve_config(
            config_path=_REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta"
            cfg = cfg.__class__(**{**cfg.__dict__, "manifest_dir": str(meta), "log_dir": str(meta)})
            paths = initialize_run_log_files(
                cfg,
                snapshot,
                command_line="./odcr step3 --task 2 --dry-run",
                started_at="2026-05-06T00:00:00Z",
                console_level="summary",
            )
            console = "\n".join(console_summary_lines(cfg, status="dry-run"))
            full = paths["full"].read_text(encoding="utf-8")
            self.assertIn("key_config", console)
            self.assertNotIn("resolved_snapshot=", console)
            self.assertIn("authoritative_full_log=true", full)
            self.assertIn("LAUNCHER AND RAW CHILD STDOUT/STDERR STREAM", full)

    def test_errors_log_lines_have_context(self) -> None:
        cfg, _sources, _snapshot = resolve_config(
            config_path=_REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta"
            cfg = cfg.__class__(**{**cfg.__dict__, "manifest_dir": str(meta), "log_dir": str(meta)})
            append_error_log(cfg, ["warning example"])
            text = run_log_paths(cfg)["errors"].read_text(encoding="utf-8")
            for term in ("run_id=", "hostname=", "pid=", "rank=", "local_rank=", "stream=", "stage=", "task="):
                self.assertIn(term, text)


if __name__ == "__main__":
    unittest.main()
