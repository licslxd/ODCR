from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import odcr  # noqa: E402
from odcr_core.config_schema import OneControlConfigError  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class TestOdcrTailNewLayout(unittest.TestCase):
    def _repo_with_run(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        tmp = tempfile.TemporaryDirectory()
        repo = Path(tmp.name)
        run_dir = repo / "runs" / "step3" / "task4" / "1"
        meta = run_dir / "meta"
        meta.mkdir(parents=True)
        (meta / "console.log").write_text("console-a\nconsole-b\n", encoding="utf-8")
        (meta / "full.log").write_text("full-a\nfull-b\n", encoding="utf-8")
        (meta / "errors.log").write_text("errors-a\nerrors-b\n", encoding="utf-8")
        _write_json(
            meta / "run_summary.json",
            {
                "run_id": "1",
                "stage": "step3",
                "task_id": 4,
                "run_dir": "runs/step3/task4/1",
                "meta_dir": "runs/step3/task4/1/meta",
                "console_log_path": "runs/step3/task4/1/meta/console.log",
                "full_log_path": "runs/step3/task4/1/meta/full.log",
                "errors_log_path": "runs/step3/task4/1/meta/errors.log",
            },
        )
        _write_json(
            repo / "runs" / "step3" / "task4" / "latest.json",
            {
                "latest_run_id": "1",
                "latest_run_dir": "runs/step3/task4/1",
                "latest_summary_path": "runs/step3/task4/1/meta/run_summary.json",
                "latest_status": "ok",
            },
        )
        return tmp, repo

    def _tail(self, repo: Path, *extra: str) -> str:
        args = odcr.build_parser().parse_args(["tail", "--stage", "step3", "--task", "4", "--lines", "1", *extra])
        buf = io.StringIO()
        with mock.patch.object(odcr, "REPO_ROOT", repo), contextlib.redirect_stdout(buf):
            odcr.cmd_tail(args)
        return buf.getvalue()

    def test_tail_uses_latest_run_summary_console_log(self) -> None:
        tmp, repo = self._repo_with_run()
        with tmp:
            out = self._tail(repo)
        self.assertIn("console.log", out)
        self.assertIn("console-b", out)
        self.assertNotIn("full-b", out)

    def test_tail_full_uses_full_log(self) -> None:
        tmp, repo = self._repo_with_run()
        with tmp:
            out = self._tail(repo, "--full")
        self.assertIn("full.log", out)
        self.assertIn("full-b", out)
        self.assertNotIn("console-b", out)

    def test_tail_errors_uses_errors_log(self) -> None:
        tmp, repo = self._repo_with_run()
        with tmp:
            out = self._tail(repo, "--errors")
        self.assertIn("errors.log", out)
        self.assertIn("errors-b", out)
        self.assertNotIn("console-b", out)

    def test_tail_missing_latest_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            repo = Path(tmp_raw)
            args = odcr.build_parser().parse_args(["tail", "--stage", "step3", "--task", "4"])
            with mock.patch.object(odcr, "REPO_ROOT", repo):
                with self.assertRaisesRegex(OneControlConfigError, "missing .*latest.json"):
                    odcr.cmd_tail(args)

    def test_tail_latest_missing_run_summary_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            repo = Path(tmp_raw)
            _write_json(
                repo / "runs" / "step3" / "task4" / "latest.json",
                {"latest_summary_path": "runs/step3/task4/1/meta/run_summary.json"},
            )
            args = odcr.build_parser().parse_args(["tail", "--stage", "step3", "--task", "4"])
            with mock.patch.object(odcr, "REPO_ROOT", repo):
                with self.assertRaisesRegex(OneControlConfigError, "latest.json pointer is damaged"):
                    odcr.cmd_tail(args)

    def test_tail_missing_target_log_fails_fast(self) -> None:
        tmp, repo = self._repo_with_run()
        with tmp:
            (repo / "runs" / "step3" / "task4" / "1" / "meta" / "console.log").unlink()
            args = odcr.build_parser().parse_args(["tail", "--stage", "step3", "--task", "4"])
            with mock.patch.object(odcr, "REPO_ROOT", repo):
                with self.assertRaisesRegex(OneControlConfigError, "new log layout did not generate"):
                    odcr.cmd_tail(args)

    def test_tail_does_not_fallback_to_retired_legacy_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            repo = Path(tmp_raw)
            (repo / "logs").mkdir()
            (repo / "logs" / "old.log").write_text("old top-level log\n", encoding="utf-8")
            (repo / "code").mkdir()
            (repo / "code" / "log.out").write_text("old code log\n", encoding="utf-8")
            for name in ("nohup.log", "fallback.log", "mirror.log"):
                (repo / name).write_text("old fallback\n", encoding="utf-8")
            args = odcr.build_parser().parse_args(["tail", "--stage", "step3", "--task", "4"])
            with mock.patch.object(odcr, "REPO_ROOT", repo):
                with self.assertRaisesRegex(OneControlConfigError, "missing .*latest.json"):
                    odcr.cmd_tail(args)

    def test_tail_rejects_run_summary_paths_outside_meta(self) -> None:
        tmp, repo = self._repo_with_run()
        with tmp:
            summary_path = repo / "runs" / "step3" / "task4" / "1" / "meta" / "run_summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["console_log_path"] = "logs/old.log"
            _write_json(summary_path, payload)
            (repo / "logs").mkdir()
            (repo / "logs" / "old.log").write_text("old log\n", encoding="utf-8")
            args = odcr.build_parser().parse_args(["tail", "--stage", "step3", "--task", "4"])
            with mock.patch.object(odcr, "REPO_ROOT", repo):
                with self.assertRaisesRegex(OneControlConfigError, "must resolve to meta/console.log"):
                    odcr.cmd_tail(args)

    def test_tail_dry_run_resolves_new_log_without_printing_content(self) -> None:
        tmp, repo = self._repo_with_run()
        with tmp:
            args = odcr.build_parser().parse_args(
                ["tail", "--stage", "step3", "--task", "4", "--dry-run"]
            )
            buf = io.StringIO()
            with mock.patch.object(odcr, "REPO_ROOT", repo), contextlib.redirect_stdout(buf):
                odcr.cmd_tail(args)
        out = buf.getvalue()
        self.assertIn("ODCR tail dry-run:", out)
        self.assertIn("console.log", out)
        self.assertNotIn("console-b", out)


if __name__ == "__main__":
    unittest.main()
