import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import OneControlConfigError, _latest_run


class LatestRunResolutionTest(unittest.TestCase):
    def _write_summary(self, repo: Path, stage: str = "step3", task_id: int = 4, run_id: str = "1") -> Path:
        meta = repo / "runs" / stage / f"task{task_id}" / run_id / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        summary = meta / "run_summary.json"
        summary.write_text(json.dumps({"run_id": run_id, "stage": stage, "task_id": task_id}), encoding="utf-8")
        return summary

    def _write_latest(self, repo: Path, summary: Path, stage: str = "step3", task_id: int = 4, run_id: str = "1") -> None:
        parent = repo / "runs" / stage / f"task{task_id}"
        parent.mkdir(parents=True, exist_ok=True)
        latest = parent / "latest.json"
        latest.write_text(
            json.dumps(
                {
                    "latest_run_id": run_id,
                    "latest_summary_path": summary.relative_to(repo).as_posix(),
                    "latest_status": "ok",
                }
            ),
            encoding="utf-8",
        )

    def test_latest_json_normal_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            summary = self._write_summary(repo, run_id="7")
            self._write_latest(repo, summary, run_id="7")
            self.assertEqual(_latest_run(repo, 4, "step3", dry_run=False), "7")

    def test_missing_latest_json_fails_fast_even_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with self.assertRaisesRegex(OneControlConfigError, "missing .*latest.json"):
                _latest_run(repo, 4, "step3", dry_run=True)

    def test_missing_run_summary_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            summary = repo / "runs" / "step3" / "task4" / "2" / "meta" / "run_summary.json"
            self._write_latest(repo, summary, run_id="2")
            with self.assertRaisesRegex(OneControlConfigError, "missing run_summary.json"):
                _latest_run(repo, 4, "step3", dry_run=False)

    def test_old_runs_task_layout_is_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            old = repo / "runs" / "task4" / "step3" / "999" / "meta"
            old.mkdir(parents=True)
            (old / "run_summary.json").write_text(json.dumps({"run_id": "999"}), encoding="utf-8")
            with self.assertRaisesRegex(OneControlConfigError, "missing .*latest.json"):
                _latest_run(repo, 4, "step3", dry_run=False)

    def test_latest_pointer_wins_over_larger_directory_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            summary = self._write_summary(repo, run_id="1")
            self._write_latest(repo, summary, run_id="1")
            self._write_summary(repo, run_id="999")
            self.assertEqual(_latest_run(repo, 4, "step3", dry_run=False), "1")


if __name__ == "__main__":
    unittest.main()
