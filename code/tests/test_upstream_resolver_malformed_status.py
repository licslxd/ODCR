from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.stage_truth_antiforgery import mutate_status, write_step3_fixture  # noqa: E402
from odcr_core.upstream_resolver import UpstreamResolutionError, resolve_upstream  # noqa: E402


class UpstreamResolverMalformedStatusTest(unittest.TestCase):
    def test_missing_selected_checkpoint_field_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            mutate_status(repo, task=2, run_id="2", mutate=lambda payload: payload.pop("selected_checkpoint", None))
            with self.assertRaisesRegex(UpstreamResolutionError, "selected_checkpoint"):
                resolve_upstream(repo_root=repo, stage="step3", task=2, consumer_stage="step4")

    def test_ready_for_missing_step4_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            mutate_status(repo, task=2, run_id="2", mutate=lambda payload: payload.__setitem__("ready_for", ["step5"]))
            with self.assertRaisesRegex(UpstreamResolutionError, "ready_for_missing_step4"):
                resolve_upstream(repo_root=repo, stage="step3", task=2, consumer_stage="step4")

    def test_schema_version_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            mutate_status(repo, task=2, run_id="2", mutate=lambda payload: payload.__setitem__("schema_version", "old"))
            with self.assertRaisesRegex(UpstreamResolutionError, "unsupported stage_status schema"):
                resolve_upstream(repo_root=repo, stage="step3", task=2, consumer_stage="step4")

    def test_probe_and_comparison_modes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            for mode in ("probe", "comparison"):
                with self.subTest(mode=mode):
                    with self.assertRaisesRegex(UpstreamResolutionError, "not implemented yet"):
                        resolve_upstream(repo_root=repo, stage="step3", task=2, mode=mode, consumer_stage="step4")


if __name__ == "__main__":
    unittest.main()
