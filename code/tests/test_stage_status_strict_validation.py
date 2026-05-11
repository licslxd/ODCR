from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.stage_status_validator import (  # noqa: E402
    STAGE_STATUS_VALIDATOR_VERSION,
    StageStatusValidationError,
    validate_stage_status_evidence,
)
from odcr_core.stage_truth_antiforgery import mutate_status, write_step3_fixture  # noqa: E402


class StageStatusStrictValidationTest(unittest.TestCase):
    def _payload(self, repo: Path) -> tuple[Path, dict]:
        run = write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
        import json

        return run, json.loads((run / "meta" / "stage_status.json").read_text(encoding="utf-8"))

    def test_valid_ready_status_has_strict_validator_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run, payload = self._payload(repo)
            self.assertEqual(payload["validator_version"], STAGE_STATUS_VALIDATOR_VERSION)
            result = validate_stage_status_evidence(
                repo_root=repo,
                stage="step3",
                task=2,
                run_id="2",
                consumer_stage="step4",
                status_payload=payload,
                run_dir=run,
                latest_payload=None,
                require_latest=False,
            )
            self.assertEqual(result.selected_checkpoint_hash, payload["selected_checkpoint_hash"])

    def test_unknown_required_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run, payload = self._payload(repo)
            payload["required_artifacts"] = list(payload["required_artifacts"]) + ["unknown_contract"]
            with self.assertRaisesRegex(StageStatusValidationError, "unknown required_artifacts"):
                validate_stage_status_evidence(
                    repo_root=repo,
                    stage="step3",
                    task=2,
                    run_id="2",
                    consumer_stage="step4",
                    status_payload=payload,
                    run_dir=run,
                )

    def test_path_to_ai_analysis_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            mutate_status(
                repo,
                task=2,
                run_id="2",
                mutate=lambda payload: (
                    payload.__setitem__("selected_checkpoint", "AI_analysis/fake.pth"),
                    payload["artifacts"]["selected_checkpoint"].__setitem__("path", "AI_analysis/fake.pth"),
                ),
            )
            import json

            run = repo / "runs" / "step3" / "task2" / "2"
            payload = json.loads((run / "meta" / "stage_status.json").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(StageStatusValidationError, "forbidden namespace AI_analysis"):
                validate_stage_status_evidence(
                    repo_root=repo,
                    stage="step3",
                    task=2,
                    run_id="2",
                    consumer_stage="step4",
                    status_payload=payload,
                    run_dir=run,
                )


if __name__ == "__main__":
    unittest.main()
