from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odcr_core.stage_status import read_stage_status
from odcr_core.stage_status_validator import validate_stage_status_evidence
from odcr_core.stage_truth_antiforgery import write_json, write_step3_fixture
from odcr_core.upstream_resolver import resolve_upstream


def test_step3_downstream_ready_comes_from_upstream_readiness_gate_not_paper_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        run = write_step3_fixture(repo, task=2, run_id="3", active=True, eligible=True)
        readiness_path = run / "meta" / "readiness_audit.json"
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        readiness["paper_eval_valid_metrics"] = {"BLEU-4": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0}
        readiness["paper_eval_test_metrics"] = {"BLEU-4": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0}
        write_json(readiness_path, readiness)

        status = read_stage_status(run)
        assert status["final_status"] == "step4_ready"
        assert status["status_source"] == "step3_upstream_readiness_gate"
        assert status["downstream_ready"] is True
        assert status["readiness_audit"].endswith("readiness_audit.json")
        validation = validate_stage_status_evidence(
            repo_root=repo,
            stage="step3",
            task=2,
            run_id="3",
            consumer_stage="step4",
            status_payload=status,
        )
        assert validation.to_payload(repo)["readiness_audit"].endswith("readiness_audit.json")
        upstream = resolve_upstream(repo_root=repo, stage="step3", task=2, consumer_stage="step4")
        assert upstream.stage_status["status_source"] == "step3_upstream_readiness_gate"
