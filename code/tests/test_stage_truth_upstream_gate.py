from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core import config_resolver  # noqa: E402
from odcr_core.index_contract import (  # noqa: E402
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    ODCR_ROUTING_TRAIN_CSV,
    STEP4_RCR_REQUIRED_COLUMNS,
    build_step4_export_lineage,
)
from odcr_core.stage_promotion import promote_upstream  # noqa: E402
from odcr_core.stage_status import build_and_write_stage_status, mark_superseded, read_stage_status  # noqa: E402
from odcr_core.stage_truth_antiforgery import write_step3_fixture  # noqa: E402
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST  # noqa: E402
from odcr_core.upstream_resolver import UpstreamResolutionError, resolve_upstream  # noqa: E402
from tools.check_one_control_guardrails import run_checks  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_step3_run(
    repo: Path,
    *,
    task: int,
    run_id: str,
    active: bool = False,
    eligible: bool = True,
    quality_blocked: bool = False,
) -> Path:
    return write_step3_fixture(
        repo,
        task=task,
        run_id=run_id,
        active=active,
        eligible=eligible,
        quality_downstream_ready=not quality_blocked,
    )


def _write_step4_run(repo: Path, *, task: int, run_id: str, from_step3: str, active: bool = False) -> Path:
    run = repo / "runs" / "step4" / f"task{task}" / run_id
    meta = run / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    row = {col: 1 for col in STEP4_RCR_REQUIRED_COLUMNS}
    row.update(
        {
            "route_reason_scorer": "rcr_scorer_clean",
            "route_reason_explainer": "rcr_explainer_rich",
            "confidence_bucket": 2,
            "preprocess_route_scorer_prior": 0,
            "preprocess_route_explainer_prior": 0,
        }
    )
    headers = list(STEP4_RCR_REQUIRED_COLUMNS)
    (run / ODCR_ROUTING_TRAIN_CSV).write_text(
        ",".join(headers) + "\n" + ",".join(str(row[col]) for col in headers) + "\n",
        encoding="utf-8",
    )
    lineage = build_step4_export_lineage(
        task_id=task,
        auxiliary_domain="A",
        target_domain="T",
        step3_checkpoint_lineage_hash="lineage",
        step4_rcr_config={"fixture": True},
        step4_run=run_id,
        frozen_step3_lineage={
            "upstream_step3_run_id": from_step3,
            "step3_checkpoint_path": f"runs/step3/task{task}/{from_step3}/model/best_observed.pth",
            "step3_checkpoint_hash": "fixture_checkpoint_hash",
            "step3_stage_status_hash": "fixture_stage_status_hash",
            "step3_eval_handoff_hash": "fixture_eval_handoff_hash",
        },
    )
    _write_json(
        run / INDEX_CONTRACT_FILENAME,
        {
            "schema_version": INDEX_CONTRACT_SCHEMA_VERSION,
            "embed_dim": 1024,
            "backbones": {
                "sentence_embed": {
                    "model_id": "fixture",
                    "local_dir": "/tmp/fixture",
                    "family": "bge_large_en",
                    "hidden_size": 1024,
                    "dual_channel": True,
                }
            },
            "step4_export_lineage": lineage,
        },
    )
    _write_json(run / STEP4_EXPORT_MANIFEST, {"schema_version": "odcr_step4_train_table/1.2", "step4_export_lineage": lineage})
    _write_json(meta / "source_table.json", {"records": []})
    _write_json(meta / "resolved_config.json", {"task": {"id": task}})
    _write_json(
        meta / "run_summary.json",
        {
            "run_id": run_id,
            "stage": "step4",
            "task_id": task,
            "status": "ok",
            "run_dir": f"runs/step4/task{task}/{run_id}",
            "meta_dir": f"runs/step4/task{task}/{run_id}/meta",
            "from_step3": from_step3,
        },
    )
    build_and_write_stage_status(repo_root=repo, stage="step4", task=task, run_id=run_id)
    if active:
        _write_json(
            repo / "runs" / "step4" / f"task{task}" / "latest.json",
            {
                "latest_run_id": run_id,
                "latest_run_dir": f"runs/step4/task{task}/{run_id}",
                "latest_summary_path": f"runs/step4/task{task}/{run_id}/meta/run_summary.json",
                "latest_status": "ok",
            },
        )
    return run


class StageTruthUpstreamGateTest(unittest.TestCase):
    def test_active_latest_run_resolves_and_quality_audit_is_not_final_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_step3_run(repo, task=2, run_id="2", active=True, eligible=True, quality_blocked=True)
            res = resolve_upstream(repo_root=repo, stage="step3", task=2, consumer_stage="step4")
            self.assertEqual(res.run_id, "2")
            self.assertTrue(res.stage_status["downstream_ready"])
            self.assertEqual(res.stage_status["final_status"], "completed_with_eval_handoff")
            self.assertTrue(res.stage_status["do_not_use_quality_audit_as_final_truth"])
            self.assertTrue((repo / "runs" / "step3" / "task2" / "2" / "meta" / "quality_audit.json.superseded_by.json").is_file())

    def test_explicit_failed_old_run_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_step3_run(repo, task=2, run_id="1", eligible=False, quality_blocked=True)
            _write_step3_run(repo, task=2, run_id="2", active=True, eligible=True, quality_blocked=True)
            with self.assertRaisesRegex(UpstreamResolutionError, "run1 is not eligible for Step4 formal upstream"):
                resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="1", consumer_stage="step4")

    def test_manual_non_latest_eligible_run_requires_promote_and_promotion_keeps_old_status_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_step3_run(repo, task=2, run_id="2", active=True, eligible=True, quality_blocked=True)
            _write_step3_run(repo, task=2, run_id="3", eligible=True, quality_blocked=True)
            with self.assertRaisesRegex(UpstreamResolutionError, "non_latest_eligible_run_requires_promote"):
                resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="3", consumer_stage="step4")
            old_status_before = read_stage_status(repo / "runs" / "step3" / "task2" / "2")
            promoted = promote_upstream(repo_root=repo, stage="step3", task=2, run_id="3", dry_run=False)
            self.assertEqual(promoted["promote_run_id"], "3")
            latest = json.loads((repo / "runs" / "step3" / "task2" / "latest.json").read_text())
            self.assertEqual(latest["latest_run_id"], "3")
            old_status = read_stage_status(repo / "runs" / "step3" / "task2" / "2")
            self.assertEqual(old_status["final_status"], old_status_before["final_status"])
            self.assertTrue(promoted["historical_stage_status_immutable"])

    def test_superseded_run_rejected_formal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_step3_run(repo, task=2, run_id="2", active=True, eligible=True, quality_blocked=True)
            _write_step3_run(repo, task=2, run_id="4", eligible=True, quality_blocked=True)
            mark_superseded(repo_root=repo, stage="step3", task=2, run_id="4", superseded_by_run_id="2")
            with self.assertRaisesRegex(UpstreamResolutionError, "superseded"):
                resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="4", consumer_stage="step4")

    def test_step4_dry_run_and_runtime_resolve_same_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_step3_run(repo, task=2, run_id="2", active=True, eligible=True, quality_blocked=True)
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                cfg_dry, _, snap_dry = config_resolver.resolve_config(
                    config_path=REPO_ROOT / "configs" / "odcr.yaml",
                    command="step4",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    from_step3="latest",
                )
                cfg_run, _, snap_run = config_resolver.resolve_config(
                    config_path=REPO_ROOT / "configs" / "odcr.yaml",
                    command="step4",
                    task_id=2,
                    set_overrides=[],
                    dry_run=False,
                    from_step3="latest",
                    run_id="9",
                )
            finally:
                config_resolver._REPO_ROOT = old_root
            self.assertEqual(snap_dry["upstream_resolution"]["run_id"], "2")
            self.assertEqual(snap_run["upstream_resolution"]["run_id"], "2")
            self.assertEqual(cfg_dry.from_run, cfg_run.from_run)

    def test_step5_reuses_upstream_resolver_for_step4_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_step3_run(repo, task=2, run_id="2", active=True, eligible=True, quality_blocked=True)
            _write_step4_run(repo, task=2, run_id="2_1", from_step3="2", active=True)
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                _cfg, _, snapshot = config_resolver.resolve_config(
                    config_path=REPO_ROOT / "configs" / "odcr.yaml",
                    command="step5",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    from_step4="latest",
                )
            finally:
                config_resolver._REPO_ROOT = old_root
            self.assertEqual(snapshot["upstream_resolution"]["producer_stage"], "step4")
            self.assertEqual(snapshot["upstream_resolution"]["run_id"], "2_1")

    def test_generic_paths_task2_runs_and_task5_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_step3_run(repo, task=2, run_id="3", eligible=True, quality_blocked=True)
            _write_step3_run(repo, task=2, run_id="4", active=True, eligible=True, quality_blocked=True)
            _write_step3_run(repo, task=5, run_id="1", active=True, eligible=True, quality_blocked=True)
            self.assertEqual(resolve_upstream(repo_root=repo, stage="step3", task=2, consumer_stage="step4").run_id, "4")
            self.assertEqual(resolve_upstream(repo_root=repo, stage="step3", task=5, consumer_stage="step4").run_id, "1")

    def test_docs_active_truth_guardrail(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        self.assertEqual(statuses.get("R112"), "PASS")


if __name__ == "__main__":
    unittest.main()
