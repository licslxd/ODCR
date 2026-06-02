from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.ablation.cli import validate_ablation_infra  # noqa: E402
from odcr_core.ablation.guards import paper_table_gate  # noqa: E402
from odcr_core.ablation.registry import load_registry, validate_registry  # noqa: E402
from odcr_core.ablation.snapshots import build_result_snapshot  # noqa: E402
from odcr_core.ablation.variants import build_ablation_dry_run_plan  # noqa: E402
from odcr_core.stage_promotion import StagePromotionError, promote_upstream  # noqa: E402
from odcr_core.upstream_resolver import UpstreamResolutionError, resolve_latest  # noqa: E402


class AblationInfraTest(unittest.TestCase):
    def test_registry_contains_only_task7_task8_full_and_three_variants(self) -> None:
        registry = load_registry(REPO_ROOT)
        result = validate_registry(registry)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["entry_count"], 8)
        self.assertEqual(sorted({row["task"] for row in result["entries"]}), [7, 8])

    def test_validate_ablation_infra_passes(self) -> None:
        result = validate_ablation_infra(REPO_ROOT)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["manifests"]["status"], "pass")

    def test_dry_run_plan_is_non_executing_and_no_latest(self) -> None:
        plan = build_ablation_dry_run_plan(REPO_ROOT, task=8, variant="wo_rcr")
        self.assertTrue(plan["dry_run_only"])
        self.assertFalse(plan["would_start_training"])
        self.assertFalse(plan["would_start_eval"])
        self.assertFalse(plan["would_write_latest"])
        self.assertEqual(plan["manifest_validation"]["forbidden_to_promote_latest"], True)

    def test_missing_snapshot_is_not_paper_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "ablations").mkdir(parents=True)
            shutil.copy2(REPO_ROOT / "ablations" / "registry.yaml", repo / "ablations" / "registry.yaml")
            snapshot = build_result_snapshot(repo, task=7, variant="wo_ccv_fca")
        self.assertEqual(snapshot["status"], "missing_artifact")
        self.assertFalse(snapshot["paper_table_allowed"])
        self.assertFalse(snapshot["paper_table_gate"]["eligible"])
        self.assertTrue(snapshot["missing_artifacts"])

    def test_paper_table_gate_requires_manual_review_clearance(self) -> None:
        gate = paper_table_gate(
            {
                "valid_complete": True,
                "test_complete": True,
                "paper_greedy_25": True,
                "task_local_rating_source": True,
                "paper_table_allowed": True,
                "requires_manual_review": True,
            }
        )
        self.assertFalse(gate["eligible"])
        self.assertIn("requires_manual_review_false", gate["missing_requirements"])

    def test_promotion_rejects_ablation_run_id_before_pointer_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with self.assertRaisesRegex(StagePromotionError, "forbidden to promote"):
                promote_upstream(
                    repo_root=repo,
                    stage="step5",
                    task=8,
                    run_id="ablation_wo_rcr_1",
                    dry_run=True,
                )
            self.assertFalse((repo / "runs" / "step5" / "task8" / "latest.json").exists())

    def test_formal_latest_rejects_ablation_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            parent = repo / "runs" / "step5" / "task8"
            parent.mkdir(parents=True)
            (parent / "latest.json").write_text(
                json.dumps(
                    {
                        "latest_run_id": "ablation_wo_rcr_1",
                        "latest_summary_path": "runs/step5/task8/ablation_wo_rcr_1/meta/run_summary.json",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(UpstreamResolutionError, "must not point to an ablation"):
                resolve_latest(repo_root=repo, stage="step5", task=8, repair=False)


if __name__ == "__main__":
    unittest.main()
