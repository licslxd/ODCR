from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from odcr_core.manifests import write_run_summary_json


SELECTED = "A_RATIO_0+B_RATIO_0+A_CF_MIX_FORMAL_HIGH_ONLY+B_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_1e-3+W0"
FALLBACK = "A_RATIO_0+B_RATIO_0+A_CF_MIX_FORMAL_HIGH_ONLY+B_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_5e-4+W1"


def _write_head_metadata(repo: Path, run_id: str, head: str) -> Path:
    run = repo / "runs" / "step5" / "task2" / run_id
    meta = run / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    resolved = {
        "run": {"from_step3": "2", "from_step4": "1", "from_step5": run_id, "step5_head": head},
        "step5_head": head,
        "selected_tuning_candidate": SELECTED,
        "fallback_tuning_candidate": FALLBACK,
        "step5_tuning": {
            "selected_tuning_candidate": SELECTED,
            "fallback_tuning_candidate": FALLBACK,
            "batch_candidate": "B224",
            "fallback_batch_candidate": "B192",
            "selected_budget_candidate": "medium",
        },
        "step5_sampler": {
            "contract_source": "step4_pool_manifest",
            "seed": 3407,
            "full_audit_default_allowed": False,
            "legacy_gold_heavy_exports_allowed": False,
        },
        "step5_prompt_templates": {
            "schema_version": "odcr_step5_prompt_template_registry/1",
            "train_policy": "controlled_canonical_deterministic",
            "valid_test_policy": "fixed_canonical",
        },
        "step5_effective_epoch": {"max_effective_epochs": 3, "early_stopping_patience": 1},
        "step5_batch_candidates": {"selected_default": "B224"},
    }
    source_table = {
        "source_table_schema_version": "1.0",
        "view": "formal",
        "field_sources": {
            "step5_head": "CLI --head",
            "selected_tuning_candidate": "step5.tuning.selected_tuning_candidate",
        },
        "records": [
            {"key": "step5_head", "source": "CLI --head"},
            {"key": "selected_tuning_candidate", "source": "step5.tuning.selected_tuning_candidate"},
        ],
    }
    (meta / "resolved_config.json").write_text(json.dumps(resolved), encoding="utf-8")
    (meta / "source_table.json").write_text(json.dumps(source_table), encoding="utf-8")
    return run


def _summary(run: Path, head: str) -> dict:
    repo = run.parents[3]
    rel_run = run.relative_to(repo).as_posix()
    return {
        "run_summary_schema_version": "1.0",
        "run_id": run.name,
        "stage": "step5",
        "task_id": 2,
        "status": "ok",
        "run_dir": rel_run,
        "meta_dir": f"{rel_run}/meta",
        "from_step3": "2",
        "from_step4": "1",
        "step5_head": head,
        "head": head,
        "formal_namespace": "head",
        "selected_tuning_candidate": SELECTED,
        "fallback_tuning_candidate": FALLBACK,
        "step5_effective_samples": {"step5A": 190646, "step5B": 426541},
        "step5_optimizer_steps": {"step5A": 426, "step5B": 953},
    }


class Step5HeadSplitArtifactsTest(unittest.TestCase):
    def test_single_head_summary_does_not_write_complete_step5_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run = _write_head_metadata(repo, "1_1_step5A", "step5A")
            write_run_summary_json(_summary(run, "step5A"), repo_root=repo, update_latest=True)
            status = json.loads((run / "meta" / "stage_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["step5_head"], "step5A")
            self.assertEqual(status["formal_namespace"], "head")
            self.assertFalse(status["downstream_ready"])
            self.assertFalse(status["downstream_ready_for_merge"])
            self.assertEqual(status["ready_for"], [])
            self.assertFalse((repo / "runs" / "step5" / "task2" / "latest.json").exists())

    def test_paired_head_without_completed_checkpoint_is_still_not_merge_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_a = _write_head_metadata(repo, "1_1_step5A", "step5A")
            run_b = _write_head_metadata(repo, "1_1_step5B", "step5B")
            write_run_summary_json(_summary(run_b, "step5B"), repo_root=repo, update_latest=True)
            write_run_summary_json(_summary(run_a, "step5A"), repo_root=repo, update_latest=True)
            status = json.loads((run_a / "meta" / "stage_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["merge_gate"]["paired_run_id"], "1_1_step5B")
            self.assertTrue(status["merge_gate"]["ready"])
            self.assertFalse(status["downstream_ready_for_merge"])
            self.assertFalse((repo / "runs" / "step5" / "task2" / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
