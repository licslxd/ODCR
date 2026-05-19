from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import odcr  # noqa: E402
from executors import step4_engine  # noqa: E402
from odcr_core import runners, step4_runtime  # noqa: E402
from odcr_core import step4_checkpoint_lineage  # noqa: E402
from odcr_core.upstream_resolver import UpstreamResolutionError, resolve_latest  # noqa: E402


class Step4PrelaunchLineageValidationTest(unittest.TestCase):
    def test_dry_run_preflight_and_formal_share_validation_function(self) -> None:
        self.assertIn(
            "validate_step4_prelaunch_lineage_for_config",
            inspect.getsource(odcr._run_resolved),
        )
        self.assertIn(
            "validate_step4_prelaunch_lineage_for_config",
            inspect.getsource(step4_runtime.run_step4_bounded_preflight),
        )
        self.assertIn(
            "validate_step4_prelaunch_lineage_for_config",
            inspect.getsource(runners.run_step4),
        )
        self.assertIn(
            "validate_step4_prelaunch_checkpoint_lineage",
            inspect.getsource(step4_engine._validate_step3_checkpoint_lineage_for_step4),
        )

    def test_preflight_dry_run_returns_lineage_validation_and_no_formal_write(self) -> None:
        src = inspect.getsource(step4_runtime.run_step4_bounded_preflight)
        self.assertIn("checkpoint_lineage_validation", src)
        self.assertIn("formal_latest_write", src)
        self.assertIn("formal_export_write", src)
        self.assertIn("preflight_dry_run", src)

    def test_no_lineage_bypass_switches_exist(self) -> None:
        roots = [
            CODE_DIR / "odcr.py",
            CODE_DIR / "odcr_core" / "step4_runtime.py",
            CODE_DIR / "odcr_core" / "step4_checkpoint_lineage.py",
            CODE_DIR / "executors" / "step4_engine.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in roots)
        for forbidden in ("allow_mismatch", "skip_lineage", "force_load", "warning-only", "warning_only"):
            self.assertNotIn(forbidden, combined)

    def test_failed_step4_run1_is_not_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            task_root = repo / "runs" / "step4" / "task2"
            run1 = task_root / "1" / "meta" / "run_summary.json"
            run1.parent.mkdir(parents=True)
            run1.write_text(
                json.dumps({"run_id": "1", "stage": "step4", "task_id": 2, "status": "failed"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(UpstreamResolutionError, "missing step4 latest.json"):
                resolve_latest(repo_root=repo, stage="step4", task=2, repair=False)

    def test_validation_report_has_diff_fields(self) -> None:
        src = inspect.getsource(step4_checkpoint_lineage.validate_step4_prelaunch_checkpoint_lineage)
        for term in (
            "checkpoint_model_architecture_payload",
            "checkpoint_state_dict_model_architecture_payload",
            "checkpoint_sidecar_model_architecture_payload",
            "expected_model_architecture_payload",
            "observed_current_loader_architecture_payload",
            "observed_current_loader_mismatch_keys",
            "sidecar_model_architecture_metadata_mismatch_keys",
            "ntoken_compatibility",
            "effective_model_ntoken",
            "source_table_hash_scope_report",
            "live_vs_frozen_step3_config_drift",
            "ignored_non_architecture_keys",
            "hash_source_paths",
        ):
            self.assertIn(term, src)

    def test_step4_runtime_binds_stage_status_selected_checkpoint(self) -> None:
        src = inspect.getsource(step4_runtime.validate_step4_prelaunch_lineage_for_config)
        self.assertIn("step3_selected_checkpoint_binding", src)
        self.assertIn("selected_checkpoint_path", src)
        engine_src = inspect.getsource(step4_engine._run_one_task)
        self.assertIn("ODCR_STEP3_SELECTED_CHECKPOINT", engine_src)
        self.assertIn("best.pth alias is not a primary", engine_src)

    def test_step4_cache_fingerprint_uses_selected_checkpoint(self) -> None:
        src = inspect.getsource(step4_engine._step4_encoded_cache_fingerprint)
        self.assertIn("ODCR_STEP3_SELECTED_CHECKPOINT", src)
        self.assertIn("stage_status.selected_checkpoint", src)
        self.assertNotIn("model\", \"best.pth", src)


if __name__ == "__main__":
    unittest.main()
