from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import OneControlConfigError, build_preprocess_config, load_yaml_config, resolve_config  # noqa: E402
from odcr_core.index_contract import (  # noqa: E402
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    ODCR_ROUTING_TRAIN_CSV,
    STEP4_RCR_REQUIRED_COLUMNS,
    build_step4_export_lineage,
)
from odcr_core.csb_contract import default_csb_contract_payload  # noqa: E402
from odcr_core.manifests import build_run_manifest  # noqa: E402
from odcr_core.preprocess_runtime import PreprocessRuntime  # noqa: E402
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST  # noqa: E402
from odcr_core.stage_truth_antiforgery import write_step3_fixture  # noqa: E402
from tools.check_one_control_guardrails import (  # noqa: E402
    D4C_PYTHON_ABS,
    GUARDRAIL_GROUPS,
    HOOK_DIAGNOSTICS_REL,
    HOOK_STOP_COMMAND,
    LEGACY_KILL_ABSENT_PATHS,
    LEGACY_KILL_ACTIVE_IMPORT_RE,
    MAINLINE_FILES,
    PRESET_READ_RE,
    RULE_GROUP_BY_ID,
    TOP_LEVEL_BLOCKS,
    format_report,
    run_checks,
    scan_evolution_snippet,
    scan_logging_artifact_snippet,
    scan_old_layout_log_snippet,
    scan_run_artifact_snippet,
)
from tools.odcr_post_edit_check import SCOPES, build_plan, plan_safety_violations  # noqa: E402
import odcr_core.config_resolver as config_resolver  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_step5_upstream_fixture(repo: Path, *, task_id: int = 2, run_id: str = "1_1_1") -> None:
    run = repo / "runs" / "step5" / f"task{task_id}" / run_id
    meta = run / "meta"
    state = run / "state"
    meta.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    _write_json(meta / "run_summary.json", {"run_id": run_id, "stage": "step5", "task_id": task_id, "status": "ok"})
    _write_json(state / "checkpoint_lineage.json", {"schema_version": "test", "stage": "step5", "run_id": run_id})
    _write_json(
        meta / "stage_status.json",
        {
            "schema_version": "odcr_stage_status/1",
            "stage": "step5",
            "task": task_id,
            "task_id": task_id,
            "run_id": run_id,
            "run_dir": f"runs/step5/task{task_id}/{run_id}",
            "final_status": "completed",
            "downstream_ready": True,
            "ready_for": ["eval", "rerank"],
            "status_source": "test_fixture",
            "rejection_reasons": [],
            "selected_checkpoint": f"runs/step5/task{task_id}/{run_id}/model/best.pth",
            "checkpoint_lineage": f"runs/step5/task{task_id}/{run_id}/state/checkpoint_lineage.json",
            "artifacts": {
                "run_summary": {
                    "path": f"runs/step5/task{task_id}/{run_id}/meta/run_summary.json",
                    "exists": True,
                    "is_file": True,
                },
                "checkpoint_lineage": {
                    "path": f"runs/step5/task{task_id}/{run_id}/state/checkpoint_lineage.json",
                    "exists": True,
                    "is_file": True,
                },
            },
        },
    )
    _write_json(
        repo / "runs" / "step5" / f"task{task_id}" / "latest.json",
        {
            "latest_run_id": run_id,
            "latest_run_dir": f"runs/step5/task{task_id}/{run_id}",
            "latest_summary_path": f"runs/step5/task{task_id}/{run_id}/meta/run_summary.json",
            "latest_status": "ok",
        },
    )


def _write_step4_upstream_fixture(repo: Path, *, task_id: int, run_id: str = "1_1") -> None:
    run = repo / "runs" / "step4" / f"task{task_id}" / run_id
    meta = run / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    export_name = ODCR_ROUTING_TRAIN_CSV
    export = run / export_name
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
    export.write_text(
        ",".join(headers) + "\n" + ",".join(str(row[col]) for col in headers) + "\n",
        encoding="utf-8",
    )
    lineage = build_step4_export_lineage(
        task_id=task_id,
        auxiliary_domain="A",
        target_domain="T",
        step3_checkpoint_lineage_hash="lineage",
        step4_rcr_config={"fixture": True},
        step4_run=run_id,
        frozen_step3_lineage={
            "upstream_step3_run_id": "2",
            "step3_checkpoint_path": f"runs/step3/task{task_id}/2/model/best_observed.pth",
            "step3_checkpoint_hash": "fixture_checkpoint_hash",
            "step3_stage_status_hash": "fixture_stage_status_hash",
            "step3_readiness_audit_hash": "fixture_readiness_audit_hash",
        },
        csb_contract=default_csb_contract_payload(),
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
    _write_json(meta / "resolved_config.json", {"task": {"id": task_id}})
    _write_json(meta / "run_summary.json", {"run_id": run_id, "stage": "step4", "task_id": task_id, "status": "ok"})
    _write_json(
        meta / "stage_status.json",
        {
            "schema_version": "odcr_stage_status/1",
            "stage": "step4",
            "task": task_id,
            "task_id": task_id,
            "run_id": run_id,
            "run_dir": f"runs/step4/task{task_id}/{run_id}",
            "final_status": "completed",
            "downstream_ready": True,
            "ready_for": ["step5"],
            "status_source": "test_fixture",
            "rejection_reasons": [],
            "selected_export": f"runs/step4/task{task_id}/{run_id}/{export_name}",
            "export_manifest": f"runs/step4/task{task_id}/{run_id}/{STEP4_EXPORT_MANIFEST}",
            "index_contract": f"runs/step4/task{task_id}/{run_id}/{INDEX_CONTRACT_FILENAME}",
            "artifacts": {
                "run_summary": {
                    "path": f"runs/step4/task{task_id}/{run_id}/meta/run_summary.json",
                    "exists": True,
                    "is_file": True,
                },
                "selected_export": {
                    "path": f"runs/step4/task{task_id}/{run_id}/{export_name}",
                    "exists": True,
                    "is_file": True,
                },
                "export_manifest": {
                    "path": f"runs/step4/task{task_id}/{run_id}/{STEP4_EXPORT_MANIFEST}",
                    "exists": True,
                    "is_file": True,
                },
                "index_contract": {
                    "path": f"runs/step4/task{task_id}/{run_id}/{INDEX_CONTRACT_FILENAME}",
                    "exists": True,
                    "is_file": True,
                },
            },
        },
    )
    _write_json(
        repo / "runs" / "step4" / f"task{task_id}" / "latest.json",
        {
            "latest_run_id": run_id,
            "latest_run_dir": f"runs/step4/task{task_id}/{run_id}",
            "latest_summary_path": f"runs/step4/task{task_id}/{run_id}/meta/run_summary.json",
            "latest_status": "ok",
        },
    )


def _resolve_step5_with_fixture(*, task_id: int, from_step4: str = "1_1", set_overrides: list[str] | None = None):
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _write_step4_upstream_fixture(repo, task_id=task_id, run_id=from_step4)
        old_root = config_resolver._REPO_ROOT
        try:
            config_resolver._REPO_ROOT = repo
            return resolve_config(
                config_path=REPO_ROOT / "configs" / "odcr.yaml",
                command="step5",
                task_id=task_id,
                set_overrides=set_overrides or [],
                dry_run=True,
                from_step4=from_step4,
                eval_profile="balanced_2gpu",
                mode="full",
            )
        finally:
            config_resolver._REPO_ROOT = old_root


class TestOneControlGuardrails(unittest.TestCase):
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def test_guardrail_script_runs_strict(self) -> None:
        proc = subprocess.run(
            [sys.executable, "code/tools/check_one_control_guardrails.py", "--strict"],
            cwd=REPO_ROOT,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("ODCR One-Control Guardrails: PASS (0 fail, 0 warn)", proc.stdout)
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        self.assertTrue(report.ok)
        self.assertEqual(report.failures, 0)
        self.assertEqual(report.warnings, 0)

    def test_config_top_level_blocks_pass(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        self.assertTrue(report.ok)
        self.assertEqual(report.warnings, 0)
        top_rule = next(item for item in report.results if item.rule_id == "R005")
        self.assertEqual(top_rule.status, "PASS")
        text = (REPO_ROOT / "configs" / "odcr.yaml").read_text(encoding="utf-8")
        for block in TOP_LEVEL_BLOCKS:
            self.assertIn(f"{block}:", text)

    def test_run_stage_entrypoint_is_absent(self) -> None:
        self.assertFalse((REPO_ROOT / "scripts" / "run_stage.sh").exists())
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        rule = next(item for item in report.results if item.rule_id == "R004")
        self.assertEqual(rule.status, "PASS")

    def test_legacy_kill_pass_files_and_active_imports_are_absent(self) -> None:
        for rel in LEGACY_KILL_ABSENT_PATHS:
            with self.subTest(path=rel):
                self.assertFalse((REPO_ROOT / rel).exists())

        active_roots = ("odcr", "code", "scripts")
        skip_prefixes = ("code/tests/",)
        skip_files = {"code/tools/check_one_control_guardrails.py"}
        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", "__pycache__", ".pytest_cache", "AI_analysis", "_archive", "data", "merged", "runs"}
            ]
            for name in files:
                path = Path(root) / name
                rel = path.relative_to(REPO_ROOT).as_posix()
                if not (rel == "odcr" or rel.startswith(active_roots[1] + "/") or rel.startswith(active_roots[2] + "/")):
                    continue
                if any(rel.startswith(prefix) for prefix in skip_prefixes) or rel in skip_files:
                    continue
                if path.suffix not in {".py", ".sh"} and rel != "odcr":
                    continue
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    self.assertIsNone(LEGACY_KILL_ACTIVE_IMPORT_RE.search(line), rel)

        report = run_checks(repo_root=REPO_ROOT, strict=True)
        rule = next(item for item in report.results if item.rule_id == "R095")
        self.assertEqual(rule.status, "PASS")
        self.assertEqual(RULE_GROUP_BY_ID.get("R095"), "legacy-cleanup")

    def test_legacy_presets_are_not_mainline_reads(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        rule = next(item for item in report.results if item.rule_id == "R001")
        self.assertEqual(rule.status, "PASS")
        for rel in MAINLINE_FILES:
            path = REPO_ROOT / rel
            if path.is_file():
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    stripped = line.strip()
                    if stripped.startswith(("#", '"""', "'''")):
                        continue
                    self.assertIsNone(PRESET_READ_RE.search(line), f"{rel}: {line}")

    def test_legacy_preset_tree_only_archived_or_analysis(self) -> None:
        self.assertFalse((REPO_ROOT / "presets").exists(), "live presets/ must not be restored")
        allowed_prefixes = ("_archive/legacy_presets_", "AI_analysis/")
        skip = {".git", "__pycache__", ".pytest_cache", "data", "merged", "runs", "artifacts"}
        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in skip]
            for name in files:
                rel = (Path(root) / name).relative_to(REPO_ROOT).as_posix()
                if "/presets/" not in rel and not rel.startswith("presets/"):
                    continue
                self.assertTrue(rel.startswith(allowed_prefixes), rel)

    def test_eval_decode_params_resolve_from_one_control_config(self) -> None:
        raw = load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        default_decode = raw["eval"]["decode"]["default"]
        mainline_decode = raw["eval"]["decode"]["mainline"]
        expected_top_p = float({**default_decode, **mainline_decode}["generate_top_p"])
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_step5_upstream_fixture(repo, task_id=2, run_id="1_1_1")
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                cfg, _, snapshot = resolve_config(
                    config_path=REPO_ROOT / "configs" / "odcr.yaml",
                    command="eval",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    from_step5="1_1_1",
                    eval_profile="balanced_2gpu",
                    mode="full",
                )
            finally:
                config_resolver._REPO_ROOT = old_root
        self.assertEqual(snapshot["field_sources"]["decode"], "eval.decode.mainline")
        self.assertEqual(float(cfg.generate_top_p), expected_top_p)
        self.assertNotEqual(float(cfg.generate_top_p), 0.9)

    def test_step4_rcr_params_resolve_from_one_control_config(self) -> None:
        raw = load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        expected = raw["step4"]["rcr"]
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                cfg, _, snapshot = resolve_config(
                    config_path=REPO_ROOT / "configs" / "odcr.yaml",
                    command="step4",
                    task_id=2,
                    set_overrides=[],
                    dry_run=True,
                    from_step3="2",
                    eval_profile="balanced_2gpu",
                    mode="full",
                )
            finally:
                config_resolver._REPO_ROOT = old_root
        self.assertEqual(snapshot["field_sources"]["step4_rcr"], "step4.rcr")
        resolved = snapshot["step4_rcr"]
        self.assertEqual(
            float(resolved["cf_reliability_weights"]["content_retention"]),
            float(expected["cf_reliability_weights"]["content_retention"]),
        )
        self.assertIn("ODCR_STEP4_RCR_CONFIG_JSON", (REPO_ROOT / "code" / "odcr_core" / "runners.py").read_text(encoding="utf-8"))
        self.assertIn("content_retention_score", cfg.step4_rcr_config_json)

    def test_step4_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R012", "R013", "R014", "R015", "R116"):
            self.assertEqual(statuses.get(rid), "PASS", rid)

    def test_no_accum_guardrail_rule_is_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        self.assertEqual(statuses.get("R117"), "PASS")
        self.assertEqual(RULE_GROUP_BY_ID.get("R117"), "no-accum-architecture")

    def test_step5_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R016", "R017", "R018", "R019", "R020", "R021", "R022", "R023", "R030"):
            self.assertEqual(statuses.get(rid), "PASS", rid)

    def test_step3_structured_loss_guardrail_rule_is_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        self.assertEqual(statuses.get("R029"), "PASS")

    def test_step3_upstream_gate_guardrail_rule_is_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        self.assertEqual(statuses.get("R097"), "PASS")
        self.assertEqual(RULE_GROUP_BY_ID.get("R097"), "step3-mainline")

    def test_step3_v0_parameter_guardrail_rule_is_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        self.assertEqual(statuses.get("R099"), "PASS")
        self.assertEqual(RULE_GROUP_BY_ID.get("R099"), "step3-mainline")

    def test_preprocess_contract_guardrail_rule_is_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        self.assertEqual(statuses.get("R024"), "PASS")

    def test_phase2_source_cleanup_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R025", "R026", "R027", "R028", "R041"):
            self.assertEqual(statuses.get(rid), "PASS", rid)

    def test_phase4a_lineage_cache_eval_guardrails_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R031", "R032", "R033", "R034", "R035"):
            self.assertEqual(statuses.get(rid), "PASS", rid)

    def test_phase4b_ddp_graph_guardrails_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R036", "R037", "R038", "R039", "R040"):
            self.assertEqual(statuses.get(rid), "PASS", rid)

    def test_evolution_protocol_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R042", "R043", "R044", "R045", "R046", "R047", "R048", "R049", "R050", "R096"):
            self.assertEqual(statuses.get(rid), "PASS", rid)
            self.assertEqual(RULE_GROUP_BY_ID.get(rid), "evolution-protocol")

    def test_post_edit_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R051", "R052", "R053", "R054", "R055", "R056"):
            self.assertEqual(statuses.get(rid), "PASS", rid)
            self.assertEqual(RULE_GROUP_BY_ID.get(rid), "post-edit-workflow")
        self.assertEqual(statuses.get("R089"), "PASS", "R089")
        self.assertEqual(RULE_GROUP_BY_ID.get("R089"), "post-edit-workflow")

    def test_run_summary_logging_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R057", "R058", "R059", "R090", "R091"):
            self.assertEqual(statuses.get(rid), "PASS", rid)
            self.assertEqual(RULE_GROUP_BY_ID.get(rid), "run-summary-logging")

    def test_p0_cache_hard_gate_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R092", "R093", "R094", "R098"):
            self.assertEqual(statuses.get(rid), "PASS", rid)
            self.assertEqual(RULE_GROUP_BY_ID.get(rid), "p0-cache-hard-gates")

    def test_logging_console_file_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R060", "R061", "R062"):
            self.assertEqual(statuses.get(rid), "PASS", rid)
            self.assertEqual(RULE_GROUP_BY_ID.get(rid), "logging-console-file")

    def test_post_edit_fast_path_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in (
            "R063",
            "R064",
            "R065",
            "R066",
            "R067",
            "R073",
            "R074",
            "R075",
            "R076",
            "R077",
        ):
            self.assertEqual(statuses.get(rid), "PASS", rid)
            self.assertEqual(RULE_GROUP_BY_ID.get(rid), "post-edit-fast-path")

    def test_logging_artifact_evolution_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R068", "R069", "R070", "R071", "R072"):
            self.assertEqual(statuses.get(rid), "PASS", rid)
            self.assertEqual(RULE_GROUP_BY_ID.get(rid), "logging-artifact-evolution")

    def test_old_layout_tail_cleanup_guardrail_rules_are_present(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        for rid in ("R084", "R085", "R086", "R087", "R088"):
            self.assertEqual(statuses.get(rid), "PASS", rid)
            self.assertEqual(RULE_GROUP_BY_ID.get(rid), "logging-old-layout-tail")

    def test_evolution_protocol_bad_snippets_are_detected(self) -> None:
        bad_snippets = {
            "R042": 'p.add_argument("--new-temperature", type=float, default=0.7)',
            "R043": 'df["new_signal_score"] = values',
            "R044": "def write_new_export(path, rows):\n    pd.DataFrame(rows).to_csv(path)\n",
            "R045": "#!/usr/bin/env bash\npython code/executors/step5_entry.py \"$@\"\n",
            "R046": 'lr = float(os.environ.get("ODCR_NEW_LR", "0.001"))',
            "R047": "def novelty_loss(x):\n    return x.mean()\n",
            "R048": "if mask.any():\n    loss = aux_loss(values[mask])\n",
            "R049": 'eta = cfg.get("eta", cfg.get("adv", 0.1))',
        }
        paths = {"R045": "scripts/new_step5.sh"}
        for rid, snippet in bad_snippets.items():
            with self.subTest(rule=rid):
                findings = scan_evolution_snippet(rid, snippet, path=paths.get(rid, "code/executors/new_feature.py"))
                self.assertTrue(findings, rid)

    def test_deprecated_config_snapshot_name_is_detected(self) -> None:
        snippet = 'open(meta_dir / "config_resolved.json", "w", encoding="utf-8")'
        findings = scan_run_artifact_snippet("R058", snippet)
        self.assertTrue(findings)
        self.assertIn("meta/resolved_config.json", findings[0].suggestion)

    def test_logging_impact_checklist_rows_are_present(self) -> None:
        text = (REPO_ROOT / "docs" / "ODCR_FEATURE_INTEGRATION_CHECKLIST.md").read_text(encoding="utf-8")
        for row in (
            "console_output_changed",
            "file_log_added",
            "metrics_file_added",
            "cache_file_added",
            "report_file_added",
            "run_summary_updated",
            "latest_pointer_updated",
            "AI_analysis_output_added",
            "artifact_role",
            "output_directory",
            "producer",
            "consumer",
            "retention_policy",
            "verbose_or_default",
            "post_edit_logging_scope",
        ):
            self.assertIn(row, text)

    def test_codex_template_has_logging_artifact_impact_section(self) -> None:
        text = (REPO_ROOT / "docs" / "CODEX_CHANGE_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
        self.assertIn("Logging / Artifact Output Impact", text)
        for term in (
            "Output role",
            "Directory rationale",
            "Duplicate or replacement",
            "run_summary indexing",
            "latest.json update",
            "Default visibility",
            "Guardrail/test needed",
        ):
            self.assertIn(term, text)

    def test_evolution_protocol_has_logging_evolution_rules(self) -> None:
        text = (REPO_ROOT / "docs" / "ODCR_EVOLUTION_PROTOCOL.md").read_text(encoding="utf-8")
        self.assertIn("Logging And Artifact Evolution", text)
        for term in (
            "new logs",
            "new reports",
            "New metrics and caches",
            "Artifact role",
            "Real run logs must not be written into `data/` or `merged/`",
            "AI_analysis/` must not become a full training log mirror",
            "Default console output must remain summary-level",
            "Formal run handoff starts at `meta/run_summary.json`",
        ):
            self.assertIn(term, text)

    def test_logging_artifact_bad_snippets_are_detected(self) -> None:
        bad_snippets = {
            "R068": 'open("runs/step3/task4/1/meta/new_report.json", "w")',
            "R069": 'meta_dir = Path("runs/step3/task4/1/meta")\n(meta_dir / "debug_report.json").write_text("{}")',
            "R070": 'shutil.copyfile(run_dir / "meta" / "full.log", "AI_analysis/full_train.log")',
            "R071": 'print("ODCR One-Control Guardrails: PASS " + "\\n".join(per_rule_pass_lines))',
            "R072": 'open("data/foo.log", "w")\nopen("merged/foo.log", "w")\nopen("code/log.out", "a")',
        }
        for rid, snippet in bad_snippets.items():
            with self.subTest(rule=rid):
                findings = scan_logging_artifact_snippet(rid, snippet, path="code/odcr_core/new_logging.py")
                self.assertTrue(findings, rid)

    def test_old_layout_log_bad_snippets_are_detected(self) -> None:
        bad_snippets = {
            "R084": 'open("logs/run.log", "w")\nopen("code/log.out", "a")',
            "R085": 'legacy_parent = Path("runs/task4/step3")\ncandidates = [meta / "train.log"]',
            "R086": 'open("nohup_worker.log", "a")\nopen("fallback.log", "a")\nopen("mirror.log", "a")',
            "R087": 'shutil.copyfile(run_dir / "meta" / "full.log", "AI_analysis/full_train.log")',
            "R088": 'open("data/foo.log", "w")\nopen("merged/foo.log", "w")',
        }
        for rid, snippet in bad_snippets.items():
            with self.subTest(rule=rid):
                findings = scan_old_layout_log_snippet(rid, snippet, path="code/odcr_core/new_logging.py")
                self.assertTrue(findings, rid)

    def test_logging_artifact_negative_docs_and_tests_are_ignored(self) -> None:
        snippet = "\n".join(
            [
                'open("data/foo.log", "w")',
                'shutil.copyfile("runs/step3/task4/1/meta/full.log", "AI_analysis/full_train.log")',
                'print("ODCR One-Control Guardrails: PASS " + details)',
            ]
        )
        for rid in ("R070", "R071", "R072"):
            self.assertFalse(scan_logging_artifact_snippet(rid, snippet, path="docs/history/logging_bad_example.md"), rid)
            self.assertFalse(scan_logging_artifact_snippet(rid, snippet, path="code/tests/test_logging_bad_example.py"), rid)

    def test_old_layout_history_docs_ai_analysis_and_tests_are_ignored(self) -> None:
        snippet = "\n".join(
            [
                'open("logs/run.log", "w")',
                'open("code/log.out", "a")',
                'open("fallback.log", "a")',
                'shutil.copyfile("runs/step3/task4/1/meta/full.log", "AI_analysis/full_train.log")',
            ]
        )
        for rid in ("R084", "R085", "R086", "R087", "R088"):
            self.assertFalse(scan_old_layout_log_snippet(rid, snippet, path="docs/history/logging_bad_example.md"), rid)
            self.assertFalse(scan_old_layout_log_snippet(rid, snippet, path="AI_analysis/ledger.md"), rid)
            self.assertFalse(scan_old_layout_log_snippet(rid, snippet, path="code/tests/test_logging_bad_example.py"), rid)

    def test_evolution_protocol_negative_docs_and_tests_are_ignored(self) -> None:
        snippet = "\n".join(
            [
                'p.add_argument("--new-temperature", type=float, default=0.7)',
                'lr = float(os.environ.get("ODCR_NEW_LR", "0.001"))',
                "if mask.any():",
                "    loss = aux_loss(values[mask])",
            ]
        )
        for rid in ("R042", "R046", "R048"):
            self.assertFalse(scan_evolution_snippet(rid, snippet, path="docs/history/bad_example.md"), rid)
            self.assertFalse(scan_evolution_snippet(rid, snippet, path="code/tests/test_bad_example.py"), rid)

    def test_codex_change_workflow_governance_docs_are_required(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        rule = next(item for item in report.results if item.rule_id == "R050")
        self.assertEqual(rule.status, "PASS")
        for rel in (
            "AGENTS.md",
            "docs/ODCR_EVOLUTION_PROTOCOL.md",
            "docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md",
            "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md",
        ):
            self.assertTrue((REPO_ROOT / rel).is_file(), rel)
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("CODEX_CHANGE_REQUEST_TEMPLATE.md", agents)
        self.assertIn("must not skip checklist", agents)
        self.assertIn("AI_analysis ledger", agents)
        template = (REPO_ROOT / "docs" / "CODEX_CHANGE_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
        self.assertIn("Change Type Selection", template)
        self.assertIn("Required Impact Surface", template)
        self.assertIn("Required Outputs", template)

    def test_post_edit_workflow_docs_require_final_response_gate(self) -> None:
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        agents_normalized = " ".join(agents.split())
        self.assertIn("post-edit validation suite", agents_normalized)
        self.assertIn("python code/tools/odcr_post_edit_check.py --scope <scope>", agents_normalized)
        self.assertIn("must not wait for git commit", agents_normalized)
        self.assertIn("does not require git commit", agents_normalized)
        self.assertIn("Codex Hooks", agents_normalized)
        self.assertIn("Stop hook", agents_normalized)
        self.assertIn("git hook / CI are optional", agents_normalized)
        self.assertIn("must not leave validation to the user", agents_normalized)
        self.assertIn("fix and", agents_normalized)
        self.assertIn("rerun", agents_normalized)
        self.assertIn("Validation block", agents_normalized)
        self.assertIn("narrowest applicable post-edit validation scope", agents_normalized)
        self.assertIn("ignored-only, dirty-workspace-only", agents_normalized)
        self.assertIn("not fixed defaults for every user-facing change", agents_normalized)
        required_checks = agents.split("## Required Checks Before Finishing", 1)[1]
        self.assertNotIn("For user-facing changes, also run", required_checks)
        self.assertIn("./odcr step3 --task 2 --dry-run", required_checks)
        self.assertIn("Step3", required_checks)

    def test_codex_template_documents_scope_selection(self) -> None:
        template = (REPO_ROOT / "docs" / "CODEX_CHANGE_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
        normalized = " ".join(template.split())
        self.assertIn("Validation block", normalized)
        self.assertIn("narrowest applicable scope", normalized)
        self.assertIn("governance-fast", normalized)
        self.assertIn("logging/path/tail/AI_analysis", normalized)
        self.assertIn("dirty-workspace-only", normalized)
        self.assertIn("Step3 dry-run is not a universal default", normalized)
        self.assertIn("cross-stage contract / manifest / lineage / cache or checkpoint hard gate / eval-rerank gate", normalized)
        self.assertIn("`--scope all` is not permanently banned", normalized)

        template = (REPO_ROOT / "docs" / "CODEX_CHANGE_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
        self.assertIn("Post-Edit Validation:", template)
        for term in (
            "- chosen scope:",
            "- commands run:",
            "- compileall:",
            "- guardrail strict:",
            "- doctor:",
            "- show/dry-run:",
            "- tests:",
            "- real training:",
            "- failures fixed:",
            "- final status:",
        ):
            self.assertIn(term, template)
        self.assertIn("Validation block", template)

    def test_gpu_tmux_policy_docs_distinguish_tmux_from_gpu(self) -> None:
        required = (
            "tmux -L odcr_gpu new-session -A -s odcr",
            "odcr-enter-gpu <JOBID>",
            "current tmux session's real-time CUDA",
            "GPU use is allowed by default",
            "repo-local validation, probe, and bounded runtime",
            "post-edit full is not a GPU prerequisite",
            "user-created, already-entered, uniquely validated GPU pane",
            "not arbitrary send-keys",
            "formal namespace guard",
            "AI_analysis",
            "old `AI_analysis` probe output",
        )
        docs = (
            "AGENTS.md",
            "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md",
            "docs/ODCR_ACTIVE_ARCHITECTURE.md",
            "docs/ODCR_EVOLUTION_PROTOCOL.md",
            "docs/ODCR_ARCHITECTURE_CONTRACT.md",
            "docs/AI_PROJECT_CANONICAL.md",
        )
        for rel in docs:
            with self.subTest(rel=rel):
                text = (REPO_ROOT / rel).read_text(encoding="utf-8")
                normalized = " ".join(text.split())
                for term in required:
                    self.assertIn(term, normalized)
                self.assertIn("Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`", normalized)
                self.assertNotRegex(normalized, r"tmux session exists\s+(?:means|equals|is equivalent to)\s+GPU")
                self.assertNotRegex(normalized, r"permanent(?:ly)?\s+(?:ban|forbid).{0,40}--scope all")

    def test_codex_stop_hook_post_edit_integration_exists(self) -> None:
        hooks_json = REPO_ROOT / ".codex" / "hooks.json"
        hook_wrapper = REPO_ROOT / ".codex" / "hooks" / "odcr_post_edit_stop.sh"
        hook_script = REPO_ROOT / ".codex" / "hooks" / "odcr_post_edit_stop.py"
        self.assertTrue(hooks_json.is_file(), ".codex/hooks.json must exist")
        self.assertTrue(hook_wrapper.is_file(), ".codex/hooks/odcr_post_edit_stop.sh must exist")
        self.assertTrue(hook_script.is_file(), ".codex/hooks/odcr_post_edit_stop.py must exist")

        hooks_text = hooks_json.read_text(encoding="utf-8")
        self.assertNotIn("/usr/bin/python3", hooks_text)
        self.assertNotIn("$(git rev-parse", hooks_text)

        config = json.loads(hooks_text)
        self.assertIn("Stop", config.get("hooks", {}))
        stop_entries = config["hooks"]["Stop"]
        commands = [
            hook.get("command", "")
            for entry in stop_entries
            for hook in entry.get("hooks", [])
            if hook.get("type") == "command"
        ]
        self.assertIn(HOOK_STOP_COMMAND, commands)
        for command in commands:
            self.assertTrue(command.startswith("/usr/bin/env bash /"), command)
            self.assertNotRegex(command, r"(^|\s)\.codex/hooks/")
            self.assertNotIn("/usr/bin/python3", command)
            self.assertNotIn("$(git rev-parse", command)
        self.assertTrue(any("timeout" in hook and hook["timeout"] <= 180 for entry in stop_entries for hook in entry.get("hooks", [])))

        bash_proc = subprocess.run(
            ["bash", "-n", ".codex/hooks/odcr_post_edit_stop.sh"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        self.assertEqual(bash_proc.returncode, 0, bash_proc.stdout)

        wrapper = hook_wrapper.read_text(encoding="utf-8")
        self.assertIn("#!/usr/bin/env bash", wrapper)
        self.assertIn("set -euo pipefail", wrapper)
        self.assertIn("BASH_SOURCE", wrapper)
        self.assertIn("../..", wrapper)
        self.assertIn("EXPECTED_REPO_ROOT", wrapper)
        self.assertIn("CONDA_PREFIX", wrapper)
        self.assertIn("command -v python3", wrapper)
        self.assertIn("command -v python", wrapper)
        self.assertIn(D4C_PYTHON_ABS, wrapper)
        self.assertIn("VERSION_CHECK", wrapper)
        self.assertIn("(v[0], v[1]) >= (3, 8)", wrapper)
        self.assertIn("python_discovery", wrapper)
        self.assertIn("selected none", wrapper)
        self.assertIn("ODCR_HOOK_SELECTED_PYTHON", wrapper)
        self.assertIn(HOOK_DIAGNOSTICS_REL, wrapper)
        self.assertIn('exec "$PYTHON_BIN" "$REPO_ROOT/.codex/hooks/odcr_post_edit_stop.py"', wrapper)
        self.assertLess(wrapper.index(D4C_PYTHON_ABS), wrapper.index("$CONDA_PREFIX/bin/python"))
        self.assertLess(wrapper.index("$CONDA_PREFIX/bin/python"), wrapper.index("command -v python3"))
        self.assertLess(wrapper.index("command -v python3"), wrapper.index("command -v python >/dev/null"))

        script = hook_script.read_text(encoding="utf-8")
        self.assertIn("code", script)
        self.assertIn("tools", script)
        self.assertIn("odcr_post_edit_check.py", script)
        self.assertIn("ODCR_HOOK_DRY_RUN", script)
        self.assertIn("AI_analysis", script)
        self.assertIn("codex_hooks", script)
        self.assertIn("runtime_last.json", script)
        self.assertIn("post_edit_returncode", script)
        self.assertIn("failure_stage", script)
        self.assertIn('RUNTIME_SCHEMA_VERSION = "odcr_codex_hook_runtime/2.2"', script)
        self.assertIn("DEFAULT_WRAPPER_TIMEOUT_SECONDS = 180", script)
        self.assertIn("DEFAULT_HOOK_MAX_SECONDS = DEFAULT_HOOK_CHILD_MAX_SECONDS", script)
        self.assertIn("ODCR_HOOK_MAX_SECONDS", script)
        self.assertIn("MAX_TOUCHED_FILES_SAMPLE = 50", script)
        self.assertIn("governance-fast", script)
        self.assertIn("IGNORED_EXACT_PATHS", script)
        self.assertIn("IGNORED_DIR_PREFIXES", script)
        self.assertIn("IGNORED_FILE_PATTERNS", script)
        self.assertIn("_ignored_only_reason", script)
        self.assertIn("ignored_only", script)
        self.assertIn("transcript_path", script)
        self.assertIn("infer_scope_for_payload", script)
        self.assertIn("no_session_touched_files", script)
        self.assertIn("unknown_session_touched_files", script)
        self.assertIn("transcript_parse_failed", script)
        self.assertIn("multi_business_stage_session_touched_files", script)
        self.assertIn("session_touched_files_count", script)
        self.assertIn("session_touched_files_sample", script)
        self.assertIn("ignored_files_count", script)
        self.assertIn("effective_scope_files_count", script)
        self.assertIn("effective_scope_files_sample", script)
        self.assertIn("ignored_files_sample", script)
        self.assertIn("workspace_dirty_detected", script)
        self.assertIn("workspace_changed_files_count", script)
        self.assertIn("workspace_git_status_used_for_scope", script)
        self.assertIn("selected_scope", script)
        self.assertIn("scope_candidates", script)
        self.assertIn("multi_stage_detected", script)
        self.assertIn('json.dumps({"continue": True}', script)
        self.assertIn("sys.stdout.write", script)
        self.assertNotIn("print(", script)

        forbidden = (
            r"(?:\./odcr|code/odcr\.py)\s+preprocess\b",
            r"(?:\./odcr|code/odcr\.py)\s+(?:eval|rerank)\b",
            r"(?:\./odcr|code/odcr\.py)\s+(?:step3|step4|step5)\b(?![^\n]*--dry-run)",
            r"\bpython(?:3)?\s+code/(?:preprocess|train|eval|rerank|executors/step[345])",
        )
        for text in (wrapper, script):
            for pattern in forbidden:
                self.assertIsNone(re.search(pattern, text), pattern)

    def test_codex_stop_hook_dry_run_stdout_json_and_runtime_schema(self) -> None:
        env = self._env()
        env["ODCR_HOOK_DRY_RUN"] = "1"
        proc = subprocess.run(
            ["bash", str(REPO_ROOT / ".codex" / "hooks" / "odcr_post_edit_stop.sh")],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), {"continue": True})
        runtime_last = REPO_ROOT / "AI_analysis" / "01_raw_logs" / "codex_hooks" / "runtime_last.json"
        self.assertTrue(runtime_last.is_file())
        payload = json.loads(runtime_last.read_text(encoding="utf-8"))
        for key in (
            "cwd",
            "repo_root",
            "PATH",
            "HOME",
            "CODEX_HOME",
            "SHELL",
            "CONDA_PREFIX",
            "selected_python",
            "selected_python_version",
            "hook_event_name",
            "stop_hook_active",
            "post_edit_command",
            "post_edit_returncode",
            "failure_stage",
            "stdout_path",
            "stderr_path",
            "timestamp",
            "schema_version",
            "inference_source",
            "inference_reason",
            "session_touched_files_count",
            "session_touched_files_sample",
            "ignored_files_count",
            "effective_scope_files_count",
            "effective_scope_files_sample",
            "ignored_files_sample",
            "workspace_dirty_detected",
            "workspace_changed_files_count",
            "workspace_git_status_used_for_scope",
            "selected_scope",
            "scope_candidates",
            "multi_stage_detected",
            "skipped",
            "skip_reason",
            "max_seconds",
        ):
            self.assertIn(key, payload)
        self.assertNotIn("changed_files", payload)
        self.assertNotIn("changed_files_total", payload)
        self.assertNotIn("changed_files_sample", payload)
        self.assertNotIn("raw_touched_files_count", payload)
        self.assertNotIn("effective_touched_files_count", payload)
        self.assertNotIn("touched_files_sample", payload)
        self.assertNotIn("git_changed_files_count", payload)
        self.assertNotIn("git_status_truncated", payload)
        self.assertLessEqual(len(payload["session_touched_files_sample"]), 50)
        self.assertLessEqual(len(payload["ignored_files_sample"]), 50)
        self.assertLessEqual(len(payload["effective_scope_files_sample"]), 50)
        self.assertEqual(payload["schema_version"], "odcr_codex_hook_runtime/2.2")
        self.assertFalse(payload["workspace_git_status_used_for_scope"])
        self.assertIn(payload["selected_scope"], {"governance-fast", "governance", "config", "logging", "preprocess", "step3", "step4", "step5", "eval", "all", "skip"})
        self.assertNotEqual(payload["inference_reason"], "multi_stage_change")
        self.assertEqual(payload["post_edit_returncode"], 0)
        if payload["selected_scope"] != "skip":
            command = payload["post_edit_command"]
            self.assertIsInstance(command, list)
            self.assertEqual(command[command.index("--scope") + 1], payload["selected_scope"])
            self.assertIn("--max-seconds", command)
            self.assertEqual(command[command.index("--max-seconds") + 1], "180")
        self.assertTrue(str(payload["stdout_path"]).startswith(str(REPO_ROOT / HOOK_DIAGNOSTICS_REL)))
        self.assertTrue(str(payload["stderr_path"]).startswith(str(REPO_ROOT / HOOK_DIAGNOSTICS_REL)))

    def test_codex_stop_hook_minimal_path_does_not_select_python2(self) -> None:
        env = {
            "HOME": os.environ.get("HOME", ""),
            "PATH": "/usr/bin:/bin",
            "ODCR_HOOK_DRY_RUN": "1",
        }
        proc = subprocess.run(
            ["bash", str(REPO_ROOT / ".codex" / "hooks" / "odcr_post_edit_stop.sh")],
            cwd=REPO_ROOT / "code",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        d4c_python = Path(D4C_PYTHON_ABS)
        if d4c_python.exists():
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout), {"continue": True})
            runtime_last = REPO_ROOT / "AI_analysis" / "01_raw_logs" / "codex_hooks" / "runtime_last.json"
            payload = json.loads(runtime_last.read_text(encoding="utf-8"))
            self.assertEqual(payload["selected_python"], D4C_PYTHON_ABS)
            major, minor, *_ = str(payload["selected_python_version"]).split(".")
            self.assertGreaterEqual((int(major), int(minor)), (3, 8))
            self.assertNotEqual(payload["selected_python"], "/usr/bin/python")
        else:
            self.assertEqual(proc.returncode, 127)
            self.assertIn("selected none", proc.stderr)
            self.assertIn("PATH=/usr/bin:/bin", proc.stderr)

    def test_post_edit_check_script_exists_and_dry_run_is_lightweight(self) -> None:
        self.assertTrue((REPO_ROOT / "code" / "tools" / "odcr_post_edit_check.py").is_file())
        for scope in SCOPES:
            with self.subTest(scope=scope):
                commands = build_plan(scope, repo_root=REPO_ROOT, python_executable=sys.executable)
                self.assertEqual(plan_safety_violations(commands), [])

        proc = subprocess.run(
            [sys.executable, "code/tools/odcr_post_edit_check.py", "--scope", "all", "--dry-run"],
            cwd=REPO_ROOT,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertNotIn("./odcr preprocess", proc.stdout)
        self.assertNotIn("./odcr eval", proc.stdout)
        self.assertNotIn("./odcr rerank", proc.stdout)
        for line in proc.stdout.splitlines():
            if "./odcr step3 " in line or "./odcr step4 " in line or "./odcr step5 " in line:
                self.assertIn("--dry-run", line)

        fast = subprocess.run(
            [sys.executable, "code/tools/odcr_post_edit_check.py", "--scope", "governance-fast", "--dry-run"],
            cwd=REPO_ROOT,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        self.assertEqual(fast.returncode, 0, fast.stdout)
        self.assertIn("python code/tools/check_one_control_guardrails.py --strict", fast.stdout)
        self.assertNotIn("compileall", fast.stdout)
        self.assertNotIn("./odcr doctor", fast.stdout)
        self.assertNotIn("./odcr step3", fast.stdout)

    def test_feature_integration_checklist_tool_prints_stage_stub(self) -> None:
        help_proc = subprocess.run(
            [sys.executable, "code/tools/print_feature_integration_checklist.py", "--help"],
            cwd=REPO_ROOT,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        self.assertEqual(help_proc.returncode, 0, help_proc.stdout)
        self.assertIn("--stage", help_proc.stdout)
        proc = subprocess.run(
            [sys.executable, "code/tools/print_feature_integration_checklist.py", "--stage", "step4"],
            cwd=REPO_ROOT,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("step4.rcr.<new_key>", proc.stdout)
        self.assertIn("Required Impact Surface", proc.stdout)
        self.assertIn("AI_analysis ledger path", proc.stdout)

    def test_all_guardrail_numbers_and_groups_are_visible(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        text = format_report(report)
        rule_ids = [item.rule_id for item in report.results]
        expected = {rid for _group, ids in GUARDRAIL_GROUPS for rid in ids}
        self.assertEqual(set(rule_ids), expected)
        self.assertEqual(len(rule_ids), len(expected))
        for group, ids in GUARDRAIL_GROUPS:
            self.assertIn(f" - {group}:", text)
            for rid in ids:
                self.assertIn(rid, text)
                self.assertIn(f"{rid} ({group})", text)

    def test_roots_models_embed_dim_resolve_from_one_control_config(self) -> None:
        raw = load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        cfg, _, snapshot = _resolve_step5_with_fixture(task_id=2, from_step4="1_1")
        self.assertEqual(snapshot["field_sources"]["data_dir"], "project.data_dir")
        self.assertEqual(snapshot["field_sources"]["models_dir"], "env.models_dir")
        self.assertEqual(snapshot["field_sources"]["embed_dim"], "env.embed_dim")
        self.assertEqual(int(cfg.embed_dim), int(raw["env"]["embed_dim"]))
        self.assertIn("runtime_roots", cfg.effective_training_payload_json)
        self.assertIn("ODCR_RESOLVED_EMBED_DIM", (REPO_ROOT / "code" / "odcr_core" / "runners.py").read_text(encoding="utf-8"))

    def test_preprocess_resolved_payload_honors_one_control_overrides(self) -> None:
        cfg = build_preprocess_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            stage_letter="b",
            set_overrides=[
                "project.data_dir=/tmp/odcr_probe_data",
                "env.sentence_embed_model=/tmp/odcr_probe_model",
            ],
            dry_run=True,
        )
        runtime = PreprocessRuntime(cfg)
        self.assertEqual(str(runtime.data_root), str(Path("/tmp/odcr_probe_data").resolve()))
        self.assertEqual(str(runtime.sentence_embed_model_path), str(Path("/tmp/odcr_probe_model").resolve()))
        self.assertEqual(cfg.resolved.data_dir, str(Path("/tmp/odcr_probe_data").resolve()))
        self.assertEqual(cfg.resolved.sentence_embed_model_path, str(Path("/tmp/odcr_probe_model").resolve()))
        child_env = runtime._child_env()
        self.assertEqual(child_env["ODCR_RESOLVED_DATA_DIR"], str(Path("/tmp/odcr_probe_data").resolve()))
        self.assertEqual(
            child_env["ODCR_RESOLVED_SENTENCE_EMBED_MODEL"],
            str(Path("/tmp/odcr_probe_model").resolve()),
        )
        self.assertNotIn("ODCR_DATA_DIR", child_env)

    def test_manifest_backbone_hidden_size_uses_resolved_config_not_user_env(self) -> None:
        old_embed_env = os.environ.pop("ODCR_EMBED_DIM", None)
        try:
            cfg, _, _ = _resolve_step5_with_fixture(task_id=4, from_step4="1_1")
            cfg = replace(cfg, step5_run=None)
            os.environ["ODCR_EMBED_DIM"] = str(int(cfg.embed_dim) + 17)
            manifest = build_run_manifest(cfg)
            self.assertEqual(
                int(manifest["backbones"]["sentence_embed"]["hidden_size"]),
                int(cfg.embed_dim),
            )
            self.assertNotEqual(
                int(manifest["backbones"]["sentence_embed"]["hidden_size"]),
                int(os.environ["ODCR_EMBED_DIM"]),
            )
        finally:
            if old_embed_env is None:
                os.environ.pop("ODCR_EMBED_DIM", None)
            else:
                os.environ["ODCR_EMBED_DIM"] = old_embed_env

    def test_manifest_backbone_hidden_size_requires_resolved_embed_dim(self) -> None:
        cfg, _, _ = _resolve_step5_with_fixture(task_id=2, from_step4="1_1")
        cfg = replace(cfg, step5_run=None)
        bad_cfg = replace(cfg, embed_dim=0)
        with self.assertRaisesRegex(RuntimeError, "cfg.embed_dim"):
            build_run_manifest(bad_cfg)

    def test_step5_innovation_params_resolve_from_one_control_config(self) -> None:
        raw = load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        expected = raw["step5"]
        cfg, _, snapshot = _resolve_step5_with_fixture(task_id=4, from_step4="1_1")
        self.assertEqual(snapshot["field_sources"]["step5_lci"], "step5.lci")
        self.assertEqual(snapshot["field_sources"]["step5_explainer_gate"], "step5.explainer_gate")
        self.assertEqual(snapshot["field_sources"]["step5_native_lora"], "step5.ccv.native_lora")
        self.assertEqual(snapshot["field_sources"]["step5_model"], "step5.model")
        self.assertEqual(
            snapshot["field_sources"]["step5_ddp_find_unused_parameters"],
            "step5.ddp.find_unused_parameters",
        )
        self.assertEqual(
            snapshot["field_sources"]["step5_ddp_find_unused_false_preflight"],
            "step5.ddp.find_unused_false_preflight",
        )
        self.assertEqual(
            snapshot["field_sources"]["step5_train_explainer_loss_weight"],
            "step5.train.explainer_loss_weight",
        )
        self.assertEqual(float(snapshot["step5"]["lci"]["weight"]), float(expected["lci"]["weight"]))
        self.assertEqual(
            float(snapshot["step5"]["explainer_gate"]["bucket_weights"]["high"]),
            float(expected["explainer_gate"]["bucket_weights"]["high"]),
        )
        self.assertEqual(
            float(snapshot["step5"]["explainer_gate"]["explainer_only_multiplier"]),
            float(expected["explainer_gate"]["explainer_only_multiplier"]),
        )
        self.assertEqual(float(snapshot["step5"]["fca"]["weight"]), float(expected["fca"]["weight"]))
        self.assertEqual(int(snapshot["step5"]["ccv"]["soft_prompt_len"]), int(expected["ccv"]["soft_prompt_len"]))
        self.assertEqual(int(snapshot["step5"]["ccv"]["native_lora"]["r"]), int(expected["ccv"]["native_lora"]["r"]))
        self.assertTrue(snapshot["step5_ddp"]["ddp_find_unused_parameters"])
        self.assertEqual(snapshot["step5_ddp"]["ddp_find_unused_false_preflight"], "synthetic_one_batch")
        self.assertEqual(int(snapshot["step5_model"]["nlayers"]), int(expected["model"]["nlayers"]))
        self.assertIn("step5_innovation", cfg.effective_training_payload_json)
        self.assertIn("step5_model", cfg.effective_training_payload_json)
        self.assertNotIn('"eta"', cfg.effective_training_payload_json)
        self.assertNotIn('"adv"', cfg.effective_training_payload_json)
        self.assertIn("content_evidence", cfg.step5_innovation_config_json)
        self.assertEqual(int(cfg.lora_r), int(expected["ccv"]["native_lora"]["r"]))
        self.assertEqual(float(cfg.lora_alpha), float(expected["ccv"]["native_lora"]["alpha"]))
        self.assertEqual(float(cfg.explainer_loss_weight), float(expected["train"]["explainer_loss_weight"]))
        self.assertEqual(int(cfg.nlayers), int(expected["model"]["nlayers"]))

    def test_step3_structured_losses_resolve_from_one_control_config(self) -> None:
        raw = load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        cfg, _, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            mode="full",
        )
        expected = raw["step3"]["structured_losses"]
        self.assertEqual(snapshot["field_sources"]["step3_structured_losses"], "step3.structured_losses")
        self.assertEqual(snapshot["field_sources"]["step3_loss_semantics"], "step3.loss_semantics")
        self.assertEqual(
            snapshot["field_sources"]["step3_ddp_find_unused_parameters"],
            "step3.ddp.find_unused_parameters",
        )
        self.assertEqual(
            float(snapshot["step3_structured_losses"]["content_alignment_weight"]),
            float(expected["content_alignment_weight"]),
        )
        self.assertFalse(snapshot["step3_ddp"]["ddp_find_unused_parameters"])
        self.assertFalse(snapshot["step3_ddp"]["ddp_static_graph"])
        self.assertTrue(snapshot["step3_ddp"]["ddp_graph_safety_preflight"])
        self.assertEqual(float(snapshot["step3_loss_semantics"]["specific_separation_margin"]), 0.6)
        self.assertIn("step3_structured_losses", cfg.effective_training_payload_json)
        self.assertIn("step3_loss_semantics", cfg.effective_training_payload_json)
        self.assertIn("step3_ddp", cfg.effective_training_payload_json)
        cfg_changed, _, snapshot_changed = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[
                "step3.structured_losses.content_alignment_weight=0.42",
                "step3.loss_semantics.specific_separation_margin=0.7",
            ],
            dry_run=True,
            mode="full",
        )
        self.assertEqual(
            float(snapshot_changed["step3_structured_losses"]["content_alignment_weight"]),
            0.42,
        )
        self.assertEqual(float(snapshot_changed["step3_loss_semantics"]["specific_separation_margin"]), 0.7)
        self.assertIn('"content_alignment_weight": 0.42', cfg_changed.effective_training_payload_json)
        self.assertIn('"specific_separation_margin": 0.7', cfg_changed.effective_training_payload_json)

    def test_step5_retired_adv_eta_fail_fast(self) -> None:
        for key in ("step5.train.eta=0.1", "step5.train.adv=0.1", "tasks.4.adv=0.1"):
            with self.subTest(key=key):
                with self.assertRaises(OneControlConfigError):
                    resolve_config(
                        config_path=REPO_ROOT / "configs" / "odcr.yaml",
                        command="step5",
                        task_id=4,
                        set_overrides=[key],
                        dry_run=True,
                        from_step4="1_1",
                        eval_profile="balanced_2gpu",
                        mode="full",
                    )

    def test_doctor_invokes_guardrail(self) -> None:
        source = (REPO_ROOT / "code" / "odcr.py").read_text(encoding="utf-8")
        self.assertIn("run_checks", source)
        proc = subprocess.run(
            [sys.executable, "code/odcr.py", "doctor"],
            cwd=REPO_ROOT,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("ODCR One-Control Guardrails: PASS (0 fail, 0 warn)", proc.stdout)
        self.assertNotIn("[WARN]", proc.stdout)
        self.assertIn("one-control guardrail passed", proc.stdout)
        self.assertIn("step3/step4 resolve checks passed through unified upstream resolver", proc.stdout)
        self.assertIn("step5/eval resolver fail-fast pending upstream", proc.stdout)
        self.assertIn("no legacy preset mainline", proc.stdout)
        self.assertIn("no scattered config", proc.stdout)
        self.assertIn("no parameter drift", proc.stdout)

    def test_guardrail_covers_cpu12_and_step3_bridge_smoke(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        self.assertTrue(report.ok)
        child_rule = next(item for item in report.results if item.rule_id == "R026")
        tmux_rule = next(item for item in report.results if item.rule_id == "R096")
        self.assertEqual(child_rule.status, "PASS")
        self.assertEqual(tmux_rule.status, "PASS")
        yaml_text = (REPO_ROOT / "configs" / "odcr.yaml").read_text(encoding="utf-8")
        self.assertIn("max_parallel_cpu: 12", yaml_text)
        self.assertNotIn("max_parallel_cpu: 16", yaml_text)
        bridge_text = (REPO_ROOT / "code" / "odcr_core" / "aux" / "runtime" / "gpu_bridge.py").read_text(encoding="utf-8")
        self.assertIn('"repo-command"', bridge_text)
        self.assertIn('"repo-script"', bridge_text)
        self.assertIn('"repo-module"', bridge_text)
        self.assertIn('"command-file"', bridge_text)
        self.assertIn("build_repo_runtime_executor_script", bridge_text)
        self.assertIn("formal_namespace_blocked", bridge_text)


if __name__ == "__main__":
    unittest.main()
