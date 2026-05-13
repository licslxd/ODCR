from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from tools.odcr_post_edit_check import CheckCommand, SCOPES, build_plan, post_edit_summary_payload, run_commands, suggest_scope_for_paths  # noqa: E402


def _hook_module():
    path = REPO_ROOT / ".codex" / "hooks" / "odcr_post_edit_stop.py"
    spec = importlib.util.spec_from_file_location("_odcr_stop_hook_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_transcript(path: Path, touched_files: list[str]) -> None:
    rows = []
    for touched in touched_files:
        rows.append(
            {
                "type": "tool_use",
                "name": "functions.apply_patch",
                "arguments": (
                    "*** Begin Patch\n"
                    f"*** Update File: {touched}\n"
                    "@@\n"
                    " unchanged\n"
                    "*** End Patch\n"
                ),
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _display_commands(scope: str) -> list[str]:
    return [command.display() for command in build_plan(scope, repo_root=REPO_ROOT, python_executable="python")]


def _is_real_stage_run(command: str) -> bool:
    stage_tokens = ("./odcr step3 ", "./odcr step4 ", "./odcr step5 ", "./odcr eval ")
    return command.startswith(stage_tokens) and "--dry-run" not in command


class TestPostEditCheck(unittest.TestCase):
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def test_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, "code/tools/odcr_post_edit_check.py", "--help"],
            cwd=REPO_ROOT,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("--scope", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--max-seconds", proc.stdout)
        for scope in SCOPES:
            self.assertIn(scope, proc.stdout)

    def test_dry_run_step3(self) -> None:
        proc = subprocess.run(
            [sys.executable, "code/tools/odcr_post_edit_check.py", "--dry-run", "--scope", "step3"],
            cwd=REPO_ROOT,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("ODCR post-edit validation", proc.stdout)
        self.assertIn("python -m compileall -q code", proc.stdout)
        self.assertIn("python code/tools/check_one_control_guardrails.py --strict", proc.stdout)
        self.assertIn("./odcr step3 --task 2 --dry-run", proc.stdout)
        self.assertIn("./odcr show --stage step3", proc.stdout)
        self.assertIn("Result: DRY-RUN", proc.stdout)

    def test_unknown_scope_fails_fast(self) -> None:
        proc = subprocess.run(
            [sys.executable, "code/tools/odcr_post_edit_check.py", "--scope", "unknown", "--dry-run"],
            cwd=REPO_ROOT,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid choice", proc.stdout)

    def test_timeout_fail_closes(self) -> None:
        command = CheckCommand(
            label="timeout probe",
            argv=(sys.executable, "-c", "import time; time.sleep(5)"),
            display_argv=("python", "-c", "sleep"),
        )
        results = run_commands([command], repo_root=REPO_ROOT, max_seconds=1)
        self.assertEqual(results[0].status, "FAIL")
        self.assertEqual(results[0].classification, "timeout")
        self.assertTrue(results[0].blocks_formal)
        self.assertEqual(post_edit_summary_payload(results)["result"], "FAIL")

    def test_stage_scopes_do_not_include_real_training_commands(self) -> None:
        for scope in ("step3", "step4", "step5"):
            with self.subTest(scope=scope):
                commands = _display_commands(scope)
                self.assertFalse([command for command in commands if _is_real_stage_run(command)])

    def test_preprocess_scope_does_not_enter_later_stages(self) -> None:
        commands = _display_commands("preprocess")
        joined = "\n".join(commands)
        self.assertNotIn("./odcr step3", joined)
        self.assertNotIn("./odcr step4", joined)
        self.assertNotIn("./odcr step5", joined)
        self.assertNotIn("./odcr eval", joined)
        self.assertFalse([command for command in commands if _is_real_stage_run(command)])

    def test_all_scope_stays_lightweight(self) -> None:
        commands = _display_commands("all")
        joined = "\n".join(commands)
        forbidden_substrings = (
            "./odcr preprocess a",
            "./odcr preprocess b",
            "./odcr preprocess c",
            "./odcr eval ",
            " eval-rerank",
            " rerank ",
        )
        for forbidden in forbidden_substrings:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, joined)
        self.assertFalse([command for command in commands if _is_real_stage_run(command)])
        self.assertIn("python -c 'step4 CSB resolver dry-run'", commands)
        self.assertIn("step5 resolver dry-run", "\n".join(commands))

    def test_governance_fast_dry_run_is_minimal(self) -> None:
        commands = _display_commands("governance-fast")
        joined = "\n".join(commands)
        self.assertIn("python code/tools/check_one_control_guardrails.py --strict", joined)
        self.assertIn("python -m py_compile code/tools/check_one_control_guardrails.py", joined)
        self.assertIn("python -m py_compile code/tools/odcr_post_edit_check.py", joined)
        self.assertNotIn("compileall", joined)
        self.assertNotIn("./odcr doctor", joined)
        self.assertNotIn("--dry-run", joined)
        self.assertFalse([command for command in commands if _is_real_stage_run(command)])

    def test_logging_scope_runs_logging_path_tests_and_doctor(self) -> None:
        commands = _display_commands("logging")
        joined = "\n".join(commands)
        self.assertIn("python -m compileall -q code", joined)
        self.assertIn("python code/tools/check_one_control_guardrails.py --strict", joined)
        self.assertIn("python code/tests/test_run_summary_logging.py -v", joined)
        self.assertIn("python code/tests/test_logging_console_file.py -v", joined)
        self.assertIn("python code/tests/test_path_layout_boundaries.py -v", joined)
        self.assertIn("./odcr doctor", commands)
        self.assertFalse([command for command in commands if _is_real_stage_run(command)])

    def test_logging_scope_path_hints(self) -> None:
        self.assertEqual(
            suggest_scope_for_paths(["code/odcr_core/path_layout.py", "docs/notes.md"]),
            "logging",
        )
        self.assertEqual(
            suggest_scope_for_paths(["AI_analysis/03_evidence_ledgers/new_report.md"]),
            "logging",
        )
        self.assertIsNone(suggest_scope_for_paths(["code/executors/step5_engine.py"]))

    def test_stop_hook_audit_log_only_skips(self) -> None:
        module = _hook_module()
        inference = module.infer_scope_for_payload(
            {"touched_files": ["audit.log"]},
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: ["code/executors/step5_engine.py"],
        )
        self.assertEqual(inference.selected_scope, "skip")
        self.assertTrue(inference.skipped)
        self.assertEqual(inference.skip_reason, "audit_runtime_only")
        self.assertEqual(inference.effective_scope_files, ())
        summary = module._inference_summary(inference)
        self.assertFalse(summary["workspace_git_status_used_for_scope"])
        payload = module._runtime_payload(
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            hook_event_name="Stop",
            command=None,
            returncode=0,
            failure_stage=None,
            stdout_path=REPO_ROOT / "AI_analysis/01_raw_logs/codex_hooks/stdout.log",
            stderr_path=REPO_ROOT / "AI_analysis/01_raw_logs/codex_hooks/stderr.log",
            inference=inference,
            max_seconds=120,
            child_timeout_seconds=120,
            wrapper_timeout_seconds=180,
            started_at="2026-05-02T00:00:00+00:00",
            finished_at="2026-05-02T00:00:01+00:00",
        )
        self.assertIsNone(payload["post_edit_command"])

    def test_stop_hook_absolute_root_audit_log_only_skips(self) -> None:
        module = _hook_module()
        inference = module.infer_scope_for_payload(
            {"touched_files": [str(REPO_ROOT / "audit.log")]},
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: [],
        )
        self.assertEqual(inference.selected_scope, "skip")
        self.assertEqual(inference.skip_reason, "audit_runtime_only")
        self.assertEqual(inference.ignored_files, ("audit.log",))
        self.assertEqual(inference.effective_scope_files, ())

    def test_stop_hook_ai_analysis_runtime_only_skips(self) -> None:
        module = _hook_module()
        inference = module.infer_scope_for_payload(
            {"touched_files": ["AI_analysis/05_final_reports/x_report.md"]},
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: [],
        )
        self.assertEqual(inference.selected_scope, "skip")
        self.assertTrue(inference.skipped)
        self.assertEqual(inference.skip_reason, "audit_runtime_only")
        self.assertEqual(inference.effective_scope_files, ())

    def test_stop_hook_audit_log_and_ai_analysis_only_skips(self) -> None:
        module = _hook_module()
        inference = module.infer_scope_for_payload(
            {"touched_files": ["audit.log", "AI_analysis/05_final_reports/x_report.md"]},
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: ["code/executors/step5_engine.py"],
        )
        self.assertEqual(inference.selected_scope, "skip")
        self.assertEqual(inference.skip_reason, "audit_runtime_only")
        self.assertEqual(inference.effective_scope_files, ())
        payload = module._runtime_payload(
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            hook_event_name="Stop",
            command=None,
            returncode=0,
            failure_stage=None,
            stdout_path=REPO_ROOT / "AI_analysis/01_raw_logs/codex_hooks/stdout.log",
            stderr_path=REPO_ROOT / "AI_analysis/01_raw_logs/codex_hooks/stderr.log",
            inference=inference,
            max_seconds=120,
            child_timeout_seconds=120,
            wrapper_timeout_seconds=180,
            started_at="2026-05-02T00:00:00+00:00",
            finished_at="2026-05-02T00:00:01+00:00",
        )
        self.assertTrue(payload["skipped"])
        self.assertIsNone(payload["post_edit_command"])

    def test_stop_hook_audit_log_plus_code_uses_code_scope(self) -> None:
        module = _hook_module()
        inference = module.infer_scope_for_payload(
            {"touched_files": ["audit.log", "code/tools/odcr_post_edit_check.py"]},
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: [],
        )
        self.assertEqual(inference.selected_scope, "governance-fast")
        self.assertFalse(inference.skipped)
        self.assertEqual(inference.ignored_files, ("audit.log",))
        self.assertEqual(inference.effective_scope_files, ("code/tools/odcr_post_edit_check.py",))

    def test_stop_hook_transcript_docs_only_selects_governance_fast(self) -> None:
        module = _hook_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            _fake_transcript(transcript, ["docs/ODCR_EVOLUTION_PROTOCOL.md"])
            inference = module.infer_scope_for_payload(
                {"transcript_path": str(transcript)},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: ["code/executors/step5_engine.py"],
            )
        self.assertEqual(inference.selected_scope, "governance-fast")
        self.assertEqual(inference.inference_source, "transcript")
        self.assertEqual(inference.inference_reason, "transcript_session_touched_files")

    def test_stop_hook_transcript_config_resolver_selects_config(self) -> None:
        module = _hook_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            _fake_transcript(transcript, ["code/odcr_core/config_resolver.py"])
            inference = module.infer_scope_for_payload(
                {"transcript_path": str(transcript)},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: [],
            )
        self.assertEqual(inference.selected_scope, "config")
        self.assertEqual(inference.scope_candidates, ("config",))

    def test_stop_hook_transcript_preprocess_selects_preprocess(self) -> None:
        module = _hook_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            _fake_transcript(transcript, ["code/compute_embeddings.py"])
            inference = module.infer_scope_for_payload(
                {"transcript_path": str(transcript)},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: [],
            )
        self.assertEqual(inference.selected_scope, "preprocess")
        self.assertEqual(inference.scope_candidates, ("preprocess",))

    def test_stop_hook_transcript_step3_selects_step3(self) -> None:
        module = _hook_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            _fake_transcript(transcript, ["code/executors/step3_train_core.py"])
            inference = module.infer_scope_for_payload(
                {"transcript_path": str(transcript)},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: [],
            )
        self.assertEqual(inference.selected_scope, "step3")
        self.assertEqual(inference.scope_candidates, ("step3",))

    def test_stop_hook_transcript_step4_selects_step4(self) -> None:
        module = _hook_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            _fake_transcript(transcript, ["code/odcr_core/step4_training_export.py"])
            inference = module.infer_scope_for_payload(
                {"transcript_path": str(transcript)},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: [],
            )
        self.assertEqual(inference.selected_scope, "step4")
        self.assertEqual(inference.scope_candidates, ("step4",))

    def test_stop_hook_transcript_step5_selects_step5(self) -> None:
        module = _hook_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            _fake_transcript(transcript, ["code/odcr_core/step5_innovation.py"])
            inference = module.infer_scope_for_payload(
                {"transcript_path": str(transcript)},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: [],
            )
        self.assertEqual(inference.selected_scope, "step5")
        self.assertFalse(inference.multi_stage_detected)

    def test_stop_hook_cross_stage_contract_selects_all(self) -> None:
        module = _hook_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            _fake_transcript(transcript, ["code/data_contract.py"])
            inference = module.infer_scope_for_payload(
                {"transcript_path": str(transcript)},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: [],
            )
        self.assertEqual(inference.selected_scope, "all")
        self.assertEqual(inference.inference_reason, "cross_stage_session_touched_files")
        self.assertEqual(inference.scope_candidates, ("all",))

    def test_stop_hook_transcript_multi_stage_selects_all(self) -> None:
        module = _hook_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            _fake_transcript(
                transcript,
                [
                    "code/executors/step4_engine.py",
                    "code/executors/step5_engine.py",
                ],
            )
            inference = module.infer_scope_for_payload(
                {"transcript_path": str(transcript)},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: [],
            )
        self.assertEqual(inference.selected_scope, "all")
        self.assertEqual(inference.inference_reason, "multi_business_stage_session_touched_files")
        self.assertTrue(inference.multi_stage_detected)

    def test_stop_hook_transcript_parse_failed_skips(self) -> None:
        module = _hook_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "bad.jsonl"
            transcript.write_text("{bad-json", encoding="utf-8")
            inference = module.infer_scope_for_payload(
                {"transcript_path": str(transcript), "touched_files": ["code/executors/step5_engine.py"]},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: ["code/executors/step5_engine.py"],
            )
        self.assertEqual(inference.selected_scope, "skip")
        self.assertTrue(inference.skipped)
        self.assertEqual(inference.inference_source, "transcript")
        self.assertEqual(inference.inference_reason, "transcript_parse_failed")
        self.assertEqual(inference.skip_reason, "transcript_parse_failed")
        payload = module._runtime_payload(
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            hook_event_name="Stop",
            command=None,
            returncode=0,
            failure_stage=None,
            stdout_path=REPO_ROOT / "AI_analysis/01_raw_logs/codex_hooks/stdout.log",
            stderr_path=REPO_ROOT / "AI_analysis/01_raw_logs/codex_hooks/stderr.log",
            inference=inference,
            max_seconds=180,
        )
        self.assertTrue(payload["skipped"])
        self.assertIsNone(payload["post_edit_command"])
        self.assertFalse(payload["workspace_git_status_used_for_scope"])

    def test_stop_hook_transcript_no_touched_files_skips(self) -> None:
        module = _hook_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "empty.jsonl"
            transcript.write_text("", encoding="utf-8")
            inference = module.infer_scope_for_payload(
                {"transcript_path": str(transcript)},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: ["code/executors/step5_engine.py"],
            )
        self.assertEqual(inference.selected_scope, "skip")
        self.assertEqual(inference.skip_reason, "transcript_no_touched_files")

    def test_stop_hook_no_transcript_no_payload_skips(self) -> None:
        module = _hook_module()
        inference = module.infer_scope_for_payload(
            {},
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: [],
        )
        self.assertEqual(inference.selected_scope, "skip")
        self.assertEqual(inference.skip_reason, "no_session_touched_files")
        self.assertEqual(inference.inference_source, "none")

    def test_stop_hook_dirty_workspace_only_skips(self) -> None:
        module = _hook_module()
        many_files = [f"code/executors/step5_engine_{idx}.py" for idx in range(201)]
        inference = module.infer_scope_for_payload(
            {},
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: many_files,
        )
        self.assertEqual(inference.selected_scope, "skip")
        self.assertNotEqual(inference.selected_scope, "governance-fast")
        self.assertNotEqual(inference.selected_scope, "all")
        self.assertEqual(inference.skip_reason, "no_session_touched_files")
        self.assertTrue(inference.workspace_dirty_detected)
        self.assertEqual(inference.workspace_changed_files_count, 201)
        self.assertFalse(inference.workspace_git_status_used_for_scope)

    def test_stop_hook_unknown_code_file_skips(self) -> None:
        module = _hook_module()
        inference = module.infer_scope_for_payload(
            {"touched_files": ["code/new_unknown_helper.py"]},
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: ["code/executors/step5_engine.py"],
        )
        self.assertEqual(inference.selected_scope, "skip")
        self.assertEqual(inference.skip_reason, "unknown_session_touched_files")
        self.assertNotEqual(inference.selected_scope, "all")

    def test_stop_hook_diagnostics_are_bounded_samples(self) -> None:
        module = _hook_module()
        inference = module.ScopeInference(
            selected_scope="governance-fast",
            inference_source="payload",
            inference_reason="payload_session_touched_files",
            session_touched_files=tuple(f"docs/file_{idx}.md" for idx in range(370)),
            ignored_files=tuple(f"AI_analysis/file_{idx}.log" for idx in range(70)),
            effective_scope_files=tuple(f"docs/file_{idx}.md" for idx in range(300)),
            scope_candidates=("governance",),
            workspace_dirty_detected=True,
            workspace_changed_files_count=300,
        )
        summary = module._inference_summary(inference)
        self.assertEqual(summary["session_touched_files_count"], 370)
        self.assertEqual(summary["ignored_files_count"], 70)
        self.assertEqual(summary["effective_scope_files_count"], 300)
        self.assertEqual(len(summary["session_touched_files_sample"]), 50)
        self.assertEqual(len(summary["ignored_files_sample"]), 50)
        self.assertEqual(len(summary["effective_scope_files_sample"]), 50)
        self.assertNotIn("touched_files", summary)
        self.assertNotIn("raw_touched_files_count", summary)
        self.assertNotIn("effective_touched_files_count", summary)
        self.assertNotIn("git_changed_files_count", summary)
        self.assertNotIn("git_status_truncated", summary)
        self.assertNotIn("changed_files_total", summary)
        self.assertNotIn("changed_files_sample", summary)

    def test_stop_hook_explicit_scope_override_records_source(self) -> None:
        module = _hook_module()
        old_value = os.environ.get("ODCR_HOOK_SCOPE")
        os.environ["ODCR_HOOK_SCOPE"] = "all"
        try:
            inference = module.infer_scope_for_payload(
                {},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: [],
            )
        finally:
            if old_value is None:
                os.environ.pop("ODCR_HOOK_SCOPE", None)
            else:
                os.environ["ODCR_HOOK_SCOPE"] = old_value
        self.assertEqual(inference.selected_scope, "all")
        self.assertEqual(inference.inference_source, "explicit_override")
        self.assertEqual(inference.override_source, "env")
        automatic = module.apply_automatic_stop_scope_policy(inference)
        self.assertEqual(automatic.original_inferred_scope, "all")
        self.assertEqual(automatic.selected_scope, "governance-fast")
        self.assertTrue(automatic.manual_followup_required)
        self.assertIn("--scope all --max-seconds 900", automatic.manual_followup_command)

    def test_stop_hook_auto_all_scope_degrades_to_governance_fast(self) -> None:
        module = _hook_module()
        inferred = module.infer_scope_for_payload(
            {
                "touched_files": [
                    "code/executors/step4_engine.py",
                    "code/executors/step5_engine.py",
                ]
            },
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: [],
        )
        self.assertEqual(inferred.selected_scope, "all")
        automatic = module.apply_automatic_stop_scope_policy(inferred)
        self.assertEqual(automatic.original_inferred_scope, "all")
        self.assertEqual(automatic.selected_scope, "governance-fast")
        self.assertEqual(automatic.inference_reason, "auto_all_scope_degraded_to_governance_fast")
        self.assertTrue(automatic.manual_followup_required)
        command = module._build_post_edit_command(
            post_edit_path=REPO_ROOT / "code" / "tools" / "odcr_post_edit_check.py",
            scope=automatic.selected_scope,
            max_seconds=module._child_timeout_seconds(module._wrapper_timeout_seconds()),
            dry_run=False,
        )
        self.assertEqual(module._command_scope(command), "governance-fast")
        self.assertNotIn("all", command)

    def test_stop_hook_runtime_payload_schema_v22_matches_scope(self) -> None:
        module = _hook_module()
        inference = module.infer_scope_for_payload(
            {"touched_files": ["code/odcr_core/step5_innovation.py", "audit.log"]},
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: [],
        )
        command = module._build_post_edit_command(
            post_edit_path=REPO_ROOT / "code" / "tools" / "odcr_post_edit_check.py",
            scope=inference.selected_scope,
            max_seconds=120,
            dry_run=True,
        )
        payload = module._runtime_payload(
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            hook_event_name="Stop",
            command=command,
            returncode=0,
            failure_stage=None,
            stdout_path=REPO_ROOT / "AI_analysis/01_raw_logs/codex_hooks/stdout.log",
            stderr_path=REPO_ROOT / "AI_analysis/01_raw_logs/codex_hooks/stderr.log",
            inference=inference,
            max_seconds=120,
            child_timeout_seconds=120,
            wrapper_timeout_seconds=180,
            started_at="2026-05-02T00:00:00+00:00",
            post_edit_started_at="2026-05-02T00:00:00+00:00",
        )
        self.assertEqual(payload["schema_version"], "odcr_codex_hook_runtime/2.2")
        self.assertEqual(payload["ignored_files_count"], 1)
        self.assertEqual(payload["effective_scope_files_count"], 1)
        self.assertEqual(payload["child_timeout_seconds"], 120)
        self.assertEqual(payload["wrapper_timeout_seconds"], 180)
        self.assertLess(payload["child_timeout_seconds"], payload["wrapper_timeout_seconds"])
        self.assertEqual(payload["started_at"], "2026-05-02T00:00:00+00:00")
        self.assertEqual(payload["post_edit_started_at"], "2026-05-02T00:00:00+00:00")
        self.assertFalse(payload["workspace_git_status_used_for_scope"])
        self.assertLessEqual(len(payload["session_touched_files_sample"]), 50)
        self.assertLessEqual(len(payload["ignored_files_sample"]), 50)
        self.assertLessEqual(len(payload["effective_scope_files_sample"]), 50)
        for legacy in (
            "raw_touched_files_count",
            "effective_touched_files_count",
            "touched_files_sample",
            "git_changed_files_count",
            "git_status_truncated",
            "changed_files_total",
            "changed_files_sample",
            "changed_files_truncated",
            "workspace_changed_files_sample",
        ):
            self.assertNotIn(legacy, payload)
        skip_payload = module._runtime_payload(
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            hook_event_name="Stop",
            command=None,
            returncode=0,
            failure_stage=None,
            stdout_path=REPO_ROOT / "AI_analysis/01_raw_logs/codex_hooks/stdout.log",
            stderr_path=REPO_ROOT / "AI_analysis/01_raw_logs/codex_hooks/stderr.log",
            inference=module.infer_scope_for_payload(
                {},
                repo_root=REPO_ROOT,
                cwd=REPO_ROOT,
                workspace_changed_files_func=lambda _root: [],
            ),
            max_seconds=120,
            child_timeout_seconds=120,
            wrapper_timeout_seconds=180,
            started_at="2026-05-02T00:00:00+00:00",
            finished_at="2026-05-02T00:00:01+00:00",
        )
        self.assertEqual(skip_payload["selected_scope"], "skip")
        self.assertIsNone(skip_payload["post_edit_command"])
        self.assertNotIn("changed_files_total", payload)
        self.assertNotIn("changed_files_sample", payload)
        self.assertEqual(module._command_scope(payload["post_edit_command"]), payload["selected_scope"])

    def test_all_scope_can_still_be_explicitly_selected(self) -> None:
        self.assertIn("all", SCOPES)
        commands = _display_commands("all")
        self.assertIn("step5 resolver dry-run", "\n".join(commands))

    def test_stop_hook_runtime_write_after_inference_contains_command_before_child(self) -> None:
        module = _hook_module()
        inference = module.infer_scope_for_payload(
            {"touched_files": ["docs/ODCR_EVOLUTION_PROTOCOL.md"]},
            repo_root=REPO_ROOT,
            cwd=REPO_ROOT,
            workspace_changed_files_func=lambda _root: [],
        )
        command = module._build_post_edit_command(
            post_edit_path=REPO_ROOT / "code" / "tools" / "odcr_post_edit_check.py",
            scope=inference.selected_scope,
            max_seconds=120,
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime_path = Path(tmp) / "runtime.json"
            runtime_last_path = Path(tmp) / "runtime_last.json"
            old_runtime = os.environ.get("ODCR_HOOK_RUNTIME_PATH")
            old_runtime_last = os.environ.get("ODCR_HOOK_RUNTIME_LAST_PATH")
            os.environ["ODCR_HOOK_RUNTIME_PATH"] = str(runtime_path)
            os.environ["ODCR_HOOK_RUNTIME_LAST_PATH"] = str(runtime_last_path)
            try:
                written = module._write_runtime(
                    repo_root=REPO_ROOT,
                    stamp="unit",
                    cwd=REPO_ROOT,
                    hook_event_name="Stop",
                    command=command,
                    returncode=None,
                    failure_stage="post_edit_running",
                    stdout_path=Path(tmp) / "stdout.log",
                    stderr_path=Path(tmp) / "stderr.log",
                    inference=inference,
                    max_seconds=120,
                    child_timeout_seconds=120,
                    wrapper_timeout_seconds=180,
                    started_at="2026-05-02T00:00:00+00:00",
                    post_edit_started_at="2026-05-02T00:00:01+00:00",
                )
            finally:
                if old_runtime is None:
                    os.environ.pop("ODCR_HOOK_RUNTIME_PATH", None)
                else:
                    os.environ["ODCR_HOOK_RUNTIME_PATH"] = old_runtime
                if old_runtime_last is None:
                    os.environ.pop("ODCR_HOOK_RUNTIME_LAST_PATH", None)
                else:
                    os.environ["ODCR_HOOK_RUNTIME_LAST_PATH"] = old_runtime_last
            payload = json.loads(written.read_text(encoding="utf-8"))
            last_payload = json.loads(runtime_last_path.read_text(encoding="utf-8"))
        self.assertEqual(payload, last_payload)
        self.assertEqual(payload["selected_scope"], "governance-fast")
        self.assertEqual(payload["effective_scope_files_count"], 1)
        self.assertEqual(payload["failure_stage"], "post_edit_running")
        self.assertEqual(payload["post_edit_command"], command)
        self.assertFalse(payload["timed_out"])
        self.assertLess(payload["child_timeout_seconds"], payload["wrapper_timeout_seconds"])


if __name__ == "__main__":
    unittest.main()
