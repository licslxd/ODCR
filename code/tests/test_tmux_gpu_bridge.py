#!/usr/bin/env python3
"""Mocked tests for the runtime-first tmux GPU bridge.

this test proves: bridge command generation supports repo-local runtime execution and stays namespace-safe.
this test does not prove: CUDA runtime speed, H2D overlap, or formal Step3 quality.
whether formal hot path is covered: no, only bridge routing and safety are covered.
whether runtime evidence is required: yes before any performance optimization can claim Level 3/4.
regression bug it prevents: arbitrary send-keys, formal namespace writes, or whitelist-only GPU gates reappearing.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))

import odcr_tmux_gpu_bridge as bridge  # noqa: E402


class FakeRunner:
    def __init__(self, socket_rows: dict[str, str], *, ps: str, step_by_job: dict[str, str], job_by_id: dict[str, str]) -> None:
        self.socket_rows = socket_rows
        self.ps = ps
        self.step_by_job = step_by_job
        self.job_by_id = job_by_id
        self.calls: list[tuple[str, ...]] = []

    def run(self, args: list[str] | tuple[str, ...], *, timeout: float | None = None) -> bridge.CommandResult:
        del timeout
        key = tuple(str(part) for part in args)
        self.calls.append(key)
        if key[:4] == ("tmux", "-S", key[2], "list-panes"):
            return bridge.CommandResult(key, 0, self.socket_rows.get(key[2], ""), "")
        if key == ("ps", "-e", "-o", "pid=", "-o", "ppid=", "-o", "args="):
            return bridge.CommandResult(key, 0, self.ps, "")
        if len(key) == 4 and key[:3] == ("scontrol", "show", "step"):
            job_id = key[3].split(".", 1)[0]
            text = self.step_by_job.get(job_id)
            return bridge.CommandResult(key, 0 if text else 1, text or "", "" if text else "missing")
        if len(key) == 4 and key[:3] == ("scontrol", "show", "job"):
            text = self.job_by_id.get(key[3])
            return bridge.CommandResult(key, 0 if text else 1, text or "", "" if text else "missing")
        if len(key) >= 6 and key[0] == "tmux" and key[3] == "send-keys":
            return bridge.CommandResult(key, 0, "", "")
        if len(key) >= 4 and key[0] == "tmux" and key[3] in {"set-buffer", "paste-buffer"}:
            return bridge.CommandResult(key, 0, "", "")
        if len(key) >= 4 and key[0] == "tmux" and key[3] == "capture-pane":
            return bridge.CommandResult(key, 0, "bash-4.2$ ", "")
        return bridge.CommandResult(key, 1, "", "unexpected command")


def pane_row(
    *,
    session: str = "odcr",
    window: str = "0",
    pane: str = "0",
    pane_id: str = "%0",
    pid: int = 100,
    command: str = "srun",
    cwd: str | None = None,
    active: str = "1",
    dead: str = "0",
    in_mode: str = "0",
) -> str:
    return "\t".join(
        (
            session,
            window,
            pane,
            pane_id,
            str(pid),
            command,
            cwd or str(bridge.REPO_ROOT),
            active,
            dead,
            in_mode,
        )
    )


def step_text(job_id: str, *, node: str = "gpu01", state: str = "RUNNING", gpu: str = "gpu:A100:1") -> str:
    return (
        f"StepId={job_id}.0 State={state} Partition=batch NodeList={node} "
        f"TRES=cpu=1,gres/gpu=1,mem=0,node=1 SrunHost:Pid=admin:200 "
        f"TresPerNode={gpu}"
    )


def job_text(job_id: str, *, node: str = "gpu01", state: str = "RUNNING", gpu: str = "gpu:A100:1") -> str:
    return (
        f"JobId={job_id} JobState={state} NodeList={node} "
        f"TRES=cpu=1,mem=1G,node=1,gres/gpu=1 TresPerNode={gpu}"
    )


@contextmanager
def patched_bridge_paths() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        old_values: dict[str, Any] = {
            "RAW_LOG_DIR": bridge.RAW_LOG_DIR,
            "SUMMARY_DIR": bridge.SUMMARY_DIR,
            "REPORT_DIR": bridge.REPORT_DIR,
            "RUNTIME_DIR": bridge.RUNTIME_DIR,
            "AI_ANALYSIS": bridge.AI_ANALYSIS,
        }
        bridge.AI_ANALYSIS = root / "AI_analysis"
        bridge.RAW_LOG_DIR = root / "AI_analysis" / "01_raw_logs"
        bridge.SUMMARY_DIR = root / "AI_analysis" / "04_phase_summaries"
        bridge.REPORT_DIR = root / "AI_analysis" / "05_final_reports"
        bridge.RUNTIME_DIR = root / "AI_analysis" / "runtime"
        try:
            yield root
        finally:
            for name, value in old_values.items():
                setattr(bridge, name, value)


@contextmanager
def patched_candidate_sockets(paths: list[Path]) -> Iterator[None]:
    old = bridge.candidate_socket_paths
    bridge.candidate_socket_paths = lambda uid=None: paths
    try:
        yield
    finally:
        bridge.candidate_socket_paths = old


def valid_runner(*, sockets: dict[str, str] | None = None, ps: str | None = None) -> FakeRunner:
    sockets = sockets or {"/sock/gpu": pane_row(pid=100)}
    ps = ps or "100 1 -bash\n200 100 srun --jobid=10 --pty bash\n"
    return FakeRunner(sockets, ps=ps, step_by_job={"10": step_text("10")}, job_by_id={"10": job_text("10")})


class UnlockPaneRunner(FakeRunner):
    def __init__(self, *, cancel_clears: bool = True, escape_clears: bool = True) -> None:
        super().__init__(
            {},
            ps="100 1 -bash\n200 100 srun --jobid=10 --pty bash\n",
            step_by_job={"10": step_text("10")},
            job_by_id={"10": job_text("10")},
        )
        self.in_mode = True
        self.cancel_clears = cancel_clears
        self.escape_clears = escape_clears

    def run(self, args: list[str] | tuple[str, ...], *, timeout: float | None = None) -> bridge.CommandResult:
        key = tuple(str(part) for part in args)
        if key[:4] == ("tmux", "-S", key[2], "list-panes"):
            self.calls.append(key)
            row = pane_row(pid=100, in_mode="1" if self.in_mode else "0")
            return bridge.CommandResult(key, 0, row + "\n", "")
        if len(key) >= 8 and key[:4] == ("tmux", "-S", "/sock/gpu", "send-keys") and key[-2:] == ("-X", "cancel"):
            self.calls.append(key)
            if self.cancel_clears:
                self.in_mode = False
            return bridge.CommandResult(key, 0, "", "")
        if len(key) >= 7 and key[:4] == ("tmux", "-S", "/sock/gpu", "send-keys") and key[-1] == "Escape":
            self.calls.append(key)
            if self.escape_clears:
                self.in_mode = False
            return bridge.CommandResult(key, 0, "", "")
        return super().run(args, timeout=timeout)


class TmuxGpuBridgeTests(unittest.TestCase):
    def test_multi_socket_selects_only_srun_gpu_candidate(self) -> None:
        sockets = {
            "/sock/admin": pane_row(pid=300, command="-bash"),
            "/sock/gpu": pane_row(pid=100),
        }
        runner = valid_runner(sockets=sockets)
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) in sockets)
        with patched_bridge_paths(), patched_candidate_sockets([Path("/sock/admin"), Path("/sock/gpu")]):
            target, discovery, source = tool.resolve_target(bridge.BridgeOptions(mode="discover"))
        self.assertEqual(source, "discovery")
        self.assertEqual(target.socket, "/sock/gpu")
        self.assertEqual(target.job_id, "10")
        self.assertEqual(len(discovery.candidates), 1)

    def test_zero_candidates_fail_fast(self) -> None:
        sockets = {"/sock/admin": pane_row(pid=300, command="-bash")}
        runner = valid_runner(sockets=sockets, ps="300 1 -bash\n")
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) in sockets)
        with patched_bridge_paths(), patched_candidate_sockets([Path("/sock/admin")]):
            with self.assertRaises(bridge.BridgeError) as ctx:
                tool.resolve_target(bridge.BridgeOptions(mode="discover"))
        self.assertEqual(ctx.exception.stop_reason, "target_not_unique")

    def test_two_candidates_fail_fast(self) -> None:
        sockets = {
            "/sock/gpu-a": pane_row(pid=100, pane_id="%0"),
            "/sock/gpu-b": pane_row(pid=101, pane_id="%1"),
        }
        ps = "100 1 -bash\n200 100 srun --jobid=10 --pty bash\n101 1 -bash\n201 101 srun --jobid=11 --pty bash\n"
        runner = FakeRunner(
            sockets,
            ps=ps,
            step_by_job={"10": step_text("10"), "11": step_text("11", node="gpu02")},
            job_by_id={"10": job_text("10"), "11": job_text("11", node="gpu02")},
        )
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) in sockets)
        with patched_bridge_paths(), patched_candidate_sockets([Path("/sock/gpu-a"), Path("/sock/gpu-b")]):
            with self.assertRaises(bridge.BridgeError) as ctx:
                tool.resolve_target(bridge.BridgeOptions(mode="discover"))
        self.assertEqual(ctx.exception.stop_reason, "target_not_unique")

    def test_explicit_env_target_mismatch_fail_fast(self) -> None:
        runner = valid_runner()
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
        with patched_bridge_paths():
            with self.assertRaises(bridge.BridgeError) as ctx:
                tool.resolve_target(bridge.BridgeOptions(mode="discover", socket="/sock/gpu", target="odcr:9.9"))
        self.assertEqual(ctx.exception.stop_reason, "target_not_unique")

    def test_stale_state_file_is_never_trusted(self) -> None:
        runner = valid_runner()
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
        with patched_bridge_paths():
            stale = bridge.RUNTIME_DIR / "gpu_pane.json"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text(json.dumps({"socket": "/sock/gpu", "target": "odcr:0.0", "job_id": "999"}), encoding="utf-8")
            target, _discovery, source = tool.resolve_target(bridge.BridgeOptions(mode="discover", socket="/sock/gpu"))
        self.assertEqual(source, "explicit")
        self.assertEqual(target.job_id, "10")

    def test_discovery_output_contains_target_fields(self) -> None:
        runner = valid_runner()
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
        result = tool.discover_socket(Path("/sock/gpu"))
        payload = result.candidates[0].to_dict()
        for key in ("socket", "session", "pane_id", "job_id", "step_id", "node", "gpu"):
            self.assertIn(key, payload)

    def test_no_arbitrary_shell_string_parser_surface(self) -> None:
        parser = bridge.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["cuda-probe", "echo arbitrary"])

    def test_denylist_blocks_dangerous_tokens(self) -> None:
        for text in (
            "nohup true",
            "echo hi &",
            "disown",
            "sbatch job.sh",
            "srun --pty bash",
            "scancel 1",
            "odcr-enter-gpu 1",
            "rm -rf data",
            "kill 1",
            "pkill python",
        ):
            with self.subTest(text=text):
                with self.assertRaises(bridge.BridgeError):
                    bridge.validate_script_safety(text)

    def test_preprocess_formal_runs_blocked_but_dryrun_allowed(self) -> None:
        for text in ("./odcr preprocess a --dry-run", "./odcr preprocess b", "./odcr preprocess c"):
            with self.subTest(text=text):
                with self.assertRaises(bridge.BridgeError):
                    bridge.validate_script_safety(text)
        bridge.validate_script_safety("./odcr preprocess b --dry-run")
        bridge.validate_script_safety("./odcr preprocess c --dry-run")

    def test_formal_step4_step5_eval_rerank_modes_blocked(self) -> None:
        for text in (
            "./odcr step3 --task 4",
            "./odcr step4 --task 4",
            "./odcr step5 --task 4",
            "./odcr eval",
            "./odcr rerank",
            "python code/odcr.py eval",
            "python code/odcr.py rerank",
        ):
            with self.subTest(text=text):
                with self.assertRaises(bridge.BridgeError):
                    bridge.validate_script_safety(text)
        bridge.validate_script_safety(
            "./odcr step4 --task 2 --preflight --max-samples 128 --validation-namespace step4_preflight_smoke"
        )

    def test_timeout_auto_for_each_mode(self) -> None:
        expected = {
            "discover": 20,
            "validate-only": 60,
            "unlock-pane": 20,
            "marker-probe": 20,
            "cuda-probe": 45,
            "preprocess-dryrun": 90,
            "bge-smoke": 240,
            "micro-benchmark": 300,
            "real-data-probe": 180,
            "step3-startup-validation": 180,
            "long-run": None,
            "collect": 20,
        }
        for mode, hard in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(bridge.resolve_timeouts(mode).hard_timeout_s, hard)
        self.assertTrue(bridge.resolve_timeouts("long-run").detached)

    def test_timeout_above_mode_limit_fails(self) -> None:
        with self.assertRaises(bridge.BridgeError):
            bridge.resolve_timeouts("cuda-probe", "121")
        with self.assertRaises(bridge.BridgeError):
            bridge.resolve_timeouts("micro-benchmark", "421")

    def test_status_schema_has_required_fields(self) -> None:
        tool = bridge.TmuxGpuBridge(runner=valid_runner(), socket_exists=lambda path: True)
        status = tool._status(
            run_id="bridge_test_cuda_probe",
            kind="cuda-probe",
            success=True,
            exit_code=0,
            elapsed_s=1.0,
            timeouts=bridge.resolve_timeouts("cuda-probe"),
            first_result_seen=True,
            stop_reason="first_cuda_probe_completed",
            metrics={"device_count": 1},
        )
        for key in (
            "run_id",
            "kind",
            "success",
            "exit_code",
            "elapsed_s",
            "startup_timeout_s",
            "first_result_timeout_s",
            "hard_timeout_s",
            "first_result_seen",
            "success_condition",
            "stop_reason",
            "metrics",
        ):
            self.assertIn(key, status)

    def test_collect_missing_end_marker_incomplete(self) -> None:
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_collect_missing")
            paths.log.parent.mkdir(parents=True, exist_ok=True)
            paths.log.write_text("ODCR_BRIDGE_BEGIN_bridge_collect_missing\n", encoding="utf-8")
            paths.status.write_text(json.dumps({"success": True}), encoding="utf-8")
            result = bridge.collect_run("bridge_collect_missing")
        self.assertFalse(result["success"])
        self.assertEqual(result["stop_reason"], "TIMEOUT_INCOMPLETE")

    def test_collect_reads_log_status_summary_report(self) -> None:
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_collect_ok")
            for path in (paths.log, paths.status, paths.summary, paths.report):
                path.parent.mkdir(parents=True, exist_ok=True)
            paths.log.write_text("ODCR_BRIDGE_BEGIN_bridge_collect_ok\nODCR_BRIDGE_END_bridge_collect_ok\n", encoding="utf-8")
            paths.status.write_text(json.dumps({"success": True}), encoding="utf-8")
            paths.summary.write_text("summary", encoding="utf-8")
            paths.report.write_text("report", encoding="utf-8")
            result = bridge.collect_run("bridge_collect_ok")
        self.assertTrue(result["success"])
        self.assertTrue(result["metrics"]["log_exists"])
        self.assertTrue(result["metrics"]["status_exists"])
        self.assertTrue(result["metrics"]["summary_exists"])
        self.assertTrue(result["metrics"]["report_exists"])

    def test_long_run_send_line_has_no_foreground_timeout(self) -> None:
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_long_run_send")
            command = bridge.send_line(paths, bridge.resolve_timeouts("long-run"))
        self.assertTrue(command.startswith("bash "))
        self.assertNotIn("timeout 900s", command)
        self.assertNotIn("timeout ", command)

    def test_long_run_script_writes_managed_launcher_artifacts(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_long_run_script")
            script = bridge.build_long_run_managed_launcher_script(
                "bridge_long_run_script",
                paths,
                bridge.resolve_timeouts("long-run"),
                target,
                bridge.BridgeOptions(mode="long-run", command_argv=("python", "-c", "print('ok')")),
            )
        bridge.validate_script_safety(script)
        self.assertIn("command.sh", script)
        self.assertIn("managed_launcher.py", script)
        self.assertIn("stdout.log", script)
        self.assertIn("stderr.log", script)
        self.assertIn("heartbeat.json", script)
        self.assertIn("pid", script)
        self.assertIn("subprocess.Popen", script)
        self.assertNotIn("timeout 900s", script)

    def test_collect_reads_completed_long_run_managed_status(self) -> None:
        with patched_bridge_paths() as root:
            paths = bridge.make_generated_paths("bridge_long_run_collect")
            managed = root / "managed" / "status.json"
            managed.parent.mkdir(parents=True, exist_ok=True)
            managed.write_text(json.dumps({"state": "completed", "returncode": 0}), encoding="utf-8")
            for path in (paths.log, paths.status, paths.summary, paths.report):
                path.parent.mkdir(parents=True, exist_ok=True)
            paths.log.write_text(
                "ODCR_BRIDGE_BEGIN_bridge_long_run_collect\nODCR_BRIDGE_END_bridge_long_run_collect\n",
                encoding="utf-8",
            )
            paths.status.write_text(
                json.dumps(
                    {
                        "kind": "long-run",
                        "success": True,
                        "metrics": {"managed_status_path": str(managed)},
                    }
                ),
                encoding="utf-8",
            )
            paths.summary.write_text("summary", encoding="utf-8")
            paths.report.write_text("report", encoding="utf-8")
            result = bridge.collect_run("bridge_long_run_collect")
        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"]["managed_status"]["state"], "completed")

    def test_dry_run_does_not_send_keys(self) -> None:
        runner = valid_runner()
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
        with patched_bridge_paths():
            result = tool.run_mode(
                bridge.BridgeOptions(mode="cuda-probe", socket="/sock/gpu", target="odcr:0.0", dry_run=True)
            )
        self.assertFalse(result["sent"])
        self.assertFalse(any("send-keys" in call for call in runner.calls))

    def test_validate_only_no_send_does_not_send_keys(self) -> None:
        runner = valid_runner()
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
        with patched_bridge_paths():
            result = tool.run_mode(
                bridge.BridgeOptions(mode="validate-only", socket="/sock/gpu", target="odcr:0.0", no_send=True)
            )
        self.assertFalse(result["sent"])
        self.assertFalse(any("send-keys" in call for call in runner.calls))

    def test_validate_only_rejects_pane_mode(self) -> None:
        sockets = {"/sock/gpu": pane_row(pid=100, in_mode="1")}
        runner = valid_runner(sockets=sockets)
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
        with patched_bridge_paths():
            result = tool.run_mode(bridge.BridgeOptions(mode="validate-only", socket="/sock/gpu", target="odcr:0.0"))
        self.assertFalse(result["success"])
        self.assertEqual(result["stop_reason"], "target_not_unique")
        details = result["metrics"]["details"]
        self.assertEqual(details["invalid_candidates"][0]["reason"], "pane_in_mode")

    def test_unlock_pane_uses_fixed_cancel_then_revalidates(self) -> None:
        runner = UnlockPaneRunner(cancel_clears=True)
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
        with patched_bridge_paths():
            result = tool.run_mode(bridge.BridgeOptions(mode="unlock-pane", socket="/sock/gpu", target="odcr:0.0"))
        self.assertTrue(result["success"])
        self.assertEqual(result["stop_reason"], "pane_unlocked")
        self.assertEqual(result["unlock_operations"], ["copy-mode-cancel"])
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "%0", "-X", "cancel"), runner.calls)
        self.assertNotIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "%0", "Escape"), runner.calls)
        self.assertFalse(result["target"]["in_mode"])

    def test_unlock_pane_retries_escape_once(self) -> None:
        runner = UnlockPaneRunner(cancel_clears=False, escape_clears=True)
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
        with patched_bridge_paths():
            result = tool.run_mode(bridge.BridgeOptions(mode="unlock-pane", socket="/sock/gpu", target="odcr:0.0"))
        self.assertTrue(result["success"])
        self.assertEqual(result["unlock_operations"], ["copy-mode-cancel", "escape"])
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "%0", "-X", "cancel"), runner.calls)
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "%0", "Escape"), runner.calls)

    def test_unlock_pane_stops_after_fixed_attempts_with_q_fallback(self) -> None:
        runner = UnlockPaneRunner(cancel_clears=False, escape_clears=False)
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
        with patched_bridge_paths():
            result = tool.run_mode(bridge.BridgeOptions(mode="unlock-pane", socket="/sock/gpu", target="odcr:0.0"))
        self.assertFalse(result["success"])
        self.assertEqual(result["stop_reason"], "pane_in_mode_unlock_failed")
        self.assertEqual(result["metrics"]["attempt_count"], 3)
        self.assertEqual(result["unlock_operations"], ["copy-mode-cancel", "escape", "q"])
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "%0", "q"), runner.calls)

    def test_send_keys_defaults_to_pane_id_literal_enter(self) -> None:
        runner = valid_runner()
        tool = bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: True)
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        command = "timeout 45s bash AI_analysis/01_raw_logs/tmux_bridge_bridge_test_cuda-probe.sh"
        tool._send_keys(target, command)
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "%0", "-l", command), runner.calls)
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "%0", "Enter"), runner.calls)

    def test_send_keys_supports_bounded_transport_methods(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        command = "timeout 20s bash AI_analysis/01_raw_logs/tmux_bridge_bridge_test_marker-probe.sh"

        runner = valid_runner()
        bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: True)._send_keys(
            target,
            command,
            "target-name-literal-enter",
        )
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "odcr:0.0", "-l", command), runner.calls)
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "odcr:0.0", "Enter"), runner.calls)

        runner = valid_runner()
        bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: True)._send_keys(target, command, "pane-id-literal-cm")
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "%0", "-l", command), runner.calls)
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "%0", "C-m"), runner.calls)

        runner = valid_runner()
        bridge.TmuxGpuBridge(runner=runner, socket_exists=lambda path: True)._send_keys(target, command, "buffer-paste-enter")
        self.assertIn(("tmux", "-S", "/sock/gpu", "set-buffer", command), runner.calls)
        self.assertIn(("tmux", "-S", "/sock/gpu", "paste-buffer", "-t", "%0"), runner.calls)
        self.assertIn(("tmux", "-S", "/sock/gpu", "send-keys", "-t", "%0", "Enter"), runner.calls)

    def test_marker_probe_script_is_short_marker_only(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_script_marker")
            script = bridge.build_marker_probe_script("bridge_script_marker", paths, bridge.resolve_timeouts("marker-probe"), target)
        bridge.validate_script_safety(script)
        self.assertIn("ODCR_BRIDGE_SEND_OK_bridge_script_marker", script)
        self.assertIn("marker_probe_completed", script)
        self.assertNotIn("nvidia-smi", script)
        self.assertNotIn("import torch", script)
        self.assertNotIn("./odcr", script)

    def test_cuda_probe_script_contains_runtime_cuda_probe_behavior(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_script_cuda")
            script = bridge.build_cuda_probe_script("bridge_script_cuda", paths, bridge.resolve_timeouts("cuda-probe"), target)
        bridge.validate_script_safety(script)
        self.assertIn("hostname", script)
        self.assertIn("pwd", script)
        self.assertIn("nvidia-smi", script)
        self.assertIn("torch_cuda_available", script)
        self.assertNotIn("sbatch", script)
        self.assertNotIn("scancel", script)
        self.assertNotIn("&", script)

    def test_bge_smoke_script_uses_transformers_local_cuda_loader(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_script_bge")
            script = bridge.build_bge_smoke_script("bridge_script_bge", paths, bridge.resolve_timeouts("bge-smoke"), target)
        bridge.validate_script_safety(script)
        self.assertNotIn("sentence_transformers", script)
        self.assertNotIn("SentenceTransformer", script)
        self.assertIn("from transformers import AutoModel, AutoTokenizer", script)
        self.assertIn("AutoTokenizer.from_pretrained(model_path, local_files_only=True)", script)
        self.assertIn("AutoModel.from_pretrained(model_path, local_files_only=True)", script)
        self.assertIn('model.to("cuda").eval()', script)
        self.assertIn("torch.no_grad()", script)
        self.assertIn("torch.cuda.amp.autocast", script)
        self.assertIn("BGE load_seconds", script)
        self.assertIn("BGE encode_seconds", script)
        self.assertIn("BGE shape", script)
        self.assertIn("BGE dtype", script)
        self.assertIn("GPU memory", script)
        self.assertIn('"load_seconds": load_seconds', script)
        self.assertIn('"encode_seconds": encode_seconds', script)
        self.assertIn('"shape": shape', script)
        self.assertIn('"dtype": dtype', script)
        self.assertIn('"norm0": norm', script)
        self.assertIn('"gpu_max_memory_allocated": memory', script)
        self.assertNotIn("./odcr preprocess", script)
        self.assertNotIn("data/", script)
        self.assertNotIn("merged/", script)

    def test_micro_benchmark_defaults_to_one_measured_batch(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_script_micro")
            script = bridge.build_micro_benchmark_script(
                "bridge_script_micro",
                paths,
                bridge.resolve_timeouts("micro-benchmark"),
                target,
                "bge-single-batch",
            )
        bridge.validate_script_safety(script)
        self.assertNotIn("sentence_transformers", script)
        self.assertNotIn("SentenceTransformer", script)
        self.assertIn("from transformers import AutoModel, AutoTokenizer", script)
        self.assertIn("AutoTokenizer.from_pretrained(model_path, local_files_only=True)", script)
        self.assertIn("AutoModel.from_pretrained(model_path, local_files_only=True)", script)
        self.assertIn("model.to(device).eval()", script)
        self.assertIn("micro_benchmark_warmup_batches\", 1", script)
        self.assertIn("micro_benchmark_measured_batches\", 1", script)
        self.assertIn("'embed_batch_size': 512", script)
        self.assertIn('"batch_size": int(benchmark_config["embed_batch_size"])', script)
        self.assertIn('"workers_used": workers_used', script)
        self.assertIn('"device_names": device_names', script)
        self.assertIn('"tokenize_s": tokenize_s', script)
        self.assertIn('"forward_s": forward_s', script)
        self.assertIn('"warmup_batches": 1', script)
        self.assertIn('"measured_batches": 1', script)

    def test_micro_benchmark_cli_accepts_parameter_overrides(self) -> None:
        parser = bridge.build_parser()
        args = parser.parse_args(
            [
                "micro-benchmark",
                "--kind",
                "bge-single-batch",
                "--embed-batch-size",
                "1536",
                "--read-chunk-rows",
                "200000",
                "--group-shard-size",
                "8192",
                "--workers",
                "2",
                "--no-bf16",
                "--tf32",
                "--grouped-text-cache",
            ]
        )
        options = bridge.options_from_args(args)
        self.assertEqual(options.embed_batch_size, 1536)
        self.assertEqual(options.read_chunk_rows, 200000)
        self.assertEqual(options.group_shard_size, 8192)
        self.assertEqual(options.workers, 2)
        self.assertFalse(options.bf16_enabled)
        self.assertTrue(options.tf32_enabled)
        self.assertTrue(options.grouped_text_cache_enabled)

    def test_real_data_probe_script_uses_fixed_ai_analysis_probe(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_script_real_data")
            script = bridge.build_real_data_probe_script(
                "bridge_script_real_data",
                paths,
                bridge.resolve_timeouts("real-data-probe"),
                target,
            )
        bridge.validate_script_safety(script)
        self.assertIn("preprocess_b_real_probe/preprocess_b_real_data_probe.py", script)
        self.assertIn('"gpu-probe"', script)
        self.assertIn("real_data_probe_completed", script)
        self.assertNotIn("bge-single-batch", script)
        self.assertNotIn("./odcr preprocess", script)
        self.assertNotIn("data/", script)
        self.assertNotIn("merged/", script)

    def test_real_data_probe_cli_accepts_preprocess_c_probe_stage(self) -> None:
        parser = bridge.build_parser()
        args = parser.parse_args(["real-data-probe", "--probe-stage", "c"])
        options = bridge.options_from_args(args)
        self.assertEqual(options.probe_stage, "c")

    def test_real_data_probe_script_can_target_preprocess_c_probe(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_script_real_data_c")
            script = bridge.build_real_data_probe_script(
                "bridge_script_real_data_c",
                paths,
                bridge.resolve_timeouts("real-data-probe"),
                target,
                "c",
            )
        bridge.validate_script_safety(script)
        self.assertIn("preprocess_c_real_probe/preprocess_c_real_data_probe.py", script)
        self.assertIn('"gpu-probe"', script)
        self.assertIn('"probe_stage": \'c\'', script)
        self.assertNotIn("preprocess_b_real_probe/preprocess_b_real_data_probe.py", script)
        self.assertNotIn("./odcr preprocess", script)
        self.assertNotIn("data/", script)
        self.assertNotIn("merged/", script)

    def test_step3_startup_validation_cli_defaults_to_task2_validation_namespace(self) -> None:
        parser = bridge.build_parser()
        args = parser.parse_args(["step3-startup-validation"])
        options = bridge.options_from_args(args)
        self.assertEqual(options.mode, "step3-startup-validation")
        self.assertEqual(options.task_id, 2)
        self.assertEqual(options.validation_slug, "step3_tmux_gpu_bridge_startup_validation_closeout")

    def test_retired_step3_bridge_modes_are_not_parser_surface(self) -> None:
        parser = bridge.build_parser()
        for mode in ("step3-ddp-smoke", "step3-short-pilot"):
            with self.subTest(mode=mode):
                with self.assertRaises(SystemExit):
                    parser.parse_args([mode])
        args = parser.parse_args(["step3-performance-probe", "--probe-type", "timing-profile-window"])
        self.assertEqual(args.mode, "step3-performance-probe")

    def test_step3_startup_validation_script_uses_fixed_internal_tool(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_step3_startup")
            script = bridge.build_step3_startup_validation_script(
                "bridge_step3_startup",
                paths,
                bridge.resolve_timeouts("step3-startup-validation"),
                target,
                task_id=2,
            )
        bridge.validate_script_safety(script)
        self.assertIn("code/tools/odcr_step3_startup_validation.py", script)
        self.assertIn("--mode startup-only", script)
        self.assertIn("--namespace validation", script)
        self.assertIn("--task 2", script)
        self.assertIn("--bridge-status-path", script)
        self.assertIn(".status.json.child.json", script)
        self.assertIn("--bridge-log-path", script)
        self.assertIn("--target-socket /sock/gpu", script)
        self.assertIn("--target-pane %0", script)
        self.assertIn("--target-job-id 10", script)
        self.assertNotIn("./odcr step3", script)
        self.assertNotIn("python code/odcr.py step3", script)
        self.assertNotIn("torchrun", script)
        self.assertNotIn("best.pth", script)
        self.assertNotIn("odcr_step3_real_data_probe.py", script)

    def test_step3_startup_validation_rejects_wrong_task_and_unsafe_slug(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_step3_limit")
            with self.assertRaises(bridge.BridgeError):
                bridge.build_step3_startup_validation_script(
                    "bridge_step3_limit",
                    paths,
                    bridge.resolve_timeouts("step3-startup-validation"),
                    target,
                    task_id=3,
                )
            with self.assertRaises(bridge.BridgeError):
                bridge.build_step3_startup_validation_script(
                    "bridge_step3_limit",
                    paths,
                    bridge.resolve_timeouts("step3-startup-validation"),
                    target,
                    validation_slug="../bad",
                )
            with self.assertRaises(bridge.BridgeError):
                bridge.make_generated_paths("../bad")

    def test_step3_performance_probe_script_is_closed_choice_validation_namespace(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_step3_perf")
            script = bridge.build_step3_performance_probe_script(
                "bridge_step3_perf",
                paths,
                bridge.resolve_timeouts("step3-performance-probe"),
                target,
                task_id=2,
                probe_type="prefetch-ab",
            )
        bridge.validate_script_safety(script)
        self.assertIn("code/tools/odcr_step3_performance_probe.py", script)
        self.assertIn("--probe-type prefetch-ab", script)
        self.assertIn("--namespace validation", script)
        self.assertIn(".status.json.child.json", script)
        self.assertNotIn("./odcr step3", script)
        self.assertNotIn("python code/odcr.py step3", script)
        self.assertNotIn("torchrun", script)
        self.assertNotIn("best.pth", script)
        self.assertNotIn("latest.json", script)

    def test_step3_performance_probe_rejects_arbitrary_probe_type(self) -> None:
        target = bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(bridge.REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        with patched_bridge_paths():
            with self.assertRaises(bridge.BridgeError):
                bridge.build_step3_performance_probe_script(
                    "bridge_step3_perf_bad",
                    bridge.make_generated_paths("bridge_step3_perf_bad"),
                    bridge.resolve_timeouts("step3-performance-probe"),
                    target,
                    task_id=2,
                    probe_type="arbitrary-shell",
                )

    def test_generated_guard_rejects_formal_step3_and_retired_probe_modes(self) -> None:
        bridge.validate_script_safety(
            "python code/tools/odcr_step3_startup_validation.py --task 2 --mode startup-only --namespace validation"
        )
        bridge.validate_script_safety(
            "python code/tools/odcr_step3_performance_probe.py --probe-type timing-profile-window --namespace validation"
        )
        for text in (
            "./odcr step3 --task 4",
            "python code/odcr.py step3 --task 4",
            "python code/executors/step3_entry.py train",
            "torchrun code/executors/step3_entry.py train",
            "best.pth",
            "checkpoint_lineage.json",
            "python code/tools/odcr_step3_real_data_probe.py --mode step3-ddp-smoke --task-id 2",
            "python code/tools/odcr_step3_real_data_probe.py --mode step3-short-pilot --task-id 2",
        ):
            with self.subTest(text=text):
                with self.assertRaises(bridge.BridgeError):
                    bridge.validate_script_safety(text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
