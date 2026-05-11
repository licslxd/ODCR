from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))

import odcr_tmux_gpu_bridge as bridge  # noqa: E402


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


def short_timeouts() -> bridge.ResolvedTimeouts:
    return bridge.ResolvedTimeouts(
        startup_timeout_s=1,
        first_result_timeout_s=2,
        hard_timeout_s=5,
    )


def final_status(*, formal_pollution: bool = False, evidence_complete: bool = True) -> dict[str, Any]:
    return {
        "schema_version": bridge.BRIDGE_STATUS_SCHEMA,
        "run_id": "bridge_watchdog",
        "kind": "repo-command",
        "success": True,
        "exit_code": 0,
        "elapsed_s": 3.0,
        "startup_timeout_s": 1,
        "first_result_timeout_s": 2,
        "hard_timeout_s": 5,
        "first_result_seen": True,
        "success_condition": bridge.OPERATION_SPECS["repo-command"].success_condition,
        "stop_reason": "repo_command_completed",
        "metrics": {
            "command_allowed_by_policy": True,
            "child_process_started": True,
            "child_process_still_running": False,
            "child_process_ok": True,
            "child_exit_code": 0,
            "child_returncode": 0,
            "runtime_evidence_ok": bool(evidence_complete),
            "evidence_complete": bool(evidence_complete),
            "final_artifact_completed": bool(evidence_complete),
            "formal_pollution": bool(formal_pollution),
            "formal_namespace_polluted": bool(formal_pollution),
        },
    }


class DelayedStatusClock:
    def __init__(self, paths: bridge.GeneratedPaths, payload: dict[str, Any] | None) -> None:
        self.now = 0.0
        self.paths = paths
        self.payload = payload

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.payload is not None and self.now >= 3.0 and not self.paths.status.exists():
            self.paths.status.parent.mkdir(parents=True, exist_ok=True)
            self.paths.status.write_text(json.dumps(self.payload), encoding="utf-8")


class GpuBridgeLongBoundedWatchdogTest(unittest.TestCase):
    def _poll(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        with patched_bridge_paths():
            paths = bridge.make_generated_paths("bridge_watchdog")
            paths.log.parent.mkdir(parents=True, exist_ok=True)
            paths.log.write_text("ODCR_BRIDGE_BEGIN_bridge_watchdog\n", encoding="utf-8")
            clock = DelayedStatusClock(paths, payload)
            tool = bridge.TmuxGpuBridge(clock=clock.clock, sleep=clock.sleep)
            return tool._poll_for_result(paths, "repo-command", short_timeouts())

    def test_first_result_timeout_then_final_success_is_recovered(self) -> None:
        status = self._poll(final_status())
        normalized = bridge.normalize_repo_runtime_success(status)
        self.assertTrue(normalized["success"], normalized)
        self.assertTrue(normalized["first_result_timeout"])
        self.assertTrue(normalized["first_result_timeout_recovered"])
        self.assertEqual(normalized["child_returncode"], 0)
        self.assertTrue(normalized["final_artifact_completed"])
        self.assertTrue(normalized["evidence_complete"])
        self.assertFalse(normalized["formal_pollution"])

    def test_first_result_timeout_without_final_artifact_fails_at_hard_timeout(self) -> None:
        status = self._poll(None)
        normalized = bridge.normalize_repo_runtime_success(status)
        self.assertFalse(normalized["success"], normalized)
        self.assertEqual(normalized["stop_reason"], "hard_timeout")
        self.assertTrue(normalized["first_result_timeout"])
        self.assertFalse(normalized["first_result_timeout_recovered"])
        self.assertFalse(normalized["final_artifact_completed"])

    def test_first_result_timeout_with_formal_pollution_fails(self) -> None:
        status = self._poll(final_status(formal_pollution=True))
        normalized = bridge.normalize_repo_runtime_success(status)
        self.assertFalse(normalized["success"], normalized)
        self.assertTrue(normalized["first_result_timeout"])
        self.assertFalse(normalized["first_result_timeout_recovered"])
        self.assertEqual(normalized["stop_reason"], "formal_namespace_polluted")

    def test_formal_step4_still_denied(self) -> None:
        classification = bridge.BridgeCommandPolicy.classify_repo_command(("./odcr", "step4", "--task", "2"))
        self.assertFalse(classification.allowed)

    def test_cuda_probe_only_is_not_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "report.json").write_text(json.dumps({"cuda_available": True}), encoding="utf-8")
            evidence = bridge.parse_step4_preflight_evidence(output_dir=root)
        self.assertTrue(evidence["gpu_transport_ok"])
        self.assertFalse(evidence["runtime_evidence_ok"])
        self.assertFalse(evidence["evidence_complete"])


if __name__ == "__main__":
    unittest.main()
