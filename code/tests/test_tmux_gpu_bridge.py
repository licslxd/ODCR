from __future__ import annotations

from pathlib import Path

from odcr_core.aux.runtime.command_registry import formal_training_command_reason
from odcr_core.aux.runtime.gpu_pane_handoff import STALE_HANDOFF_STOP_REASON
from odcr_core.aux.runtime import tmux_gpu_bridge as bridge_mod
from odcr_core.aux.runtime.pane_discovery import DiscoveryResult, PaneCandidate
from odcr_core.aux.runtime.tmux_gpu_bridge import BridgeError, BridgeOptions, TmuxGpuBridge, build_parser


def _gpu_pane(
    *,
    socket: str = "/tmp/odcr_gpu",
    target: str = "odcr:0.0",
    pane_id: str = "%0",
    pane_pid: int = 123,
) -> PaneCandidate:
    return PaneCandidate(
        socket=socket,
        session="odcr",
        target=target,
        window_index="0",
        window_name="srun",
        pane_index="0",
        pane_id=pane_id,
        pane_pid=pane_pid,
        pane_command="srun",
        cwd="/public/home/zhangliml/lc/ODCR/ODCR-main",
        active=True,
        dead=False,
        in_mode=False,
        cwd_match_repo=True,
        command_class="srun",
    )


def _gpu_discovery(pane: PaneCandidate) -> DiscoveryResult:
    return DiscoveryResult(
        candidates=(pane,),
        invalid=(),
        sockets_considered=(pane.socket,),
        sockets=({"socket": pane.socket, "exists": True},),
        sessions=({"socket": pane.socket, "session": pane.session},),
        windows=({"socket": pane.socket, "session": pane.session, "window_index": pane.window_index},),
        panes=(pane,),
    )


def _gpu_discovery_many(*panes: PaneCandidate) -> DiscoveryResult:
    return DiscoveryResult(
        candidates=tuple(panes),
        invalid=(),
        sockets_considered=tuple(pane.socket for pane in panes),
        sockets=tuple({"socket": pane.socket, "exists": True} for pane in panes),
        sessions=tuple({"socket": pane.socket, "session": pane.session} for pane in panes),
        windows=tuple({"socket": pane.socket, "session": pane.session, "window_index": pane.window_index} for pane in panes),
        panes=tuple(panes),
    )


class _NoopWriter:
    def runtime_diagnostic(self, *args, **kwargs) -> None:
        return None


def test_bridge_exec_parser_accepts_background_step3_rating_eval() -> None:
    args = build_parser().parse_args(
        [
            "bridge",
            "exec",
            "--background",
            "--stdout",
            "test/step3_rating_task2_eval_5seed_run2.driver.nohup.log",
            "--pid-file",
            "test/step3_rating_task2_eval_5seed_run2.driver.pid",
            "--",
            "./odcr",
            "step3-rating",
            "--task",
            "2",
            "--mode",
            "multi",
            "--run-id",
            "2",
        ]
    )
    assert args.runtime_command == "bridge"
    assert args.bridge_command == "exec"
    assert args.background is True
    assert args.exec_argv[-7:] == ["step3-rating", "--task", "2", "--mode", "multi", "--run-id", "2"]


def test_formal_train_string_guard_removed_and_allows_rating_eval() -> None:
    assert formal_training_command_reason(("./odcr", "step3", "--task", "2")) is None
    assert formal_training_command_reason(("./odcr", "step3", "--task", "2", "--mode", "full")) is None
    assert formal_training_command_reason(("./odcr", "step3", "--task", "2", "--mode", "train_only")) is None
    assert formal_training_command_reason(("./odcr", "step3", "--task", "2", "--mode", "eval_only")) is None
    assert formal_training_command_reason(("./odcr", "step3", "--task", "2", "--cache-check")) is None
    assert formal_training_command_reason(("./odcr", "step3-rating", "--task", "2", "--mode", "multi", "--run-id", "2")) is None


def test_formal_train_string_guard_allows_pipeline_and_direct_train_executors() -> None:
    assert formal_training_command_reason(("./odcr", "pipeline", "--task", "2", "--to", "eval")) is None
    assert formal_training_command_reason(("torchrun", "--standalone", "executors/step3_entry.py", "train")) is None
    assert formal_training_command_reason(("python", "code/executors/step5_entry.py", "train")) is None


def test_bridge_discover_multiple_candidates_is_success_not_blocker(monkeypatch) -> None:
    bridge = TmuxGpuBridge()
    panes = (
        _gpu_pane(socket="/tmp/odcr_gpu_a", target="odcr:0.0", pane_id="%0", pane_pid=123),
        _gpu_pane(socket="/tmp/odcr_gpu_b", target="odcr:0.1", pane_id="%1", pane_pid=124),
    )
    monkeypatch.setattr(bridge_mod, "discover_panes", lambda **_kwargs: _gpu_discovery_many(*panes))
    monkeypatch.setattr(bridge_mod, "get_writer", lambda _repo_root: _NoopWriter())

    result = bridge.discover(BridgeOptions(mode="discover"))

    assert result["success"] is True
    assert result["runnable_candidate_count"] == 2
    assert result["selection_required"] is True
    assert result["non_unique_discovery_is_not_blocker"] is True


def test_compute_app_guard_is_audit_only_not_cuda_blocker() -> None:
    bridge = TmuxGpuBridge()
    pane = _gpu_pane()
    status = {
        "success": True,
        "hostname": "gpu-node-1",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "torch.cuda.is_available": True,
        "torch.cuda.device_count": 2,
        "nvidia-smi": {"returncode": 0},
        "nvidia-smi-compute-apps": {"returncode": 0, "stdout": "123, python, 1024", "stderr": ""},
    }

    guard = bridge._compute_app_guard_from_status(status=status, candidate=pane, target_source="live_discovery_cuda_probe")
    ok, reasons = bridge._live_cuda_status_ok(status, guard)

    assert guard["pass"] is True
    assert guard["blocked"] is False
    assert guard["audit_only"] is True
    assert guard["status"] == "audit_only_active_compute_apps"
    assert ok is True
    assert "compute_app_guard_blocked" not in reasons


def test_ambiguous_live_cuda_candidates_choose_deterministic_candidate() -> None:
    bridge = TmuxGpuBridge()
    pane_a = _gpu_pane(socket="/tmp/odcr_gpu_a", target="odcr:0.0", pane_id="%0", pane_pid=123)
    pane_b = _gpu_pane(socket="/tmp/odcr_gpu_b", target="odcr:0.1", pane_id="%1", pane_pid=124)
    records = [
        {"pane": pane_b.to_dict(), "live_cuda_eligible": True, "child_status": {"SLURM_JOB_ID": "88"}},
        {"pane": pane_a.to_dict(), "live_cuda_eligible": True, "child_status": {"SLURM_JOB_ID": "88"}},
    ]

    selected = bridge._choose_live_candidate(records=records, handoff=None)

    assert selected["pane"]["socket"] == "/tmp/odcr_gpu_a"
    assert selected["live_cuda_ambiguity_resolved"] is True
    assert selected["ambiguity_policy"] == "deterministic_first_live_cuda_candidate"


def test_bridge_exec_stale_handoff_uses_live_cuda_discovery(monkeypatch, tmp_path: Path) -> None:
    bridge = TmuxGpuBridge()
    pane = _gpu_pane()
    discovery = _gpu_discovery(pane)
    sent: list[str] = []

    def stale_target(_options: BridgeOptions):
        raise BridgeError(
            "stale handoff",
            stop_reason=STALE_HANDOFF_STOP_REASON,
            details={"handoff": {"exists": True, "valid": False}},
        )

    def live_target(_options: BridgeOptions, *, stale_handoff, timeout_s: int):
        assert stale_handoff == {"exists": True, "valid": False}
        assert timeout_s == 77
        return pane, discovery, "live_discovery_cuda_probe", {
            "live_discovery_cuda_probe_success": True,
            "stale_handoff_detected": True,
        }

    monkeypatch.setattr(bridge, "resolve_target_with_recovery", stale_target)
    monkeypatch.setattr(bridge, "_resolve_live_discovery_cuda_target", live_target)
    monkeypatch.setattr(bridge, "_run_handshake_on_target", lambda *args, **kwargs: {"success": True})
    monkeypatch.setattr(bridge, "_write_exec_script", lambda **kwargs: tmp_path / "driver.sh")
    monkeypatch.setattr(bridge, "_send_bridge_line", lambda _target, line: sent.append(line))
    monkeypatch.setattr(bridge, "_wait_optional_pid_file", lambda _pid_path: "456")

    result = bridge.run_exec_mode(
        BridgeOptions(
            mode="exec",
            exec_argv=("./odcr", "step5", "--task", "2", "--run-id", "1_3"),
            background=True,
            require_cuda=True,
            timeout=77,
        )
    )

    assert result["success"] is True
    assert result["target_source"] == "live_discovery_cuda_probe"
    assert result["stale_handoff_blocked_execution"] is False
    assert result["live_discovery_cuda_probe_success"] is True
    assert result["cuda_preflight"]["success"] is True
    assert sent and "driver.sh" in sent[0]
