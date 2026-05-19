from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from odcr_core.aux.runtime.pane_discovery import CommandResult, SubprocessRunner, candidate_socket_paths, discover_panes
from odcr_core.aux.runtime.gpu_handshake import write_handshake
from odcr_core.aux.runtime.gpu_pane_handoff import STALE_HANDOFF_STOP_REASON, current_handoff_path
from odcr_core.aux.runtime.tmux_gpu_bridge import BridgeError, BridgeOptions, TmuxGpuBridge


class FakeRunner(SubprocessRunner):
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run(self, args, *, timeout=None):  # type: ignore[override]
        key = tuple(str(part) for part in args)
        self.calls.append(key)
        if key[:4] == ("tmux", "-S", "/sock/gpu", "list-panes"):
            row = _pane_row(pane_id="%0", pid="100", command="srun")
            return CommandResult(key, 0, row + "\n", "")
        if key[:4] == ("tmux", "-S", "/sock/gpu", "send-keys"):
            return CommandResult(key, 0, "", "")
        return CommandResult(key, 1, "", "unexpected")


def _pane_row(
    *,
    session: str = "odcr",
    window: str = "0",
    pane: str = "0",
    pane_id: str = "%0",
    pid: str = "100",
    command: str = "bash",
    cwd: str | None = None,
    active: str = "1",
    dead: str = "0",
    in_mode: str = "0",
) -> str:
    return "\t".join(
        (
            session,
            window,
            "main",
            pane,
            pane_id,
            pid,
            command,
            cwd or str(Path.cwd()),
            active,
            dead,
            in_mode,
            "title",
            "/dev/pts/1",
            command,
        )
    )


def _handoff_payload(socket: str = "/sock/gpu", pane: str = "%1", *, hostname: str = "gpu03") -> dict:
    return {
        "schema_version": "odcr_current_gpu_pane_handoff/2",
        "source": "odcr-enter-gpu",
        "generated_at_utc": "2026-05-18T00:00:00Z",
        "repo_root": str(Path.cwd()),
        "admin_tmux": {
            "raw_TMUX": f"{socket},54534,0",
            "socket": socket,
            "pane": pane,
            "server_pid_diagnostic_only": "54534",
            "session_id_diagnostic_only": "0",
            "selection_uses_pid": False,
            "session": "odcr",
            "target": "odcr:0.0",
            "pane_id": pane,
            "pane_current_path": str(Path.cwd()),
            "pane_current_command": "bash",
            "pane_in_mode": False,
            "captured_on_host": "admin",
        },
        "gpu_runtime": {
            "hostname": hostname,
            "slurm_job_id": "205571",
            "cuda_visible_devices": "0,1",
            "cwd": str(Path.cwd()),
            "nvidia_smi_L_returncode": 0,
            "nvidia_smi_L_stdout": "GPU 0\nGPU 1",
            "compute_apps_returncode": 0,
            "compute_apps_stdout": "",
            "torch_cuda_available": True,
            "torch_cuda_device_count": 2,
            "torch_cuda_device_names": ["NVIDIA A100-PCIE-40GB", "NVIDIA A100-PCIE-40GB"],
        },
        "valid_for_bridge_selection": True,
        "selection_key": {
            "socket": socket,
            "pane": pane,
            "hostname": hostname,
            "slurm_job_id": "205571",
            "cuda_visible_devices": "0,1",
        },
    }


def _patch_handoff_root(monkeypatch, tmp_path: Path):
    import odcr_core.aux.runtime.tmux_gpu_bridge as bridge_module

    monkeypatch.setattr(bridge_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bridge_module, "CURRENT_HANDOFF_PATH", current_handoff_path(tmp_path))
    monkeypatch.setattr(bridge_module, "STATE_HINT_PATH", tmp_path / "AI_analysis" / "runtime" / "gpu_pane.json")
    monkeypatch.setattr(bridge_module, "REPORT_DIR", tmp_path / "AI_analysis" / "05_final_reports")
    monkeypatch.setattr(bridge_module, "RAW_LOG_DIR", tmp_path / "AI_analysis" / "01_raw_logs")
    return bridge_module


@pytest.fixture(autouse=True)
def _isolate_handoff_state(monkeypatch, tmp_path: Path):
    _patch_handoff_root(monkeypatch, tmp_path)


class MultiSocketInventoryRunner(SubprocessRunner):
    def __init__(self, *, cwd: str | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.cwd = cwd

    def run(self, args, *, timeout=None):  # type: ignore[override]
        key = tuple(str(part) for part in args)
        self.calls.append(key)
        socket = key[2] if len(key) > 2 else ""
        command = key[3] if len(key) > 3 else ""
        if command == "list-sessions":
            return CommandResult(key, 0, "odcr\t$0\t1\t1\n", "")
        if command == "list-windows":
            return CommandResult(key, 0, "odcr\t0\tmain\t1\t1\n", "")
        if command == "list-panes":
            if socket == "/sock/admin":
                return CommandResult(key, 0, _pane_row(pane_id="%0", command="bash", cwd=self.cwd) + "\n", "")
            if socket == "/sock/gpu":
                return CommandResult(key, 0, _pane_row(pane_id="%1", pid="200", command="bash", cwd=self.cwd) + "\n", "")
        if command == "capture-pane":
            return CommandResult(key, 0, "safe last line\n", "")
        return CommandResult(key, 1, "", "unexpected")


class EmptyRunner(SubprocessRunner):
    def run(self, args, *, timeout=None):  # type: ignore[override]
        key = tuple(str(part) for part in args)
        if key[:4] == ("tmux", "-S", "/sock/gpu", "list-panes"):
            return CommandResult(key, 0, "", "")
        return CommandResult(key, 1, "", "unexpected")


class InModeThenRecoveredRunner(SubprocessRunner):
    def __init__(self, *, recover_after: int = 1, compute_apps: str = "") -> None:
        self.calls: list[tuple[str, ...]] = []
        self.mode_exit_keys: list[str] = []
        self.recover_after = recover_after
        self.compute_apps = compute_apps

    def run(self, args, *, timeout=None):  # type: ignore[override]
        key = tuple(str(part) for part in args)
        self.calls.append(key)
        if key[:4] == ("tmux", "-S", "/sock/gpu", "list-panes"):
            in_mode = "0" if len(self.mode_exit_keys) >= self.recover_after else "1"
            row = "\t".join(("odcr", "0", "0", "%0", "100", "srun", str(Path.cwd()), "1", "0", in_mode))
            return CommandResult(key, 0, row + "\n", "")
        if key and key[0] == "nvidia-smi":
            return CommandResult(key, 0, self.compute_apps, "")
        if key[:4] == ("tmux", "-S", "/sock/gpu", "send-keys"):
            if key[-1] in {"Escape", "q"}:
                self.mode_exit_keys.append(key[-1])
            return CommandResult(key, 0, "", "")
        return CommandResult(key, 1, "", "unexpected")


class ActivePaneCommandRunner(InModeThenRecoveredRunner):
    def run(self, args, *, timeout=None):  # type: ignore[override]
        key = tuple(str(part) for part in args)
        self.calls.append(key)
        if key[:4] == ("tmux", "-S", "/sock/gpu", "list-panes"):
            row = "\t".join(("odcr", "0", "0", "%0", "100", "torchrun", str(Path.cwd()), "1", "0", "1"))
            return CommandResult(key, 0, row + "\n", "")
        if key and key[0] == "nvidia-smi":
            return CommandResult(key, 0, "", "")
        if key[:4] == ("tmux", "-S", "/sock/gpu", "send-keys"):
            self.mode_exit_keys.append(key[-1])
            return CommandResult(key, 0, "", "")
        return CommandResult(key, 1, "", "unexpected")


class FastClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(float(seconds), 1.0)


class TimeoutBridge(TmuxGpuBridge):
    pass


class SuccessBridge(TmuxGpuBridge):
    def _wait_status(self, paths, timeout_s):  # type: ignore[override]
        return {
            "schema_version": "odcr_runtime_gpu_handshake/1",
            "success": True,
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "SLURM_JOB_ID": "205900",
            "nvidia-smi": {"returncode": 0},
            "nvidia-smi-compute-apps": {"returncode": 0, "stdout": ""},
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        }


def test_discover_selects_unique_current_pane() -> None:
    runner = FakeRunner()
    tool = TmuxGpuBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
    target, discovery, source = tool.resolve_target(BridgeOptions(mode="discover", socket="/sock/gpu"))
    assert source == "cli_explicit"
    assert target.pane_id == "%0"
    assert len(discovery.candidates) == 1


def test_candidate_socket_paths_prioritize_env_and_current_tmux_before_default(monkeypatch) -> None:
    monkeypatch.setenv("ODCR_GPU_TMUX_SOCKET", "/sock/env_gpu")
    monkeypatch.setenv("TMUX", "/sock/current_gpu,123,0")
    paths = [str(path) for path in candidate_socket_paths(uid=2080)]
    assert paths.index("/sock/env_gpu") < paths.index("/sock/current_gpu")
    assert paths.index("/sock/current_gpu") < paths.index("/tmp/tmux-2080/odcr_gpu")


def test_cli_explicit_socket_beats_default_tmp_socket() -> None:
    runner = MultiSocketInventoryRunner()
    tool = TmuxGpuBridge(
        runner=runner,
        socket_exists=lambda path: str(path) in {"/sock/gpu", "/tmp/tmux-2080/odcr_gpu"},
    )
    target, _discovery, source = tool.resolve_target(BridgeOptions(mode="validate-only", socket="/sock/gpu", target="%1"))
    assert source == "cli_explicit"
    assert target.socket == "/sock/gpu"
    assert target.pane_id == "%1"
    assert not any(call[:4] == ("tmux", "-S", "/tmp/tmux-2080/odcr_gpu", "list-panes") for call in runner.calls)


def test_state_file_and_current_tmux_do_not_select_target(monkeypatch, tmp_path) -> None:
    state = tmp_path / "gpu_pane.json"
    state.write_text(
        '{"socket": "/sock/admin", "pane_id": "%0", "target": "odcr:0.0", "node": "gpu_old"}\n',
        encoding="utf-8",
    )
    import odcr_core.aux.runtime.tmux_gpu_bridge as bridge_module

    monkeypatch.setattr(bridge_module, "STATE_HINT_PATH", state)
    monkeypatch.setenv("TMUX", "/sock/gpu,1,0")
    monkeypatch.setenv("TMUX_PANE", "%1")
    tool = TmuxGpuBridge(
        runner=MultiSocketInventoryRunner(),
        socket_exists=lambda path: str(path) in {"/sock/admin", "/sock/gpu"},
    )
    with pytest.raises(BridgeError) as exc:
        tool.resolve_target(BridgeOptions(mode="validate-only"))
    assert exc.value.stop_reason == "missing_current_gpu_pane_handoff"


def test_current_gpu_pane_handoff_beats_old_state_and_default_tmp(monkeypatch, tmp_path) -> None:
    _patch_handoff_root(monkeypatch, tmp_path)
    handoff = current_handoff_path(tmp_path)
    handoff.parent.mkdir(parents=True)
    json_dumps = json.dumps(_handoff_payload()) + "\n"
    handoff.write_text(json_dumps, encoding="utf-8")
    assert "54534" in json_dumps
    old_state = tmp_path / "AI_analysis" / "runtime" / "gpu_pane.json"
    old_state.write_text('{"socket": "/sock/admin", "pane_id": "%0"}\n', encoding="utf-8")
    runner = MultiSocketInventoryRunner(cwd=str(tmp_path))
    tool = TmuxGpuBridge(
        runner=runner,
        socket_exists=lambda path: str(path) in {"/sock/admin", "/sock/gpu", f"/tmp/tmux-{os.getuid()}/odcr_gpu"},
    )
    target, _discovery, source = tool.resolve_target(BridgeOptions(mode="cuda-probe"))
    assert source == "current_gpu_pane_handoff"
    assert target.socket == "/sock/gpu"
    assert target.pane_id == "%1"
    assert not any(call[:4] == ("tmux", "-S", f"/tmp/tmux-{os.getuid()}/odcr_gpu", "list-panes") for call in runner.calls)


def test_stale_current_gpu_pane_handoff_no_longer_short_circuits_discovery(monkeypatch, tmp_path) -> None:
    _patch_handoff_root(monkeypatch, tmp_path)
    handoff = current_handoff_path(tmp_path)
    handoff.parent.mkdir(parents=True)
    payload = _handoff_payload(hostname="admin")
    payload["generated_at_utc"] = "2026-05-16T00:00:00Z"
    handoff.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    runner = MultiSocketInventoryRunner(cwd=str(tmp_path))

    class StaleFallbackBridge(SuccessBridge):
        def _send_registered_command(self, target, command):  # type: ignore[override]
            self.last_target = target

        def _wait_status(self, paths, timeout_s):  # type: ignore[override]
            target = getattr(self, "last_target", None)
            if target is not None and target.socket == "/sock/gpu":
                return {
                    "schema_version": "odcr_runtime_gpu_handshake/1",
                    "success": True,
                    "hostname": "gpu03",
                    "SLURM_JOB_ID": "205900",
                    "CUDA_VISIBLE_DEVICES": "0,1",
                    "nvidia-smi": {"returncode": 0},
                    "nvidia-smi-compute-apps": {"returncode": 0, "stdout": ""},
                    "torch.cuda.is_available": True,
                    "torch.cuda.device_count": 2,
                }
            return {
                "schema_version": "odcr_runtime_gpu_handshake/1",
                "success": False,
                "hostname": "admin",
                "CUDA_VISIBLE_DEVICES": "",
                "nvidia-smi": {"returncode": 1},
                "nvidia-smi-compute-apps": {"returncode": 1, "stdout": ""},
                "torch.cuda.is_available": False,
                "torch.cuda.device_count": 0,
            }

    import odcr_core.aux.runtime.pane_discovery as pane_discovery

    monkeypatch.setattr(pane_discovery, "candidate_socket_paths", lambda **_kwargs: (Path("/sock/admin"), Path("/sock/gpu")))
    tool = StaleFallbackBridge(runner=runner, socket_exists=lambda path: str(path) in {"/sock/admin", "/sock/gpu"})
    result = tool.run_bridge_mode(BridgeOptions(mode="cuda-probe"))
    assert result["success"] is True
    assert result["target_source"] == "live_discovery_cuda_probe"
    assert result["stale_handoff_detected"] is True
    assert result["stale_handoff_blocked_execution"] is False
    assert result["stale_state_used"] is False
    assert result["selected_cuda_socket"] == "/sock/gpu"
    assert result["selected_cuda_hostname"] == "gpu03"


def test_current_handoff_has_priority_over_cli_explicit_and_records_conflict(monkeypatch, tmp_path) -> None:
    _patch_handoff_root(monkeypatch, tmp_path)
    handoff = current_handoff_path(tmp_path)
    handoff.parent.mkdir(parents=True)
    handoff.write_text(json.dumps(_handoff_payload(socket="/sock/gpu", pane="%1")) + "\n", encoding="utf-8")
    tool = TmuxGpuBridge(
        runner=MultiSocketInventoryRunner(),
        socket_exists=lambda path: str(path) in {"/sock/admin", "/sock/gpu", "/sock/other"},
    )
    result = tool.run_bridge_mode(BridgeOptions(mode="validate-only", socket="/sock/other", target="%9", dry_run=True))
    assert result["success"] is True
    assert result["target_source"] == "current_gpu_pane_handoff"
    assert result["handoff_cli_conflict"]["cli_overrides_handoff"] is True
    assert result["handoff_cli_conflict"]["conflicts"]["socket"]["cli"] == "/sock/other"
    assert result["target"]["socket"] == "/sock/gpu"


def test_old_gpu_pane_json_is_historical_hint_only(monkeypatch, tmp_path) -> None:
    bridge_module = _patch_handoff_root(monkeypatch, tmp_path)
    old_state = tmp_path / "AI_analysis" / "runtime" / "gpu_pane.json"
    old_state.parent.mkdir(parents=True)
    old_state.write_text('{"socket": "/sock/admin", "pane_id": "%0"}\n', encoding="utf-8")
    tool = TmuxGpuBridge(
        runner=MultiSocketInventoryRunner(),
        socket_exists=lambda path: str(path) in {"/sock/admin", "/sock/gpu"},
    )
    with pytest.raises(BridgeError) as exc:
        tool.resolve_target(BridgeOptions(mode="validate-only"))
    assert exc.value.stop_reason == "missing_current_gpu_pane_handoff"
    assert bridge_module.OLD_GPU_PANE_STATE_ROLE == "historical_hint_only"


def test_step5_runtime_probe_refuses_missing_current_handoff(monkeypatch, tmp_path) -> None:
    bridge_module = _patch_handoff_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        bridge_module,
        "write_probe_report",
        lambda stage, task, *, handshake, probe_result, repo_root, target_source=None, stale_state_used=False, handoff_admission=None: {
            "schema_version": "odcr_runtime_bounded_probe/1",
            "success": False,
            "stage": stage,
            "task": task,
            "target_source": target_source,
            "stale_state_used": stale_state_used,
            "handoff_admission": handoff_admission,
        },
    )
    tool = TmuxGpuBridge(runner=MultiSocketInventoryRunner(), socket_exists=lambda path: True)
    result = tool.run_probe("step5A", 2, timeout=1, from_step4="1")
    assert result["success"] is False
    assert result["bridge"]["stop_reason"] == "missing_current_gpu_pane_handoff"
    assert result["bridge"]["formal_train_command_emitted"] is False
    assert result["bridge"]["synthetic_batch_used_for_formal_gate"] is False


def test_admin_pane_does_not_override_explicit_gpu_pane(monkeypatch) -> None:
    monkeypatch.setenv("ODCR_GPU_TMUX_SOCKET", "/sock/admin")
    monkeypatch.setenv("ODCR_GPU_TMUX_TARGET", "%0")
    tool = TmuxGpuBridge(
        runner=MultiSocketInventoryRunner(),
        socket_exists=lambda path: str(path) in {"/sock/admin", "/sock/gpu"},
    )
    target, _discovery, source = tool.resolve_target(BridgeOptions(mode="cuda-probe", socket="/sock/gpu", target="%1"))
    assert source == "cli_explicit"
    assert target.socket == "/sock/gpu"
    assert target.pane_id == "%1"


def test_global_discovery_inventory_enumerates_multiple_sockets_sessions_windows_panes() -> None:
    runner = MultiSocketInventoryRunner()
    discovery = discover_panes(
        runner=runner,
        socket_paths=(Path("/sock/admin"), Path("/sock/gpu")),
        socket_exists=lambda path: str(path) in {"/sock/admin", "/sock/gpu"},
        all_sockets=True,
        include_filtered=True,
        capture_hash=True,
        repo_root=Path.cwd(),
    )
    assert len(discovery.sockets_considered) == 2
    assert len(discovery.sessions) == 2
    assert len(discovery.windows) == 2
    assert len(discovery.panes) == 2
    assert all(item.last_visible_line_hash for item in discovery.panes)


def test_global_cuda_probe_execution_is_retired_fail_fast() -> None:
    tool = TmuxGpuBridge(
        runner=MultiSocketInventoryRunner(),
        socket_exists=lambda path: str(path) in {"/sock/admin", "/sock/gpu"},
    )
    with pytest.raises(BridgeError) as exc:
        tool.run_bridge_mode(BridgeOptions(mode="cuda-probe", global_discovery=True, all_sockets=True, all_panes=True))
    assert exc.value.stop_reason == "global_target_selection_retired"


def test_probe_global_scan_is_retired_fail_fast(monkeypatch, tmp_path) -> None:
    bridge_module = _patch_handoff_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        bridge_module,
        "write_probe_report",
        lambda stage, task, *, handshake, probe_result, repo_root, target_source=None, stale_state_used=False, handoff_admission=None: {
            "schema_version": "odcr_runtime_bounded_probe/1",
            "success": False,
            "stage": stage,
            "task": task,
            "target_source": target_source,
            "stale_state_used": stale_state_used,
            "handoff_admission": handoff_admission,
        },
    )
    handoff = current_handoff_path(tmp_path)
    handoff.parent.mkdir(parents=True)
    handoff.write_text(json.dumps(_handoff_payload()) + "\n", encoding="utf-8")
    tool = TmuxGpuBridge(runner=MultiSocketInventoryRunner(), socket_exists=lambda path: str(path) == "/sock/gpu")
    result = tool.run_probe("step5A", 2, scan=True)
    assert result["bridge"]["stop_reason"] == "global_target_selection_retired"


def test_current_shell_cuda_false_does_not_block_bridge_dispatch() -> None:
    tool = SuccessBridge(runner=FakeRunner(), socket_exists=lambda path: str(path) == "/sock/gpu")
    result = tool.run_bridge_mode(BridgeOptions(mode="cuda-probe", socket="/sock/gpu"))
    assert result["success"] is True
    assert result["fresh_discover"] is True
    assert result["stale_state_used"] is False


def test_runtime_probe_propagates_explicit_socket_and_target(monkeypatch) -> None:
    import odcr_core.aux.runtime.tmux_gpu_bridge as bridge_module

    selected_pane = {
        "socket": "/sock/gpu",
        "session": "odcr",
        "target": "odcr:0.0",
        "window_index": "0",
        "window_name": "main",
        "pane_index": "0",
        "pane_id": "%1",
        "pane_pid": 200,
        "pane_command": "bash",
        "cwd": str(Path.cwd()),
        "active": True,
        "dead": False,
        "in_mode": False,
        "cwd_match_repo": True,
        "command_class": "shell",
    }

    class ProbePropagationBridge(TmuxGpuBridge):
        def __init__(self) -> None:
            super().__init__()
            self.bridge_options: BridgeOptions | None = None
            self.sent_target = None
            self.sent_command = None

        def run_bridge_mode(self, options):  # type: ignore[override]
            self.bridge_options = options
            return {
                "success": True,
                "target_source": "cli_explicit",
                "stale_state_used": False,
                "selected_cuda_pane": "/sock/gpu|odcr:0.0|%1",
                "selected_cuda_socket": "/sock/gpu",
                "selected_cuda_pane_id": "%1",
                "selected_cuda_candidate": {"pane": selected_pane},
                "child_status": {
                    "success": True,
                    "hostname": "gpu03",
                    "CUDA_VISIBLE_DEVICES": "0,1",
                    "torch.cuda.is_available": True,
                    "torch.cuda.device_count": 2,
                },
            }

        def _send_registered_command(self, target, command):  # type: ignore[override]
            self.sent_target = target
            self.sent_command = command

        def _wait_status(self, paths, timeout_s):  # type: ignore[override]
            return {
                "schema_version": "odcr_step5_e4_bounded_probe/1",
                "success": True,
                "evidence_level": "E4_gpu_shard_forward_bounded",
                "forward_executed": True,
                "loss_backward_executed": True,
                "optimizer_step_executed": True,
                "real_forward_backward_executed": True,
                "real_data_batch_used": True,
                "real_ccv_packet_used": True,
                "synthetic_batch_used_for_formal_gate": False,
                "memory_truth": {"reserved_is_diagnostic_only": True},
                "candidate_decision": {"reserved_memory_used_for_rejection": False},
            }

    monkeypatch.setattr(
        bridge_module,
        "write_probe_report",
        lambda stage, task, *, handshake, probe_result, repo_root, target_source=None, stale_state_used=False, handoff_admission=None: {
            "success": bool(probe_result and probe_result.get("success")),
            "stage": stage,
            "task": task,
            "target_source": target_source,
            "stale_state_used": stale_state_used,
        },
    )
    tool = ProbePropagationBridge()
    result = tool.run_probe(
        "step5A",
        2,
        socket="/sock/gpu",
        target="%1",
        candidate_id="B224_C2_R0_real_batch_confirmed_pane",
        timeout=900,
        from_step4="1",
    )
    assert tool.bridge_options is not None
    assert tool.bridge_options.socket == "/sock/gpu"
    assert tool.bridge_options.target == "%1"
    assert tool.sent_target.socket == "/sock/gpu"
    assert tool.sent_target.pane_id == "%1"
    assert "--candidate-id" in tool.sent_command
    assert result["target_source"] == "cli_explicit"
    assert result["selected_cuda_socket"] == "/sock/gpu"
    assert result["selected_cuda_pane_id"] == "%1"


def test_bridge_timeout_reports_repair_path_without_user_handoff() -> None:
    clock = FastClock()
    tool = TimeoutBridge(
        runner=FakeRunner(),
        socket_exists=lambda path: str(path) == "/sock/gpu",
        clock=clock.now,
        sleep=clock.sleep,
    )
    result = tool.run_bridge_mode(BridgeOptions(mode="validate-only", socket="/sock/gpu", timeout=1))
    assert result["success"] is False
    assert result["stop_reason"] == "timeout"
    assert result["bridge_repair_required"] is True
    assert "odcr-enter-gpu" not in result["error"]


def test_fresh_discover_ambiguity_has_no_manual_gpu_handoff() -> None:
    tool = TmuxGpuBridge(runner=EmptyRunner(), socket_exists=lambda path: str(path) == "/sock/gpu")
    with pytest.raises(BridgeError) as exc:
        tool.resolve_target(BridgeOptions(mode="validate-only", socket="/sock/gpu"))
    assert "odcr-enter-gpu" not in str(exc.value)
    assert "did not resolve to exactly one runnable tmux pane" in str(exc.value)


def test_pane_in_mode_recovery_sends_only_safe_mode_exit_key() -> None:
    runner = InModeThenRecoveredRunner(recover_after=1)
    tool = SuccessBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
    result = tool.run_bridge_mode(BridgeOptions(mode="validate-only", socket="/sock/gpu"))
    assert result["success"] is True
    recovery = result["pane_mode_recovery"]
    assert recovery["before_in_mode"] is True
    assert recovery["after_in_mode"] is False
    assert recovery["recovery_keys_sent"] == ["Escape"]
    assert recovery["stale_state_used"] is False
    assert recovery["tmux_session_control_used"] is False
    joined = " ".join(" ".join(call) for call in runner.calls)
    assert "odcr-enter-gpu" not in joined
    assert "sbatch" not in joined
    assert "scancel" not in joined
    assert "new-session" not in joined
    assert "kill-session" not in joined
    assert "attach-session" not in joined
    assert "switch-client" not in joined


def test_pane_in_mode_recovery_retries_are_bounded() -> None:
    runner = InModeThenRecoveredRunner(recover_after=99)
    tool = SuccessBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
    result = tool.run_bridge_mode(BridgeOptions(mode="validate-only", socket="/sock/gpu"))
    assert result["success"] is False
    details = result["details"]
    assert details["result"] == "pane_remains_in_mode_after_safe_recovery_attempts"
    assert details["recovery_keys_sent"] == ["Escape", "q"]
    assert runner.mode_exit_keys == ["Escape", "q"]
    assert "odcr-enter-gpu" not in result["error"]


def test_pane_in_mode_recovery_compute_app_guard_blocks_send() -> None:
    runner = InModeThenRecoveredRunner(compute_apps="123, python, 1024\n")
    tool = SuccessBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
    result = tool.run_bridge_mode(BridgeOptions(mode="validate-only", socket="/sock/gpu"))
    assert result["success"] is False
    details = result["details"]
    assert details["result"] == "blocked_by_compute_app_guard"
    assert details["compute_app_guard"]["reason"] == "nvidia_smi_compute_apps_nonempty"
    assert runner.mode_exit_keys == []


def test_pane_in_mode_recovery_refuses_active_worker_command() -> None:
    runner = ActivePaneCommandRunner()
    tool = SuccessBridge(runner=runner, socket_exists=lambda path: str(path) == "/sock/gpu")
    result = tool.run_bridge_mode(BridgeOptions(mode="validate-only", socket="/sock/gpu"))
    assert result["success"] is False
    details = result["details"]
    assert details["result"] == "blocked_by_compute_app_guard"
    assert details["compute_app_guard"]["reason"] == "pane_command_looks_like_active_compute_app"
    assert runner.mode_exit_keys == []


def test_unregistered_bridge_mode_fails_fast() -> None:
    tool = TmuxGpuBridge(runner=FakeRunner(), socket_exists=lambda path: str(path) == "/sock/gpu")
    with pytest.raises(BridgeError):
        tool.run_bridge_mode(BridgeOptions(mode="repo-command", socket="/sock/gpu"))


def test_gpu_handshake_uses_ai_analysis_writer(tmp_path) -> None:
    code = write_handshake(
        kind="unit",
        require_cuda=False,
        status_path=tmp_path / "AI_analysis" / "01_raw_logs" / "aux_runtime_gpu_handshake.status.json",
        log_path=tmp_path / "AI_analysis" / "01_raw_logs" / "aux_runtime_gpu_handshake.log",
        report_path=tmp_path / "AI_analysis" / "05_final_reports" / "aux_runtime_gpu_validation_report.md",
        repo_root=tmp_path,
        stage="step5A",
        task="2",
    )
    assert code == 0
    status = tmp_path / "AI_analysis" / "01_raw_logs" / "aux_runtime_gpu_handshake.status.json"
    log = tmp_path / "AI_analysis" / "01_raw_logs" / "aux_runtime_gpu_handshake.log"
    report = tmp_path / "AI_analysis" / "05_final_reports" / "aux_runtime_gpu_validation_report.md"
    assert status.is_file()
    assert log.is_file()
    assert report.is_file()
    assert '"source": "gpu_handshake"' in status.read_text(encoding="utf-8")
    assert "schema_version" in log.read_text(encoding="utf-8")
    assert "schema_version" in report.read_text(encoding="utf-8")


def test_gpu_handshake_failure_has_no_manual_gpu_handoff(monkeypatch) -> None:
    import odcr_core.aux.runtime.gpu_handshake as gpu_handshake

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    payload = gpu_handshake.collect_handshake(kind="cuda-probe", require_cuda=True, stage="step5A", task="2")
    assert payload["success"] is False
    assert "odcr-enter-gpu" not in str(payload.get("error"))
