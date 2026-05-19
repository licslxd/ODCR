from __future__ import annotations

import json
from pathlib import Path

import pytest

from odcr_core.aux.runtime.bounded_probe import build_probe_payload
from odcr_core.aux.runtime.gpu_pane_handoff import current_handoff_path
from odcr_core.aux.runtime.pane_discovery import CommandResult, SubprocessRunner
from odcr_core.aux.runtime.tmux_gpu_bridge import BridgeOptions, TmuxGpuBridge


def _pane_row(
    *,
    socket_name: str = "gpu",
    session: str = "odcr",
    pane_id: str = "%0",
    pid: str = "100",
    command: str = "bash",
    cwd: str | None = None,
    in_mode: str = "0",
) -> str:
    return "\t".join(
        (
            session,
            "0",
            "main",
            "0",
            pane_id,
            pid,
            command,
            cwd or str(Path.cwd()),
            "1",
            "0",
            in_mode,
            socket_name,
            "/dev/pts/1",
            command,
        )
    )


def _stale_handoff(socket: str = "/sock/old", pane: str = "%9") -> dict:
    return {
        "schema_version": "odcr_current_gpu_pane_handoff/2",
        "generated_at_utc": "2026-05-16T00:00:00Z",
        "valid_for_bridge_selection": True,
        "admin_tmux": {
            "raw_TMUX": f"{socket},1,0",
            "socket": socket,
            "pane": pane,
            "pane_id": pane,
            "target": "odcr:0.0",
            "selection_uses_pid": False,
        },
        "gpu_runtime": {
            "hostname": "gpu-old",
            "slurm_job_id": "200000",
            "cuda_visible_devices": "0,1",
            "nvidia_smi_L_returncode": 0,
            "torch_cuda_available": True,
            "torch_cuda_device_count": 2,
        },
        "selection_key": {"socket": socket, "pane": pane},
    }


def _patch_bridge_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import odcr_core.aux.runtime.tmux_gpu_bridge as bridge_module

    monkeypatch.setattr(bridge_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bridge_module, "AI_ANALYSIS", tmp_path / "AI_analysis")
    monkeypatch.setattr(bridge_module, "RAW_LOG_DIR", tmp_path / "AI_analysis" / "01_raw_logs")
    monkeypatch.setattr(bridge_module, "REPORT_DIR", tmp_path / "AI_analysis" / "05_final_reports")
    monkeypatch.setattr(bridge_module, "CURRENT_HANDOFF_PATH", current_handoff_path(tmp_path))
    monkeypatch.setattr(bridge_module, "STATE_HINT_PATH", tmp_path / "AI_analysis" / "runtime" / "gpu_pane.json")


class InventoryRunner(SubprocessRunner):
    def __init__(self, panes: dict[str, str], *, compute_apps: str = "") -> None:
        self.panes = panes
        self.compute_apps = compute_apps
        self.calls: list[tuple[str, ...]] = []
        self.mode_exit_keys: list[str] = []

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
            row = self.panes.get(socket, "")
            return CommandResult(key, 0, (row + "\n") if row else "", "")
        if command == "capture-pane":
            return CommandResult(key, 0, "last line\n", "")
        if command == "send-keys":
            if key[-1] in {"Escape", "q"}:
                self.mode_exit_keys.append(key[-1])
            return CommandResult(key, 0, "", "")
        if key and key[0] == "nvidia-smi":
            return CommandResult(key, 0, self.compute_apps, "")
        return CommandResult(key, 1, "", "unexpected")


class LiveProbeBridge(TmuxGpuBridge):
    def __init__(self, *, runner: SubprocessRunner, statuses: dict[str, dict]) -> None:
        super().__init__(runner=runner, socket_exists=lambda path: str(path) in statuses)
        self.statuses = statuses
        self.last_target = None

    def _send_registered_command(self, target, command):  # type: ignore[override]
        self.last_target = target

    def _wait_status(self, paths, timeout_s):  # type: ignore[override]
        target = self.last_target
        assert target is not None
        return dict(self.statuses[target.socket])


def _status(
    *,
    hostname: str = "gpu03",
    slurm: str = "205900",
    cuda_visible: str = "0,1",
    count: int = 2,
    compute_apps: str = "",
    success: bool = True,
) -> dict:
    return {
        "schema_version": "odcr_runtime_gpu_handshake/1",
        "success": success,
        "hostname": hostname,
        "SLURM_JOB_ID": slurm,
        "CUDA_VISIBLE_DEVICES": cuda_visible,
        "nvidia-smi": {"returncode": 0 if success else 1},
        "nvidia-smi-compute-apps": {"returncode": 0, "stdout": compute_apps},
        "torch.cuda.is_available": bool(success and count > 0),
        "torch.cuda.device_count": count,
    }


def _prepare_stale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sockets: tuple[str, ...]) -> None:
    _patch_bridge_root(monkeypatch, tmp_path)
    handoff = current_handoff_path(tmp_path)
    handoff.parent.mkdir(parents=True, exist_ok=True)
    handoff.write_text(json.dumps(_stale_handoff()) + "\n", encoding="utf-8")
    (tmp_path / "AI_analysis" / "runtime" / "gpu_pane.json").write_text(
        '{"socket": "/sock/admin", "pane_id": "%0"}\n',
        encoding="utf-8",
    )
    import odcr_core.aux.runtime.pane_discovery as pane_discovery

    monkeypatch.setattr(pane_discovery, "candidate_socket_paths", lambda **_kwargs: tuple(Path(item) for item in sockets))


def test_stale_handoff_triggers_live_cuda_discovery_without_admin_selection(monkeypatch, tmp_path) -> None:
    _prepare_stale(monkeypatch, tmp_path, ("/sock/admin", "/sock/gpu"))
    runner = InventoryRunner(
        {
            "/sock/admin": _pane_row(socket_name="admin", pane_id="%0", command="bash", cwd=str(tmp_path)),
            "/sock/gpu": _pane_row(socket_name="gpu", pane_id="%1", command="srun", cwd=str(tmp_path)),
        }
    )
    bridge = LiveProbeBridge(
        runner=runner,
        statuses={
            "/sock/admin": _status(hostname="admin01", cuda_visible="", count=0, success=False),
            "/sock/gpu": _status(hostname="gpu03"),
        },
    )

    result = bridge.run_bridge_mode(BridgeOptions(mode="cuda-probe"))

    assert result["success"] is True
    assert result["target_source"] == "live_discovery_cuda_probe"
    assert result["selected_cuda_socket"] == "/sock/gpu"
    assert result["selected_cuda_hostname"] == "gpu03"
    assert result["stale_handoff_detected"] is True
    assert result["stale_handoff_blocked_execution"] is False
    assert result["stale_state_used"] is False


def test_srun_pane_static_command_does_not_reject_successful_live_cuda_probe(monkeypatch, tmp_path) -> None:
    _prepare_stale(monkeypatch, tmp_path, ("/sock/gpu",))
    runner = InventoryRunner({"/sock/gpu": _pane_row(pane_id="%1", command="srun", cwd=str(tmp_path))})
    bridge = LiveProbeBridge(runner=runner, statuses={"/sock/gpu": _status(hostname="gpu03")})

    result = bridge.run_bridge_mode(BridgeOptions(mode="cuda-probe"))

    selected = result["live_discovery_selection"]["selected"]
    assert result["success"] is True
    assert selected["pane"]["command_class"] == "srun"
    assert selected["live_cuda_probe_success"] is True
    assert "command_not_shell_or_srun" not in selected["skip_reasons"]


def test_pane_in_mode_can_recover_before_live_cuda_probe(monkeypatch, tmp_path) -> None:
    _prepare_stale(monkeypatch, tmp_path, ("/sock/gpu",))

    class RecoverRunner(InventoryRunner):
        def run(self, args, *, timeout=None):  # type: ignore[override]
            key = tuple(str(part) for part in args)
            if len(key) > 3 and key[3] == "list-panes":
                in_mode = "0" if self.mode_exit_keys else "1"
                return CommandResult(
                    key,
                    0,
                    _pane_row(pane_id="%1", command="srun", in_mode=in_mode, cwd=str(tmp_path)) + "\n",
                    "",
                )
            return super().run(args, timeout=timeout)

    runner = RecoverRunner({"/sock/gpu": _pane_row(pane_id="%1", command="srun", in_mode="1", cwd=str(tmp_path))})
    bridge = LiveProbeBridge(runner=runner, statuses={"/sock/gpu": _status(hostname="gpu03")})

    result = bridge.run_bridge_mode(BridgeOptions(mode="cuda-probe"))

    assert result["success"] is True
    assert runner.mode_exit_keys == ["Escape"]
    assert result["live_discovery_selection"]["selected"]["pane_mode_recovery"]["success"] is True


def test_ambiguous_live_cuda_panes_fail_without_random_selection(monkeypatch, tmp_path) -> None:
    _prepare_stale(monkeypatch, tmp_path, ("/sock/gpu1", "/sock/gpu2"))
    runner = InventoryRunner(
        {
            "/sock/gpu1": _pane_row(pane_id="%1", command="srun", cwd=str(tmp_path)),
            "/sock/gpu2": _pane_row(pane_id="%2", command="srun", cwd=str(tmp_path)),
        }
    )
    bridge = LiveProbeBridge(
        runner=runner,
        statuses={
            "/sock/gpu1": _status(hostname="gpu03", slurm="205900"),
            "/sock/gpu2": _status(hostname="gpu04", slurm="205900"),
        },
    )

    result = bridge.run_bridge_mode(BridgeOptions(mode="cuda-probe"))

    assert result["success"] is False
    assert result["stop_reason"] == "ambiguous_live_cuda_panes"
    assert result["stale_state_used"] is False


def test_compute_app_guard_blocks_unknown_live_cuda_process(monkeypatch, tmp_path) -> None:
    _prepare_stale(monkeypatch, tmp_path, ("/sock/gpu",))
    runner = InventoryRunner({"/sock/gpu": _pane_row(pane_id="%1", command="srun", cwd=str(tmp_path))})
    bridge = LiveProbeBridge(
        runner=runner,
        statuses={"/sock/gpu": _status(hostname="gpu03", compute_apps="12345, python, 4096")},
    )

    result = bridge.run_bridge_mode(BridgeOptions(mode="cuda-probe"))

    assert result["success"] is False
    assert result["stop_reason"] == "no_live_cuda_pane_after_stale_handoff"
    guard_path = tmp_path / "AI_analysis" / "05_final_reports" / "gpu_bridge_live_discovery_compute_app_guard.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    assert guard["pass"] is False
    assert guard["kill_attempted"] is False
    assert guard["scancel_attempted"] is False


def test_step5_readiness_requires_e4_with_validation_for_live_discovery_source() -> None:
    handshake = {
        "hostname": "gpu03",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "torch.cuda.is_available": True,
        "torch.cuda.device_count": 2,
    }
    code_only = build_probe_payload("step5A", 2, handshake=handshake, target_source="live_discovery_cuda_probe")
    assert code_only["success"] is False
    assert code_only["target_source_allowed_for_step5_gate"] is True

    e4 = {
        "success": True,
        "evidence_level": "E4_gpu_shard_forward_bounded_formal_entry_with_validation",
        "formal_entry_lifecycle": True,
        "forward_executed": True,
        "loss_backward_executed": True,
        "optimizer_step_executed": True,
        "preflight_executed": True,
        "scratch_cleanup_status": "pass",
        "graph_tensor_audit_status": "pass",
        "graph_scratch_before_ema": [],
        "ema_init_pass": True,
        "ema_init_executed_in_E4": True,
        "ddp_wrap_pass": True,
        "first_train_step_pass": True,
        "validation_pass_executed": True,
        "validation_forward_pass": True,
        "validation_loss_finite": True,
        "validation_oom": False,
        "step5A_validation_scorer_only": True,
        "flan_explainer_called_in_step5A_validation": False,
        "out_logits_materialized_in_step5A_validation": False,
        "valid_forward_micro_batch_size": 192,
        "train_per_gpu_batch_size": 224,
        "all_trainable_grad_status": "pass",
        "trainable_param_count": 2,
        "grad_present_count": 2,
        "lora_trainable_count": 1,
        "lora_grad_present_count": 1,
        "missing_grad_params": [],
        "real_forward_backward_executed": True,
        "real_task_data_used": True,
        "real_ccv_packet_used": True,
        "synthetic_batch_used_for_formal_gate": False,
        "memory_truth": {"reserved_is_diagnostic_only": True},
        "candidate_decision": {"reserved_memory_used_for_rejection": False},
    }
    payload = build_probe_payload(
        "step5A",
        2,
        handshake=handshake,
        probe_result=e4,
        target_source="live_discovery_cuda_probe",
    )
    assert payload["success"] is True
    assert payload["evidence_level"] == "E4_gpu_shard_forward_bounded_formal_entry_with_validation"


def test_step5_e4_source_table_records_live_discovery_gpu_target(tmp_path) -> None:
    source_table = tmp_path / "source_table.json"
    source_table.write_text(json.dumps({"records": []}), encoding="utf-8")
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    probe_result = {
        "source_table_path": str(source_table),
        "output_dir": str(tmp_path),
        "step5A_validation_scorer_only": True,
        "validation_oom": False,
        "all_trainable_grad": {"evidence_context": {"evidence_id": "E4-live"}},
    }
    bridge_result = {
        "target_source": "live_discovery_cuda_probe",
        "stale_handoff_detected": True,
        "stale_handoff_blocked_execution": False,
        "live_discovery_cuda_probe_success": True,
        "selected_cuda_socket": "/sock/gpu",
        "selected_cuda_pane_id": "%1",
        "selected_cuda_hostname": "gpu03",
        "selected_cuda_visible_devices": "0,1",
        "selected_device_count": 2,
        "compute_app_guard_status": "pass",
        "selected_cuda_candidate": {"child_status": {}, "compute_app_guard": {"status": "pass"}},
    }

    patched = TmuxGpuBridge()._patch_step5_probe_gpu_target_contract(
        probe_result=probe_result,
        bridge_result=bridge_result,
    )

    assert patched["gpu_target_source"] == "live_discovery_cuda_probe"
    records = {item["key"]: item["value"] for item in json.loads(source_table.read_text(encoding="utf-8"))["records"]}
    assert records["gpu_target_source"] == "live_discovery_cuda_probe"
    assert records["stale_handoff_detected"] is True
    assert records["stale_handoff_blocked_execution"] is False
    assert records["live_discovery_cuda_probe_success"] is True
    assert records["selected_cuda_socket"] == "/sock/gpu"
    assert records["selected_cuda_pane"] == "%1"
    assert records["selected_cuda_hostname"] == "gpu03"
    assert records["selected_cuda_visible_devices"] == "0,1"
    assert records["selected_device_count"] == 2
    assert records["compute_app_guard_status"] == "pass"
    assert records["validation_e4_evidence_id"] == "E4-live"
    assert records["validation_scorer_only"] is True
    assert records["validation_oom"] is False
