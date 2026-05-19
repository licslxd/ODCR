from __future__ import annotations

import json
from pathlib import Path

import pytest

from odcr_core.aux.runtime.gpu_pane_handoff import (
    ADMIN_PART_SCHEMA_VERSION,
    CommandResult,
    HandoffError,
    admin_part_path,
    build_gpu_post_srun_payload,
    current_handoff_path,
    current_handoff_tmp_path,
    failed_handoff_path,
    parse_tmux_env,
    selection_from_handoff_payload,
    write_admin_pre_srun,
    write_gpu_post_srun,
)


def _env(**overrides: str) -> dict[str, str]:
    env = {
        "TMUX": "/public/home/zhangliml/tmp/codex/tmux-2080/odcr_gpu,54534,0",
        "TMUX_PANE": "%0",
        "SLURM_JOB_ID": "205571",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }
    env.update(overrides)
    return env


def _runner(args) -> CommandResult:
    key = tuple(str(part) for part in args)
    if key[:4] == ("tmux", "-S", "/public/home/zhangliml/tmp/codex/tmux-2080/odcr_gpu", "display-message"):
        stdout = "\t".join(("odcr", "odcr:0.0", "%0", "/repo", "bash", "0"))
        return CommandResult(key, 0, stdout, "")
    if key == ("nvidia-smi", "-L"):
        return CommandResult(key, 0, "GPU 0: NVIDIA A100-PCIE-40GB\nGPU 1: NVIDIA A100-PCIE-40GB", "")
    if key[:1] == ("nvidia-smi",):
        return CommandResult(key, 0, "", "")
    return CommandResult(key, 1, "", "unexpected")


def _torch_probe() -> dict:
    return {
        "torch_cuda_available": True,
        "torch_cuda_device_count": 2,
        "torch_cuda_device_names": ["NVIDIA A100-PCIE-40GB", "NVIDIA A100-PCIE-40GB"],
    }


def test_parse_tmux_socket_ignores_server_pid_for_selection() -> None:
    parsed = parse_tmux_env("/public/home/zhangliml/tmp/codex/tmux-2080/odcr_gpu,54534,0", "%0")
    assert parsed["socket"] == "/public/home/zhangliml/tmp/codex/tmux-2080/odcr_gpu"
    assert parsed["server_pid_diagnostic_only"] == "54534"
    assert parsed["session_id_diagnostic_only"] == "0"
    assert parsed["pane"] == "%0"
    assert parsed["selection_uses_pid"] is False


def test_two_phase_current_gpu_pane_is_atomic_and_replaces_stale_file(tmp_path: Path) -> None:
    path = current_handoff_path(tmp_path)
    tmp = current_handoff_tmp_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"stale": true}\n', encoding="utf-8")
    tmp.write_text('{"old_tmp": true}\n', encoding="utf-8")

    admin = write_admin_pre_srun(
        repo_root=tmp_path,
        source="odcr-enter-gpu",
        job_id="205571",
        selected_node="gpu03",
        env=_env(),
        runner=_runner,
        hostname="admin",
        cwd="/repo",
        user="zhangliml",
        generated_at_utc="2026-05-17T00:00:00Z",
    )
    assert admin["schema_version"] == ADMIN_PART_SCHEMA_VERSION
    assert admin_part_path(tmp_path).is_file()
    assert admin["admin_tmux"]["raw_TMUX"].endswith(",54534,0")
    assert admin["admin_tmux"]["selection_uses_pid"] is False

    payload = write_gpu_post_srun(
        repo_root=tmp_path,
        source="odcr-enter-gpu",
        env=_env(),
        runner=_runner,
        torch_probe=_torch_probe,
        hostname="gpu03",
        cwd="/repo",
        generated_at_utc="2026-05-17T00:00:01Z",
    )

    assert path.is_file()
    assert not tmp.exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == payload
    assert written["schema_version"] == "odcr_current_gpu_pane_handoff/2"
    assert written["source"] == "odcr-enter-gpu"
    assert written["admin_tmux"]["raw_TMUX"].endswith(",54534,0")
    assert written["admin_tmux"]["socket"] == "/public/home/zhangliml/tmp/codex/tmux-2080/odcr_gpu"
    assert written["admin_tmux"]["selection_uses_pid"] is False
    assert written["gpu_runtime"]["torch_cuda_device_count"] == 2
    assert written["selection_key"] == {
        "socket": "/public/home/zhangliml/tmp/codex/tmux-2080/odcr_gpu",
        "pane": "%0",
        "hostname": "gpu03",
        "slurm_job_id": "205571",
        "cuda_visible_devices": "0,1",
    }
    selection = selection_from_handoff_payload(written)
    assert selection["source"] == "current_gpu_pane_handoff"
    assert selection["socket"] == written["selection_key"]["socket"]
    assert selection["target"] == "%0"


def test_gpu_post_srun_refuses_admin_and_deletes_stale_file(tmp_path: Path) -> None:
    path = current_handoff_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"stale": true}\n', encoding="utf-8")
    write_admin_pre_srun(
        repo_root=tmp_path,
        source="odcr-enter-gpu",
        job_id="205571",
        env=_env(),
        runner=_runner,
        hostname="admin",
    )
    with pytest.raises(HandoffError):
        write_gpu_post_srun(
            repo_root=tmp_path,
            source="odcr-enter-gpu",
            env=_env(),
            runner=_runner,
            torch_probe=_torch_probe,
            hostname="admin",
        )
    assert not path.exists()
    assert failed_handoff_path(tmp_path).is_file()


def test_admin_pre_srun_refuses_empty_tmux_without_cuda_probe(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(args) -> CommandResult:
        calls.append(tuple(str(part) for part in args))
        return _runner(args)

    with pytest.raises(HandoffError):
        write_admin_pre_srun(
            repo_root=tmp_path,
            source="odcr-enter-gpu",
            job_id="205571",
            env=_env(TMUX="", TMUX_PANE=""),
            runner=runner,
            hostname="gpu03",
        )
    assert not any(call and call[0] == "nvidia-smi" for call in calls)


def test_gpu_post_srun_refuses_device_count_less_than_two(tmp_path: Path) -> None:
    write_admin_pre_srun(
        repo_root=tmp_path,
        source="odcr-enter-gpu",
        job_id="205571",
        env=_env(),
        runner=_runner,
        hostname="admin",
    )
    with pytest.raises(HandoffError):
        write_gpu_post_srun(
            repo_root=tmp_path,
            source="odcr-enter-gpu",
            env=_env(),
            runner=_runner,
            torch_probe=lambda: {
                "torch_cuda_available": True,
                "torch_cuda_device_count": 1,
                "torch_cuda_device_names": ["NVIDIA A100-PCIE-40GB"],
            },
            hostname="gpu03",
        )


def test_gpu_post_srun_does_not_execute_tmux(tmp_path: Path) -> None:
    write_admin_pre_srun(
        repo_root=tmp_path,
        source="odcr-enter-gpu",
        job_id="205571",
        env=_env(),
        runner=_runner,
        hostname="admin",
    )
    calls: list[tuple[str, ...]] = []

    def runner(args) -> CommandResult:
        key = tuple(str(part) for part in args)
        calls.append(key)
        if key and key[0] == "tmux":
            raise AssertionError("gpu-post-srun must not call tmux")
        return _runner(args)

    payload = build_gpu_post_srun_payload(
        repo_root=tmp_path,
        source="odcr-enter-gpu",
        env=_env(),
        runner=runner,
        torch_probe=_torch_probe,
        hostname="gpu03",
        generated_at_utc="2026-05-17T00:00:00Z",
    )
    assert payload["valid_for_bridge_selection"] is True
    assert not any(call and call[0] == "tmux" for call in calls)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"SLURM_JOB_ID": ""}, "SLURM_JOB_ID"),
        ({"CUDA_VISIBLE_DEVICES": ""}, "CUDA_VISIBLE_DEVICES"),
    ],
)
def test_gpu_post_srun_requires_slurm_and_cuda_visible_devices(tmp_path: Path, override: dict[str, str], message: str) -> None:
    write_admin_pre_srun(
        repo_root=tmp_path,
        source="odcr-enter-gpu",
        job_id="205571",
        env=_env(),
        runner=_runner,
        hostname="admin",
    )
    with pytest.raises(HandoffError, match=message):
        write_gpu_post_srun(
            repo_root=tmp_path,
            source="odcr-enter-gpu",
            env=_env(**override),
            runner=_runner,
            torch_probe=_torch_probe,
            hostname="gpu03",
        )


def test_gpu_post_srun_requires_torch_cuda_true(tmp_path: Path) -> None:
    write_admin_pre_srun(
        repo_root=tmp_path,
        source="odcr-enter-gpu",
        job_id="205571",
        env=_env(),
        runner=_runner,
        hostname="admin",
    )
    with pytest.raises(HandoffError, match="torch.cuda.is_available"):
        write_gpu_post_srun(
            repo_root=tmp_path,
            source="odcr-enter-gpu",
            env=_env(),
            runner=_runner,
            torch_probe=lambda: {
                "torch_cuda_available": False,
                "torch_cuda_device_count": 0,
                "torch_cuda_device_names": [],
            },
            hostname="gpu03",
        )
