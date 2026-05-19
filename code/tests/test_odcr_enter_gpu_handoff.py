from __future__ import annotations

import subprocess
from pathlib import Path


ODCR_ENTER_GPU = Path("/public/home/zhangliml/.local/bin/odcr-enter-gpu")


def _script_text() -> str:
    assert ODCR_ENTER_GPU.is_file()
    return ODCR_ENTER_GPU.read_text(encoding="utf-8")


def test_odcr_enter_gpu_script_is_syntax_valid() -> None:
    result = subprocess.run(["bash", "-n", str(ODCR_ENTER_GPU)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_odcr_enter_gpu_runs_admin_pre_srun_before_srun() -> None:
    text = _script_text()
    admin_index = text.index("--mode admin-pre-srun")
    srun_index = text.index("\n  srun \\")
    assert admin_index < srun_index
    assert "--job-id \"$jobid\"" in text
    assert "--selected-node \"$selected_node\"" in text
    assert "tmux display-message" not in text


def test_odcr_enter_gpu_runs_gpu_post_srun_inside_gpu_shell() -> None:
    text = _script_text()
    srun_index = text.index("\n  srun \\")
    gpu_index = text.index("--mode gpu-post-srun")
    assert srun_index < gpu_index
    assert "python code/tools/write_current_gpu_pane.py --mode gpu-post-srun" in text


def test_handoff_failure_does_not_block_gpu_shell_entry() -> None:
    text = _script_text()
    assert "if ! python code/tools/write_current_gpu_pane.py --mode gpu-post-srun" in text
    assert "GPU pane handoff not written; Codex bridge must not use stale GPU state." in text
    assert "exec bash --noprofile --norc" in text
    failure_warning = text.index("GPU pane handoff not written; Codex bridge must not use stale GPU state.")
    shell_exec = text.rindex("exec bash --noprofile --norc")
    assert failure_warning < shell_exec


def test_odcr_enter_gpu_deletes_stale_handoff_state_at_start() -> None:
    text = _script_text()
    for name in (
        "current_gpu_pane.json",
        "current_gpu_pane.json.tmp",
        "current_gpu_pane.failed.json",
        "current_gpu_pane.admin_part.json",
        "current_gpu_pane.admin_part.json.tmp",
    ):
        assert name in text


def test_odcr_enter_gpu_does_not_use_pid_for_selection() -> None:
    text = _script_text()
    assert "pane_pid" not in text
    assert "server_pid" not in text
    assert "TMUX_PANE" not in text
