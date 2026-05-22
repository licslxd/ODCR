from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from odcr_core.aux.runtime.command_registry import (
    FORMAL_TRAIN_DETECTOR_VERSION,
    formal_training_command_reason,
)
from odcr_core.aux.templates.command_templates import bridge_command


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bridge_exec_is_public_runtime_bridge_command() -> None:
    assert bridge_command("exec") == ("./odcr", "runtime", "bridge", "exec")
    proc = subprocess.run(
        ["./odcr", "runtime", "bridge", "exec", "--help"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0
    assert "dispatch a command" in proc.stdout


def test_no_closed_whitelist_blocks_eval_or_nohup_wrapped_eval() -> None:
    assert formal_training_command_reason(("nohup", "./odcr", "step3-rating", "--task", "2", "--mode", "multi", "--run-id", "2")) is None
    assert formal_training_command_reason(("python", "code/odcr.py", "eval", "--task", "2", "--from-step5", "latest")) is None


def test_formal_training_string_blocker_removed_for_user_authorized_gpu_runs() -> None:
    assert formal_training_command_reason(("nohup", "./odcr", "step3", "--task", "2", "--mode", "full")) is None
    assert formal_training_command_reason(("./odcr", "step5", "--task", "2", "--run-id", "1_2")) is None
    assert FORMAL_TRAIN_DETECTOR_VERSION == "odcr_bridge_formal_train_audit/2"
