from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from odcr_core.aux.runtime.command_registry import LEGACY_FORBIDDEN_COMMANDS


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["repo-command", "repo-script", "repo-module", "command-file"])
def test_legacy_bridge_cli_modes_fail_fast(mode: str) -> None:
    proc = subprocess.run(
        [sys.executable, "code/tools/odcr_tmux_gpu_bridge.py", mode],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 2
    assert "retired" in proc.stderr
    assert mode in LEGACY_FORBIDDEN_COMMANDS
