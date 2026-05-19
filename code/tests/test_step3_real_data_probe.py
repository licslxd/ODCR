from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_retired_step3_real_data_probe_fails_fast() -> None:
    proc = subprocess.run(
        [sys.executable, "code/tools/odcr_step3_real_data_probe.py", "--help"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0
    run = subprocess.run(
        [sys.executable, "code/tools/odcr_step3_real_data_probe.py"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert run.returncode == 2
    assert "./odcr runtime probe --stage step3" in run.stderr
