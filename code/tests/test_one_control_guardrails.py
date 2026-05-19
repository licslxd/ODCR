from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from helpers.cache_cleanup import remove_active_python_caches
from tools.check_one_control_guardrails import RULE_GROUP_BY_ID, format_report, run_checks


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_guardrail_script_runs_strict() -> None:
    remove_active_python_caches(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "code/tools/check_one_control_guardrails.py", "--strict"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ODCR One-Control Guardrails: PASS (0 fail, 0 warn)" in proc.stdout


def test_aux_guardrail_rules_are_present() -> None:
    for rid in [f"R{idx}" for idx in range(119, 132)]:
        assert RULE_GROUP_BY_ID.get(rid) == "aux-infrastructure"
    for rid in [f"R{idx}" for idx in range(137, 143)]:
        assert RULE_GROUP_BY_ID.get(rid) == "step5-innovation"


def test_format_report_summary_level() -> None:
    remove_active_python_caches(REPO_ROOT)
    report = run_checks(repo_root=REPO_ROOT, strict=True)
    text = format_report(report)
    assert text.startswith("ODCR One-Control Guardrails: PASS")
