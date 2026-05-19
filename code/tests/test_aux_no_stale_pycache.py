from __future__ import annotations

from pathlib import Path

from helpers.cache_cleanup import remove_active_python_caches
from tools.check_one_control_guardrails import run_checks


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_active_tree_pycache_guardrail_passes() -> None:
    remove_active_python_caches(REPO_ROOT)
    result = next(item for item in run_checks(strict=True).results if item.rule_id == "R126")
    assert result.status == "PASS"
