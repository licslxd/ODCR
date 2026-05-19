from __future__ import annotations

from pathlib import Path

from helpers.cache_cleanup import remove_active_python_caches
from tools.check_one_control_guardrails import run_checks


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cleanup_guardrails_pass() -> None:
    remove_active_python_caches(REPO_ROOT)
    results = {item.rule_id: item for item in run_checks(strict=True).results}
    for rule_id in ("R123", "R129", "R130", "R131"):
        assert results[rule_id].status == "PASS"
