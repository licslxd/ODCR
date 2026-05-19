from __future__ import annotations

from pathlib import Path

from helpers.cache_cleanup import remove_active_python_caches
from odcr_core.aux.governance.post_edit_registry import SCOPES
from odcr_core.aux.governance.rule_registry import RULE_GROUP_BY_ID
from tools.check_one_control_guardrails import run_checks


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_aux_rules_registered() -> None:
    for rule_id in [f"R{idx}" for idx in range(119, 132)]:
        assert RULE_GROUP_BY_ID[rule_id] == "aux-infrastructure"
    for rule_id in [f"R{idx}" for idx in range(137, 142)]:
        assert RULE_GROUP_BY_ID[rule_id] == "step5-innovation"


def test_single_scope_registry_contains_expected_scopes() -> None:
    assert "governance-fast" in SCOPES
    assert "step5" in SCOPES
    assert "all" in SCOPES


def test_guardrail_strict_passes_aux_rules() -> None:
    remove_active_python_caches(REPO_ROOT)
    report = run_checks(strict=True)
    assert report.ok
