from __future__ import annotations

import re
from pathlib import Path

import pytest

from odcr_core.config_resolver import resolve_config
from odcr_core.config_schema import OneControlConfigError


REPO_ROOT = Path(__file__).resolve().parents[2]


def _active_files() -> list[Path]:
    files: list[Path] = []
    for rel_root in ("code", "configs", ".codex"):
        root = REPO_ROOT / rel_root
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    files.extend(path for path in (REPO_ROOT / "odcr", REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md") if path.is_file())
    return sorted(files)


def test_no_production_synthetic_step5_formal_path_terms() -> None:
    banned = re.compile(
        r"synthetic_one_batch|_step5_synthetic_preflight_batch|synthetic_preflight_role|find_unused_false_preflight.*synthetic"
    )
    offenders: list[str] = []
    for path in _active_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith("code/tests/") or rel == "code/odcr_core/aux/governance/guardrail_runner.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if banned.search(text):
            offenders.append(rel)
    assert offenders == []


def test_synthetic_policy_is_rejected_by_resolver() -> None:
    with pytest.raises(OneControlConfigError):
        resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step5",
            task_id=2,
            set_overrides=["step5.ddp.find_unused_false_preflight=synthetic_one_batch"],
            dry_run=True,
            from_step4="1",
            run_id="auto",
            step5_head="step5A",
        )


def test_artifact_only_probe_cannot_mark_e4() -> None:
    text = (REPO_ROOT / "code" / "odcr_core" / "step5_runtime_probe.py").read_text(encoding="utf-8")
    block_start = text.index('"artifact_build_only": True')
    block = text[block_start : block_start + 1200]
    assert '"evidence_level": E3_GPU_TRANSPORT' in block
    assert '"forward_executed": False' in block
    assert '"loss_backward_executed": False' in block
    assert '"optimizer_step_executed": False' in block
