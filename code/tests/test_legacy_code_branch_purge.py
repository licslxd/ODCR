from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_code1_code2_reference_source_trees_deleted() -> None:
    assert not (REPO_ROOT / "code1").exists()
    assert not (REPO_ROOT / "code2").exists()


def test_legacy_tmux_bridge_wrapper_is_fail_fast_shim() -> None:
    text = (REPO_ROOT / "code" / "tools" / "odcr_tmux_gpu_bridge.py").read_text(encoding="utf-8")
    assert "retired and fail-fast" in text
    assert "odcr_core.aux.runtime.tmux_gpu_bridge" not in text
    assert "subprocess" not in text


def test_runtime_bridge_has_no_old_target_selection_branches() -> None:
    text = (REPO_ROOT / "code" / "odcr_core" / "aux" / "runtime" / "tmux_gpu_bridge.py").read_text(encoding="utf-8")
    for forbidden in (
        "_run_global_bridge_mode",
        "_default_socket_path",
        "_read_state_hint",
        "_state_hint_socket_target",
        "TARGET_SOURCE_ENV",
        "TARGET_SOURCE_CURRENT_TMUX",
        "TARGET_SOURCE_DEFAULT",
    ):
        assert forbidden not in text
    assert "global_target_selection_retired" in text


def test_current_gpu_pane_handoff_accepts_v2_only() -> None:
    text = (REPO_ROOT / "code" / "odcr_core" / "aux" / "runtime" / "gpu_pane_handoff.py").read_text(encoding="utf-8")
    assert "COMPATIBLE_SCHEMA_VERSIONS = {SCHEMA_VERSION}" in text
    assert "handoff/1" not in text
