from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_help_is_under_odcr_entrypoint() -> None:
    odcr_source = (REPO_ROOT / "code" / "odcr.py").read_text(encoding="utf-8")
    runtime_source = (
        REPO_ROOT / "code" / "odcr_core" / "aux" / "control" / "cli_runtime.py"
    ).read_text(encoding="utf-8")
    bridge_source = (
        REPO_ROOT / "code" / "odcr_core" / "aux" / "runtime" / "tmux_gpu_bridge.py"
    ).read_text(encoding="utf-8")
    assert 'sub.add_parser("runtime"' in odcr_source
    assert "cmd_runtime(args)" in odcr_source
    assert "runtime_command" in runtime_source
    assert 'add_parser("bridge"' in bridge_source
    assert 'add_parser("probe"' in bridge_source
