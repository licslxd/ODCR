from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_aux_architecture_has_active_sources() -> None:
    required = [
        "runtime/tmux_gpu_bridge.py",
        "runtime/pane_discovery.py",
        "runtime/gpu_handshake.py",
        "runtime/command_registry.py",
        "governance/rule_registry.py",
        "evidence/ai_analysis_writer.py",
        "artifacts/path_policy.py",
        "control/cli_runtime.py",
    ]
    for rel in required:
        path = REPO_ROOT / "code" / "odcr_core" / "aux" / rel
        assert path.is_file(), rel
        assert path.read_text(encoding="utf-8").strip(), rel
