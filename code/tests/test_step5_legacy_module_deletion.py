from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STEP5_ENGINE = REPO_ROOT / "code" / "executors" / "step5_engine.py"


def test_step5_active_model_deletes_retired_legacy_modules() -> None:
    source = STEP5_ENGINE.read_text(encoding="utf-8")
    for needle in (
        "self.recommender",
        "PETER_MLP",
        "self.hidden2token",
        "self.flan_soft_prompt_stack",
    ):
        assert needle not in source


def test_step5_ccv_missing_packet_is_fail_fast_not_soft_prompt_fallback() -> None:
    source = STEP5_ENGINE.read_text(encoding="utf-8")
    assert "Step5 CCVControlPacket is required" in source
    assert "no-control soft-prompt fallback is not an active path" in source
    assert "flan_soft_prompt_stack" not in source
