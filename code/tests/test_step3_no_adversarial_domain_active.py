from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_step3_active_path_has_no_adversarial_domain_training_controls() -> None:
    active_files = [
        REPO_ROOT / "code" / "executors" / "step3_train_core.py",
        REPO_ROOT / "code" / "odcr_core" / "odcr_representation.py",
        REPO_ROOT / "configs" / "odcr.yaml",
    ]
    forbidden = re.compile(
        r"domain_adversarial|gradient reversal|\bGRL\b|\bDANN\b|domain discriminator|adversarial training",
        re.IGNORECASE,
    )
    for path in active_files:
        text = path.read_text(encoding="utf-8")
        assert not forbidden.search(text), path


def test_z_domain_is_kept_as_sidecar_structural_latent() -> None:
    text = (REPO_ROOT / "code" / "odcr_core" / "odcr_representation.py").read_text(encoding="utf-8")
    assert "z_domain = self.csb_domain_norm" in text
    assert "domain_vec.detach()" in text

