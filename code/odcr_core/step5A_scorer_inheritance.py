"""Retired Step5A partial scorer transplant diagnostic.

This module is kept only for negative/structural-copy diagnostics.  It is not a
Step5A readiness gate; functional parity with ``Step3FrozenTeacher`` is the
required active evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from odcr_core.file_atomic import atomic_write_json


STEP5A_SCORER_INHERITANCE_SCHEMA_VERSION = "odcr_step5A_scorer_inheritance/1"
DEFAULT_STEP3_SCORER_CHECKPOINT = "runs/step3/task2/2/model/best_observed.pth"
DEFAULT_STEP3_SCORER_SHA256 = "9089ac53b138c12ba1260370aed3d637b305f7f7f6a98a7bcbc7721eb5559017"


class Step5AScorerInheritanceError(RuntimeError):
    """Raised when Step5A cannot prove Step3 scorer inheritance."""


class Step3InheritedRatingScorer(nn.Module):
    """Step3 PETER_MLP-compatible scorer wrapped in the Step5 scorer interface."""

    def __init__(self, hidden_size: int):
        super().__init__()
        h = int(hidden_size)
        self.linear1 = nn.Linear(h, h)
        self.linear2 = nn.Linear(h, 1)
        self.sigmoid = nn.Sigmoid()
        self.last_hidden: torch.Tensor | None = None

    def forward(
        self,
        shared_latent: torch.Tensor,
        content_profile: torch.Tensor | None = None,
        specific_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del content_profile, specific_latent
        hidden = self.sigmoid(self.linear1(shared_latent))
        self.last_hidden = hidden
        return self.linear2(hidden).view(-1)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise Step5AScorerInheritanceError(f"Step3 checkpoint root is not a state_dict mapping: {path}")
    return state


def transplant_step3_scorer_into_step5A(
    model: nn.Module,
    *,
    repo_root: str | Path,
    checkpoint_path: str | Path = DEFAULT_STEP3_SCORER_CHECKPOINT,
    expected_sha256: str = DEFAULT_STEP3_SCORER_SHA256,
    report_path: str | Path | None = None,
    strict_hash: bool = True,
) -> dict[str, Any]:
    """Load Step3 embeddings and scorer-compatible recommender weights for diagnostics only."""

    root = Path(repo_root).expanduser().resolve()
    ckpt = Path(checkpoint_path).expanduser()
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    ckpt = ckpt.resolve()
    if not ckpt.is_file():
        raise Step5AScorerInheritanceError(f"Step3 scorer checkpoint missing: {ckpt}")
    actual_sha = sha256_file(ckpt)
    if strict_hash and expected_sha256 and actual_sha != str(expected_sha256):
        raise Step5AScorerInheritanceError(
            f"Step3 scorer checkpoint hash mismatch: expected={expected_sha256} actual={actual_sha} path={ckpt}"
        )
    state = _load_state(ckpt)
    missing: list[str] = []
    unexpected: list[str] = []
    inherited: list[str] = []

    def _copy(src_key: str, dst_tensor: torch.Tensor, *, module_name: str) -> None:
        src = state.get(src_key)
        if not isinstance(src, torch.Tensor):
            missing.append(src_key)
            return
        if tuple(src.shape) != tuple(dst_tensor.shape):
            missing.append(f"{src_key}:shape {tuple(src.shape)} != {tuple(dst_tensor.shape)}")
            return
        with torch.no_grad():
            dst_tensor.copy_(src.to(device=dst_tensor.device, dtype=dst_tensor.dtype))
        inherited.append(module_name)

    if hasattr(model, "user_embeddings"):
        _copy("user_embeddings.weight", model.user_embeddings.weight, module_name="user_embeddings.weight")
    if hasattr(model, "item_embeddings"):
        _copy("item_embeddings.weight", model.item_embeddings.weight, module_name="item_embeddings.weight")
    scorer = getattr(model, "odcr_scorer", None)
    if not isinstance(scorer, Step3InheritedRatingScorer):
        missing.append("model.odcr_scorer:expected Step3InheritedRatingScorer")
    else:
        _copy("recommender.linear1.weight", scorer.linear1.weight, module_name="odcr_scorer.linear1.weight")
        _copy("recommender.linear1.bias", scorer.linear1.bias, module_name="odcr_scorer.linear1.bias")
        _copy("recommender.linear2.weight", scorer.linear2.weight, module_name="odcr_scorer.linear2.weight")
        _copy("recommender.linear2.bias", scorer.linear2.bias, module_name="odcr_scorer.linear2.bias")
    needed = {
        "user_embeddings.weight",
        "item_embeddings.weight",
        "recommender.linear1.weight",
        "recommender.linear1.bias",
        "recommender.linear2.weight",
        "recommender.linear2.bias",
    }
    unexpected = sorted(str(k) for k in state.keys() if str(k) in needed and str(k) not in needed)
    report = {
        "schema_version": STEP5A_SCORER_INHERITANCE_SCHEMA_VERSION,
        "step3_checkpoint_path": str(ckpt),
        "step3_checkpoint_hash": actual_sha,
        "expected_step3_checkpoint_hash": str(expected_sha256),
        "strict_hash": bool(strict_hash),
        "inherited_modules": sorted(set(inherited)),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "scorer_init_source": "step3_transplant" if not missing else "incomplete_step3_transplant",
        "distillation_enabled": False,
        "distillation_weight": 0.0,
        "inheritance_or_distillation_pass": not missing,
        "structural_copy_only": True,
        "readiness_pass": False,
        "readiness_required_report": "step3_functional_parity_report",
        "active_step5A_readiness_gate": "Step3FrozenTeacher parity, not partial transplant",
    }
    if not report["inheritance_or_distillation_pass"]:
        raise Step5AScorerInheritanceError(
            "Step5A scorer inheritance failed: " + json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
    setattr(model, "_step5a_scorer_inheritance_report", report)
    if report_path is not None:
        # DDP ranks all need the transplanted weights, but a single rank owns
        # the shared evidence file to avoid same-tmp atomic write races.
        if int(os.environ.get("RANK", "0") or 0) == 0:
            atomic_write_json(Path(report_path), report)
    return report


def inheritance_report_for_model(model: nn.Module) -> dict[str, Any]:
    report = getattr(model, "_step5a_scorer_inheritance_report", None)
    if isinstance(report, Mapping):
        return dict(report)
    return {
        "schema_version": STEP5A_SCORER_INHERITANCE_SCHEMA_VERSION,
        "inheritance_or_distillation_pass": False,
        "scorer_init_source": "unknown",
        "distillation_enabled": False,
        "distillation_weight": 0.0,
        "structural_copy_only": True,
        "readiness_pass": False,
        "readiness_required_report": "step3_functional_parity_report",
    }


__all__ = [
    "DEFAULT_STEP3_SCORER_CHECKPOINT",
    "DEFAULT_STEP3_SCORER_SHA256",
    "STEP5A_SCORER_INHERITANCE_SCHEMA_VERSION",
    "Step3InheritedRatingScorer",
    "Step5AScorerInheritanceError",
    "inheritance_report_for_model",
    "sha256_file",
    "transplant_step3_scorer_into_step5A",
]
