"""Frozen Step3 rating teacher used by the Step5A scorer branch."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from odcr_core.file_atomic import atomic_write_json
from odcr_core.training_checkpoint import stable_hash


STEP5A_FROZEN_TEACHER_SCHEMA_VERSION = "odcr_step5A_frozen_step3_teacher/1"
DEFAULT_STEP3_TEACHER_CHECKPOINT = "runs/step3/task2/2/model/best_observed.pth"
DEFAULT_STEP3_TEACHER_SHA256 = "9089ac53b138c12ba1260370aed3d637b305f7f7f6a98a7bcbc7721eb5559017"


class Step5AFrozenTeacherError(RuntimeError):
    """Raised when the frozen Step3 teacher cannot be constructed exactly."""


@dataclass(frozen=True)
class Step3FrozenTeacherOutput:
    pred_rating: torch.Tensor
    shared_latent: torch.Tensor
    specific_latent: torch.Tensor
    shared_proj: torch.Tensor
    specific_proj: torch.Tensor
    packet: dict[str, torch.Tensor]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise Step5AFrozenTeacherError(f"Step3 teacher checkpoint is not a state_dict mapping: {path}")
    normalized = {}
    for key, value in state.items():
        k = str(key)
        if k.startswith("module."):
            k = k[len("module.") :]
        normalized[k] = value
    return normalized


def _as_profile_tuple(
    profile_tensors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(profile_tensors) != 6:
        raise Step5AFrozenTeacherError(f"Step3 teacher requires six profile tensors, got {len(profile_tensors)}")
    dc, ds, uc, us, ic, ist = profile_tensors
    return dc.detach(), ds.detach(), uc.detach(), us.detach(), ic.detach(), ist.detach()


class Step3FrozenTeacher(nn.Module):
    """Frozen Step3 rating subgraph with the original Step3 latent forward path."""

    def __init__(
        self,
        *,
        nuser: int,
        nitem: int,
        ntoken: int,
        emsize: int,
        nhead: int,
        nhid: int,
        nlayers: int,
        dropout: float,
        profile_tensors: Sequence[torch.Tensor],
        checkpoint_path: str | Path = DEFAULT_STEP3_TEACHER_CHECKPOINT,
        expected_sha256: str = DEFAULT_STEP3_TEACHER_SHA256,
        evidence_max_length: int = 48,
        repo_root: str | Path | None = None,
        strict_hash: bool = True,
        report_path: str | Path | None = None,
        source_table_hash: str | None = None,
        resolved_config_hash: str | None = None,
        lineage_hashes: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        from executors.step3_train_core import Model as Step3Model

        root = Path(repo_root or ".").expanduser().resolve()
        ckpt = Path(checkpoint_path).expanduser()
        if not ckpt.is_absolute():
            ckpt = root / ckpt
        ckpt = ckpt.resolve()
        if not ckpt.is_file():
            raise Step5AFrozenTeacherError(f"Step3 teacher checkpoint missing: {ckpt}")
        actual_sha = sha256_file(ckpt)
        if strict_hash and expected_sha256 and actual_sha != str(expected_sha256):
            raise Step5AFrozenTeacherError(
                f"Step3 teacher checkpoint hash mismatch: expected={expected_sha256} actual={actual_sha} path={ckpt}"
            )
        state = _load_state_dict(ckpt)
        try:
            nuser = int(state["user_embeddings.weight"].shape[0])
            nitem = int(state["item_embeddings.weight"].shape[0])
            ntoken = int(state["word_embeddings.weight"].shape[0])
            emsize = int(state["word_embeddings.weight"].shape[1])
        except Exception as exc:
            raise Step5AFrozenTeacherError("Step3 teacher checkpoint is missing core embedding shapes") from exc
        dc, ds, uc, us, ic, ist = _as_profile_tuple(profile_tensors)
        self.step3_model = Step3Model(
            int(nuser),
            int(nitem),
            int(ntoken),
            int(emsize),
            int(nhead),
            int(nhid),
            int(nlayers),
            float(dropout),
            uc,
            us,
            ic,
            ist,
            dc,
            ds,
        )
        self.step3_model.evidence_length = int(evidence_max_length)
        incompatible = self.step3_model.load_state_dict(state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise Step5AFrozenTeacherError(
                "Step3 teacher strict state load failed: "
                + json.dumps({"missing_keys": missing, "unexpected_keys": unexpected}, sort_keys=True)
            )
        self.step3_model.eval()
        for param in self.step3_model.parameters():
            param.requires_grad_(False)
        self.checkpoint_path = str(ckpt)
        self.checkpoint_hash = actual_sha
        self.expected_checkpoint_hash = str(expected_sha256)
        self.output_schema = {
            "pred_rating": "[B]",
            "shared_latent": "[B,H]",
            "specific_latent": "[B,H]",
            "shared_proj": "[B,H]",
            "specific_proj": "[B,H]",
            "packet": "detached Step3 evidence/latent diagnostics",
        }
        self.load_report = self._build_load_report(
            missing_keys=missing,
            unexpected_keys=unexpected,
            source_table_hash=source_table_hash,
            resolved_config_hash=resolved_config_hash,
            lineage_hashes=lineage_hashes,
        )
        if report_path is not None:
            atomic_write_json(Path(report_path), self.load_report)

    def _build_load_report(
        self,
        *,
        missing_keys: Sequence[str],
        unexpected_keys: Sequence[str],
        source_table_hash: str | None,
        resolved_config_hash: str | None,
        lineage_hashes: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        frozen_param_count = sum(int(p.numel()) for p in self.step3_model.parameters() if not p.requires_grad)
        trainable_param_count = sum(int(p.numel()) for p in self.step3_model.parameters() if p.requires_grad)
        module_names = [
            "user_embeddings",
            "item_embeddings",
            "word_embeddings",
            "odcr_disentangler",
            "shared_stream_attn",
            "specific_stream_attn",
            "shared_id_adapter",
            "specific_id_adapter",
            "polarity_embedding",
            "evidence_pool_norm",
            "recommender",
            "transformer_encoder",
        ]
        payload = {
            "schema_version": STEP5A_FROZEN_TEACHER_SCHEMA_VERSION,
            "step3_checkpoint_path": self.checkpoint_path,
            "step3_checkpoint_hash": self.checkpoint_hash,
            "expected_step3_checkpoint_hash": self.expected_checkpoint_hash,
            "loaded_module_names": module_names,
            "missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys),
            "frozen_param_count": frozen_param_count,
            "trainable_param_count_should_be_0": trainable_param_count,
            "teacher_forward_smoke_status": "not_run",
            "teacher_output_schema": dict(self.output_schema),
            "config_hashes": {
                "resolved_config_hash": str(resolved_config_hash or ""),
                "source_table_hash": str(source_table_hash or ""),
                "lineage_hashes": dict(lineage_hashes or {}),
            },
        }
        payload["teacher_contract_hash"] = stable_hash(payload)
        return payload

    def refresh_smoke_status(self, output: Step3FrozenTeacherOutput) -> None:
        self.load_report["teacher_forward_smoke_status"] = "PASS"
        self.load_report["teacher_forward_smoke_shapes"] = {
            "pred_rating": list(output.pred_rating.shape),
            "shared_latent": list(output.shared_latent.shape),
            "specific_latent": list(output.specific_latent.shape),
        }

    def forward(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        domain_idx: torch.Tensor,
        *,
        content_anchor: torch.Tensor | None = None,
        style_anchor: torch.Tensor | None = None,
        content_evidence_ids: torch.Tensor | None = None,
        style_evidence_ids: torch.Tensor | None = None,
        domain_style_anchor_ids: torch.Tensor | None = None,
        local_style_hint_ids: torch.Tensor | None = None,
        polarity_ids: torch.Tensor | None = None,
        evidence_quality_prior: torch.Tensor | None = None,
    ) -> Step3FrozenTeacherOutput:
        with torch.no_grad():
            latents, guides, evidence = self.step3_model._compute_latents(
                user.long(),
                item.long(),
                domain_idx.long(),
                content_anchor=content_anchor,
                style_anchor=style_anchor,
                content_evidence_ids=content_evidence_ids,
                style_evidence_ids=style_evidence_ids,
                domain_style_anchor_ids=domain_style_anchor_ids,
                local_style_hint_ids=local_style_hint_ids,
                polarity_ids=polarity_ids,
                evidence_quality_prior=evidence_quality_prior,
            )
            pred = self.step3_model.recommender(latents.shared).detach()
            packet = {
                "content_guide": guides["content_guide"].detach(),
                "style_guide": guides["style_guide"].detach(),
                "domain_style_guide": guides["domain_style_guide"].detach(),
                "local_style_guide": guides["local_style_guide"].detach(),
                "evidence_quality_prior": evidence.evidence_quality_prior.detach(),
            }
            out = Step3FrozenTeacherOutput(
                pred_rating=pred,
                shared_latent=latents.shared.detach(),
                specific_latent=latents.specific.detach(),
                shared_proj=latents.shared_proj.detach(),
                specific_proj=latents.specific_proj.detach(),
                packet=packet,
            )
            if self.load_report.get("teacher_forward_smoke_status") != "PASS":
                self.refresh_smoke_status(out)
            return out


def assert_teacher_frozen(teacher: nn.Module) -> dict[str, Any]:
    trainable = [name for name, param in teacher.named_parameters() if param.requires_grad]
    frozen = [name for name, param in teacher.named_parameters() if not param.requires_grad]
    return {
        "schema_version": STEP5A_FROZEN_TEACHER_SCHEMA_VERSION,
        "teacher_params_frozen": not trainable,
        "trainable_parameter_names": trainable,
        "frozen_parameter_count": len(frozen),
    }


__all__ = [
    "DEFAULT_STEP3_TEACHER_CHECKPOINT",
    "DEFAULT_STEP3_TEACHER_SHA256",
    "STEP5A_FROZEN_TEACHER_SCHEMA_VERSION",
    "Step3FrozenTeacher",
    "Step3FrozenTeacherOutput",
    "Step5AFrozenTeacherError",
    "assert_teacher_frozen",
    "sha256_file",
]
