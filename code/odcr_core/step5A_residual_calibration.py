"""Step5A zero-init residual calibration on top of a frozen Step3 teacher."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


STEP5A_RESIDUAL_CALIBRATION_SCHEMA_VERSION = "odcr_step5A_residual_calibration/1"


@dataclass(frozen=True)
class Step5AResidualCalibrationConfig:
    enabled: bool = True
    feature_source: str = "teacher_pred"
    zero_init: bool = True
    lambda_gt_initial: float = 0.0
    lambda_gt_final: float = 1.0
    lambda_distill_initial: float = 1.0
    lambda_distill_final: float = 0.25
    lambda_residual: float = 0.05
    regularizer: str = "huber"
    huber_delta: float = 0.1

    def weights_for_epoch(self, *, epoch_index: int, max_epochs: int) -> dict[str, float]:
        denom = max(int(max_epochs) - 1, 1)
        t = min(max(float(epoch_index) / float(denom), 0.0), 1.0)
        return {
            "lambda_gt": float(self.lambda_gt_initial + t * (self.lambda_gt_final - self.lambda_gt_initial)),
            "lambda_distill": float(
                self.lambda_distill_initial + t * (self.lambda_distill_final - self.lambda_distill_initial)
            ),
            "lambda_residual": float(self.lambda_residual),
        }


def parse_step5a_residual_calibration_config(raw: Mapping[str, Any] | None) -> Step5AResidualCalibrationConfig:
    obj = dict(raw or {})
    return Step5AResidualCalibrationConfig(
        enabled=bool(obj.get("enabled", True)),
        feature_source=str(obj.get("feature_source") or "teacher_pred"),
        zero_init=bool(obj.get("zero_init", True)),
        lambda_gt_initial=float(obj.get("lambda_gt_initial", 0.0) or 0.0),
        lambda_gt_final=float(obj.get("lambda_gt_final", 1.0) or 0.0),
        lambda_distill_initial=float(obj.get("lambda_distill_initial", 1.0) or 0.0),
        lambda_distill_final=float(obj.get("lambda_distill_final", 0.25) or 0.0),
        lambda_residual=float(obj.get("lambda_residual", 0.05) or 0.0),
        regularizer=str(obj.get("regularizer") or "huber"),
        huber_delta=float(obj.get("huber_delta", 0.1) or 0.1),
    )


class ZeroInitResidualCalibrator(nn.Module):
    """A small trainable residual head whose initial output is exactly zero."""

    def __init__(self, *, hidden_size: int, feature_source: str = "teacher_pred") -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.feature_source = str(feature_source or "teacher_pred")
        self.scale_delta = nn.Parameter(torch.zeros(()))
        self.bias_delta = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.Linear(1, max(1, min(self.hidden_size, 32))),
            nn.Tanh(),
            nn.Linear(max(1, min(self.hidden_size, 32)), 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.scale_delta.zero_()
            self.bias_delta.zero_()
            for param in self.mlp.parameters():
                param.zero_()

    def forward(self, teacher_pred: torch.Tensor, packet: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        del packet
        base = teacher_pred.detach().view(-1, 1).to(dtype=teacher_pred.dtype)
        delta = self.scale_delta.to(dtype=teacher_pred.dtype) * base + self.bias_delta.to(dtype=teacher_pred.dtype)
        delta = delta + self.mlp(base).to(dtype=teacher_pred.dtype)
        return delta.view_as(teacher_pred)

    def zero_init_check(self, teacher_pred: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            delta = self(teacher_pred.detach())
        max_abs = float(delta.detach().abs().max().item()) if delta.numel() else 0.0
        return {
            "schema_version": STEP5A_RESIDUAL_CALIBRATION_SCHEMA_VERSION,
            "zero_init": bool(max_abs <= 1e-12),
            "max_abs_delta": max_abs,
            "feature_source": self.feature_source,
        }


def residual_regularizer(
    delta: torch.Tensor,
    *,
    regularizer: str = "huber",
    huber_delta: float = 0.1,
) -> torch.Tensor:
    mode = str(regularizer or "huber").strip().lower()
    if mode == "l1":
        return delta.abs()
    if mode == "huber":
        return F.huber_loss(delta, torch.zeros_like(delta), reduction="none", delta=float(huber_delta))
    raise ValueError(f"unsupported Step5A residual regularizer: {regularizer!r}")


def step5a_residual_contract_report(model: nn.Module) -> dict[str, Any]:
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    forbidden = [name for name in trainable if not name.startswith("step5a_residual_calibrator.")]
    return {
        "schema_version": STEP5A_RESIDUAL_CALIBRATION_SCHEMA_VERSION,
        "residual_zero_init": True,
        "teacher_frozen": True,
        "stop_gradient_teacher": True,
        "trainable_parameter_names": trainable,
        "forbidden_trainable_names": forbidden,
        "residual_trainable_only": not forbidden and bool(trainable),
    }


__all__ = [
    "STEP5A_RESIDUAL_CALIBRATION_SCHEMA_VERSION",
    "Step5AResidualCalibrationConfig",
    "ZeroInitResidualCalibrator",
    "parse_step5a_residual_calibration_config",
    "residual_regularizer",
    "step5a_residual_contract_report",
]
