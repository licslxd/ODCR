# -*- coding: utf-8 -*-
"""
训练期诊断相关环境变量与 payload 解析（无 torch 依赖）。

供 manifest、config fingerprint、RUN_SNAPSHOT / config_resolved 与 train_diagnostics 共用，
保证父进程 manifest 与子进程行为使用同一套解析规则。

指纹分层（勿再混写）：

- **training_semantic_fingerprint** / **generation_semantic_fingerprint**（见 ``odcr_core.config_resolver``）：
  训练主身份与生成/评测语义分离
  （``training_payload``、``hardware_profile``、decode 预设摘要、``ddp_world_size``、eval batch、
  ``train_label_max_length`` 等）。**不包含** finite_check / grad_topk 等诊断开关。

- **runtime_diagnostics_fingerprint**：仅哈希 :func:`runtime_diagnostics_fingerprint_source` 的返回值
  （``ODCR_FINITE_CHECK_MODE``、``ODCR_GRAD_TOPK`` 及其解析结果）。与训练数学目标解耦。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def parse_odcr_finite_check_mode() -> tuple[str, Optional[str]]:
    """
    解析 ``ODCR_FINITE_CHECK_MODE``。

    返回 ``(mode, warn)``：mode ∈ off | loss_only | full_word_dist；
    非法取值时 mode 为 loss_only，warn 为说明字符串。
    """
    raw = (os.environ.get("ODCR_FINITE_CHECK_MODE", "") or "").strip().lower()
    if not raw:
        return "loss_only", None
    if raw in ("off", "none", "0", "false", "no"):
        return "off", None
    if raw in ("loss_only", "loss"):
        return "loss_only", None
    if raw in ("full", "full_word_dist", "word_dist"):
        return "full_word_dist", None
    return "loss_only", f"未知 ODCR_FINITE_CHECK_MODE={raw!r}，已按 loss_only 处理"


def odcr_grad_topk() -> int:
    try:
        return max(0, int(os.environ.get("ODCR_GRAD_TOPK", "0")))
    except ValueError:
        return 0


def ddp_find_unused_requested_from_training_payload_json(payload_json: str) -> Optional[bool]:
    """
    从 ``ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON`` 解析 ``training_row.ddp_find_unused_parameters``。

    仅当 ``training_row`` 中**显式存在**该键时返回 bool；否则返回 None（不伪造请求意图）。
    """
    raw = (payload_json or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    row = obj.get("training_row")
    if not isinstance(row, dict):
        return None
    if "ddp_find_unused_parameters" not in row:
        return None
    return bool(row["ddp_find_unused_parameters"])


def ddp_find_unused_from_training_payload_json(payload_json: str) -> bool:
    """缺键或非法 payload 时视为 False。"""
    v = ddp_find_unused_requested_from_training_payload_json(payload_json)
    return False if v is None else v


def runtime_diagnostics_fingerprint_source() -> Dict[str, Any]:
    """
    进入 **runtime_diagnostics_fingerprint** 的唯一结构化输入（与 manifest 中诊断块字段对齐）。
    """
    mode, warn = parse_odcr_finite_check_mode()
    raw_fc = (os.environ.get("ODCR_FINITE_CHECK_MODE", "") or "").strip() or None
    raw_gt = (os.environ.get("ODCR_GRAD_TOPK", "") or "").strip() or None
    gt = odcr_grad_topk()
    return {
        "finite_check_mode": mode,
        "finite_check_mode_env": raw_fc,
        "finite_check_mode_parse_warning": warn,
        "odcr_grad_topk": int(gt),
        "odcr_grad_topk_env": raw_gt,
    }


def training_diagnostics_snapshot(
    *,
    diagnostics_scope: str,
    effective_training_payload_json: str = "",
    ddp_find_unused_parameters_requested: Optional[bool] = None,
    ddp_find_unused_parameters_effective: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    写入 manifest / 快照的稳定结构。

    - ``diagnostics_scope``：``parent`` = 父进程 torchrun 前（manifest / 父指纹）；
      ``child`` = 子进程 FinalTrainingConfig / DDP 真值侧。
    - ``ddp_find_unused_parameters_*``：仅写入非 None 的键；requested 缺省时会尝试从 payload 解析
      （仅当 ``training_row`` 含该键时才有 requested）。
    """
    mode, warn = parse_odcr_finite_check_mode()
    raw_fc = (os.environ.get("ODCR_FINITE_CHECK_MODE", "") or "").strip() or None
    raw_gt = (os.environ.get("ODCR_GRAD_TOPK", "") or "").strip() or None
    gt = odcr_grad_topk()
    if ddp_find_unused_parameters_requested is None:
        ddp_find_unused_parameters_requested = ddp_find_unused_requested_from_training_payload_json(
            effective_training_payload_json
        )
    out: Dict[str, Any] = {
        "diagnostics_scope": diagnostics_scope,
        "finite_check_mode": mode,
        "finite_check_mode_env": raw_fc,
        "finite_check_mode_parse_warning": warn,
        "odcr_grad_topk": gt,
        "odcr_grad_topk_env": raw_gt,
    }
    if ddp_find_unused_parameters_requested is not None:
        out["ddp_find_unused_parameters_requested"] = bool(ddp_find_unused_parameters_requested)
    if ddp_find_unused_parameters_effective is not None:
        out["ddp_find_unused_parameters_effective"] = bool(ddp_find_unused_parameters_effective)
    return out
