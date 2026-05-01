# -*- coding: utf-8 -*-
"""
训练期诊断：步级日志、梯度范数、checkpoint 记录、DDP 心跳与调试、数值与 batch 校验。
依赖 train_logging 的路由字段 odcr_route（detail / summary / both）。
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from train_logging import ROUTE_DETAIL, ROUTE_SUMMARY, log_route_extra
from odcr_core.training_diagnostics import odcr_grad_topk, parse_odcr_finite_check_mode

_LOGGER = logging.getLogger("odcr")


def odcr_cuda_bf16_autocast_enabled() -> bool:
    """
    CUDA 且 ``torch.cuda.is_bf16_supported()`` 为真时默认启用 bf16 混合精度前向。
    由 runtime preset 注入 ``ODCR_RUNTIME_PRECISION_MODE`` 控制（bf16/fp16/fp32）；
    Step3 由 runners 注入 ``ODCR_TRAINING_STAGE=step3`` 强制关闭 bf16。
    """
    if (os.environ.get("ODCR_TRAINING_STAGE") or "").strip().lower() == "step3":
        return False
    v = os.environ.get("ODCR_RUNTIME_PRECISION_MODE", "bf16").strip().lower()
    if v in ("fp32", "fp16"):
        return False
    if not torch.cuda.is_available():
        return False
    return bool(torch.cuda.is_bf16_supported())


@contextmanager
def odcr_cuda_bf16_autocast():
    """训练 / 验证 / 推理前向的统一 bf16 autocast（不支持或已关闭时等价于禁用）。"""
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=odcr_cuda_bf16_autocast_enabled(),
    ):
        yield


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, "").strip())
        return max(1, v)
    except (TypeError, ValueError):
        return default


def odcr_log_step_interval() -> int:
    return _env_int("ODCR_LOG_STEP_INTERVAL", 50)


def odcr_log_grad_interval() -> int:
    v = os.environ.get("ODCR_LOG_GRAD_INTERVAL", "").strip()
    if v:
        return _env_int("ODCR_LOG_GRAD_INTERVAL", 50)
    return odcr_log_step_interval()


def odcr_grad_warn_norm() -> float:
    try:
        return float(os.environ.get("ODCR_GRAD_WARN_NORM", "100"))
    except ValueError:
        return 100.0


def odcr_debug_grad_diff() -> bool:
    v = os.environ.get("ODCR_DEBUG_GRAD_DIFF", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def odcr_ddp_epoch_end_barrier() -> bool:
    """
    是否在每 epoch 末调用 ``dist.barrier()``。

    默认 **关闭**，避免无业务语义的全局屏障开销；仅排查跨 rank 节奏/IO 时开启。
    评测收尾等「rank0 聚合后其余 rank 须等待」的路径仍保留必要 barrier。
    """
    v = (os.environ.get("ODCR_DDP_EPOCH_END_BARRIER") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def odcr_debug_grad_diff_interval() -> int:
    return _env_int("ODCR_DEBUG_GRAD_DIFF_INTERVAL", 200)


def odcr_log_step_loss_parts() -> bool:
    v = os.environ.get("ODCR_LOG_STEP_LOSS_PARTS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def collect_distributed_env_for_meta() -> Dict[str, str]:
    """NCCL / torch.distributed / 主节点等与多卡相关的环境变量（键有序）。"""
    keys_pick = frozenset(
        {
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCH_DISTRIBUTED_BACKEND",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "TORCH_NCCL_BLOCKING_WAIT",
        }
    )
    pref = ("NCCL_", "TORCH_DISTRIBUTED", "GLOO_", "TPU_", "MPI_", "AWS_", "ETH_", "IBV_")
    out: Dict[str, str] = {}
    for k, v in os.environ.items():
        if k in keys_pick or k.startswith(pref):
            out[k] = v
    return dict(sorted(out.items()))


@contextmanager
def odcr_timing_phase(
    logger: Optional[logging.Logger],
    phase: str,
    *,
    route: str,
    rank0_only: bool = True,
    rank: int = 0,
):
    """关键阶段耗时：入口/出口各一行。"""
    lg = logger or _LOGGER
    t0 = time.perf_counter()
    if (not rank0_only) or rank == 0:
        try:
            lg.info("[Timing] %s start", phase, extra=log_route_extra(lg, route))
        except Exception:
            lg.info("[Timing] %s start", phase)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if (not rank0_only) or rank == 0:
            try:
                lg.info("[Timing] %s end elapsed_s=%.3f", phase, dt, extra=log_route_extra(lg, route))
            except Exception:
                lg.info("[Timing] %s end elapsed_s=%.3f", phase, dt)


def grad_norm_total(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms: List[torch.Tensor] = []
    for p in parameters:
        if p.grad is not None and p.grad.is_sparse is False:
            norms.append(torch.norm(p.grad.detach(), 2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def grad_topk_param_norms(model: nn.Module, k: int) -> List[Tuple[str, float]]:
    """
    在设备上计算各参数 grad L2 norm，用 ``torch.topk`` 取前 k 个，最后只对 top-k 做少量 CPU 同步。
    避免「每个参数一次 .item()」带来的同步风暴。
    """
    if k <= 0:
        return []
    names: List[str] = []
    norms: List[torch.Tensor] = []
    for name, p in model.named_parameters():
        if p.grad is None or p.grad.is_sparse:
            continue
        names.append(name)
        norms.append(torch.norm(p.grad.detach(), 2))
    if not norms:
        return []
    stacked = torch.stack(norms)
    kk = min(k, int(stacked.numel()))
    vals, indices = torch.topk(stacked, kk)
    idx_list = indices.detach().cpu().tolist()
    val_list = vals.detach().cpu().tolist()
    return [(names[int(i)], float(v)) for i, v in zip(idx_list, val_list)]


def log_grad_monitor(
    logger: Optional[logging.Logger],
    model: nn.Module,
    *,
    global_step: int,
    epoch: int,
    route_detail: str,
    warn_norm: Optional[float] = None,
    grad_norm_pre_clip: Optional[float] = None,
    grad_norm_post_clip: Optional[float] = None,
    current_accum: Optional[int] = None,
    is_tail_window: Optional[bool] = None,
    skip_param_topk: bool = False,
) -> None:
    if logger is None:
        return
    warn_norm = odcr_grad_warn_norm() if warn_norm is None else warn_norm
    params = list(model.parameters())
    if grad_norm_pre_clip is not None and grad_norm_post_clip is not None:
        pre_v = float(grad_norm_pre_clip)
        post_v = float(grad_norm_post_clip)
        total = post_v
    else:
        pre_v = grad_norm_total(params)
        post_v = pre_v
        total = pre_v
    topk = odcr_grad_topk()
    parts = [
        f"global_step={global_step}",
        f"epoch={epoch}",
        f"grad_norm_pre_clip={pre_v:.6g}",
        f"grad_norm_post_clip={post_v:.6g}",
    ]
    if current_accum is not None:
        parts.append(f"current_accum={int(current_accum)}")
    if is_tail_window is not None:
        parts.append(f"is_tail_window={bool(is_tail_window)}")
    if not skip_param_topk and topk > 0:
        tops = grad_topk_param_norms(model, topk)
        if tops:
            parts.append("top_params=" + json.dumps(tops, ensure_ascii=False))
    msg = "[Grad] " + " ".join(parts)
    bad = (not math.isfinite(total)) or (total > warn_norm)
    lvl = logging.WARNING if bad else logging.INFO
    if not math.isfinite(total):
        msg += " (non-finite grad_norm)"
    try:
        logger.log(lvl, msg, extra=log_route_extra(logger, route_detail))
    except Exception:
        logger.log(lvl, msg)


def odcr_save_checkpoint(
    state_dict: Any,
    path: str,
    *,
    epoch: int,
    reason: str,
    logger: Optional[logging.Logger] = None,
    route_detail: str = ROUTE_DETAIL,
    is_last: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    p = os.path.abspath(os.path.expanduser(path))
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(state_dict, p)
    if not os.path.isfile(p):
        raise RuntimeError(f"[Checkpoint] save failed: file missing after torch.save: {p}")
    sz = os.path.getsize(p)
    if metadata is not None:
        meta_path = os.path.splitext(p)[0] + ".meta.json"
        try:
            payload: Dict[str, Any] = {
                "epoch": int(epoch),
                "reason": str(reason),
                "checkpoint_path": p,
                "checkpoint_size_bytes": int(sz),
            }
            payload.update(metadata)
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(payload, mf, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
    if is_last:
        msg = f"[Checkpoint] saved last checkpoint to {p} size={sz}"
    else:
        msg = f"[Checkpoint] saved checkpoint to {p} size={sz} epoch={int(epoch)} reason={reason}"
    if logger is not None:
        try:
            logger.info(msg, extra=log_route_extra(logger, route_detail))
        except Exception:
            logger.info(msg)
    else:
        print(msg, flush=True)


def ddp_heartbeat(
    logger: Optional[logging.Logger],
    tag: str,
    *,
    rank: int,
    epoch: Optional[int] = None,
    extra_kv: Optional[str] = None,
    route_summary: str = ROUTE_SUMMARY,
) -> None:
    if logger is None or rank != 0:
        return
    tail = f" epoch={epoch}" if epoch is not None else ""
    x = f" {extra_kv}" if extra_kv else ""
    try:
        logger.info("[DDP] heartbeat %s%s%s", tag, tail, x, extra=log_route_extra(logger, route_summary))
    except Exception:
        logger.info("[DDP] heartbeat %s%s%s", tag, tail, x)


def maybe_log_grad_norm_diff_ddp(
    model: nn.Module,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    global_step: int,
    logger: Optional[logging.Logger],
    route_detail: str,
) -> None:
    if (
        world_size != 2
        or not odcr_debug_grad_diff()
        or global_step <= 0
        or global_step % odcr_debug_grad_diff_interval() != 0
    ):
        return
    if not dist.is_available() or not dist.is_initialized():
        return
    name_sel = ""
    gn = torch.zeros(1, device=device, dtype=torch.float64)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name_sel = name
        if p.grad is not None:
            gn[0] = float(p.grad.detach().float().norm(2).item())
        break
    if not name_sel:
        return
    bufs = [torch.zeros_like(gn) for _ in range(world_size)]
    dist.all_gather(bufs, gn)
    if rank == 0 and logger is not None:
        vals = [float(b[0].item()) for b in bufs]
        diff = abs(vals[0] - vals[1]) if len(vals) >= 2 else 0.0
        try:
            logger.info(
                "[DDP] grad_norm_diff_debug param=%s rank_norms=%s abs_diff=%.6g global_step=%d",
                name_sel,
                vals,
                diff,
                global_step,
                extra=log_route_extra(logger, route_detail),
            )
        except Exception:
            logger.info(
                "[DDP] grad_norm_diff_debug param=%s rank_norms=%s abs_diff=%.6g global_step=%d",
                name_sel,
                vals,
                diff,
                global_step,
            )


def run_training_finite_checks(
    mode: str,
    loss: torch.Tensor,
    word_dist: torch.Tensor,
    logger: Optional[logging.Logger],
    *,
    global_step: int,
    epoch: int,
    route_detail: str = ROUTE_DETAIL,
) -> None:
    """按模式执行步级 finite 检查；主线默认 ``loss_only``（不扫整图 word_dist）。"""
    if mode == "off":
        return
    check_finite_loss(loss, logger, global_step=global_step, epoch=epoch, route_detail=route_detail)
    if mode == "full_word_dist":
        check_finite_tensor(
            word_dist,
            logger,
            name="word_dist",
            global_step=global_step,
            epoch=epoch,
            route_detail=route_detail,
        )


def check_finite_loss(
    loss: torch.Tensor,
    logger: Optional[logging.Logger],
    *,
    global_step: int,
    epoch: int,
    tag: str = "loss",
    route_detail: str = ROUTE_DETAIL,
) -> bool:
    if logger is None:
        return bool(torch.isfinite(loss).all().item())
    ok = bool(torch.isfinite(loss).all().item())
    if not ok:
        try:
            logger.warning(
                "[Data] non-finite %s at global_step=%d epoch=%d value=%s",
                tag,
                global_step,
                epoch,
                loss.detach(),
                extra=log_route_extra(logger, route_detail),
            )
        except Exception:
            logger.warning("[Data] non-finite %s at global_step=%d epoch=%d", tag, global_step, epoch)
    return ok


def check_finite_tensor(
    t: torch.Tensor,
    logger: Optional[logging.Logger],
    *,
    name: str,
    global_step: int,
    epoch: int,
    route_detail: str = ROUTE_DETAIL,
) -> bool:
    if logger is None:
        return bool(torch.isfinite(t).all().item())
    ok = bool(torch.isfinite(t).all().item())
    if not ok:
        try:
            logger.warning(
                "[Data] non-finite tensor %s at global_step=%d epoch=%d",
                name,
                global_step,
                epoch,
                extra=log_route_extra(logger, route_detail),
            )
        except Exception:
            logger.warning("[Data] non-finite tensor %s at global_step=%d epoch=%d", name, global_step, epoch)
    return ok


def warn_empty_batch(
    logger: Optional[logging.Logger],
    *,
    global_step: int,
    epoch: int,
    bsz: int,
    route_detail: str = ROUTE_DETAIL,
) -> None:
    if logger is None or bsz > 0:
        return
    try:
        logger.warning(
            "[Data] empty batch global_step=%d epoch=%d",
            global_step,
            epoch,
            extra=log_route_extra(logger, route_detail),
        )
    except Exception:
        logger.warning("[Data] empty batch global_step=%d epoch=%d", global_step, epoch)


def log_step_sample(
    logger: Optional[logging.Logger],
    *,
    global_step: int,
    epoch: int,
    lr: float,
    train_loss_batch: float,
    route_detail: str = ROUTE_DETAIL,
    extra: Optional[Dict[str, Any]] = None,
    as_json: bool = False,
) -> None:
    if logger is None:
        return
    if as_json or os.environ.get("ODCR_LOG_STEP_JSON", "").strip().lower() in ("1", "true", "yes", "on"):
        body: Dict[str, Any] = {
            "global_step": global_step,
            "epoch": epoch,
            "lr": lr,
            "train_loss_batch": train_loss_batch,
        }
        if extra:
            body.update(extra)
        msg = "[Step] " + json.dumps(body, ensure_ascii=False)
    else:
        msg = (
            f"[Step] global_step={global_step} epoch={epoch} lr={lr:.6g} train_loss_batch={train_loss_batch:.6g}"
        )
        if extra:
            msg += " " + json.dumps(extra, ensure_ascii=False)
    try:
        logger.info(msg, extra=log_route_extra(logger, route_detail))
    except Exception:
        logger.info(msg)


def log_training_crash(logger: Optional[logging.Logger], exc: BaseException, route_detail: str = ROUTE_DETAIL) -> None:
    if logger is None:
        return
    tb = traceback.format_exc()
    try:
        logger.error(
            "[Train] 训练中断: %s\n%s",
            exc,
            tb,
            extra=log_route_extra(logger, route_detail),
        )
    except Exception:
        logger.error("[Train] 训练中断: %s\n%s", exc, tb)


def log_bf16_amp_note(logger: Optional[logging.Logger], use_bf16_autocast: bool, has_grad_scaler: bool) -> None:
    if logger is None:
        return
    if use_bf16_autocast and not has_grad_scaler:
        msg = "[Train] bf16 autocast=ON，未使用 GradScaler；AMP overflow 计数不可用（仅记录说明）。"
    elif has_grad_scaler:
        msg = "[Train] 使用 GradScaler 时可观察 scaler.get_scale() / overflow（当前步进路径若扩展可接入）。"
    else:
        return
    try:
        logger.info(msg, extra=log_route_extra(logger, ROUTE_SUMMARY))
    except Exception:
        logger.info(msg)
