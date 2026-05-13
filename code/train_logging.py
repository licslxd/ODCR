# -*- coding: utf-8 -*-
"""
训练/评估统一日志：标准 logging、按 run 分文件、DDP 仅 rank0 写文件。
环境变量：
  ODCR_LOG_DIR      日志目录（mainline 由 odcr.py 显式传入 runs/.../meta/full.log）
  ODCR_CONSOLE_LEVEL 控制台级别，默认 INFO（非 rank0 恒为 WARNING）
  ODCR_FILE_LEVEL    文件级别，默认 DEBUG（仅 rank0 有 FileHandler）
  ODCR_LOG_CONSOLE=1 rank0 在写日志文件时仍向 stdout 镜像（默认关闭：仅 FileHandler 写文件，避免与 shell 重定向/tee 双写）
  ODCR_MIRROR_LOG   已退役；不会创建 fallback mirror log
  ODCR_LOG_PRETTY=0  关闭多行缩进 JSON（默认开启 RUN_META / RUN_CONFIG 多行缩进，便于阅读）
  ODCR_LOG_STRUCTURED_CONSOLE=1  结构化块（RUN_*）同时打到控制台；默认仅写入日志文件，避免与 FileHandler
                                并存且 stderr 被 tee/重定向到同一文件时出现重复行。
  ODCR_EVAL_SUMMARY=0           关闭 eval 自动汇总（默认开启）：任务级写入 runs/task{T}/vN/meta/eval_registry.*；跨任务全局仅写入 runs/global/vN/meta/eval_registry_all.*（见 path_layout 边界说明）。
  ODCR_ITERATION_META_DIR       由 odcr 注入，指向 runs/task{T}/vN/meta/；表头字段见 _EVAL_REGISTRY_CSV_FIELDS。
  ODCR_EVAL_SUMMARY_GLOBAL_DIR  可选；覆盖全局汇总目录（默认同 path_layout.get_global_meta_dir：仅 eval_registry_all.* 等跨任务元数据）。
  ODCR_STEP3_ALL_SHELL_LOG  （仅 shell）旧 Step3 批量脚本 --all 前台 tee 汇总路径；未设置时默认为 runs/global/vN/meta/shell_logs/step3_optimized_all_<秒级时间戳>.log
  ODCR_LOG_SILENT_STDIO_WARN=1  关闭 setup_train_logging 对 stdout/stderr 与 --log_file 同路径的告警
  ODCR_DUAL_TRAIN_LOG=1       双文件：--log_file 为 full.log（细粒度：Step/Grad/Checkpoint/Timing/数据告警），
                              另写同目录 console.log（RUN_*、epoch 块、DDP 心跳、性能汇总）。可用 ODCR_SUMMARY_LOG 覆盖摘要路径。
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import socket
import string
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from odcr_core import path_layout
from paths_config import get_odcr_root

LOGGER_NAME = "odcr"
RUN_END_LINE = "========== RUN END =========="

# 日志路由：双文件模式下 full.log 接收 detail+summary+both，console.log 仅 summary+both；单文件无 Filter，等价于 both。
ROUTE_DETAIL = "detail"
ROUTE_SUMMARY = "summary"
ROUTE_BOTH = "both"


class _OdcrRouteFilter(logging.Filter):
    """仅放行 odcr_route 属于 allowed 或 both 的记录。"""

    def __init__(self, allowed: FrozenSet[str]) -> None:
        super().__init__()
        self._allowed = allowed

    def filter(self, record: logging.LogRecord) -> bool:
        r = getattr(record, "odcr_route", ROUTE_BOTH)
        if r == ROUTE_BOTH:
            return True
        return r in self._allowed


def _dual_log_enabled() -> bool:
    v = os.environ.get("ODCR_DUAL_TRAIN_LOG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _summary_log_path(detail_path: str) -> str:
    ex = os.environ.get("ODCR_SUMMARY_LOG", "").strip()
    if ex:
        return os.path.abspath(os.path.expanduser(ex))
    return os.path.join(os.path.dirname(os.path.abspath(detail_path)), "console.log")


def _plain_route_visible(handler: logging.Handler, route: str) -> bool:
    allowed = getattr(handler, "_odcr_routes_allowed", None)
    if allowed is None:
        return True
    if route == ROUTE_BOTH:
        return True
    return route in allowed


class _StreamSuppressFileOnlyFilter(logging.Filter):
    """带 odcr_file_only 的记录不输出到 StreamHandler，仍由 FileHandler 写入（若存在）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "odcr_file_only", False)


def generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{rnd}"


def _log_dir() -> str:
    return os.path.abspath(
        os.path.expanduser(os.environ.get("ODCR_LOG_DIR") or os.path.join(get_odcr_root(), "runs", "internal"))
    )


def create_run_paths(
    task_idx: int,
    explicit_log_file: Optional[str] = None,
) -> Tuple[str, str]:
    """
    返回 (log_path, run_id)。
    explicit_log_file 为有效路径且不是占位符 log.out 时，直接使用该路径；否则生成
    runs/internal/task{task_idx}/{run_id}/meta/full.log
    """
    run_id = generate_run_id()
    ex = (explicit_log_file or "").strip()
    if ex and ex != "log.out":
        return os.path.abspath(os.path.expanduser(ex)), run_id
    base = _log_dir()
    path = os.path.join(base, f"task{task_idx}", run_id, "meta", "full.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path, run_id


def _parse_level(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)


def _pretty_log_enabled() -> bool:
    """默认 True（多行缩进）；设 ODCR_LOG_PRETTY=0/false/no/off 为单行 JSON。"""
    v = os.environ.get("ODCR_LOG_PRETTY", "").strip().lower()
    if not v:
        return True
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return True


def _structured_console_enabled() -> bool:
    """为 True 时结构化 RUN_* 也镜像到控制台（默认 False）。"""
    v = os.environ.get("ODCR_LOG_STRUCTURED_CONSOLE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _console_mirror_enabled() -> bool:
    """rank0 写日志文件时是否仍附加 StreamHandler（默认 False，仅文件写入）。"""
    v = os.environ.get("ODCR_LOG_CONSOLE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def logger_has_file_handler(logger: Optional[logging.Logger]) -> bool:
    """用于判断当前是否由 FileHandler 写文件（避免 print 与 logging 双份）。"""
    if logger is None:
        return False
    return any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def _structured_log_extra(logger: logging.Logger) -> Dict[str, Any]:
    """
    若同时存在 FileHandler 与 StreamHandler 且未要求结构化块上控制台，则标记 odcr_file_only，
    由 StreamHandler 过滤器抑制，避免 tee/2>&1 与文件为同一路径时大块 JSON 重复。
    """
    if _structured_console_enabled():
        return {}
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if has_file and has_stream:
        return {"odcr_file_only": True}
    return {}


def log_route_extra(logger: logging.Logger, route: str) -> Dict[str, Any]:
    """合并结构化控制台抑制与路由字段（供 train_diagnostics 等复用）。"""
    e = _structured_log_extra(logger)
    e["odcr_route"] = route
    return e


def _realpath_resolved(path: str) -> str:
    try:
        return os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    except OSError:
        return os.path.abspath(os.path.expanduser(path))


def _warn_if_stdio_points_to_log_file(logger: logging.Logger, log_path: str) -> None:
    """若 stdout/stderr 已重定向到与 log_path 同一文件，则告警（易导致双 fd 写 full.log 乱序）。"""
    v = os.environ.get("ODCR_LOG_SILENT_STDIO_WARN", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return
    target = _realpath_resolved(log_path)
    for stream_name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        try:
            if stream.isatty():
                continue
            nm = getattr(stream, "name", None)
            if not isinstance(nm, str) or not nm or nm.startswith("<"):
                continue
            if _realpath_resolved(nm) == target:
                logger.warning(
                    "[ODCR] %s 与 --log_file 解析为同一路径 (%s)，可能导致重复写入或乱序；"
                    "请避免将终端重定向到 full.log，或关闭 ODCR_LOG_CONSOLE。",
                    stream_name,
                    target,
                )
        except Exception:
            pass


def _json_safe_sorted_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for k in sorted(data.keys()):
        v = data[k]
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = repr(v)
    return safe


def setup_train_logging(
    *,
    log_file: Optional[str],
    task_idx: int,
    rank: int = 0,
    world_size: int = 1,
    run_id: Optional[str] = None,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
) -> Dict[str, Any]:
    """
    配置名为 odcr 的 logger：rank0 默认仅 FileHandler 写文件（不重定向终端也可单份）；
    设 ODCR_LOG_CONSOLE=1 时 rank0 额外附加 StreamHandler；非 rank0 仅 StreamHandler（WARNING+）。
    """
    if run_id is None:
        run_id = generate_run_id()
    console_level = _parse_level(console_level or os.environ.get("ODCR_CONSOLE_LEVEL", "INFO"))
    file_level = _parse_level(file_level or os.environ.get("ODCR_FILE_LEVEL", "DEBUG"))

    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    context_fmt = logging.Formatter(
        (
            "%(asctime)s - %(levelname)s - "
            f"rank={rank} local_rank={os.environ.get('LOCAL_RANK', '') or rank} "
            f"pid={os.getpid()} hostname={socket.gethostname()} run_id={run_id} - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = log_file
    mirror = _console_mirror_enabled()
    # rank0 且有日志路径时默认不挂 StreamHandler，避免与 FileHandler 双写同一文件（含 shell 重定向）
    want_stream = (rank != 0) or (not (rank == 0 and log_path)) or mirror

    stream_handler: Optional[logging.StreamHandler] = None
    if want_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        if rank == 0:
            sh.setLevel(console_level)
        else:
            sh.setLevel(logging.WARNING)
        if rank == 0 and log_path:
            sh.addFilter(_StreamSuppressFileOnlyFilter())
        logger.addHandler(sh)
        stream_handler = sh

    summary_log_path: Optional[str] = None
    dual_log = False
    if rank == 0 and log_path:
        d = os.path.dirname(os.path.abspath(log_path))
        if d:
            os.makedirs(d, exist_ok=True)
        use_dual = _dual_log_enabled()
        sp: Optional[str] = _summary_log_path(log_path) if use_dual else None
        if use_dual and sp and _realpath_resolved(sp) != _realpath_resolved(log_path):
            dual_log = True
            summary_log_path = sp
            sd = os.path.dirname(os.path.abspath(sp))
            if sd:
                os.makedirs(sd, exist_ok=True)
            allow_d: FrozenSet[str] = frozenset({ROUTE_DETAIL, ROUTE_SUMMARY, ROUTE_BOTH})
            allow_s: FrozenSet[str] = frozenset({ROUTE_SUMMARY, ROUTE_BOTH})
            fh_d = logging.FileHandler(log_path, encoding="utf-8")
            fh_d.setLevel(file_level)
            fh_d.setFormatter(fmt)
            fh_d._odcr_routes_allowed = allow_d  # type: ignore[attr-defined]
            fh_d.addFilter(_OdcrRouteFilter(allow_d))
            logger.addHandler(fh_d)
            fh_s = logging.FileHandler(sp, encoding="utf-8")
            fh_s.setLevel(file_level)
            fh_s.setFormatter(fmt)
            fh_s._odcr_routes_allowed = allow_s  # type: ignore[attr-defined]
            fh_s.addFilter(_OdcrRouteFilter(allow_s))
            logger.addHandler(fh_s)
            if stream_handler is not None:
                stream_handler.addFilter(_OdcrRouteFilter(allow_s))
            _warn_if_stdio_points_to_log_file(logger, log_path)
        else:
            if use_dual and sp and _realpath_resolved(sp) == _realpath_resolved(log_path):
                logger.warning(
                    "[ODCR] ODCR_DUAL_TRAIN_LOG=1 但摘要路径与主日志相同，已回退为单文件写入。"
                )
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(file_level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            _warn_if_stdio_points_to_log_file(logger, log_path)
        errors_path = os.path.join(os.path.dirname(os.path.abspath(log_path)), "errors.log")
        fh_e = logging.FileHandler(errors_path, encoding="utf-8")
        fh_e.setLevel(logging.WARNING)
        fh_e.setFormatter(context_fmt)
        logger.addHandler(fh_e)

    return {
        "logger": logger,
        "run_id": run_id,
        "log_path": log_path,
        "rank": rank,
        "world_size": world_size,
        "summary_log_path": summary_log_path,
        "dual_log": dual_log,
    }


def log_run_header(logger: logging.Logger, meta: Dict[str, Any]) -> None:
    """RUN_META：默认多行缩进 JSON（键有序）；ODCR_LOG_PRETTY=0 时为单行。"""
    safe = _json_safe_sorted_dict(meta)
    pretty = _pretty_log_enabled()
    payload = json.dumps(safe, ensure_ascii=False, indent=2 if pretty else None)
    extra = log_route_extra(logger, ROUTE_DETAIL)
    if pretty:
        logger.info("RUN_META\n%s", payload, extra=extra)
    else:
        logger.info("RUN_META %s", payload, extra=extra)


def log_config_snapshot(
    logger: logging.Logger,
    config: Dict[str, Any],
    *,
    exclude_keys: Tuple[str, ...] = ("logger",),
) -> None:
    """RUN_CONFIG：与 RUN_META 相同展示规则；不可 JSON 序列化的值用 repr。"""
    trimmed = {k: v for k, v in config.items() if k not in exclude_keys}
    safe = _json_safe_sorted_dict(trimmed)
    pretty = _pretty_log_enabled()
    payload = json.dumps(safe, ensure_ascii=False, indent=2 if pretty else None)
    extra = log_route_extra(logger, ROUTE_DETAIL)
    if pretty:
        logger.info("RUN_CONFIG\n%s", payload, extra=extra)
    else:
        logger.info("RUN_CONFIG %s", payload, extra=extra)


def log_run_snapshot(
    logger: logging.Logger,
    meta: Dict[str, Any],
    config: Dict[str, Any],
    *,
    exclude_keys: Tuple[str, ...] = ("logger",),
) -> None:
    """单次 RUN_SNAPSHOT：meta（运行/CLI 上下文）+ config（完整训练配置），避免重复键与双份 RUN_*。"""
    trimmed = {k: v for k, v in config.items() if k not in exclude_keys}
    body = {"meta": _json_safe_sorted_dict(meta), "config": _json_safe_sorted_dict(trimmed)}
    pretty = _pretty_log_enabled()
    payload = json.dumps(body, ensure_ascii=False, indent=2 if pretty else None)
    extra = log_route_extra(logger, ROUTE_DETAIL)
    if pretty:
        logger.info("RUN_SNAPSHOT\n%s", payload, extra=extra)
    else:
        logger.info("RUN_SNAPSHOT %s", payload, extra=extra)


def format_epoch_line(
    epoch: int,
    time_str: str,
    lr: float,
    train_loss: float,
    valid_loss: Optional[float] = None,
) -> str:
    parts = [
        f"epoch={epoch}",
        f"time={time_str}",
        f"lr={lr:.6g}",
        f"train_loss={train_loss:.4f}",
    ]
    if valid_loss is not None:
        parts.append(f"valid_loss={valid_loss:.4f}")
    return " | ".join(parts)


def _fmt_duration_sec(t: float) -> str:
    if t >= 3600:
        return f"{t/3600:.1f}h"
    if t >= 60:
        return f"{t/60:.1f}m"
    return f"{t:.1f}s"


def format_epoch_summary_lines(
    *,
    epoch: int,
    train_loss_total_epoch: float,
    train_loss_r_epoch: float,
    train_loss_c_epoch: float,
    train_loss_e_epoch: float,
    valid_loss_total_epoch: float,
    valid_loss_r_epoch: float,
    valid_loss_c_epoch: float,
    valid_loss_e_epoch: float,
    lr: float,
    quick_bleu4: Optional[float] = None,
    full_bleu_monitor_bleu4: Optional[float] = None,
    meteor: Optional[float] = None,
) -> str:
    """紧凑 [Epoch Summary] 块（明文，便于 grep）。full_bleu_monitor_bleu4 为训练期 full BLEU 监控分，非正式 eval。"""
    lines = [
        "[Epoch Summary]",
        f"epoch={epoch}",
        f"train_loss_total_epoch={train_loss_total_epoch:.6g}",
        f"train_loss_r_epoch={train_loss_r_epoch:.6g}",
        f"train_loss_c_epoch={train_loss_c_epoch:.6g}",
        f"train_loss_e_epoch={train_loss_e_epoch:.6g}",
        f"valid_loss_total_epoch={valid_loss_total_epoch:.6g}",
        f"valid_loss_r_epoch={valid_loss_r_epoch:.6g}",
        f"valid_loss_c_epoch={valid_loss_c_epoch:.6g}",
        f"valid_loss_e_epoch={valid_loss_e_epoch:.6g}",
        f"lr={lr:.6g}",
    ]
    if quick_bleu4 is not None:
        lines.append(f"quick_bleu4={quick_bleu4:.6g}")
    if full_bleu_monitor_bleu4 is not None:
        lines.append(f"full_bleu_monitor_bleu4={full_bleu_monitor_bleu4:.6g}")
    if meteor is not None:
        lines.append(f"meteor={meteor:.6g}")
    return "\n".join(lines) + "\n\n"


def format_epoch_training_block(
    *,
    time_str: str,
    epoch: int,
    epoch_time_s: float,
    total_time_s: float,
    step_time_s: float,
    gpu_util: str,
    gpu_mem: str,
    cpu_used: str,
    cpu_total: str,
    cpu_util: str,
    lr: float,
    train_loss: float,
    valid_loss: Optional[float] = None,
    bleu_line: Optional[str] = None,
    lr_schedule_detail: Optional[str] = None,
) -> str:
    """每 epoch 一块多行纯文本（无 logging 时间戳/级别前缀），与 perf_monitor 的 rec 字段对齐。

    第一行为 time_str，第二行为「Epoch n」，其后指标行统一缩进 4 空格；块末多一个空行与下一 epoch 分隔。
    """
    et = _fmt_duration_sec(epoch_time_s)
    tt = _fmt_duration_sec(total_time_s)
    step_ms = step_time_s * 1000
    line3 = f"epoch_time={et}\t|\ttotal={tt}\t\t|\tstep={step_ms:.0f}ms |"
    line4 = f"GPU={gpu_util}\t|\tMem={gpu_mem}\t|\tCPU={cpu_used}/{cpu_total} {cpu_util}"
    parts5: List[str] = [f"lr={lr:.6g}", f"train_loss={train_loss:.4f}"]
    if valid_loss is not None:
        parts5.append(f"valid_loss={valid_loss:.4f}")
    line5 = "\t|\t".join(parts5)
    detail = [line3, line4, line5]
    if lr_schedule_detail is not None:
        detail.append(lr_schedule_detail.rstrip())
    if bleu_line is not None:
        detail.append(bleu_line.rstrip())
    indent = "    "
    lines: List[str] = [time_str, f"Epoch {epoch}"] + [indent + ln for ln in detail]
    # 块末空行：与下一 epoch 的时间戳分隔，便于阅读
    return "\n".join(lines) + "\n\n"


def log_epoch_training_block(logger: Optional[logging.Logger], text: str) -> None:
    """写入 format_epoch_training_block 生成的多行块（双文件模式下写入 console 摘要）。"""
    _write_plain_log_block(logger, text, route=ROUTE_SUMMARY)


def log_epoch_summary_compact(logger: Optional[logging.Logger], text: str) -> None:
    """[Epoch Summary] 等紧凑块，与 epoch 训练块相同路由（摘要侧）。"""
    _write_plain_log_block(logger, text, route=ROUTE_SUMMARY)


def format_collapse_summary_lines(collapse: Dict[str, Any]) -> List[str]:
    """[Collapse Summary] 明文块。"""
    top10 = collapse.get("top10_pred_texts_with_count") or []
    parts = []
    for i, item in enumerate(top10, 1):
        t = (item.get("text") or "").replace("\n", " ")[:120]
        parts.append(f"  #{i} count={item.get('count')} text={t!r}")
    warn = collapse.get("collapse_warnings") or []
    wline = f"collapse_warnings={warn}" if warn else "collapse_warnings=[]"
    if warn:
        wline = "[Collapse warning] " + wline
    lines = [
        "[Collapse Summary]",
        f"top1_pred={(collapse.get('top1_pred_text') or '')[:160]!r}",
        f"top1_count={collapse.get('top1_pred_count')}",
        f"top1_ratio={collapse.get('top1_pred_ratio')}",
        f"unique_count={collapse.get('pred_unique_count')}",
        f"unique_ratio={collapse.get('pred_unique_ratio')}",
        f"mean_pred_len_tokens={collapse.get('mean_pred_len_tokens')}",
        f"mean_ref_len_tokens={collapse.get('mean_ref_len_tokens')}",
        "top10=",
        *parts,
        wline,
    ]
    return lines


def format_eval_summary_lines(
    *,
    decode_cfg: Dict[str, Any],
    final: Dict[str, Any],
    collapse: Optional[Dict[str, Any]] = None,
    eval_run_tag: str = "",
) -> List[str]:
    """[Eval Summary] 紧凑一行式关键指标（decode + 主分数 + 塌缩摘要）。

    dist1_evaluate_text/dist2_evaluate_text 与 FINAL RESULTS 中 paper-compatible DIST 一致；
    ext_* 为 extended_text_metrics_bundle，诊断用，非论文主表 DIST。
    """
    ex = final.get("explanation") or {}
    bl = ex.get("bleu") or {}
    rg = ex.get("rouge") or {}
    lines = [
        "[Eval Summary]",
        f"eval_run_tag={eval_run_tag}" if eval_run_tag else "eval_run_tag=",
        f"decode={decode_cfg.get('decode_strategy')}",
        f"temp={decode_cfg.get('generate_temperature')}",
        f"top_p={decode_cfg.get('generate_top_p')}",
        f"penalty={decode_cfg.get('repetition_penalty')}",
        f"decode_seed={decode_cfg.get('decode_seed')}",
        f"label_smoothing={decode_cfg.get('label_smoothing')}",
        f"max_explanation_length={decode_cfg.get('max_explanation_length')}",
        f"mae={(final.get('recommendation') or {}).get('mae')}",
        f"rmse={(final.get('recommendation') or {}).get('rmse')}",
        f"bleu4={bl.get('4')}",
        f"meteor={ex.get('meteor')}",
        f"rouge_l={rg.get('l')}",
    ]
    ext = final.get("text_metrics_corpus_and_sentence") or {}
    corp = ext.get("corpus_level") or {}
    sent = ext.get("sentence_level_mean") or {}
    di = ex.get("dist") or {}
    if ex.get("text_metrics_skipped"):
        lines.append("text_metrics=skipped")
    lines.extend(
        [
            f"dist1_evaluate_text={di.get('1')}",
            f"dist2_evaluate_text={di.get('2')}",
            f"ext_corpus_distinct_1_pct={corp.get('distinct_1_pct')}",
            f"ext_corpus_distinct_2_pct={corp.get('distinct_2_pct')}",
            f"ext_sentence_distinct_1_pct={sent.get('distinct_1_pct')}",
            f"ext_sentence_distinct_2_pct={sent.get('distinct_2_pct')}",
            f"ext_unigram_repetition={sent.get('unigram_repetition_ratio')}",
            f"ext_trigram_repetition={sent.get('trigram_repetition_ratio')}",
            f"ext_mean_pred_len_words={corp.get('mean_pred_len_words')}",
            f"ext_mean_ref_len_words={corp.get('mean_ref_len_words')}",
        ]
    )
    if collapse:
        lines.append(f"top1_ratio={collapse.get('top1_pred_ratio')}")
        lines.append(f"unique_ratio={collapse.get('pred_unique_ratio')}")
    return lines


def format_eval_metrics_ext_lines(final: Dict[str, Any]) -> List[str]:
    """[Eval metrics ext]：extended_text_metrics_bundle，仅供诊断；非论文主表 DIST-1/DIST-2。"""
    ext = final.get("text_metrics_corpus_and_sentence") or {}
    corp = ext.get("corpus_level") or {}
    sent = ext.get("sentence_level_mean") or {}
    tab = "\t"
    return [
        "[Eval metrics ext] (extended_text_metrics_bundle; 诊断：塌缩/重复/句内句间多样性；"
        "勿与主表 DIST-1/DIST-2 混同，不参与与论文主表的横向对比)",
        f"{tab}ext_corpus_distinct_1_pct={corp.get('distinct_1_pct')}",
        f"{tab}ext_corpus_distinct_2_pct={corp.get('distinct_2_pct')}",
        f"{tab}ext_sentence_distinct_1_pct={sent.get('distinct_1_pct')}",
        f"{tab}ext_sentence_distinct_2_pct={sent.get('distinct_2_pct')}",
        f"{tab}ext_unigram_repetition={sent.get('unigram_repetition_ratio')}",
        f"{tab}ext_trigram_repetition={sent.get('trigram_repetition_ratio')}",
        f"{tab}ext_mean_pred_len_words={corp.get('mean_pred_len_words')}",
        f"{tab}ext_mean_ref_len_words={corp.get('mean_ref_len_words')}",
    ]


def format_final_results_lines(
    final: Dict[str, Any],
    *,
    task_description: Optional[str] = None,
    start_time: Optional[str] = None,
    decode_cfg: Optional[Dict[str, Any]] = None,
    collapse_stats: Optional[Dict[str, Any]] = None,
    eval_run_tag: str = "",
) -> List[str]:
    """构建 FINAL RESULTS 文本行（无 log 前缀；指标行用制表符缩进）。

    Explanation 块中 DIST-1/DIST-2 为 evaluate_text 语料级 distinct（论文可比）；其后的 [Eval metrics ext] 为诊断扩展，非主表 DIST。

    task_description: 可选，在评估结果块最上方增加一行「任务说明：…」（位于 FINAL RESULTS 分隔线之上）。
    start_time: 可选，eval 开始时间字符串；未传则用当前时间。
    """
    current_time = start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tab = "\t"
    lines: List[str] = []
    if task_description:
        lines.append(f"任务说明：{task_description}")
    lines.extend(
        [
            "------------------------------------------FINAL RESULTS------------------------------------------",
            current_time,
            "[Recommendation]",
        ]
    )
    lines.extend(
        [
            f"{tab}MAE = {(final.get('recommendation') or {}).get('mae')} | RMSE = {(final.get('recommendation') or {}).get('rmse')} ",
            "[Explanation]",
        ]
    )
    ex = final.get("explanation") or {}
    if ex.get("text_metrics_skipped"):
        lines.append(f"{tab}Text metrics skipped for this eval protocol.")
    else:
        rouge = ex.get("rouge") or {}
        bleu = ex.get("bleu") or {}
        dist = ex.get("dist") or {}
        lines.extend(
            [
                f"{tab}ROUGE: {rouge.get('1')}, {rouge.get('2')}, {rouge.get('l')} ",
                f"{tab}BLEU: {bleu.get('1')}, {bleu.get('2')}, {bleu.get('3')}, {bleu.get('4')} ",
                f"{tab}DIST-1/DIST-2 (evaluate_text, paper-compatible): {dist.get('1')}, {dist.get('2')}",
                f"{tab}METEOR: {ex.get('meteor')} ",
            ]
        )
    pm = final.get("paper_metrics")
    if isinstance(pm, dict) and pm.get("bleu"):
        bl = pm["bleu"]
        lines.append(
            f"{tab}[paper_metrics] BLEU4(统一分词)={bl.get('4')} ROUGE-L={pm.get('rouge', {}).get('rouge_l_f')} "
            f"DIST2={pm.get('distinct_corpus', {}).get('scale_percent_0_100', {}).get('2')} "
            f"schema={pm.get('schema_version')}"
        )
    lines.extend(format_eval_metrics_ext_lines(final))
    if decode_cfg is not None:
        lines.append("")
        lines.extend(format_eval_summary_lines(decode_cfg=decode_cfg, final=final, collapse=collapse_stats, eval_run_tag=eval_run_tag))
    if collapse_stats:
        lines.append("")
        lines.extend(format_collapse_summary_lines(collapse_stats))
    return lines


def _write_plain_log_block(
    logger: Optional[logging.Logger],
    text: str,
    *,
    route: str = ROUTE_BOTH,
) -> None:
    """写入多行纯文本（不经 Formatter）：按 odcr 路由写入对应 FileHandler / StreamHandler。"""
    if not text.endswith("\n"):
        text = text + "\n"
    if logger is None:
        print(text, end="", flush=True)
        return
    for h in logger.handlers:
        # FileHandler 继承 StreamHandler；与 logger.info 一样走 Handler 锁，避免与 emit 交错
        if isinstance(h, logging.StreamHandler) and _plain_route_visible(h, route):
            try:
                h.acquire()
                try:
                    h.stream.write(text)
                    h.flush()
                finally:
                    h.release()
            except Exception:
                pass


def log_final_results_block(logger: Optional[logging.Logger], lines: list) -> None:
    """FINAL RESULTS 多行块（双文件模式下写入摘要侧）。"""
    text = "\n".join(lines)
    _write_plain_log_block(logger, text, route=ROUTE_SUMMARY)


def flush_preset_load_events(logger: Optional[logging.Logger]) -> None:
    """将 import config 阶段记录的 presets YAML 加载结果刷入训练日志（摘要路由）。"""
    if logger is None:
        return
    try:
        import config as cfg

        ev = getattr(cfg, "PRESET_LOAD_EVENTS", None) or []
        for line in ev:
            logger.info("[PresetYAML] %s", line, extra=log_route_extra(logger, ROUTE_SUMMARY))
    except Exception:
        pass


def finalize_run_log(logger: Optional[logging.Logger], extra: Optional[str] = None) -> None:
    _write_plain_log_block(logger, RUN_END_LINE + "\n", route=ROUTE_BOTH)
    if logger is not None:
        if extra:
            logger.info("%s", extra, extra=log_route_extra(logger, ROUTE_BOTH))
    elif extra:
        print(extra, flush=True)


def flush_odcr_file_handlers(logger: Optional[logging.Logger]) -> None:
    """将 odcr logger 各 Handler 缓冲刷盘（读取日志尾部前调用，避免 tail 缺最新行）。"""
    if logger is None:
        return
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass


# 误写入 train.log 的常见 shell 行（历史 bug 或手工重定向）
_TRAIN_LOG_SHELL_MARKERS = (
    "---------- Task ",
    "========== 跳过 Task ",
)
_EPOCH_HEAD_LINE_RE = re.compile(r"^Epoch (\d+)\s*$")


def audit_train_log_file(path: str) -> Dict[str, Any]:
    """轻量自检（启发式）：是否混入 shell 包装行、Epoch 行序列是否严格递增 1。

    适用于单次连续训练；同文件若含多段 train/eval，可能出现重复 Epoch 编号，结果仅供参考。
    """
    result: Dict[str, Any] = {
        "path": path,
        "shell_hits": [],
        "epoch_numbers": [],
        "epoch_sequence_gaps": [],
    }
    if not path or not os.path.isfile(path):
        result["error"] = "not_a_file"
        return result
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fp:
            lines = fp.readlines()
    except OSError as e:
        result["error"] = str(e)
        return result
    for lineno, ln in enumerate(lines, 1):
        for marker in _TRAIN_LOG_SHELL_MARKERS:
            if marker in ln:
                result["shell_hits"].append(
                    {"line": lineno, "marker": marker, "snippet": ln.strip()[:160]}
                )
                break
    for ln in lines:
        m = _EPOCH_HEAD_LINE_RE.match(ln.strip())
        if m:
            result["epoch_numbers"].append(int(m.group(1)))
    nums = result["epoch_numbers"]
    if len(nums) >= 2:
        for a, b in zip(nums, nums[1:]):
            if b != a + 1:
                result["epoch_sequence_gaps"].append({"after_epoch": a, "next_seen": b})
    result["epoch_line_count"] = len(nums)
    result["epoch_max"] = max(nums) if nums else None
    return result


def broadcast_run_paths_ddp(
    log_path: Optional[str],
    run_id: Optional[str],
    rank: int,
) -> Tuple[str, str]:
    """分布式初始化后由 rank0 生成路径并广播到各 rank。"""
    import torch.distributed as dist

    if not dist.is_initialized():
        return log_path or "", run_id or ""
    obj = [log_path, run_id] if rank == 0 else [None, None]
    dist.broadcast_object_list(obj, src=0)
    return obj[0], obj[1]


# --- eval 结果自动汇总（任务 runs/task{T}/vN/meta + 跨任务 runs/global/vN/meta；见 path_layout 边界）---

_EVAL_REGISTRY_TXT = "eval_registry.txt"
_EVAL_REGISTRY_JSONL = "eval_registry.jsonl"
_EVAL_REGISTRY_CSV = "eval_registry.csv"
_EVAL_REGISTRY_GLOBAL_TXT = "eval_registry_all.txt"
_EVAL_REGISTRY_GLOBAL_JSONL = "eval_registry_all.jsonl"
_EVAL_REGISTRY_GLOBAL_CSV = "eval_registry_all.csv"

_EVAL_REGISTRY_CSV_FIELDS: Tuple[str, ...] = (
    "ts",
    "run_id",
    "task_idx",
    "pipeline",
    "method",
    "method_name",
    "experiment_profile",
    "ablation_profile",
    "domain_from",
    "domain_to",
    "log_file",
    "save_file",
    "task_description",
    "eval_export_tag",
    "decode_strategy",
    "generate_temperature",
    "generate_top_p",
    "repetition_penalty",
    "decode_seed",
    "label_smoothing",
    "max_explanation_length",
    "mae",
    "rmse",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "dist_1",
    "dist_2",
    "meteor",
    "ext_corpus_distinct_1",
    "ext_corpus_distinct_2",
    "ext_sentence_distinct_1",
    "ext_sentence_distinct_2",
    "ext_unigram_repetition",
    "ext_trigram_repetition",
    "ext_mean_pred_len_words",
    "ext_mean_ref_len_words",
    "collapse_unique_count",
    "collapse_unique_ratio",
    "collapse_top1_ratio",
    "collapse_mean_pred_len_tokens",
    "eval_elapsed_s",
)


def _eval_summary_enabled() -> bool:
    v = os.environ.get("ODCR_EVAL_SUMMARY", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _global_eval_registry_meta_dir() -> str:
    """跨任务全局 eval 汇总目录（仅 ``eval_registry_all.*`` 等）；单任务注册表须写 ``runs/task{T}/vN/meta/``。

    默认路径由 ``path_layout.get_global_meta_dir`` 给出；可用 ``ODCR_EVAL_SUMMARY_GLOBAL_DIR`` 覆盖。
    """
    g = os.environ.get("ODCR_EVAL_SUMMARY_GLOBAL_DIR", "").strip()
    if g:
        return os.path.abspath(os.path.expanduser(g))
    _root = get_odcr_root()
    it = os.environ.get("ODCR_ITER", "v1").strip() or "v1"
    return str(path_layout.get_global_meta_dir(Path(_root), it))


def flatten_final_metrics_for_summary(final: Dict[str, Any]) -> Dict[str, Any]:
    """将 FINAL RESULTS + ext + collapse 摊平为可写入 CSV/JSON 的标量。

    dist_1/dist_2 来自 evaluate_text（论文主表口径）；ext_* 列为诊断扩展指标，勿与 dist_* 混读。
    """
    def _f(x: Any) -> float:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)

    r = final["recommendation"]
    e = final.get("explanation") or {}
    if e.get("text_metrics_skipped"):
        d: Dict[str, Any] = {
            "mae": _f(r["mae"]),
            "rmse": _f(r["rmse"]),
            "rouge_1": "",
            "rouge_2": "",
            "rouge_l": "",
            "bleu_1": "",
            "bleu_2": "",
            "bleu_3": "",
            "bleu_4": "",
            "dist_1": "",
            "dist_2": "",
            "meteor": "",
        }
    else:
        rg = e["rouge"]
        bl = e["bleu"]
        di = e["dist"]

        d = {
            "mae": _f(r["mae"]),
            "rmse": _f(r["rmse"]),
            "rouge_1": _f(rg["1"]),
            "rouge_2": _f(rg["2"]),
            "rouge_l": _f(rg["l"]),
            "bleu_1": _f(bl["1"]),
            "bleu_2": _f(bl["2"]),
            "bleu_3": _f(bl["3"]),
            "bleu_4": _f(bl["4"]),
            "dist_1": _f(di["1"]),
            "dist_2": _f(di["2"]),
            "meteor": _f(e["meteor"]),
        }

    ext = final.get("text_metrics_corpus_and_sentence") or {}
    corp = ext.get("corpus_level") or {}
    sent = ext.get("sentence_level_mean") or {}
    d["ext_corpus_distinct_1"] = corp.get("distinct_1_pct", "")
    d["ext_corpus_distinct_2"] = corp.get("distinct_2_pct", "")
    d["ext_sentence_distinct_1"] = sent.get("distinct_1_pct", "")
    d["ext_sentence_distinct_2"] = sent.get("distinct_2_pct", "")
    d["ext_unigram_repetition"] = sent.get("unigram_repetition_ratio", "")
    d["ext_trigram_repetition"] = sent.get("trigram_repetition_ratio", "")
    d["ext_mean_pred_len_words"] = corp.get("mean_pred_len_words", "")
    d["ext_mean_ref_len_words"] = corp.get("mean_ref_len_words", "")

    cs = final.get("collapse_stats") or {}
    if cs:
        d["collapse_unique_count"] = cs.get("pred_unique_count", "")
        d["collapse_unique_ratio"] = cs.get("pred_unique_ratio", "")
        d["collapse_top1_ratio"] = cs.get("top1_pred_ratio", "")
        d["collapse_mean_pred_len_tokens"] = cs.get("mean_pred_len_tokens", "")
    else:
        d["collapse_unique_count"] = ""
        d["collapse_unique_ratio"] = ""
        d["collapse_top1_ratio"] = ""
        d["collapse_mean_pred_len_tokens"] = ""
    return d


def _append_text(path: str, text: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    _append_text(path, line)


def _append_csv_row(path: str, row: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    need_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_EVAL_REGISTRY_CSV_FIELDS), extrasaction="ignore")
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _EVAL_REGISTRY_CSV_FIELDS})


def append_train_epoch_metrics_jsonl(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append one structured training metric row beside the run ``full.log``."""
    if not log_file:
        return
    try:
        d = os.path.dirname(os.path.abspath(os.path.expanduser(log_file)))
        if not d:
            return
        path = os.path.join(d, path_layout.metrics_filename("metrics"))
        _append_jsonl(path, row)
    except Exception:
        pass


def _run_meta_metric_path(log_file: Optional[str], kind: str) -> Optional[str]:
    if not log_file:
        return None
    try:
        d = os.path.dirname(os.path.abspath(os.path.expanduser(log_file)))
        if not d:
            return None
        return os.path.join(d, path_layout.metrics_filename(kind))
    except Exception:
        return None


def append_step3_loss_breakdown_jsonl(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append Step3 component loss details to ``loss_breakdown.jsonl``."""
    path = _run_meta_metric_path(log_file, "loss_breakdown")
    if not path:
        return
    try:
        _append_jsonl(path, row)
    except Exception:
        pass


def append_step3_timing_profile_jsonl(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append Step3 safe-interval timing detail to ``timing_profile.jsonl``."""
    path = _run_meta_metric_path(log_file, "timing_profile")
    if not path:
        return
    try:
        _append_jsonl(path, row)
    except Exception:
        pass


def append_step3_gpu_profile_jsonl(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append Step3 GPU allocation/reservation profile to ``gpu_profile.jsonl``."""
    path = _run_meta_metric_path(log_file, "gpu_profile")
    if not path:
        return
    try:
        _append_jsonl(path, row)
    except Exception:
        pass


def append_step3_scheduler_events_jsonl(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append Step3 validation-aware LR damping events to ``scheduler_events.jsonl``."""
    path = _run_meta_metric_path(log_file, "scheduler_events")
    if not path:
        return
    try:
        _append_jsonl(path, row)
    except Exception:
        pass


def append_step3_damping_events_jsonl(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append explicit Step3 damping events to ``damping_events.jsonl``."""
    path = _run_meta_metric_path(log_file, "damping_events")
    if not path:
        return
    try:
        _append_jsonl(path, row)
    except Exception:
        pass


def append_step3_objective_drift_jsonl(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append Step3 V3 objective-drift decisions."""
    path = _run_meta_metric_path(log_file, "objective_drift")
    if not path:
        return
    try:
        _append_jsonl(path, row)
    except Exception:
        pass


def append_step3_recovery_events_jsonl(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append Step3 V3 recovery-controller events."""
    path = _run_meta_metric_path(log_file, "recovery_events")
    if not path:
        return
    try:
        _append_jsonl(path, row)
    except Exception:
        pass


def append_step3_training_effectiveness_jsonl(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append Step3 epoch-level training effectiveness diagnostics."""
    path = _run_meta_metric_path(log_file, "training_effectiveness")
    if not path:
        return
    try:
        _append_jsonl(path, row)
    except Exception:
        pass


def write_step3_training_effectiveness_summary_json(*, log_file: Optional[str], payload: Dict[str, Any]) -> None:
    """Write the latest Step3 training effectiveness summary."""
    path = _run_meta_metric_path(log_file, "training_effectiveness_summary")
    if not path:
        return
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    except Exception:
        pass


def append_step3_samples_jsonl(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append diagnostic-only Step3 samples to ``samples.jsonl``."""
    path = _run_meta_metric_path(log_file, "samples")
    if not path:
        return
    try:
        _append_jsonl(path, row)
    except Exception:
        pass


def write_step3_collapse_stats_json(*, log_file: Optional[str], payload: Dict[str, Any]) -> None:
    """Write Step3 diagnostic collapse stats to ``collapse_stats.json``."""
    path = _run_meta_metric_path(log_file, "collapse_stats")
    if not path:
        return
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    except Exception:
        pass


_STEP3_EPOCH_SUMMARY_FIELDS: Tuple[str, ...] = (
    "epoch",
    "train_loss",
    "valid_loss",
    "best_metric",
    "delta_from_best",
    "delta_recent",
    "lr_base",
    "lr_effective",
    "base_min_lr",
    "effective_min_lr",
    "damping_event",
    "objective_drift_status",
    "loss_phase",
    "checkpoint_improved",
    "effective_improvement_status",
    "recommended_action",
    "elapsed_s",
    "samples_per_sec",
    "checkpoint_path",
    "status",
)


def append_step3_epoch_summary_csv(*, log_file: Optional[str], row: Dict[str, Any]) -> None:
    """Append Step3 epoch summary to ``epoch_summary.csv``."""
    path = _run_meta_metric_path(log_file, "epoch_summary")
    if not path:
        return
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        need_header = not os.path.isfile(path) or os.path.getsize(path) == 0
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(_STEP3_EPOCH_SUMMARY_FIELDS), extrasaction="ignore")
            if need_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in _STEP3_EPOCH_SUMMARY_FIELDS})
    except Exception:
        pass


def write_step3_loss_component_epoch_summary_csv(*, log_file: Optional[str], rows: list[Dict[str, Any]]) -> None:
    """Write Step3 per-epoch loss component dashboard rows."""
    path = _run_meta_metric_path(log_file, "loss_component_epoch_summary")
    if not path:
        return
    fieldnames = ("schema_version", "epoch", "loss_name", "raw_mean", "weighted_mean", "count")
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({key: row.get(key, "") for key in fieldnames})
    except Exception:
        pass


def write_step3_loss_component_trends_json(*, log_file: Optional[str], payload: Dict[str, Any]) -> None:
    """Write Step3 loss component trend dashboard."""
    path = _run_meta_metric_path(log_file, "loss_component_trends")
    if not path:
        return
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    except Exception:
        pass


def write_step3_component_contribution_summary_md(*, log_file: Optional[str], text: str) -> None:
    """Write a compact Step3 component contribution markdown summary."""
    path = _run_meta_metric_path(log_file, "component_contribution_summary")
    if not path:
        return
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text).rstrip() + "\n")
    except Exception:
        pass


def append_eval_run_summaries(
    final: Dict[str, Any],
    *,
    task_idx: int,
    run_id: str,
    pipeline: str,
    domain_from: str,
    domain_to: str,
    log_file: Optional[str] = None,
    save_file: Optional[str] = None,
    task_description: Optional[str] = None,
    start_time: Optional[str] = None,
    eval_elapsed: Optional[float] = None,
    decode_cfg: Optional[Dict[str, Any]] = None,
    eval_export_tag: str = "",
) -> None:
    """将一次 eval 的指标追加到任务级 ``runs/task{T}/vN/meta/eval_registry.*``，并追加跨任务全局
    ``runs/global/vN/meta/eval_registry_all.*``（目录可由 ``ODCR_EVAL_SUMMARY_GLOBAL_DIR`` 覆盖；
    语义同 ``path_layout`` 中 global vs task meta 边界）。

    设 ODCR_EVAL_SUMMARY=0 可关闭。失败时静默忽略。
    """
    if not _eval_summary_enabled():
        return
    try:
        metrics = flatten_final_metrics_for_summary(final)
    except Exception:
        return

    ts = start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    desc_one_line = (task_description or "").replace("\n", " ").strip()
    dc = decode_cfg or {}
    experiment_profile = str(dc.get("experiment_profile") or "csb_odcr_full")
    ablation_profile = str(dc.get("ablation_profile") or experiment_profile)
    row: Dict[str, Any] = {
        "ts": ts,
        "run_id": run_id or "",
        "task_idx": task_idx,
        "pipeline": pipeline,
        "method": "CSB-ODCR",
        "method_name": "CSB-ODCR",
        "experiment_profile": experiment_profile,
        "ablation_profile": ablation_profile,
        "domain_from": domain_from,
        "domain_to": domain_to,
        "log_file": os.path.abspath(os.path.expanduser(log_file)) if log_file else "",
        "save_file": os.path.abspath(os.path.expanduser(save_file)) if save_file else "",
        "task_description": desc_one_line,
        "eval_export_tag": eval_export_tag or "",
        "decode_strategy": dc.get("decode_strategy", ""),
        "generate_temperature": dc.get("generate_temperature", ""),
        "generate_top_p": dc.get("generate_top_p", ""),
        "repetition_penalty": dc.get("repetition_penalty", ""),
        "decode_seed": dc.get("decode_seed", ""),
        "label_smoothing": dc.get("label_smoothing", ""),
        "max_explanation_length": dc.get("max_explanation_length", ""),
        **metrics,
        "eval_elapsed_s": round(eval_elapsed, 1) if eval_elapsed is not None else "",
    }

    lines_block = format_final_results_lines(
        final,
        task_description=task_description,
        start_time=start_time,
        decode_cfg=decode_cfg,
        collapse_stats=final.get("collapse_stats"),
        eval_run_tag=eval_export_tag,
    )
    if eval_elapsed is not None:
        _m, _s = divmod(int(eval_elapsed), 60)
        lines_block.append(f"Eval elapsed: {_m}m {_s}s ({eval_elapsed:.1f}s)")
    plain_sep = (
        "================================================================================\n"
        f"{ts} | run_id={run_id} | task_idx={task_idx} | pipeline={pipeline}\n"
        f"eval_export_tag={eval_export_tag} decode={dc.get('decode_strategy')} temp={dc.get('generate_temperature')} top_p={dc.get('generate_top_p')}\n"
        f"{domain_from} -> {domain_to}\n"
        f"log_file={row['log_file']}\n"
        f"save_file={row['save_file']}\n"
        "--------------------------------------------------------------------------------\n"
        + "\n".join(lines_block)
        + "\n================================================================================\n"
    )

    try:
        meta = os.environ.get("ODCR_ITERATION_META_DIR", "").strip()
        if meta:
            _append_jsonl(os.path.join(meta, _EVAL_REGISTRY_JSONL), row)
            _append_csv_row(os.path.join(meta, _EVAL_REGISTRY_CSV), row)
            _append_text(os.path.join(meta, _EVAL_REGISTRY_TXT), plain_sep)
        gdir = _global_eval_registry_meta_dir()
        _append_jsonl(os.path.join(gdir, _EVAL_REGISTRY_GLOBAL_JSONL), row)
        _append_csv_row(os.path.join(gdir, _EVAL_REGISTRY_GLOBAL_CSV), row)
        _append_text(os.path.join(gdir, _EVAL_REGISTRY_GLOBAL_TXT), plain_sep)
    except Exception:
        pass
