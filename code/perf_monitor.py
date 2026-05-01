"""
轻量级 PyTorch 训练性能监控模块
- epoch_end(emit_log=False) 时仅记录 metrics 并返回 rec，由 train_logging.format_epoch_training_block 统一多行输出
- emit_log=True（默认）时保留单行 INFO 兼容轻量调用
- 训练结束输出一张美观的总汇总表格
- 无侵入、不影响训练速度

DDP 多进程注意：仅 rank0 创建 PerfMonitor 时，epoch_end 内用 torch.cuda.max_memory_allocated(其它卡)
在本进程往往恒为 0（显存实际在其它 rank 上）。多卡训练请在每个 epoch 由 **全体 rank** 调用
gather_ddp_gpu_stats_for_epoch_log，再让 rank0 把返回值写入 rec 的 gpu_util / gpu_mem。
"""

import time
import sys
import os
import logging

from typing import Optional, Tuple

from cpu_utils import effective_cpu_count
from paths_config import append_log_dual

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

_pynvml_ok = False
_nvml = None
try:
    import pynvml
    pynvml.nvmlInit()
    _pynvml_ok = True
except Exception:
    pass

_psutil_ok = False
try:
    import psutil
    _psutil_ok = True
except Exception:
    pass


def _get_num_cpu():
    """返回当前进程可见的 CPU 数（与 cgroup 限制后的 nproc 通常一致）"""
    try:
        return effective_cpu_count()
    except Exception:
        return None


def _get_ram_usage():
    """返回系统内存使用量，如 '12.3G/32G (38%)' 或 '-'"""
    if not _psutil_ok:
        return "-"
    try:
        v = psutil.virtual_memory()
        used_g = v.used / 1024**3
        total_g = v.total / 1024**3
        pct = v.percent
        return f"{used_g:.1f}G/{total_g:.0f}G ({pct:.0f}%)"
    except Exception:
        return "-"


def _get_cpu_used_and_util():
    """
    返回 (利用核心数, 总核心数, 利用率字符串)
    利用核心数 = round(利用率% * 总核心数 / 100)，近似有多少核在忙
    """
    total = _get_num_cpu()
    total_str = str(total) if total is not None else "?"
    if not _psutil_ok:
        return "?", total_str, "-"
    try:
        pct = psutil.cpu_percent(interval=None)
        util_str = f"{pct:.0f}%"
        if total is not None and total > 0:
            used = max(1, round(pct / 100 * total))
            return str(used), total_str, util_str
        return "?", total_str, util_str
    except Exception:
        return "?", total_str, "-"


def _get_gpu_util(device_id=0):
    if not _pynvml_ok:
        return "-"
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        return f"{util.gpu}%"
    except Exception:
        return "-"


def _get_gpu_util_multi(device_ids):
    """多卡利用率，如 '0:85% 1:72%' 或 '0:85%'"""
    if not _pynvml_ok or not device_ids:
        return "-"
    parts = []
    for did in device_ids:
        try:
            u = _get_gpu_util(did)
            parts.append(f"{did}:{u}")
        except Exception:
            pass
    return " ".join(parts) if parts else "-"


def _get_gpu_mem(device_id=0):
    """返回 (显示字符串, 字节数或None)。

    优先用 NVML 的显存占用，与 nvidia-smi 一致，且在多进程 DDP 下仍能对「非本进程绑定的卡」
    给出正确读数。若仅用 torch.cuda.max_memory_allocated，则只有当前进程在该 device 上
    分配过张量才有非零值，另一张卡常会误显示为 0。
    """
    if _pynvml_ok:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            b = mem.used
            return f"{b / 1024**3:.2f}G", b
        except Exception:
            pass
    if not HAS_CUDA or not torch.cuda.is_available():
        return "-", None
    try:
        b = torch.cuda.max_memory_allocated(device_id)
        return f"{b / 1024**3:.2f}G", b
    except Exception:
        return "-", None


def _get_gpu_mem_multi(device_ids):
    """多卡显存：返回 (显示字符串, 所有卡总字节数)"""
    if not HAS_CUDA or not torch.cuda.is_available() or not device_ids:
        return "-", None
    parts = []
    total_bytes = 0
    for did in device_ids:
        try:
            s, b = _get_gpu_mem(did)
            parts.append(f"{did}:{s}")
            if b is not None:
                total_bytes += b
        except Exception:
            pass
    return " ".join(parts) if parts else "-", total_bytes if total_bytes else None


def _nvml_physical_index_for_torch_cuda_device(torch_cuda_idx: int) -> Optional[int]:
    """
    将当前进程内可见的 cuda 设备序号映射到 NVML 的 GPU 索引。
    未设置 CUDA_VISIBLE_DEVICES 时二者一致；设置为 \"4,5\" 时 cuda:0 -> 4。
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        return int(torch_cuda_idx)
    parts = [p.strip() for p in vis.split(",") if p.strip()]
    if torch_cuda_idx < 0 or torch_cuda_idx >= len(parts):
        return int(torch_cuda_idx)
    p = parts[torch_cuda_idx]
    if p.isdigit():
        return int(p)
    return int(torch_cuda_idx)


def _local_rank_gpu_util_percent(torch_cuda_idx: int) -> float:
    """当前 rank 对应 GPU 的利用率 0–100；不可用返回 -1.0。"""
    if not _pynvml_ok:
        return -1.0
    phy = _nvml_physical_index_for_torch_cuda_device(torch_cuda_idx)
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(int(phy))
        return float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
    except Exception:
        return -1.0


def gather_ddp_gpu_stats_for_epoch_log(
    rank: int, world_size: int, cuda_device_index: int
) -> Tuple[str, str, Optional[int]]:
    """
    DDP 训练每个 epoch 在同一点由 **所有 rank** 调用一次。

    - 各 rank 读取 **本进程当前 device** 的 torch.cuda.memory_allocated / memory_reserved
      以及（若 NVML 可用）对应物理 GPU 的利用率。
    - 经 all_gather 汇总后，返回全 rank 拼好的日志字符串与峰值字节（供 PerfMonitor 汇总表）。

    单卡或未初始化分布式时：不调用 collective，等价于 world_size==1。

    Returns:
        (gpu_util_str, gpu_mem_str, peak_bytes)
    """
    if not HAS_CUDA or not torch.cuda.is_available():
        return "-", "-", None

    import torch.distributed as dist

    local_idx = int(torch.cuda.current_device())
    dev = torch.device(f"cuda:{int(cuda_device_index)}")
    alloc_b = float(torch.cuda.memory_allocated(local_idx))
    resv_b = float(torch.cuda.memory_reserved(local_idx))
    util = _local_rank_gpu_util_percent(local_idx)

    use_dist = (
        world_size is not None
        and int(world_size) > 1
        and dist.is_available()
        and dist.is_initialized()
    )

    if not use_dist:
        u_str = f"{int(util)}%" if util >= 0.0 else "-"
        gpu_util = f"0:{u_str}"
        gpu_mem = (
            f"alloc 0:{alloc_b / 1024**3:.2f}G | resv 0:{resv_b / 1024**3:.2f}G"
        )
        peak = int(max(alloc_b, resv_b))
        return gpu_util, gpu_mem, peak

    t = torch.tensor([alloc_b, resv_b, util], dtype=torch.float64, device=dev)
    bufs = [torch.empty(3, dtype=torch.float64, device=dev) for _ in range(int(world_size))]
    dist.all_gather(bufs, t)

    util_parts = []
    alloc_parts = []
    resv_parts = []
    peak = 0.0
    for r in range(int(world_size)):
        a = bufs[r][0].item()
        rv = bufs[r][1].item()
        u = bufs[r][2].item()
        peak = max(peak, a, rv)
        alloc_parts.append(f"{r}:{a / 1024**3:.2f}G")
        resv_parts.append(f"{r}:{rv / 1024**3:.2f}G")
        util_parts.append(f"{r}:{int(u)}%" if u >= 0.0 else f"{r}:-")
    gpu_util = " ".join(util_parts)
    gpu_mem = "alloc " + " ".join(alloc_parts) + " | resv " + " ".join(resv_parts)
    return gpu_util, gpu_mem, int(peak)


def _fmt(t):
    if t >= 3600:
        return f"{t/3600:.1f}h"
    elif t >= 60:
        return f"{t/60:.1f}m"
    else:
        return f"{t:.1f}s"


def _display_width(s):
    """返回字符串显示宽度：中文/全角/表格线=2，ASCII=1，保证│等符号对齐"""
    w = 0
    for c in s:
        # CJK、全角标点、表格线（║│═╔╗╚╝╠╣）在终端多为双列显示
        if ('\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f' or
                '\u2500' <= c <= '\u257f'):
            w += 2
        else:
            w += 1
    return w


def _pad_to_width(s, width, align='left'):
    """按显示宽度填充到指定宽度，align: left/right"""
    w = _display_width(s)
    if w >= width:
        return s
    pad = width - w
    return (' ' * pad + s) if align == 'right' else (s + ' ' * pad)


class PerfMonitor:
    """轻量级训练性能监控，支持单卡或多卡"""

    def __init__(
        self,
        device=0,
        log_file=None,
        num_proc=None,
        device_ids=None,
        train_num_workers: Optional[int] = None,
        valid_num_workers: Optional[int] = None,
        test_num_workers: Optional[int] = None,
        training_logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            device: 主设备 ID（单卡或兼容用）
            log_file: 日志文件路径
            num_proc: datasets.map（Tokenize）阶段使用的并行进程数，不是 DataLoader
            device_ids: 多卡时传入 [0,1,...]，不传则只监控 device
            train_num_workers: 训练 DataLoader 的 num_workers（主训练循环）
            valid_num_workers: 验证 DataLoader 的 num_workers；无验证集或未创建时可不传
            test_num_workers: 测试/推理 DataLoader 的 num_workers
        """
        self.device = device if isinstance(device, int) else (device.index if hasattr(device, 'index') else 0)
        self.device_ids = device_ids if device_ids is not None else [self.device]
        self.log_file = log_file
        self.num_proc = num_proc  # Tokenize / datasets.map
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers
        self._logger = training_logger
        self.total_start = None
        self._epoch_start_time = None
        self.records = []
        self._num_cpu = _get_num_cpu()

    def start(self):
        self.total_start = time.perf_counter()
        self.records = []

    def epoch_start(self):
        self._epoch_start_time = time.perf_counter()
        if HAS_CUDA and torch.cuda.is_available():
            for did in self.device_ids:
                try:
                    torch.cuda.reset_peak_memory_stats(did)
                except Exception:
                    pass

    def epoch_end(self, epoch, n_steps, emit_log=True):
        epoch_time = time.perf_counter() - self._epoch_start_time
        total_time = time.perf_counter() - self.total_start
        step_time = epoch_time / n_steps if n_steps > 0 else 0
        # 多卡：汇总各卡 GPU 利用率与显存（格式如 "0:85% 1:72%" "0:22.11G 1:22.11G"）
        gpu_util = _get_gpu_util_multi(self.device_ids)
        gpu_mem_str, _ = _get_gpu_mem_multi(self.device_ids)
        gpu_mems = [_get_gpu_mem(did) for did in self.device_ids]
        gpu_mem_bytes = max((m[1] or 0) for m in gpu_mems) if gpu_mems else None
        cpu_used, cpu_total, cpu_util = _get_cpu_used_and_util()
        ram_usage = _get_ram_usage()

        rec = {
            "epoch": epoch,
            "epoch_time": epoch_time,
            "total_time": total_time,
            "step_time": step_time,
            "n_steps": n_steps,
            "gpu_util": gpu_util,
            "gpu_mem": gpu_mem_str,
            "gpu_mem_bytes": gpu_mem_bytes,
            "cpu_used": cpu_used,
            "cpu_total": cpu_total,
            "cpu_util": cpu_util,
            "ram_usage": ram_usage,
        }
        self.records.append(rec)

        if not emit_log:
            return rec

        lines_out = [
            f"Epoch {epoch}",
            "",
            f"time: {_fmt(epoch_time)} | Total: {_fmt(total_time)} | Step: {step_time*1000:.0f}ms",
            "",
            f"GPU: {gpu_util} | Mem: {gpu_mem_str} | RAM: {ram_usage}",
            "",
            f"CPU: {cpu_used}/{cpu_total} cores used | {cpu_util}",
            "",
        ]
        block = "\n".join(lines_out)
        one = (
            f"Epoch {epoch} | epoch_time={_fmt(epoch_time)} | total={_fmt(total_time)} | "
            f"step={step_time*1000:.0f}ms | GPU={gpu_util} | Mem={gpu_mem_str} | "
            f"CPU={cpu_used}/{cpu_total} {cpu_util}"
        )
        if self._logger is not None:
            self._logger.debug(block + "\n")
            try:
                from train_logging import ROUTE_SUMMARY, log_route_extra

                _ex = log_route_extra(self._logger, ROUTE_SUMMARY)
            except Exception:
                _ex = {}
            self._logger.info(one, extra=_ex)
        else:
            print(block, flush=True)
            append_log_dual(self.log_file, block + "\n")
        return rec

    def finish(self):
        if not self.records:
            return
        total = self.records[-1]["total_time"]
        n_epochs = len(self.records)
        avg_epoch = sum(r["epoch_time"] for r in self.records) / n_epochs
        avg_step = sum(r["step_time"] for r in self.records) / n_epochs
        mem_bytes = [r["gpu_mem_bytes"] for r in self.records if r.get("gpu_mem_bytes")]
        max_mem = f"{max(mem_bytes)/1024**3:.2f}G" if mem_bytes else "-"
        cpu_used, cpu_total, cpu_util = _get_cpu_used_and_util()

        # 列宽：按字符数填充，确保 | 上下严格对齐（monospace 下一字符=一列）
        LABEL_LEN, VALUE_LEN = 20, 28  # 字符数

        def _row(label, value):
            lbl = (label + " " * LABEL_LEN)[:LABEL_LEN]  # 左对齐，截断
            val = (" " * VALUE_LEN + value)[-VALUE_LEN:]  # 右对齐
            return f"║ {lbl} │ {val} ║"

        rows = [
            _row("总训练轮数", f"{n_epochs} epochs"),
            _row("总训练时间", _fmt(total)),
            _row("平均单轮耗时", _fmt(avg_epoch)),
            _row("平均单步耗时", f"{avg_step*1000:.0f} ms"),
            _row("GPU 峰值显存", max_mem),
            _row("CPU 利用核心", f"{cpu_used}/{cpu_total} cores" + (f" | {cpu_util}" if cpu_util != "-" else "")),
        ]
        if self.num_proc is not None:
            rows.append(_row("Tokenize (map) num_proc", str(self.num_proc)))
        if self.train_num_workers is not None:
            rows.append(_row("DataLoader train workers", str(self.train_num_workers)))
        if self.valid_num_workers is not None:
            rows.append(_row("DataLoader valid workers", str(self.valid_num_workers)))
        if self.test_num_workers is not None:
            rows.append(_row("DataLoader test workers", str(self.test_num_workers)))

        inner_len = 1 + LABEL_LEN + 3 + VALUE_LEN + 1  # " " + lbl + " │ " + val + " "
        border = "═" * inner_len
        lines = [
            "",
            f"╔{border}╗",
            "║" + (" 训练性能汇总 (Performance Summary) ".ljust(inner_len)) + "║",
            f"╠{border}╣",
            *rows,
            f"╚{border}╝",
            "",
        ]
        out = "\n".join(lines)
        if self._logger is not None:
            try:
                from train_logging import ROUTE_SUMMARY, log_route_extra

                _ex = log_route_extra(self._logger, ROUTE_SUMMARY)
            except Exception:
                _ex = {}
            self._logger.info(out, extra=_ex)
        else:
            print(out, flush=True)
            append_log_dual(self.log_file, out + "\n")


def _shutdown_pynvml():
    global _pynvml_ok
    if _pynvml_ok:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        _pynvml_ok = False
