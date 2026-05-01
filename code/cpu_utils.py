"""
CPU 可见核心数（与 shell 里 `nproc` 在多数作业调度场景下更一致）

- 优先 `os.sched_getaffinity(0)`：反映当前进程被允许使用的 CPU 集合（cpuset/cgroup 限制后通常就是你能用的核数）
- 否则回退 `os.cpu_count()`
- 环境变量 `RUNNING_CPU_COUNT` 可显式覆盖（整数，>=1），便于集群脚本统一指定
"""

from __future__ import annotations

import os


def effective_cpu_count() -> int:
    env = os.environ.get("RUNNING_CPU_COUNT", "").strip()
    if env:
        try:
            n = int(env)
            return max(1, n)
        except ValueError:
            pass
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:
        n = os.cpu_count()
        return max(1, n if n is not None else 1)
    except Exception:
        return 1
