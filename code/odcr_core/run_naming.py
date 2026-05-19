"""vN / run 目录名 / packNN 解析与在同父目录下的下一个可用编号分配（禁止时间戳作主键）。

run 目录名仅接受 **slug**：无下划线的非负整数字符串，或由若干整数字段用单个下划线连接
（如 ``1``、``2``、``2_1``、``2_1_1``）。不接受以 ``run`` 为前缀的旧式命名。

自动递增 ``next_run_id`` 时只统计**纯数字**单段目录名（忽略含下划线的 slug 与其它名称）。

multi_seed、eval、matrix 等与 train 共用 ``next_run_id`` / ``allocate_child_dir(..., kind='run')``。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_RE_ITER = re.compile(r"^v(\d+)$", re.IGNORECASE)
# 实验 run：纯十进制段用下划线连接，如 1、2、2_1、2_1_1
_RE_RUN_SLUG = re.compile(r"^(\d+(?:_\d+)*)$")
_RE_PACK = re.compile(r"^pack(\d+)$", re.IGNORECASE)
STEP5_HEAD_CHOICES = ("step5A", "step5B", "combined")
_STEP5_HEAD_BY_LOWER = {head.lower(): head for head in STEP5_HEAD_CHOICES}
_RE_STEP5_HEAD_SLUG = re.compile(r"^(\d+(?:_\d+)*)_(step5A|step5B|combined)$", re.IGNORECASE)
_STEP5_RUN_ID_ERROR = (
    "Step5 run-id must include consumed Step4 run prefix, e.g. 1_1, or use --run-id auto."
)


def normalize_iteration_id(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("iteration id 不能为空")
    m = _RE_ITER.match(s)
    if not m:
        raise ValueError(f"iteration id 须为 vN 形式，例如 v1；当前: {raw!r}")
    return f"v{int(m.group(1))}"


def parse_run_id(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("run id 不能为空")
    m = _RE_RUN_SLUG.match(s)
    if m:
        slug = m.group(1)
        parts = slug.split("_")
        norm = "_".join(str(int(p)) for p in parts)
        return norm
    raise ValueError(
        "run id 须为 slug：非负整数或由整数段用下划线连接（如 1、2、2_1、2_1_1）；"
        f"当前: {raw!r}"
    )


def parse_step5_head(raw: str | None) -> str:
    s = str(raw or "combined").strip()
    if not s:
        s = "combined"
    head = _STEP5_HEAD_BY_LOWER.get(s.lower())
    if head is None:
        raise ValueError(f"Step5 --head must be one of {', '.join(STEP5_HEAD_CHOICES)}; got {raw!r}")
    return head


def _split_step5_head_suffix(raw: str) -> tuple[str, str | None]:
    s = (raw or "").strip()
    m = _RE_STEP5_HEAD_SLUG.match(s)
    if not m:
        return s, None
    return m.group(1), parse_step5_head(m.group(2))


def step5_head_from_run_id(raw: str | None, *, default: str = "combined") -> str:
    _, head = _split_step5_head_suffix(str(raw or ""))
    return head or parse_step5_head(default)


def parse_step5_run_id(
    raw: str,
    *,
    head: str | None = None,
    require_head_suffix: bool = False,
) -> str:
    """Parse a Step5 run id.

    New formal head runs are ``{step4_run}_{step5_seq}_{head}``, for example
    ``1_1_step5A``. Legacy numeric ``{step4_run}_{step5_seq}`` remains accepted
    for already-written combined runs and explicit compatibility reads.
    """
    s = (raw or "").strip()
    if not s:
        raise ValueError("run id 不能为空")
    expected_head = parse_step5_head(head) if head else None
    numeric, parsed_head = _split_step5_head_suffix(s)
    try:
        numeric_slug = parse_run_id(numeric)
    except ValueError as exc:
        raise ValueError(_STEP5_RUN_ID_ERROR) from exc
    parts = numeric_slug.split("_")
    if len(parts) < 2:
        raise ValueError(_STEP5_RUN_ID_ERROR)
    if parsed_head is None:
        if require_head_suffix or expected_head in {"step5A", "step5B"}:
            raise ValueError(
                f"{_STEP5_RUN_ID_ERROR} Head-specific formal runs must use a suffix such as "
                f"{numeric_slug}_{expected_head or 'step5A'}."
            )
        return numeric_slug
    if expected_head is not None and parsed_head != expected_head:
        raise ValueError(f"Step5 run-id head suffix {parsed_head!r} does not match --head {expected_head!r}.")
    return f"{numeric_slug}_{parsed_head}"


def step5_numeric_slug(run_id: str) -> str:
    numeric, _head = _split_step5_head_suffix(str(run_id or ""))
    return parse_run_id(numeric)


def normalize_step5_run_id_for_step4(raw: str, *, step4_run: str, head: str) -> str:
    normalized = parse_step5_run_id(raw, head=head, require_head_suffix=parse_step5_head(head) in {"step5A", "step5B"})
    inferred = step4_slug_from_step5_slug(normalized)
    expected = parse_run_id(step4_run)
    if inferred != expected:
        raise ValueError(
            f"{_STEP5_RUN_ID_ERROR} Consumed Step4 run is {expected}, but run-id {normalized!r} implies {inferred!r}."
        )
    return normalized


def parse_pack_id(raw: str) -> str:
    s = (raw or "").strip().lower()
    m = _RE_PACK.match(s)
    if not m:
        raise ValueError(f"pack id 须为 packNN 形式，例如 pack01；当前: {raw!r}")
    return f"pack{int(m.group(1)):02d}"


def _max_suffix(parent: Path, pattern: re.Pattern[str], prefix: str) -> int:
    if not parent.is_dir():
        return 0
    best = 0
    for p in parent.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def _max_flat_run_index(parent: Path) -> int:
    """同级中 **纯数字** 单段目录名的最大序号（忽略含下划线的 slug 及其它名称）。"""
    if not parent.is_dir():
        return 0
    best = 0
    for p in parent.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        name = p.name
        if "_" in name:
            continue
        m = re.fullmatch(r"(\d+)", name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def next_run_id(parent: Path) -> str:
    return str(_max_flat_run_index(parent) + 1)


def next_pack_id(parent: Path) -> str:
    n = _max_suffix(parent, _RE_PACK, "pack") + 1
    return f"pack{n:02d}"


def allocate_child_dir(
    parent: Path,
    *,
    requested: Optional[str],
    kind: str,
) -> str:
    """
    kind: \"run\" | \"pack\"
    requested 为 None / \"\" / \"auto\" 时分配下一个目录名（默认递增 ``1``、``2``、…）；否则校验格式且目录不得已存在。
    """
    req = (requested or "").strip().lower()
    if kind == "run":
        if not req or req == "auto":
            rid = next_run_id(parent)
        else:
            rid = parse_run_id(requested)
        target = parent / rid
        if target.exists():
            raise FileExistsError(f"已存在目录（禁止覆盖）: {target}")
        return rid
    if kind == "pack":
        if not req or req == "auto":
            pid = next_pack_id(parent)
        else:
            pid = parse_pack_id(requested)
        target = parent / pid
        if target.exists():
            raise FileExistsError(f"已存在目录（禁止覆盖）: {target}")
        return pid
    raise ValueError(f"未知 kind: {kind!r}")


def allocate_multi_seed_run_id(multi_seed_parent: Path, requested: Optional[str]) -> str:
    """在 ``runs/task{T}/vN/meta/multi_seed/`` 下分配子目录名（与 ``allocate_child_dir(..., kind='run')`` 等价）。"""
    return allocate_child_dir(multi_seed_parent, requested=requested, kind="run")


def allocate_step5_run_id(step5_parent: Path, step4_run_parsed: str, *, head: str = "combined") -> str:
    """在 ``runs/step5`` 下按 ``{step4}_{n}_{head}`` 递增，避免 Step5A/Step5B 覆盖。"""
    base = parse_run_id(step4_run_parsed)
    head_norm = parse_step5_head(head)
    n = 1
    while True:
        cand = f"{base}_{n}_{head_norm}"
        target = step5_parent / cand
        if not target.exists():
            return cand
        n += 1


def step4_slug_from_step5_slug(step5_run: str) -> str:
    """由 Step5 目录名反推 Step4 目录名：去掉 Step5 序号与可选 head 后缀。

    例：``2_1_10`` → ``2_1``，``1_1_step5A`` → ``1``。

    **唯一约定**：全仓库凡涉及「step5_run → step4_run」须调用本函数（或下方别名），禁止各模块自写解析。
    """
    slug = step5_numeric_slug(step5_run)
    parts = slug.split("_")
    if len(parts) < 2:
        raise ValueError(_STEP5_RUN_ID_ERROR)
    return "_".join(parts[:-1])


def parse_stage_run_id(stage: str, raw: str) -> str:
    return parse_step5_run_id(raw) if str(stage or "").strip().lower() == "step5" else parse_run_id(raw)


def inferred_step4_slug_from_step5_run(step5_run: str) -> str:
    """与 :func:`step4_slug_from_step5_slug` 等价；供 manifest/校验层显式引用「反推规则入口」。"""
    return step4_slug_from_step5_slug(step5_run)


def allocate_step4_run_id(step4_parent: Path, step3_run_parsed: str) -> str:
    """在 ``train/step4/`` 下按 ``{step3}_{n}`` 递增分配首个不存在的目录名（如 step3=2 → 2_1, 2_2, …）。"""
    base = step3_run_parsed
    n = 1
    while True:
        cand = f"{base}_{n}"
        target = step4_parent / cand
        if not target.exists():
            return cand
        n += 1
