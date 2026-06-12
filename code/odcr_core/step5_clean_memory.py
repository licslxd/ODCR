"""Train-only memory controls for Step5 explanation generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from odcr_core.index_contract import GLOBAL_COL_ITEM, GLOBAL_COL_USER
from odcr_core.training_checkpoint import stable_hash


STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION = "odcr_step5_clean_memory_controls/2"
STEP5_CLEAN_MEMORY_EVAL_CONTROL_SCHEMA_VERSION = "odcr_step5_clean_memory_eval_control/1.0"
STEP5_CLEAN_MEMORY_SOURCE = "train_only_memory_controls"
STEP5_CLEAN_MEMORY_MODE = "train_memory_controls"
STEP5_CLEAN_MEMORY_MODEL_NAME = "ODCR-CleanMemory"
STEP5_CLEAN_MEMORY_PROTOCOL = "train_memory"
STEP5_CLEAN_MEMORY_EVIDENCE_SOURCE = "train_only_user_item_domain_memory"
STEP5_CLEAN_MEMORY_STEP3_SOURCE = "train_history_rating_prior_clean_fallback"
STEP5_CLEAN_MEMORY_STEP4_SOURCE = "train_memory_route_reliability_prior"
STEP5_CLEAN_MEMORY_AUDIT_STATUS = "clean_gate_passed"

_TEXT_COLS = (
    "user",
    "item",
    "rating",
    "review",
    "explanation",
    "content_evidence",
    "style_evidence",
    "clean_text",
    "polarity_anchor",
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
    "user_idx",
    "item_idx",
)

_REQUIRED_CLEAN_MEMORY_COLUMNS = (
    "content_evidence",
    "style_evidence",
    "domain_style_anchor",
    "local_style_residual_hint",
    "polarity_anchor",
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
    "sample_weight_hint",
    "route_scorer",
    "route_explainer",
    "step5_clean_control_source",
    "step5_clean_control_contract_version",
    "step5_control_mode",
    "step5_control_source",
    "step5_control_contract_version",
    "step5_leave_one_out_memory",
)


@dataclass(frozen=True)
class _MemoryEntry:
    signature: str
    content: str
    style: str
    polarity: str
    rating: float


@dataclass(frozen=True)
class _DomainMemory:
    user_entries: dict[int, list[_MemoryEntry]]
    item_entries: dict[int, list[_MemoryEntry]]
    domain_content: str
    domain_style: str
    default_rating: float


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def _clip_text(text: str, *, max_chars: int = 180) -> str:
    clean = " ".join(_as_text(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _clip01(v: Any, default: float = 0.5) -> float:
    try:
        x = float(v)
    except Exception:
        x = float(default)
    return max(0.0, min(1.0, x))


def _rating_to_polarity(rating: float) -> str:
    if rating >= 3.75:
        return "positive"
    if rating <= 2.75:
        return "negative"
    return "neutral"


def _signature(row: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "user": _as_text(row.get("user")),
            "item": _as_text(row.get("item")),
            "user_idx": _as_text(row.get("user_idx")),
            "item_idx": _as_text(row.get("item_idx")),
            "review": _as_text(row.get("review")),
            "target_text": _as_text(row.get("explanation")) or _as_text(row.get("clean_text")),
        },
        length=24,
    )


def _available_columns(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    cols = set(str(c) for c in header.columns)
    return [c for c in _TEXT_COLS if c in cols]


def _append_limited(bucket: dict[int, list[_MemoryEntry]], key: int, entry: _MemoryEntry, *, limit: int) -> None:
    values = bucket.setdefault(int(key), [])
    if len(values) < int(limit):
        values.append(entry)


def _build_domain_memory(
    path: Path,
    *,
    needed_users: set[int],
    needed_items: set[int],
    max_entries_per_key: int = 8,
    domain_prior_rows: int = 2048,
) -> _DomainMemory:
    if not path.is_file():
        return _DomainMemory({}, {}, "none", "none", 3.0)
    cols = _available_columns(path)
    user_entries: dict[int, list[_MemoryEntry]] = {}
    item_entries: dict[int, list[_MemoryEntry]] = {}
    content_prior: list[str] = []
    style_prior: list[str] = []
    ratings: list[float] = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=100_000):
        if "user_idx" in chunk.columns or "item_idx" in chunk.columns:
            mask = pd.Series(False, index=chunk.index)
            if needed_users and "user_idx" in chunk.columns:
                mask = mask | chunk["user_idx"].isin(needed_users)
            if needed_items and "item_idx" in chunk.columns:
                mask = mask | chunk["item_idx"].isin(needed_items)
            prior_need = len(content_prior) < int(domain_prior_rows) or len(style_prior) < int(domain_prior_rows)
            scan_df = chunk if prior_need else chunk.loc[mask]
        else:
            scan_df = chunk
            mask = pd.Series(True, index=chunk.index)
        for row in scan_df.to_dict("records"):
            content = _clip_text(_as_text(row.get("content_evidence")) or _as_text(row.get("review")))
            style = _clip_text(
                _as_text(row.get("style_evidence"))
                or _as_text(row.get("explanation"))
                or _as_text(row.get("clean_text"))
            )
            if len(content_prior) < int(domain_prior_rows) and content:
                content_prior.append(content)
            if len(style_prior) < int(domain_prior_rows) and style:
                style_prior.append(style)
            if len(ratings) < int(domain_prior_rows):
                try:
                    ratings.append(float(row.get("rating")))
                except Exception:
                    pass
        if "user_idx" in chunk.columns or "item_idx" in chunk.columns:
            chunk = chunk.loc[mask]
        for row in chunk.to_dict("records"):
            content = _clip_text(_as_text(row.get("content_evidence")) or _as_text(row.get("review")))
            style = _clip_text(
                _as_text(row.get("style_evidence"))
                or _as_text(row.get("explanation"))
                or _as_text(row.get("clean_text"))
            )
            if not content and not style:
                continue
            try:
                rating = float(row.get("rating"))
            except Exception:
                rating = 3.0
            ratings.append(rating)
            entry = _MemoryEntry(
                signature=_signature(row),
                content=content or "none",
                style=style or "none",
                polarity=_as_text(row.get("polarity_anchor")) or _rating_to_polarity(rating),
                rating=rating,
            )
            try:
                user_idx = int(row.get("user_idx"))
                _append_limited(user_entries, user_idx, entry, limit=max_entries_per_key)
            except Exception:
                pass
            try:
                item_idx = int(row.get("item_idx"))
                _append_limited(item_entries, item_idx, entry, limit=max_entries_per_key)
            except Exception:
                pass
    return _DomainMemory(
        user_entries=user_entries,
        item_entries=item_entries,
        domain_content=_clip_text(" ; ".join(content_prior[:8]), max_chars=240) or "none",
        domain_style=_clip_text(" ; ".join(style_prior[:8]), max_chars=240) or "none",
        default_rating=(sum(ratings) / len(ratings)) if ratings else 3.0,
    )


def _select(entries: Sequence[_MemoryEntry], *, exclude_signature: str, limit: int) -> list[_MemoryEntry]:
    out: list[_MemoryEntry] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.signature == exclude_signature or entry.signature in seen:
            continue
        seen.add(entry.signature)
        out.append(entry)
        if len(out) >= int(limit):
            break
    return out


def _domain_role(row: Mapping[str, Any], *, split_label: str) -> str:
    raw = _as_text(row.get("domain")).lower()
    if raw in {"auxiliary", "target"}:
        return raw
    return "target" if str(split_label).lower() in {"valid", "test"} else "target"


def _local_index_from_row(
    row: Mapping[str, Any],
    *,
    global_col: str,
    local_col: str,
    offset: int,
) -> int | None:
    if global_col in row and _as_text(row.get(global_col)) != "":
        try:
            return int(row.get(global_col)) - int(offset)
        except Exception:
            return None
    if local_col in row and _as_text(row.get(local_col)) != "":
        try:
            return int(row.get(local_col))
        except Exception:
            return None
    return None


def _local_indices(row: Mapping[str, Any], *, role: str, index_contract: Mapping[str, Any]) -> tuple[int | None, int | None]:
    user_offset = int(index_contract.get(f"{role}_user_offset") or 0)
    item_offset = int(index_contract.get(f"{role}_item_offset") or 0)
    user_local = _local_index_from_row(
        row,
        global_col=GLOBAL_COL_USER,
        local_col="user_idx",
        offset=user_offset,
    )
    item_local = _local_index_from_row(
        row,
        global_col=GLOBAL_COL_ITEM,
        local_col="item_idx",
        offset=item_offset,
    )
    return user_local, item_local


def _build_row_controls(
    row: Mapping[str, Any],
    *,
    role: str,
    memory: _DomainMemory,
    index_contract: Mapping[str, Any],
    split_label: str,
    leave_one_out: bool,
) -> dict[str, Any]:
    user_local, item_local = _local_indices(row, role=role, index_contract=index_contract)
    current_sig = _signature(row) if leave_one_out else ""
    user_hist = _select(memory.user_entries.get(int(user_local), []) if user_local is not None else [], exclude_signature=current_sig, limit=2)
    item_hist = _select(memory.item_entries.get(int(item_local), []) if item_local is not None else [], exclude_signature=current_sig, limit=3)
    selected = [*item_hist, *user_hist]
    if selected:
        rating_pred = sum(e.rating for e in selected) / float(len(selected))
    elif leave_one_out:
        rating_pred = 3.0
    else:
        rating_pred = float(memory.default_rating)
    content_bits: list[str] = []
    if item_hist:
        content_bits.append("ITEM_HISTORY: " + " ; ".join(e.content for e in item_hist))
    if user_hist:
        content_bits.append("USER_HISTORY: " + " ; ".join(e.content for e in user_hist))
    if not content_bits:
        if leave_one_out:
            content_bits.append("DOMAIN_PRIOR: train_memory_domain_prior_without_row_specific_text")
        else:
            content_bits.append("DOMAIN_PRIOR: " + memory.domain_content)
    style_bits: list[str] = []
    if user_hist:
        style_bits.append("USER_STYLE: " + " ; ".join(e.style for e in user_hist))
    if item_hist:
        style_bits.append("ITEM_STYLE: " + " ; ".join(e.style for e in item_hist[:2]))
    if not style_bits:
        if leave_one_out:
            style_bits.append("DOMAIN_STYLE: train_memory_domain_style_prior_without_row_specific_text")
        else:
            style_bits.append("DOMAIN_STYLE: " + memory.domain_style)
    coverage = min(1.0, 0.18 * len(user_hist) + 0.22 * len(item_hist) + (0.20 if selected else 0.0))
    content_anchor = max(0.15 if selected else 0.05, coverage)
    style_anchor = max(0.15 if selected else 0.05, min(1.0, 0.16 * len(user_hist) + 0.14 * len(item_hist) + 0.20))
    evidence_quality = min(1.0, 0.62 * content_anchor + 0.38 * style_anchor)
    uncertainty = 1.0 - evidence_quality
    sampler_weight = _clip01(row.get("sampler_weight", 1.0), default=1.0)
    sample_weight = max(0.05, evidence_quality) * sampler_weight
    return {
        "rating": round(float(rating_pred), 4),
        "content_evidence": _clip_text(" | ".join(content_bits), max_chars=420),
        "style_evidence": _clip_text(" | ".join(style_bits), max_chars=360),
        "domain_style_anchor": f"{role}:train_memory_style_prior:{_rating_to_polarity(rating_pred)}",
        "local_style_residual_hint": (
            f"source=train_only_memory; split={split_label}; "
            f"user_history={len(user_hist)}; item_history={len(item_hist)}"
        ),
        "polarity_anchor": _rating_to_polarity(float(rating_pred)),
        "content_anchor_score": round(float(content_anchor), 4),
        "style_anchor_score": round(float(style_anchor), 4),
        "evidence_quality_prior": round(float(evidence_quality), 4),
        "uncertainty_score": round(float(uncertainty), 4),
        "entropy_score": round(float(uncertainty), 4),
        "confidence_bucket": 2 if evidence_quality >= 0.67 else (1 if evidence_quality >= 0.34 else 0),
        "route_scorer": 1,
        "route_explainer": 1,
        "sample_weight_hint": round(float(sample_weight), 6),
        "cf_reliability_score": round(float(evidence_quality), 4),
        "content_retention_score": round(float(content_anchor), 4),
        "style_shift_score": round(float(max(0.0, 1.0 - style_anchor)), 4),
        "rating_stability_score": round(float(evidence_quality), 4),
        "text_quality_score": round(float(evidence_quality), 4),
        "step5_clean_control_source": STEP5_CLEAN_MEMORY_SOURCE,
        "step5_clean_control_contract_version": STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION,
        "step5_control_mode": STEP5_CLEAN_MEMORY_MODE,
        "step5_control_source": STEP5_CLEAN_MEMORY_SOURCE,
        "step5_control_contract_version": STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION,
        "step5_leave_one_out_memory": bool(leave_one_out),
    }


def apply_step5_clean_memory_controls(
    df: pd.DataFrame,
    *,
    repo_root: str | Path,
    target_domain: str,
    auxiliary_domain: str,
    index_contract: Mapping[str, Any],
    split_label: str,
    leave_one_out: bool,
) -> pd.DataFrame:
    """Replace Step5 control columns with train-only memory controls."""

    if df.empty:
        return df.copy()
    root = Path(repo_root)
    needed: dict[str, dict[str, set[int]]] = {
        "target": {"users": set(), "items": set()},
        "auxiliary": {"users": set(), "items": set()},
    }
    for row in df.to_dict("records"):
        role = _domain_role(row, split_label=split_label)
        user_local, item_local = _local_indices(row, role=role, index_contract=index_contract)
        if user_local is not None and user_local >= 0:
            needed[role]["users"].add(int(user_local))
        if item_local is not None and item_local >= 0:
            needed[role]["items"].add(int(item_local))
    memories = {
        "target": _build_domain_memory(
            root / "data" / str(target_domain) / "train.csv",
            needed_users=needed["target"]["users"],
            needed_items=needed["target"]["items"],
        ),
        "auxiliary": _build_domain_memory(
            root / "data" / str(auxiliary_domain) / "train.csv",
            needed_users=needed["auxiliary"]["users"],
            needed_items=needed["auxiliary"]["items"],
        ),
    }
    out = df.copy()
    rows = out.to_dict("records")
    updates: list[dict[str, Any]] = []
    for row in rows:
        role = _domain_role(row, split_label=split_label)
        memory = memories.get(role) or memories["target"]
        updates.append(
            _build_row_controls(
                row,
                role=role,
                memory=memory,
                index_contract=index_contract,
                split_label=split_label,
                leave_one_out=bool(leave_one_out),
            )
        )
    for key in updates[0].keys():
        out[key] = [row[key] for row in updates]
    return out


def require_step5_clean_memory_controls(df: pd.DataFrame, *, ctx: str, require_leave_one_out: bool = False) -> None:
    missing = [col for col in _REQUIRED_CLEAN_MEMORY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{ctx} missing CleanMemory control columns {missing}; run Step5 clean-memory control construction first."
        )
    bad_source = df["step5_clean_control_source"].astype(str).str.strip() != STEP5_CLEAN_MEMORY_SOURCE
    if bool(bad_source.any()):
        raise ValueError(f"{ctx} has non-clean Step5 controls; expected {STEP5_CLEAN_MEMORY_SOURCE}.")
    bad_contract = (
        df["step5_clean_control_contract_version"].astype(str).str.strip()
        != STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION
    )
    if bool(bad_contract.any()):
        raise ValueError(
            f"{ctx} has stale CleanMemory control contract; expected {STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION}."
        )
    bad_mode = df["step5_control_mode"].astype(str).str.strip() != STEP5_CLEAN_MEMORY_MODE
    if bool(bad_mode.any()):
        raise ValueError(f"{ctx} has non-CleanMemory control mode; expected {STEP5_CLEAN_MEMORY_MODE}.")
    bad_control_source = df["step5_control_source"].astype(str).str.strip() != STEP5_CLEAN_MEMORY_SOURCE
    if bool(bad_control_source.any()):
        raise ValueError(f"{ctx} has non-CleanMemory control source; expected {STEP5_CLEAN_MEMORY_SOURCE}.")
    bad_control_contract = (
        df["step5_control_contract_version"].astype(str).str.strip()
        != STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION
    )
    if bool(bad_control_contract.any()):
        raise ValueError(
            f"{ctx} has stale Step5 control contract; expected {STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION}."
        )
    if require_leave_one_out:
        vals = df["step5_leave_one_out_memory"].astype(str).str.strip().str.lower()
        if not bool(vals.isin({"true", "1", "yes"}).all()):
            raise ValueError(f"{ctx} has rows without leave-one-out train-memory controls.")
