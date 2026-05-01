"""torchrun 子进程启动时的单行配置探测日志（可 grep，与 odcr 主链 ENV 对齐）。"""
from __future__ import annotations

import json
import os
from pathlib import Path

_MANIFEST_NAME = "manifest.json"


def _read_manifest_fp(data: dict, key: str) -> str:
    fp = data.get(key)
    if fp is None or str(fp).strip() == "":
        return "<missing>"
    return str(fp)


def _read_training_semantic_fingerprint() -> str:
    env_fp = (os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT") or "").strip()
    if env_fp:
        return env_fp
    md = (os.environ.get("ODCR_MANIFEST_DIR") or "").strip()
    if not md:
        return "<missing>"
    path = Path(md) / _MANIFEST_NAME
    if not path.is_file():
        return "<missing>"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "<missing>"
    return _read_manifest_fp(data, "training_semantic_fingerprint")


def _read_generation_semantic_fingerprint() -> str:
    env_fp = (os.environ.get("ODCR_GENERATION_SEMANTIC_FINGERPRINT") or "").strip()
    if env_fp:
        return env_fp
    md = (os.environ.get("ODCR_MANIFEST_DIR") or "").strip()
    if not md:
        return "<missing>"
    path = Path(md) / _MANIFEST_NAME
    if not path.is_file():
        return "<missing>"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "<missing>"
    return _read_manifest_fp(data, "generation_semantic_fingerprint")


def _read_runtime_diagnostics_fingerprint() -> str:
    env_fp = (os.environ.get("ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT") or "").strip()
    if env_fp:
        return env_fp
    md = (os.environ.get("ODCR_MANIFEST_DIR") or "").strip()
    if not md:
        return "<missing>"
    path = Path(md) / _MANIFEST_NAME
    if not path.is_file():
        return "<missing>"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "<missing>"
    fp = data.get("runtime_diagnostics_fingerprint")
    if fp is None or str(fp).strip() == "":
        return "<missing>"
    return str(fp)


def _classify_training_payload(*, required: bool) -> str:
    raw = (os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON") or "").strip()
    if not raw:
        return "missing" if required else "absent"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "missing"
    if not isinstance(obj, dict):
        return "missing"
    return "loaded"


def _classify_decode_profile(*, required: bool) -> str:
    raw = (os.environ.get("ODCR_DECODE_PROFILE_JSON") or "").strip()
    if not raw:
        return "missing" if required else "absent"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "missing"
    if not isinstance(obj, dict):
        return "missing"
    return "loaded"


def _classify_rerank_profile(*, active: bool) -> str:
    if not active:
        return "absent"
    raw = (os.environ.get("ODCR_RERANK_PROFILE_JSON") or "").strip()
    if not raw:
        return "missing"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "missing"
    if not isinstance(obj, dict):
        return "missing"
    return "loaded"


def _read_full_bleu_eval_startup_line(*, required: bool) -> str:
    raw = (os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON") or "").strip()
    if not raw:
        return "[full_bleu_eval] payload=missing" if required else "[full_bleu_eval] payload=absent"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "[full_bleu_eval] payload=invalid_json"
    row = obj.get("training_row")
    if not isinstance(row, dict):
        return "[full_bleu_eval] training_row=missing"
    try:
        from config import format_full_bleu_eval_resolved_log_line, resolve_full_bleu_eval_from_training_row

        sched = resolve_full_bleu_eval_from_training_row(row)
        return format_full_bleu_eval_resolved_log_line(sched)
    except Exception as e:
        return f"[full_bleu_eval] resolve_error={e!s}"


def print_startup_config_check(*, stage: str, command: str) -> None:
    """仅在 rank0 打印一行；stage 为 step3|step4|step5，command 为子命令或 run。"""
    if os.environ.get("RANK", "0") != "0":
        return
    need_tp = stage in ("step3", "step5")
    need_decode = stage == "step5"
    rerank_active = stage == "step5" and command == "eval-rerank"
    tfp = _read_training_semantic_fingerprint()
    gfp = _read_generation_semantic_fingerprint()
    rdfp = _read_runtime_diagnostics_fingerprint()
    tp = _classify_training_payload(required=need_tp)
    dec = _classify_decode_profile(required=need_decode)
    rr = _classify_rerank_profile(active=rerank_active)
    fbe = _read_full_bleu_eval_startup_line(required=need_tp)
    print(
        "[startup_config_check] "
        f"stage={stage} command={command} "
        f"training_semantic_fingerprint={tfp} "
        f"generation_semantic_fingerprint={gfp} "
        f"runtime_diagnostics_fingerprint={rdfp} "
        f"effective_training_payload_loaded={tp} "
        f"decode_profile_loaded={dec} "
        f"rerank_profile_loaded={rr} "
        f"{fbe}",
        flush=True,
    )
