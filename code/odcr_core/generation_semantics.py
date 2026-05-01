"""生成/评测语义快照与 SHA256 指纹（无 torch/nltk），供 manifest、metrics、phase1 共用。"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Tuple


def _generation_cfg_opt_int(raw: Any) -> Optional[int]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("null", "none"):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def compute_generation_semantic_family_tag(core: Mapping[str, Any]) -> str:
    """与 generation_semantic_resolved 核心字段一致的人类可读标签（目录对照与 summary）。"""
    st = str(core.get("strategy") or "").strip().lower() or "unknown"
    r = float(core.get("repetition_penalty") or 0.0)
    rs = str(r).replace(".", "p")
    ml = int(core.get("max_explanation_length") or 0)
    nr = core.get("no_repeat_ngram_size")
    nrs = "nrna" if nr is None else f"nr{int(nr)}"
    mn = core.get("min_len")
    mns = "minna" if mn is None else f"min{int(mn)}"
    if st == "greedy":
        base = "greedy"
    elif st == "uncertainty_low_temp_top_k":
        t = float(core.get("temperature") or 0.0)
        g = float(core.get("gap_threshold") or 0.0)
        pf = int(core.get("prefix_greedy_steps") or 0)
        kk = int(core.get("top_k") or 0)
        ts = str(t).replace(".", "p")
        gs = str(g).replace(".", "p")
        base = f"uncertainty_t{ts}_g{gs}_pre{pf}_k{kk}"
    else:
        t = float(core.get("temperature") or 0.0)
        p = float(core.get("top_p") or 0.0)
        ts = str(t).replace(".", "p")
        ps = str(p).replace(".", "p")
        base = f"nucleus_t{ts}_p{ps}"
    return f"{base}_r{rs}_l{ml}_{nrs}_{mns}"


def build_generation_semantic_resolved_and_fingerprint(
    decode_cfg: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """
    人类可读的生成语义快照 + 稳定指纹（SHA256 hex）。

    权威字段含 strategy / temperature / top_p / repetition_penalty / max_explanation_length /
    label_smoothing / decode_seed / no_repeat_ngram_size / min_len / generation_semantic_family_tag；
    缺失的整型约束在 resolved 中显式为 null（JSON）。
    """
    seed_raw = decode_cfg.get("decode_seed")
    dec_seed: Optional[int]
    if seed_raw is None or seed_raw == "":
        dec_seed = None
    else:
        try:
            dec_seed = int(seed_raw)
        except (TypeError, ValueError):
            dec_seed = None
    _pfx = _generation_cfg_opt_int(decode_cfg.get("prefix_greedy_steps"))
    _topk = _generation_cfg_opt_int(decode_cfg.get("top_k"))
    core: Dict[str, Any] = {
        "strategy": str(decode_cfg.get("decode_strategy", "") or "").strip().lower(),
        "temperature": float(decode_cfg.get("generate_temperature", 0.0)),
        "top_p": float(decode_cfg.get("generate_top_p", 0.0)),
        "gap_threshold": float(decode_cfg.get("gap_threshold", 0.35)),
        "prefix_greedy_steps": 4 if _pfx is None else max(0, int(_pfx)),
        "top_k": 5 if _topk is None else max(1, int(_topk)),
        "repetition_penalty": float(decode_cfg.get("repetition_penalty", 0.0)),
        "max_explanation_length": int(decode_cfg.get("max_explanation_length", 0)),
        "label_smoothing": float(decode_cfg.get("label_smoothing", 0.0)),
        "decode_seed": dec_seed,
        "no_repeat_ngram_size": _generation_cfg_opt_int(decode_cfg.get("no_repeat_ngram_size")),
        "min_len": _generation_cfg_opt_int(decode_cfg.get("min_len")),
        "soft_max_len": _generation_cfg_opt_int(decode_cfg.get("soft_max_len")),
        "hard_max_len": _generation_cfg_opt_int(decode_cfg.get("hard_max_len")),
        "eos_boost_start": _generation_cfg_opt_int(decode_cfg.get("eos_boost_start")),
        "eos_boost_value": float(decode_cfg.get("eos_boost_value", 0.0)),
        "tail_temperature": float(decode_cfg.get("tail_temperature", -1.0)),
        "tail_top_p": float(decode_cfg.get("tail_top_p", -1.0)),
        "forbid_eos_after_open_quote": bool(decode_cfg.get("forbid_eos_after_open_quote", True)),
        "forbid_eos_after_open_bracket": bool(decode_cfg.get("forbid_eos_after_open_bracket", True)),
        "forbid_bad_terminal_tokens": bool(decode_cfg.get("forbid_bad_terminal_tokens", True)),
        "decode_token_repeat_window": _generation_cfg_opt_int(decode_cfg.get("decode_token_repeat_window")),
        "decode_token_repeat_max": _generation_cfg_opt_int(decode_cfg.get("decode_token_repeat_max")),
    }
    resolved: Dict[str, Any] = {
        **core,
        "generation_semantic_family_tag": compute_generation_semantic_family_tag(core),
    }

    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fp = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return resolved, fp


__all__ = [
    "build_generation_semantic_resolved_and_fingerprint",
    "compute_generation_semantic_family_tag",
]
