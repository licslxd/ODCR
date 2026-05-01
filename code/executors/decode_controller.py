"""
手写解码控制器：与 HF generate 解耦，统一 nucleus / greedy + 尾部退火 + 终端约束。

由 ``step5_engine.Model`` 调用；有效参数字典见 ``build_generate_kwargs_effective_v2``。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace as dc_replace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


PAPER_DECODE_CONTROLLER_SCHEMA = "odcr_decode_controller/1.0"
DECODE_BACKEND_LEGACY = "legacy_full_recompute"
DECODE_BACKEND_KV_FAST = "sdpa_kv_fast"
DECODE_BACKEND_KV_SAFE = "sdpa_kv_safe"
# 非兼容迁移：旧名 sdpa_kv 不再作为合法 resolved 值（见 resolve_decode_backend_name）。

DECODE_BACKEND_FALLBACK_RAISE = "raise"
DECODE_BACKEND_FALLBACK_SYNC_THEN_FALLBACK = "sync_then_fallback"

# 解码诊断/统计张量（entropy、log_probs 等）固定 fp32，避免在 bf16 autocast 下与 logits 混 dtype 触发 index_put 失败。
_DIAG_DTYPE = torch.float32


def decode_as_inference_enhancement(strategy: str) -> bool:
    """
    Step5 ODCR 主线约束：decode/rerank 仅作为 inference enhancement，不参与 scorer 主训练图。
    """
    return str(strategy or "").strip().lower() in {"greedy", "nucleus", "uncertainty_low_temp_top_k"}


def _sampling_ctx_str(ctx: Optional[Mapping[str, Any]], key: str, default: str = "unknown") -> str:
    if not ctx:
        return default
    return str(ctx.get(key, default))


def _assign_diag_by_index1d(
    dst: torch.Tensor,
    indices: torch.Tensor,
    src: torch.Tensor,
    *,
    decode_strategy: str,
    sampling_diag_context: Optional[Mapping[str, Any]],
) -> None:
    """按一维索引写入诊断张量：显式对齐 dtype/device，并在仍失败时附带 decode 上下文。"""
    if indices.numel() == 0:
        return
    _tev = _sampling_ctx_str(sampling_diag_context, "train_time_eval")
    _be = _sampling_ctx_str(sampling_diag_context, "backend")
    aligned = src.to(device=dst.device, dtype=dst.dtype)
    try:
        dst[indices] = aligned
    except RuntimeError as exc:
        raise RuntimeError(
            "decode diagnostic index_put failed: "
            f"decode_strategy={decode_strategy!r}, backend={_be!r}, train_time_eval={_tev!r}, "
            f"dst_dtype={dst.dtype}, src_dtype={src.dtype}, aligned_dtype={aligned.dtype}, "
            f"dst_shape={tuple(dst.shape)}, src_shape={tuple(src.shape)}, aligned_shape={tuple(aligned.shape)}"
        ) from exc


def empty_uncertainty_decode_aggregate() -> Dict[str, Any]:
    """跨 batch / 跨 rank 累加用容器（由 mainline valid decode 等路径填充）。"""
    return {
        "uncertainty_total_decision_count": 0,
        "uncertainty_trigger_count": 0,
        "uncertainty_first_trigger_steps": [],  # type: ignore[dict-item]
        "uncertainty_trigger_entropy_sum": 0.0,
        "uncertainty_trigger_entropy_count": 0,
        "uncertainty_trigger_entropy_values": [],  # type: ignore[dict-item]
    }


def merge_uncertainty_run_into_aggregate(
    agg: MutableMapping[str, Any],
    run: Optional[Mapping[str, Any]],
) -> None:
    """将单次 generate（一个 micro-batch）的统计合并进 agg。"""
    if not run:
        return
    agg["uncertainty_total_decision_count"] = int(agg.get("uncertainty_total_decision_count", 0)) + int(
        run.get("total_decision_count", 0)
    )
    agg["uncertainty_trigger_count"] = int(agg.get("uncertainty_trigger_count", 0)) + int(
        run.get("trigger_count", 0)
    )
    agg["uncertainty_trigger_entropy_sum"] = float(agg.get("uncertainty_trigger_entropy_sum", 0.0)) + float(
        run.get("trigger_entropy_sum", 0.0)
    )
    agg["uncertainty_trigger_entropy_count"] = int(agg.get("uncertainty_trigger_entropy_count", 0)) + int(
        run.get("trigger_entropy_count", 0)
    )
    tev = run.get("trigger_entropy_values") or []
    if tev:
        agg.setdefault("uncertainty_trigger_entropy_values", [])
        agg["uncertainty_trigger_entropy_values"].extend(float(x) for x in tev)  # type: ignore[union-attr]
    steps = run.get("first_trigger_steps") or []
    if steps:
        agg.setdefault("uncertainty_first_trigger_steps", [])
        agg["uncertainty_first_trigger_steps"].extend(int(x) for x in steps)  # type: ignore[union-attr]


def reduce_uncertainty_aggregates(partials: Sequence[Optional[Mapping[str, Any]]]) -> Dict[str, Any]:
    """DDP 各 rank 局部 agg 合并（与 empty_uncertainty_decode_aggregate 同结构）。"""
    out = empty_uncertainty_decode_aggregate()
    for p in partials:
        if not p:
            continue
        out["uncertainty_total_decision_count"] += int(p.get("uncertainty_total_decision_count", 0))
        out["uncertainty_trigger_count"] += int(p.get("uncertainty_trigger_count", 0))
        out["uncertainty_trigger_entropy_sum"] += float(p.get("uncertainty_trigger_entropy_sum", 0.0))
        out["uncertainty_trigger_entropy_count"] += int(p.get("uncertainty_trigger_entropy_count", 0))
        ev = p.get("uncertainty_trigger_entropy_values") or []
        if ev:
            out.setdefault("uncertainty_trigger_entropy_values", [])
            out["uncertainty_trigger_entropy_values"].extend(float(x) for x in ev)  # type: ignore[union-attr]
        lst = p.get("uncertainty_first_trigger_steps") or []
        out["uncertainty_first_trigger_steps"].extend(int(x) for x in lst)  # type: ignore[union-attr]
    return out


def resolve_decode_backend_name(raw: Any) -> str:
    """将配置/覆盖中的原始字符串解析为 canonical backend 名（fast / safe / legacy）。"""
    s = str(raw or "").strip().lower()
    if not s:
        return DECODE_BACKEND_KV_FAST
    if s == "kv_cache":
        return DECODE_BACKEND_KV_FAST
    if s in (DECODE_BACKEND_LEGACY, DECODE_BACKEND_KV_FAST, DECODE_BACKEND_KV_SAFE):
        return s
    raise ValueError(
        f"未知 decode_backend={raw!r}；允许: {DECODE_BACKEND_LEGACY}, {DECODE_BACKEND_KV_FAST}, {DECODE_BACKEND_KV_SAFE}（kv_cache→fast）"
    )


def resolve_decode_backend_fallback_policy(raw: Any) -> str:
    s = str(raw or DECODE_BACKEND_FALLBACK_RAISE).strip().lower()
    if s in (DECODE_BACKEND_FALLBACK_RAISE, DECODE_BACKEND_FALLBACK_SYNC_THEN_FALLBACK):
        return s
    raise ValueError(
        f"未知 decode_backend_fallback_policy={raw!r}；允许: {DECODE_BACKEND_FALLBACK_RAISE}, {DECODE_BACKEND_FALLBACK_SYNC_THEN_FALLBACK}"
    )


def decode_backend_uses_kv_cache(name: str) -> bool:
    b = resolve_decode_backend_name(name)
    return b in (DECODE_BACKEND_KV_FAST, DECODE_BACKEND_KV_SAFE)


def decode_exception_blocks_fallback(exc: BaseException) -> bool:
    """为 True 时禁止用 legacy 掩盖（CUDA / SDPA / 典型 device 侧失败）。"""
    _cuda_err = getattr(torch.cuda, "CudaError", ())
    if _cuda_err and isinstance(exc, _cuda_err):  # type: ignore[arg-type]
        return True
    msg = str(exc).lower()
    if "cuda" in msg:
        return True
    needles = (
        "device-side assert",
        "device assert",
        "cudnn",
        "scaled_dot_product_attention",
        "sdpa",
        "index out of range",
        "index out of bounds",
        "index_put",
        "acceleratorerror",
        "nccl",
    )
    return any(n in msg for n in needles)


@dataclass
class GenerateConfig:
    """单次生成运行时的解码配置（由 FinalTrainingConfig / yaml 填充）。"""

    strategy: str = "greedy"
    temperature: float = 0.8
    top_p: float = 0.9
    # uncertainty_low_temp_top_k：top1-top2 logit 差低于阈值时在已约束 logits 上做低温 top-k 采样
    gap_threshold: float = 0.35
    prefix_greedy_steps: int = 4
    top_k: int = 5
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 0
    min_len: int = 0
    soft_max_len: int = 0
    hard_max_len: int = 25
    eos_boost_start: int = 9999
    eos_boost_value: float = 0.0
    tail_temperature: float = -1.0
    tail_top_p: float = -1.0
    forbid_eos_after_open_quote: bool = True
    forbid_eos_after_open_bracket: bool = True
    forbid_bad_terminal_tokens: bool = True
    bad_terminal_token_ids: Tuple[int, ...] = ()
    token_repeat_window: int = 4
    token_repeat_max: int = 2
    decode_seed: Optional[int] = None
    # uncertainty_low_temp_top_k：对参与低温 top-k 的分布计算 trigger entropy 时用 log(p+eps)
    uncertainty_entropy_eps: float = 1e-8
    # None：沿用 Model.decode_backend / Model.decode_backend_fallback_policy
    decode_backend: Optional[str] = None
    decode_backend_fallback_policy: Optional[str] = None
    # 供日志区分训练内 valid 监控等入口（如 train_time_eval）
    decode_run_context: Optional[str] = None


def merge_generate_config_with_override(base: GenerateConfig, override: Mapping[str, Any]) -> GenerateConfig:
    """将 override 合并进 base（仅允许 GenerateConfig 已有字段）；用于单次 generate 临时覆盖，不回写模型状态。"""
    if not override:
        return base
    names = {f.name for f in fields(GenerateConfig)}
    unknown = set(override.keys()) - names
    if unknown:
        raise ValueError(f"cfg_override 含未知字段 {sorted(unknown)}；允许: {sorted(names)}")
    return dc_replace(base, **dict(override))


def coerce_generate_cfg_override(
    base: GenerateConfig, cfg_override: Optional[Union[GenerateConfig, Mapping[str, Any]]]
) -> Optional[GenerateConfig]:
    """与 generate / generate_with_token_logprobs 对齐：None、GenerateConfig 或 dict 映射为单次解码用 GenerateConfig。"""
    if cfg_override is None:
        return None
    if isinstance(cfg_override, GenerateConfig):
        return cfg_override
    return merge_generate_config_with_override(base, cfg_override)


@dataclass
class GenerationState:
    decoder_input_ids: torch.Tensor
    active: torch.Tensor
    step: int
    recent_tokens: List[List[int]] = field(default_factory=list)


def _effective_tail_scalar(base: float, tail: float, step: int, soft: int, hard: int) -> float:
    if tail < 0 or hard <= soft or step < soft:
        return float(base)
    alpha = (float(step) - float(soft)) / float(max(1, hard - soft))
    alpha = min(1.0, max(0.0, alpha))
    return float(base) * (1.0 - alpha) + float(tail) * alpha


def _unbalanced_delimiters(text: str) -> bool:
    """轻量括号/引号未闭合检测（字符级，供 forbid_eos）。"""
    s = text or ""
    p = 0
    b = 0
    quote = False
    esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            quote = not quote
            continue
        if quote:
            continue
        if ch == "(":
            p += 1
        elif ch == ")":
            p = max(0, p - 1)
        elif ch == "[":
            b += 1
        elif ch == "]":
            b = max(0, b - 1)
    return quote or p > 0 or b > 0


def prepare_logits(hidden_last: torch.Tensor, hidden2token: torch.nn.Module) -> torch.Tensor:
    return hidden2token(hidden_last)


def apply_repetition_penalty_logits(
    logits: torch.Tensor, decoder_input_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    if penalty <= 1.0:
        return logits
    appeared = torch.zeros(logits.size(0), logits.size(1), device=logits.device, dtype=torch.bool)
    appeared.scatter_(1, decoder_input_ids, True)
    adjusted = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(appeared, adjusted, logits)


def apply_no_repeat_ngram_logits(
    logits: torch.Tensor, decoder_input_ids: torch.Tensor, ngram_size: int
) -> None:
    if ngram_size <= 0:
        return
    B, V = logits.shape
    for b in range(B):
        seq = decoder_input_ids[b].tolist()
        cur_len = len(seq)
        if cur_len + 1 < ngram_size:
            continue
        generated = set()
        for i in range(cur_len - ngram_size + 1):
            generated.add(tuple(seq[i : i + ngram_size]))
        prefix = tuple(seq[cur_len - (ngram_size - 1) : cur_len])
        banned = {ng[-1] for ng in generated if ng[:-1] == prefix}
        if not banned:
            continue
        idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
        logits[b, idx] = torch.finfo(logits.dtype).min


def apply_token_repeat_suppression(
    logits: torch.Tensor,
    recent_rows: Sequence[Sequence[int]],
    *,
    window: int,
    max_same: int,
) -> None:
    if window <= 0 or max_same <= 0:
        return
    B = logits.size(0)
    for b in range(B):
        hist = list(recent_rows[b])[-window:]
        if len(hist) < max_same:
            continue
        t = hist[-1]
        if sum(1 for x in hist if x == t) >= max_same:
            logits[b, t] = torch.finfo(logits.dtype).min


def apply_min_len_eos_mask(
    logits: torch.Tensor,
    *,
    eos_id: int,
    gen_so_far: int,
    min_len: int,
) -> None:
    if eos_id < 0 or min_len <= 0:
        return
    if gen_so_far < min_len:
        logits[:, eos_id] = torch.finfo(logits.dtype).min


def apply_unbalanced_delimiter_eos_mask(
    logits: torch.Tensor,
    *,
    eos_id: int,
    decoded_texts: Sequence[str],
    cfg: GenerateConfig,
) -> None:
    if eos_id < 0 or not decoded_texts:
        return
    if not (cfg.forbid_eos_after_open_bracket or cfg.forbid_eos_after_open_quote):
        return
    for b in range(logits.size(0)):
        t = decoded_texts[b] if b < len(decoded_texts) else ""
        if _unbalanced_delimiters(t):
            logits[b, eos_id] = torch.finfo(logits.dtype).min


def forbid_eos_if_bad_tail_token(
    logits: torch.Tensor,
    *,
    eos_id: int,
    tail_token_ids: torch.Tensor,
    bad_ids: Tuple[int, ...],
) -> None:
    if eos_id < 0 or not bad_ids:
        return
    # tail_token_ids: (B,)
    for b in range(logits.size(0)):
        tid = int(tail_token_ids[b].item())
        if tid in bad_ids:
            logits[b, eos_id] = torch.finfo(logits.dtype).min


def apply_sampling_schedule(
    cfg: GenerateConfig, step: int
) -> Tuple[float, float]:
    hard = max(1, int(cfg.hard_max_len))
    soft = int(cfg.soft_max_len)
    if soft <= 0:
        soft = max(1, int(hard * 0.65))
    soft = min(soft, hard - 1) if hard > 1 else 1
    t_base = float(cfg.temperature)
    p_base = float(cfg.top_p)
    t_tail = float(cfg.tail_temperature) if cfg.tail_temperature >= 0 else t_base
    p_tail = float(cfg.tail_top_p) if cfg.tail_top_p >= 0 else p_base
    eff_t = _effective_tail_scalar(t_base, t_tail, step, soft, hard)
    eff_p = _effective_tail_scalar(p_base, p_tail, step, soft, hard)
    eff_t = max(eff_t, 1e-8)
    eff_p = min(1.0, max(1e-6, eff_p))
    return eff_t, eff_p


def apply_eos_boost(
    logits: torch.Tensor,
    *,
    eos_id: int,
    step: int,
    cfg: GenerateConfig,
) -> None:
    if eos_id < 0 or cfg.eos_boost_value == 0.0:
        return
    if step >= int(cfg.eos_boost_start):
        logits[:, eos_id] = logits[:, eos_id] + float(cfg.eos_boost_value)


def assert_topk_slot_indices_valid_for_gather(
    sampled_inner: torch.Tensor,
    inds: torch.Tensor,
    *,
    decode_strategy: str,
    decode_top_k: int,
    train_time_eval: str,
    backend: str,
    policy: str,
) -> torch.Tensor:
    """将 ``sampled_inner`` 规范为 ``(B,1)`` long，并在 ``gather(inds,1,…)`` 前做槽位范围校验。"""
    si = sampled_inner
    if si.dim() == 1:
        si = si.unsqueeze(1)
    si = si.long()
    kcols = int(inds.size(1))
    if kcols <= 0:
        raise RuntimeError(
            f"top-k gather: empty inds.size(1); decode_strategy={decode_strategy!r}, decode_top_k={decode_top_k}, "
            f"inds.shape={tuple(inds.shape)}, train_time_eval={train_time_eval!r}, backend={backend!r}, policy={policy!r}"
        )
    min_v = int(si.min().item())
    max_v = int(si.max().item())
    if min_v < 0 or max_v >= kcols:
        raise RuntimeError(
            f"top-k slot index out of range for torch.gather(inds, 1, sampled_inner): "
            f"decode_strategy={decode_strategy!r}, decode_top_k={decode_top_k}, inds.shape={tuple(inds.shape)}, "
            f"sampled_inner.shape={tuple(sampled_inner.shape)}, sampled_inner.min/max=({min_v},{max_v}) "
            f"(valid slots [0,{kcols - 1}]), train_time_eval={train_time_eval!r}, backend={backend!r}, policy={policy!r}"
        )
    return si


def sample_next_token(
    logits: torch.Tensor,
    *,
    strategy: str,
    temperature: float,
    top_p: float,
    generator: Optional[torch.Generator],
    gen_so_far: int = 0,
    gap_threshold: float = 0.35,
    prefix_greedy_steps: int = 4,
    top_k: int = 5,
    row_active: Optional[torch.Tensor] = None,
    uncertainty_entropy_eps: float = 1e-8,
    sampling_diag_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
    """返回 (output_id (B,1), entropy (B,), log_probs, uncertainty_step_diag|None)。

    uncertainty_step_diag 仅当 strategy=uncertainty_low_temp_top_k 且本步已过 prefix_greedy_steps 时出现，
    含 post_prefix_decisions / trigger_count / trigger_mask(B,)（见主线 decode 统计）。
    """
    B = logits.size(0)
    device = logits.device
    diag_dtype = _DIAG_DTYPE
    st = strategy.lower()
    if st == "nucleus":
        logits_scaled = logits / temperature
        logits_scaled_f = logits_scaled.to(dtype=diag_dtype)
        log_probs = F.log_softmax(logits_scaled_f, dim=-1)
        probs = F.softmax(logits_scaled_f, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        nucleus = sorted_probs.masked_fill(mask, 0.0)
        nucleus = nucleus / (nucleus.sum(dim=-1, keepdim=True) + 1e-12)
        ent = -(nucleus * nucleus.clamp_min(1e-12).log()).sum(dim=-1)
        sampled_inner = torch.multinomial(nucleus, 1, generator=generator)
        next_ids = torch.gather(sorted_idx, -1, sampled_inner)
        return next_ids, ent, log_probs, None
    if st == "uncertainty_low_temp_top_k":
        _dc = sampling_diag_context or {}
        _tev = str(_dc.get("train_time_eval", "unknown"))
        _be = str(_dc.get("backend", "unknown"))
        _pol = str(_dc.get("policy", "unknown"))
        if int(top_k) <= 0:
            raise ValueError(
                f"decode_top_k 须为正整数（uncertainty_low_temp_top_k），当前={int(top_k)!r}；"
                f"decode_strategy={strategy!r}, train_time_eval={_tev!r}, backend={_be!r}, policy={_pol!r}"
            )
        log_probs = F.log_softmax(logits.to(dtype=diag_dtype), dim=-1)
        if int(gen_so_far) < int(prefix_greedy_steps):
            next_ids = logits.argmax(dim=-1, keepdim=True)
            return next_ids, torch.zeros(B, device=device, dtype=diag_dtype), log_probs, None
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"logits 含 NaN/Inf（uncertainty_low_temp_top_k）；decode_strategy={strategy!r}, "
                f"decode_top_k={int(top_k)}, logits.shape={tuple(logits.shape)}, "
                f"train_time_eval={_tev!r}, backend={_be!r}, policy={_pol!r}"
            )
        if row_active is None:
            eligible = torch.ones(B, dtype=torch.bool, device=device)
        else:
            eligible = row_active.to(device=device, dtype=torch.bool)
        V = int(logits.size(-1))
        if V <= 0:
            raise RuntimeError(
                f"empty vocab in uncertainty_low_temp_top_k: logits.shape={tuple(logits.shape)}, "
                f"decode_strategy={strategy!r}, decode_top_k={int(top_k)}, "
                f"train_time_eval={_tev!r}, backend={_be!r}, policy={_pol!r}"
            )
        effective_k = min(int(top_k), V)
        effective_k = max(1, effective_k)
        n_top2 = min(2, V)
        top2_vals, _ = torch.topk(logits, n_top2, dim=-1)
        if n_top2 >= 2:
            gap = (top2_vals[:, 0] - top2_vals[:, 1]).to(dtype=diag_dtype)
        else:
            gap = torch.full((B,), float("inf"), device=device, dtype=diag_dtype)
        # 触发：已过 prefix ∧ gap 低于阈值 ∧ 本步对该行执行了低温 top-k 多项式采样（与 need_sample 一致）
        need_sample = (gap < float(gap_threshold)) & eligible
        post_prefix_decisions = int(eligible.sum().item())
        trigger_count = int(need_sample.sum().item())
        greedy_ids = logits.argmax(dim=-1, keepdim=True)
        next_ids = greedy_ids.clone()
        ent = torch.zeros(B, device=device, dtype=diag_dtype)
        if bool(need_sample.any()):
            u = need_sample.nonzero(as_tuple=False).squeeze(-1)
            Lu = logits.index_select(0, u)
            if not torch.isfinite(Lu).all():
                raise RuntimeError(
                    f"subselect logits 含 NaN/Inf（uncertainty_low_temp_top_k）；decode_strategy={strategy!r}, "
                    f"decode_top_k={int(top_k)}, Lu.shape={tuple(Lu.shape)}, "
                    f"train_time_eval={_tev!r}, backend={_be!r}, policy={_pol!r}"
                )
            Lu_scaled = Lu / max(float(temperature), 1e-8)
            vals, inds = torch.topk(Lu_scaled, effective_k, dim=-1)
            if effective_k == 1:
                chosen = inds[:, 0:1]
                ent_u = torch.zeros(Lu.size(0), device=device, dtype=diag_dtype)
            else:
                vals_f = vals.to(dtype=diag_dtype)
                probs_topk = F.softmax(vals_f, dim=-1)
                if not torch.isfinite(probs_topk).all():
                    raise RuntimeError(
                        f"top-k probs 非有限（uncertainty_low_temp_top_k）；decode_strategy={strategy!r}, "
                        f"decode_top_k={int(top_k)}, effective_k={effective_k}, probs_topk.shape={tuple(probs_topk.shape)}, "
                        f"train_time_eval={_tev!r}, backend={_be!r}, policy={_pol!r}"
                    )
                eps_h = max(float(uncertainty_entropy_eps), 1e-30)
                ent_u = -(probs_topk * torch.log(probs_topk + eps_h)).sum(dim=-1)
                sampled_inner = torch.multinomial(probs_topk, 1, generator=generator)
                si = assert_topk_slot_indices_valid_for_gather(
                    sampled_inner,
                    inds,
                    decode_strategy=strategy,
                    decode_top_k=int(top_k),
                    train_time_eval=_tev,
                    backend=_be,
                    policy=_pol,
                )
                chosen = torch.gather(inds, 1, si)
            next_ids[u] = chosen
            _assign_diag_by_index1d(
                ent,
                u,
                ent_u,
                decode_strategy=strategy,
                sampling_diag_context=_dc,
            )
        diag = {
            "post_prefix_decisions": post_prefix_decisions,
            "trigger_count": trigger_count,
            "trigger_mask": need_sample,
        }
        return next_ids, ent, log_probs, diag
    log_probs = F.log_softmax(logits.to(dtype=diag_dtype), dim=-1)
    next_ids = logits.argmax(dim=-1, keepdim=True)
    ent = torch.zeros(B, device=device, dtype=diag_dtype)
    return next_ids, ent, log_probs, None


def update_generation_state(
    state: GenerationState,
    output_id: torch.Tensor,
    eos_id: int,
) -> None:
    state.decoder_input_ids = torch.cat([state.decoder_input_ids, output_id], dim=-1)
    state.step += 1
    rows = output_id.squeeze(-1).tolist()
    for b, tid in enumerate(rows):
        if b >= len(state.recent_tokens):
            state.recent_tokens.append([])
        state.recent_tokens[b].append(int(tid))
    if eos_id >= 0:
        active = output_id.squeeze(-1) != eos_id
        state.active = state.active & active


def build_candidate_generation_specs(
    base: GenerateConfig,
    family: str,
    *,
    k_cli: int,
    include_diverse: bool,
) -> List[Tuple[str, Optional[GenerateConfig]]]:
    """返回 (family_tag, cfg_override|None)；None 表示使用模型当前默认 GenerateConfig。"""
    fam = (family or "balanced").strip().lower()
    out: List[Tuple[str, Optional[GenerateConfig]]] = []
    if fam == "mixed":
        for _ in range(2):
            c = dc_replace(
                base,
                temperature=max(1e-8, float(base.temperature) * 0.82),
                top_p=min(1.0, float(base.top_p) * 0.92),
            )
            out.append(("conservative", c))
        for _ in range(2):
            out.append(("balanced", None))
        if include_diverse:
            c = dc_replace(
                base,
                temperature=float(base.temperature) * 1.1,
                top_p=min(0.98, float(base.top_p) * 1.03),
            )
            out.append(("diverse", c))
        return out
    k = max(1, int(k_cli))
    for j in range(k):
        if fam == "conservative":
            c = dc_replace(
                base,
                temperature=max(1e-8, float(base.temperature) * (0.88 + 0.02 * j)),
                top_p=min(1.0, float(base.top_p) * (0.94 + 0.005 * j)),
            )
            out.append(("conservative", c))
        elif fam == "diverse":
            c = dc_replace(
                base,
                temperature=float(base.temperature) * (1.02 + 0.03 * j),
                top_p=min(0.99, float(base.top_p) + 0.02 * j),
            )
            out.append(("diverse", c))
        else:
            c = dc_replace(
                base,
                temperature=max(1e-8, float(base.temperature) * (0.97 + 0.02 * (j - k / 2.0))),
            )
            out.append(("balanced", c))
    return out


def build_generate_kwargs_effective_v2(cfg: GenerateConfig, *, eos_token_id: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "schema_version": PAPER_DECODE_CONTROLLER_SCHEMA,
        "strategy": cfg.strategy,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "gap_threshold": cfg.gap_threshold,
        "prefix_greedy_steps": cfg.prefix_greedy_steps,
        "top_k": cfg.top_k,
        "repetition_penalty": cfg.repetition_penalty,
        "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
        "min_len": cfg.min_len,
        "soft_max_len": cfg.soft_max_len,
        "hard_max_len": cfg.hard_max_len,
        "eos_boost_start": cfg.eos_boost_start,
        "eos_boost_value": cfg.eos_boost_value,
        "tail_temperature": cfg.tail_temperature,
        "tail_top_p": cfg.tail_top_p,
        "forbid_eos_after_open_quote": cfg.forbid_eos_after_open_quote,
        "forbid_eos_after_open_bracket": cfg.forbid_eos_after_open_bracket,
        "forbid_bad_terminal_tokens": cfg.forbid_bad_terminal_tokens,
        "bad_terminal_token_ids": list(cfg.bad_terminal_token_ids),
        "token_repeat_window": cfg.token_repeat_window,
        "token_repeat_max": cfg.token_repeat_max,
        "decode_seed": cfg.decode_seed,
        "uncertainty_entropy_eps": float(cfg.uncertainty_entropy_eps),
    }
    if eos_token_id >= 0:
        out["eos_token_id"] = eos_token_id
    if cfg.decode_backend is not None:
        out["decode_backend"] = str(cfg.decode_backend)
    if cfg.decode_backend_fallback_policy is not None:
        out["decode_backend_fallback_policy"] = str(cfg.decode_backend_fallback_policy)
    if cfg.decode_run_context is not None:
        out["decode_run_context"] = str(cfg.decode_run_context)
    return out
