from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from odcr_core.generation.cache_types import DecodeStepOutput, LayerKVCache, PastKeyValues

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - 极旧 torch
    sdpa_kernel = None  # type: ignore[misc, assignment]
    SDPBackend = None  # type: ignore[misc, assignment]


def _align_new_kv_to_cache(
    ref_key: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在 torch.cat 前将本步 K/V 与 cache.key 对齐 dtype/device，禁止隐式 float32 混入 bf16 路径。"""
    if k_new.dtype != ref_key.dtype or k_new.device != ref_key.device:
        k_new = k_new.to(device=ref_key.device, dtype=ref_key.dtype)
    if v_new.dtype != ref_key.dtype or v_new.device != ref_key.device:
        v_new = v_new.to(device=ref_key.device, dtype=ref_key.dtype)
    return k_new, v_new


def _sdpa_forward_context(sdpa_variant: str):
    """safe：decode 段强制 MATH SDP 后端，避免部分融合内核在 bf16 autocast 训练内 eval 下不稳定。"""
    v = str(sdpa_variant or "fast").strip().lower()
    if v == "safe":
        if sdpa_kernel is None or SDPBackend is None:
            return nullcontext()
        return sdpa_kernel([SDPBackend.MATH])
    return nullcontext()


@dataclass
class DecoderKVBackend:
    model: torch.nn.Module
    sdpa_variant: str = "fast"
    _sdpa_cores: list[torch.nn.Module] = field(default_factory=list)

    class _SDPASelfAttentionCore(torch.nn.Module):
        def __init__(self, embed_dim: int, num_heads: int, bias: bool) -> None:
            super().__init__()
            self.embed_dim = int(embed_dim)
            self.num_heads = int(num_heads)
            self.head_dim = int(embed_dim // num_heads)
            self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
            self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
            self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
            self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        def load_from_mha(self, old_attn: torch.nn.Module) -> None:
            q_w, k_w, v_w = old_attn.in_proj_weight.chunk(3, dim=0)
            self.q_proj.weight.data.copy_(q_w)
            self.k_proj.weight.data.copy_(k_w)
            self.v_proj.weight.data.copy_(v_w)
            if old_attn.in_proj_bias is None:
                if self.q_proj.bias is not None:
                    self.q_proj.bias.data.zero_()
                    self.k_proj.bias.data.zero_()
                    self.v_proj.bias.data.zero_()
            else:
                q_b, k_b, v_b = old_attn.in_proj_bias.chunk(3, dim=0)
                self.q_proj.bias.data.copy_(q_b)
                self.k_proj.bias.data.copy_(k_b)
                self.v_proj.bias.data.copy_(v_b)
            self.out_proj.weight.data.copy_(old_attn.out_proj.weight.data)
            self.out_proj.bias.data.copy_(old_attn.out_proj.bias.data)

    def _ensure_sdpa_cores(self) -> None:
        if self._sdpa_cores:
            return
        cores: list[torch.nn.Module] = []
        for layer in self.model.transformer_encoder.layers:
            old_attn = layer.self_attn
            core = self._SDPASelfAttentionCore(
                embed_dim=int(old_attn.embed_dim),
                num_heads=int(old_attn.num_heads),
                bias=old_attn.in_proj_bias is not None,
            )
            core.load_from_mha(old_attn)
            core.to(
                device=old_attn.in_proj_weight.device,
                dtype=old_attn.in_proj_weight.dtype,
            )
            core.eval()
            cores.append(core)
        self._sdpa_cores = cores

    def _split_qkv(
        self,
        core: torch.nn.Module,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = int(core.embed_dim)
        h = int(core.num_heads)
        dh = int(core.head_dim)
        q = core.q_proj(x)
        k = core.k_proj(x)
        v = core.v_proj(x)
        q = q.view(x.size(0), x.size(1), h, dh).transpose(1, 2)
        k = k.view(x.size(0), x.size(1), h, dh).transpose(1, 2)
        v = v.view(x.size(0), x.size(1), h, dh).transpose(1, 2)
        return q, k, v

    def _layer_decode_step(
        self,
        layer: torch.nn.Module,
        core: torch.nn.Module,
        x_t: torch.Tensor,
        cache: Optional[LayerKVCache],
    ) -> tuple[torch.Tensor, LayerKVCache]:
        q, k_new, v_new = self._split_qkv(core, x_t)
        if cache is None:
            k_cat = k_new
            v_cat = v_new
        else:
            k_new, v_new = _align_new_kv_to_cache(cache.key, k_new, v_new)
            k_cat = torch.cat([cache.key, k_new], dim=2)
            v_cat = torch.cat([cache.value, v_new], dim=2)
        if q.dtype != k_cat.dtype or q.device != k_cat.device:
            q = q.to(device=k_cat.device, dtype=k_cat.dtype)
        with _sdpa_forward_context(self.sdpa_variant):
            ctx = F.scaled_dot_product_attention(
                q,
                k_cat,
                v_cat,
                is_causal=False,
            )
        ctx = ctx.transpose(1, 2).contiguous().view(x_t.size(0), x_t.size(1), -1)
        src2 = core.out_proj(ctx)
        src = x_t + layer.dropout1(src2)
        src = layer.norm1(src)
        src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
        src = src + layer.dropout2(src2)
        src = layer.norm2(src)
        return src, LayerKVCache(key=k_cat, value=v_cat)

    def prefill(
        self,
        prefix_hidden: torch.Tensor,
    ) -> PastKeyValues:
        self._ensure_sdpa_cores()
        layers = list(self.model.transformer_encoder.layers)
        cache = PastKeyValues.empty(num_layers=len(layers))
        if int(prefix_hidden.size(1)) <= 0:
            return cache
        seq = prefix_hidden
        for li, layer in enumerate(layers):
            core = self._sdpa_cores[li]
            q, k, v = self._split_qkv(core, seq)
            if q.dtype != k.dtype or q.device != k.device:
                q = q.to(device=k.device, dtype=k.dtype)
            with _sdpa_forward_context(self.sdpa_variant):
                ctx = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=False,
                )
            ctx = ctx.transpose(1, 2).contiguous().view(seq.size(0), seq.size(1), -1)
            src2 = core.out_proj(ctx)
            src = seq + layer.dropout1(src2)
            src = layer.norm1(src)
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
            seq = src + layer.dropout2(src2)
            seq = layer.norm2(seq)
            cache.layers[li] = LayerKVCache(key=k, value=v)
        cache.prefix_len = int(prefix_hidden.size(1))
        cache.generated_len = 0
        cache.validate()
        return cache

    def decode_step(
        self,
        last_token: torch.Tensor,
        past_key_values: Optional[PastKeyValues],
        *,
        embed_token_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_to_logits_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> DecodeStepOutput:
        self._ensure_sdpa_cores()
        layers = list(self.model.transformer_encoder.layers)
        if past_key_values is None:
            cache = PastKeyValues.empty(num_layers=len(layers))
        else:
            cache = past_key_values
        x_t = embed_token_fn(last_token)
        for li, layer in enumerate(layers):
            core = self._sdpa_cores[li]
            x_t, layer_cache = self._layer_decode_step(layer, core, x_t, cache.layers[li])
            cache.layers[li] = layer_cache
        cache.generated_len = int(cache.generated_len) + 1
        cache.validate()
        logits = hidden_to_logits_fn(x_t)
        return DecodeStepOutput(logits=logits, past_key_values=cache, hidden_last=x_t)
