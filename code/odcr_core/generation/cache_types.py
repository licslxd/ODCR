from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class LayerKVCache:
    # (B, H, S, Dh)
    key: torch.Tensor
    value: torch.Tensor

    @property
    def seq_len(self) -> int:
        return int(self.key.shape[2])

    def shape_meta(self) -> Dict[str, int]:
        return {
            "batch": int(self.key.shape[0]),
            "heads": int(self.key.shape[1]),
            "seq": int(self.key.shape[2]),
            "head_dim": int(self.key.shape[3]),
        }

    def dtype_device_meta(self) -> Dict[str, Any]:
        return {
            "dtype": self.key.dtype,
            "device": self.key.device,
        }


@dataclass
class PastKeyValues:
    layers: List[LayerKVCache]
    prefix_len: int = 0
    generated_len: int = 0

    @classmethod
    def empty(cls, num_layers: int) -> "PastKeyValues":
        return cls(layers=[None] * int(num_layers))  # type: ignore[list-item]

    def validate(self) -> None:
        if not self.layers:
            return
        ref: Optional[Dict[str, int]] = None
        ref_dtype: Optional[torch.dtype] = None
        ref_device: Optional[torch.device] = None
        for i, layer in enumerate(self.layers):
            if layer is None:
                continue
            meta = layer.shape_meta()
            if layer.key.dtype != layer.value.dtype:
                raise ValueError(
                    f"layer {i} KV dtype mismatch: key={layer.key.dtype} value={layer.value.dtype}"
                )
            if layer.key.device != layer.value.device:
                raise ValueError(
                    f"layer {i} KV device mismatch: key={layer.key.device} value={layer.value.device}"
                )
            if ref is None:
                ref = meta
                ref_dtype = layer.key.dtype
                ref_device = layer.key.device
            else:
                if meta["batch"] != ref["batch"] or meta["heads"] != ref["heads"] or meta["head_dim"] != ref["head_dim"]:
                    raise ValueError(f"inconsistent kv shape at layer {i}: {meta} vs {ref}")
                if layer.key.dtype != ref_dtype:
                    raise ValueError(
                        f"inconsistent kv dtype at layer {i}: {layer.key.dtype} vs ref {ref_dtype}"
                    )
                if layer.key.device != ref_device:
                    raise ValueError(
                        f"inconsistent kv device at layer {i}: {layer.key.device} vs ref {ref_device}"
                    )

    def debug_shapes(self) -> List[Dict[str, int]]:
        out: List[Dict[str, int]] = []
        for layer in self.layers:
            if layer is None:
                out.append({"batch": 0, "heads": 0, "seq": 0, "head_dim": 0})
            else:
                out.append(layer.shape_meta())
        return out


@dataclass
class DecodeStepOutput:
    logits: torch.Tensor
    past_key_values: Optional[PastKeyValues]
    hidden_last: Optional[torch.Tensor] = None
