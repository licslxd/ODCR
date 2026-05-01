# -*- coding: utf-8 -*-
"""验证集 explanation BLEU-4（DDP 全量）：实现已收口至 odcr_core.bleu_runtime，本模块保持导入路径稳定。"""
from __future__ import annotations

from odcr_core.bleu_runtime import bleu4_explanation_full_valid_ddp

__all__ = ["bleu4_explanation_full_valid_ddp"]
