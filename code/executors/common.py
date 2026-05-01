"""跨 Step 共享：torchrun 薄壳名（单一来源 odcr_core.dispatch）。"""

from odcr_core.dispatch import (
    TORCHRUN_STEP3_SCRIPT,
    TORCHRUN_STEP4_SCRIPT,
    TORCHRUN_STEP5_SCRIPT,
)

__all__ = [
    "TORCHRUN_STEP3_SCRIPT",
    "TORCHRUN_STEP4_SCRIPT",
    "TORCHRUN_STEP5_SCRIPT",
]
