"""Active ODCR auxiliary infrastructure.

The aux package owns non-training infrastructure: runtime/tmux validation,
governance checks, AI_analysis evidence writing, path policy helpers, and CLI
control facades. Training semantics stay in the existing stage modules.
"""

from __future__ import annotations

__all__ = ["artifacts", "control", "evidence", "governance", "runtime", "templates"]
