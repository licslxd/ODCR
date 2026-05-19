"""Repo-local Python startup policy.

ODCR keeps bytecode caches out of the active tree; validation that needs
compilation uses compileall or py_compile explicitly.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
