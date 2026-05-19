"""Fail-close timeout policy for hook and post-edit execution."""

from __future__ import annotations

from dataclasses import dataclass

from .post_edit_registry import DEFAULT_HOOK_CHILD_MAX_SECONDS, DEFAULT_WRAPPER_TIMEOUT_SECONDS


@dataclass(frozen=True)
class TimeoutPolicy:
    wrapper_timeout_s: int = DEFAULT_WRAPPER_TIMEOUT_SECONDS
    child_timeout_s: int = DEFAULT_HOOK_CHILD_MAX_SECONDS

    def normalized_child_timeout(self, requested: int | None = None) -> int:
        value = self.child_timeout_s if requested is None else int(requested)
        if value <= 0:
            raise ValueError("timeout must be positive")
        if value >= self.wrapper_timeout_s:
            raise ValueError("child timeout must be less than wrapper timeout")
        return value


def classify_timeout(returncode: int | None, *, timed_out: bool) -> str:
    if timed_out:
        return "timeout_fail_close"
    if returncode is None:
        return "unknown_fail_close"
    return "pass" if returncode == 0 else "fail_close"
