"""Runtime-first Stage2 GPU validation flow helpers.

These helpers encode the governance order for Stage2 candidate selection. They
do not launch formal training and do not write formal latest/checkpoint files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FAST_SANITY_COMMANDS: tuple[str, ...] = (
    "python -m compileall -q code",
    "./odcr doctor",
    "python code/tools/check_one_control_guardrails.py --strict",
    "./odcr show --stage step3 --task 2",
    "./odcr step3 --task 2 --dry-run",
)

GPU_VALIDATION_STEPS: tuple[str, ...] = (
    "discover",
    "validate-only",
    "marker-probe",
    "cuda-probe",
)

PREREQUISITE_RUNTIME_PROBES: tuple[str, ...] = (
    "quality-checkpoint-window",
    "grad-monitor-window",
    "timing-profile-window",
    "memory-phase-window",
    "prefetch-ab",
    "ddp-gather-sync-window",
)

CANDIDATE_PROBES: tuple[str, ...] = ("G1-S", "G1S", "G1-M", "G2-C")
DEFAULT_SKIPPED_CANDIDATES: tuple[str, ...] = ("G3",)


@dataclass(frozen=True)
class Stage2CandidateVerdict:
    verdict: str
    selected: str | None
    rejected: tuple[str, ...]
    skipped: tuple[str, ...]
    runtime_verified: bool
    evidence_complete: bool
    reason: str


def runtime_first_flow_order() -> tuple[str, ...]:
    return (
        "fast_sanity",
        "fresh_gpu_discover_validate",
        "marker_cuda_probe",
        "real_runtime_probes",
        "candidate_probes",
        "candidate_selection",
        "post_edit_full_diagnostic_optional",
        "final_verdict",
    )


def post_edit_diagnostic_blocks_gpu_probe(diagnostic: Mapping[str, Any] | None) -> bool:
    """Post-edit diagnostics are never a GPU prerequisite in runtime-first flow."""

    _ = diagnostic
    return False


def post_edit_diagnostic_blocks_formal(diagnostic: Mapping[str, Any] | None) -> bool:
    if not diagnostic:
        return False
    classification = str(diagnostic.get("classification") or diagnostic.get("result_class") or "").strip()
    return classification in {"P0_semantic_blocker", "semantic_fail", "test_fail"}


def can_enter_gpu_runtime(*, fast_sanity_pass: bool, post_edit_diagnostic: Mapping[str, Any] | None = None) -> bool:
    return bool(fast_sanity_pass) and not post_edit_diagnostic_blocks_gpu_probe(post_edit_diagnostic)


def _runtime_ok(item: Mapping[str, Any] | None) -> bool:
    if not item:
        return False
    return bool(item.get("runtime_verified")) and bool(item.get("evidence_complete")) and not bool(
        item.get("formal_namespace_polluted") or item.get("formal_pollution")
    )


def prerequisite_evidence_complete(probe_results: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(_runtime_ok(probe_results.get(name)) for name in PREREQUISITE_RUNTIME_PROBES)


def select_stage2_candidate(
    *,
    prerequisite_results: Mapping[str, Mapping[str, Any]],
    candidate_results: Mapping[str, Mapping[str, Any]],
    post_edit_diagnostic: Mapping[str, Any] | None = None,
) -> Stage2CandidateVerdict:
    prerequisites_ok = prerequisite_evidence_complete(prerequisite_results)
    if not prerequisites_ok:
        return Stage2CandidateVerdict(
            verdict="B",
            selected=None,
            rejected=tuple(CANDIDATE_PROBES),
            skipped=DEFAULT_SKIPPED_CANDIDATES,
            runtime_verified=False,
            evidence_complete=False,
            reason="candidate selection requires complete real runtime prerequisite evidence",
        )
    if post_edit_diagnostic_blocks_formal(post_edit_diagnostic):
        return Stage2CandidateVerdict(
            verdict="C",
            selected=None,
            rejected=tuple(CANDIDATE_PROBES),
            skipped=DEFAULT_SKIPPED_CANDIDATES,
            runtime_verified=True,
            evidence_complete=True,
            reason="semantic P0 post-edit diagnostic blocks formal candidate",
        )
    verified = {name: result for name, result in candidate_results.items() if _runtime_ok(result)}
    if not verified:
        return Stage2CandidateVerdict(
            verdict="B",
            selected=None,
            rejected=tuple(CANDIDATE_PROBES),
            skipped=DEFAULT_SKIPPED_CANDIDATES,
            runtime_verified=True,
            evidence_complete=False,
            reason="candidate probes did not produce complete runtime evidence",
        )
    ordered_preference: Sequence[str] = ("G1-M", "G1S", "G1-S", "G2-C")
    selected = next((name for name in ordered_preference if name in verified), None)
    rejected = tuple(name for name in CANDIDATE_PROBES if name != selected)
    return Stage2CandidateVerdict(
        verdict="A" if selected else "B",
        selected=selected,
        rejected=rejected,
        skipped=DEFAULT_SKIPPED_CANDIDATES,
        runtime_verified=True,
        evidence_complete=bool(selected),
        reason="selected fastest safe preferred candidate with complete runtime evidence" if selected else "no eligible candidate",
    )
