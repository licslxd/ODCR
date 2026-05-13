"""Codex workflow rules shared by ODCR governance docs."""

CODEX_RUNTIME_RULES = (
    "No interim status updates when the request forbids them.",
    "All audit and handoff artifacts go under AI_analysis.",
    "Use ./odcr runtime for tmux GPU bridge, probe, and preflight work.",
    "Do not run odcr-enter-gpu, srun, sbatch, scancel, or manage tmux sessions.",
    "Use short registered runtime commands, never arbitrary shell dispatch.",
    "Dry-run and probe evidence are not formal train evidence.",
    "Do not create per-run AI_analysis subdirectories for summaries; use the canonical buckets.",
)

