# ODCR GPU Runtime-First Execution Contract

## Scope

GPU use is allowed by default for Codex repo-local validation, probe, and
bounded runtime when the current tmux pane is user-created, already-entered,
uniquely validated, and exposes real-time CUDA. This contract replaces the old
GPU whitelist hard blocker with the `./odcr runtime` stage-dispatch allowlist
and deletes the post-edit full pre-GPU gate.

## Runtime-First Rule

There is no arbitrary repo-command shell dispatch. The tmux bridge may execute
only registered stage-dispatch commands and generated command files under
validation directories. The bridge still sends one generated command file to
the validated pane; arbitrary send-keys remain forbidden.

formal full train still requires user confirmation. runtime evidence takes
priority over static full-suite instability.

post-edit full is not a GPU prerequisite. The GPU preflight is fast sanity:

- `python -m compileall -q code`
- `./odcr doctor`
- `python code/tools/check_one_control_guardrails.py --strict`
- `./odcr show --stage step3 --task 2`
- `./odcr step3 --task 2 --dry-run`

After fast sanity, Codex may fresh discover/validate the GPU pane and run
marker, CUDA, runtime, and bounded candidate probes. post-edit full may run
afterward as a diagnostic or hygiene report. `exit -9`, timeout, or resource
kill is classified as resource instability, not as a GPU prohibition. Only a
semantic P0 can block a future formal candidate.

## Formal Boundary

The formal namespace guard remains mandatory. Validation/probe commands must
not write:

- `runs/step3/task2/latest.json`
- formal Step3 checkpoints such as `model/best.pth`
- formal checkpoint lineage/state
- Step4/Step5/eval/rerank outputs
- paper/final metrics

Validation/probe output defaults to:

- `AI_analysis/06_probe_evidence/...`
- `runs/step3_validation/...`

Formal full train still requires explicit user confirmation in a future
request.

## Stage2

Stage2 candidate selection uses real runtime probes as the source of truth.
Runtime evidence takes priority over static full-suite instability. Candidate
recommendations require `runtime_verified=true`, `evidence_complete=true`,
complete timing/memory/prefetch/grad/DDP evidence, finite gradients, closed
timing, safe memory, and no formal namespace pollution.
