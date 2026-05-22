# ODCR GPU Runtime-First Execution Contract

## Scope

GPU use is allowed by default for Codex repo-local validation, probe, and
bounded runtime when the current tmux pane is user-created, already-entered,
uniquely validated, and exposes real-time CUDA. This contract deletes the old
GPU whitelist hard blocker and the post-edit full pre-GPU gate.

## Runtime-First Rule

There is no GPU whitelist hard blocker and no closed-choice-only runtime
restriction. The tmux bridge may execute non-formal GPU commands through
`./odcr runtime bridge exec -- ...`, including repo-local Python modules,
repo-local scripts, generated command files under validation directories,
direct repo-local commands, eval, rerank, and diagnostics. The only hard
runtime blocker is ODCR formal model training. The bridge still sends a
generated command file to the validated pane; direct manual tmux control is not
the execution contract.

No GPU whitelist hard blocker is allowed in the active execution path; formal
full train still requires user confirmation; runtime evidence takes priority
over static full-suite instability.

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

- `AI_analysis/01_raw_logs/...`
- `AI_analysis/05_final_reports/...`

Formal full train still requires explicit user confirmation in a future
request.

## Stage2

Stage2 candidate selection uses real runtime probes as the source of truth.
Runtime evidence takes priority over static full-suite instability. Candidate
recommendations require `runtime_verified=true`, `evidence_complete=true`,
complete timing/memory/prefetch/grad/DDP evidence, finite gradients, closed
timing, safe memory, and no formal namespace pollution.

Safe memory means real memory truth, not PyTorch reserved-pool headroom.
Candidate admission may reject OOM, allocator failure, failed forward/backward,
non-finite loss, DDP graph failure, rank imbalance, configured allocated-memory
ratio, long-window allocated-memory creep, nvidia-smi process-memory
instability, formal namespace pollution, resolver bypass, or missing required
fields. `max_memory_reserved_gb` and reserved-minus-allocated are retained only
as allocator/cache diagnostics and must not be hard gates, ranking blockers, or
larger-batch skip rules.
## Current Runtime Bridge Contract

The first execution protocol is now registry-driven:

```bash
./odcr runtime bridge discover
./odcr runtime bridge validate-only
./odcr runtime bridge marker-probe
./odcr runtime bridge cuda-probe
./odcr runtime bridge exec -- <command> [args...]
```

The user must already be inside the GPU allocation in the same tmux pane. Codex
must not run ODCR formal model training through the bridge. GPU evidence is
current-pane evidence only and is written to
`AI_analysis/01_raw_logs/aux_runtime_gpu_handshake.log` and
`AI_analysis/05_final_reports/aux_runtime_gpu_validation_report.md`; bridge exec
driver/status logs are written under `AI_analysis/01_raw_logs/runtime_bridge_exec`
unless the caller supplies explicit stdout/status/pid paths.

The user workflow remains exactly two commands:

```bash
tmux -L odcr_gpu new-session -A -s odcr
odcr-enter-gpu <JOBID>
```

`odcr-enter-gpu` automatically writes the bridge handoff. It captures tmux
metadata on the admin side before `srun`, then captures CUDA metadata on the GPU
side after `srun`; the GPU-side writer must not rely on `tmux display-message`
or any tmux metadata probe. On success it atomically writes
`AI_analysis/runtime/current_gpu_pane.json` with schema
`odcr_current_gpu_pane_handoff/2`. On handoff failure it deletes stale active
handoff state, writes a failure report when possible, prints a warning, and
continues into the GPU shell. The bridge must read only a fresh
`current_gpu_pane.json`, then rerun validate/cuda-probe; the old
`AI_analysis/runtime/gpu_pane.json` is historical hint material only.
