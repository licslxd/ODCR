# ODCR Step3 Quality Evidence Performance Contract

## Scope

This contract separates Step3 code wiring from runtime proof. A passing unit
test is not a formal runtime pass. A dry-run, show, doctor, or guardrail pass is
not evidence that Step3 quality or A100 performance improved in a full run.
The runtime-probe truth rules live in
`docs/ODCR_STEP3_RUNTIME_PROBE_TRUTH_CONTRACT.md`; they require bounded hot-path
execution plus complete evidence rows before `step3-performance-probe` can pass.

## Evidence Levels

Evidence Level 1: code exists. It proves classes, helpers, schemas, or config
keys are present.

Evidence Level 2: active path is wired. It proves the resolver or formal train
loop can call the code path and records source-table/run-summary fields.

Evidence Level 3: runtime behavior verified. It requires a controlled runtime
validation or probe artifact, such as timing closure, grad finite skip, or H2D
overlap evidence.

Evidence Level 4: formal run evidence. It requires a full formal Step3 run
artifact set showing quality and performance behavior.

Evidence Level 5: downstream eligibility. It requires `quality_status=pass`,
`downstream_ready=true`, and a selected checkpoint with a matching sidecar hash.

## Why run1 Is Blocked

`runs/step3/task2/1` remains preserved as historical formal evidence, but it is
quality blocked. The global best valid loss was epoch 2
(`4.8578043832568705`), while `best.pth` and the legacy `best_event.json`
pointed to epoch 7 (`8.155338486101646`). The old rolling comparison treated
epoch 7 as best because it compared against the previous epoch instead of the
global best. The same run also has mass `grad_norm_pre_clip=inf` events,
diagnostic collapse, and missing or empty diagnostic samples.

## Checkpoint Policy

Step3 checkpoint selection uses global `valid_loss` minimization from epoch 1.
The default downstream checkpoint is `model/best_observed.pth`. Conservative
comparison is written separately to `model/best_after_min_epochs.pth`.
`model/latest.pth` records the latest epoch. `model/topk/` keeps ranked
epoch/metric-named candidates. `model/best.pth`, when retained for compatibility,
is only an alias/copy of `best_observed.pth` and must carry sidecar metadata
that says so.

Every checkpoint sidecar records epoch, metric, direction, scope, file hash,
global best, after-min best, run hashes, grad nonfinite counts, optimizer hash
when saved, and code commit. `state/checkpoint_lineage.json` is an event ledger;
it must never silently overwrite global best evidence.

## Quality Gate

`status=ok` only means the process completed. Downstream consumption requires
`quality_status=pass` and `downstream_ready=true`. Hard blocks include missing
global best evidence, best checkpoint mismatch, missing sidecar hash, excessive
nonfinite gradients, continuous nonfinite steps, unexplained post-clip zero
events, missing samples, diagnostic collapse, severe valid-loss deterioration,
missing metrics/timing/GPU files, and failed or partial run status.

Step4, Step5, eval, and rerank must reject blocked Step3 latest pointers before
loading a checkpoint.

## Grad Finite Policy

Step3 records `grad_check_ms`, `grad_norm_compute_ms`, `grad_clip_ms`,
`grad_monitor_ms`, nonfinite parameter top-k, optimizer/scheduler execution
booleans, skipped-step reason, and continuous nonfinite counts. Nonfinite
gradients skip `optimizer.step()` by default. Scheduler stepping on skipped
optimizer steps is a config-controlled behavior and defaults to false.

## Timing Closure

Timing evidence is wall-clock wrapped around the full step and broken into
loader, H2D, prefetch, forward, loss, gather, finite sync, DDP/backward, grad
check/norm/clip/monitor, optimizer, EMA, zero-grad, scheduler, I/O, checkpoint,
CUDA sync, unknown, and closure ratio fields. Timing closure is Level 3 only
when controlled validation shows unknown time below the configured threshold.

## Memory Attribution

Memory evidence records phase-level allocated, reserved, peak allocated, peak
reserved, reserved-minus-allocated, inactive split, non-releasable, malloc retry,
OOM count, and optional snapshot paths. Reserved memory is not treated as batch
headroom without phase evidence, and it must not be a hard reject or batch-skip
gate. Runtime admission should fail on real OOM/allocator failure, failed
forward/backward, non-finite loss, graph/rank instability, configured
allocated-memory ratio, or long-window allocated-memory creep. Reserved memory
remains diagnostic evidence for the PyTorch caching allocator only.

## Prefetch Evidence

The prefetcher records code-present, active-path, stream creation, double-buffer
configuration and activity, device-buffer count, record-stream count,
compute-wait count, H2D timing, hidden-by-compute ratio, overlap verification,
and fallback reason. `double_buffer=true` must be consumed behaviorally or
reported inactive; silent fallback is forbidden.

## A100 Candidate Governance

Low-risk instrumentation and safety gates may be adopted as defaults. Fused or
foreach AdamW, DDP bucket options, static graph, compile, CUDA graphs, allocator
candidates, chunked losses, compact gather tuning, worker sweeps, and batch
ladders are probe-only until runtime evidence passes. DALI, FlashAttention,
2:4 sparsity, activation checkpointing as speed optimization, bare torchrun,
G2 direct formal, and grad accumulation are rejected or not default.

Batch ladder candidates G1-S, G1-M, G2-C, and G3 are future probes only. They
cannot appear as default formal show output and require quality pass, finite
gradients, timing closure, safe peak allocation and reservation, no retry burst,
verified prefetch overlap, and no checkpoint policy bug before promotion.

## Step3 Diagnostic Eval

`odcr_step3_diagnostic` and `code1_target_only_comparable` are diagnostic-only
protocols and are not final paper metrics. `full_pipeline_final` is available
only after Step4/Step5/eval/rerank. Diagnostic samples and collapse stats are
risk evidence for Step3, not a substitute for the final pipeline.

## Performance-Probe Bridge Scope

`step3-performance-probe` is a validation-only tmux GPU bridge operation. It
requires a fresh validated GPU pane and writes validation namespace evidence
only under `AI_analysis/01_raw_logs` and `AI_analysis/05_final_reports`. It must
not update formal latest pointers, create formal checkpoints, start
Step4/Step5/eval/rerank, run preprocess, or execute arbitrary send-keys. GPU
use is allowed by default for repo-local validation, probe, and bounded
runtime; post-edit full is not a GPU prerequisite.
This performance-probe bridge is validation-only and cannot claim formal
performance evidence by itself.
