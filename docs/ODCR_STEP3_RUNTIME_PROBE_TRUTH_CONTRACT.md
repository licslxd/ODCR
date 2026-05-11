# ODCR Step3 Runtime Probe Truth Contract

## Scope

This contract fixes the Step3 performance-probe false-positive class. The
runtime probe is a validation-only bounded Step3 hot-path check. It is not a
formal train, not a paper metric, not a batch ladder decision, and not
Step4/Step5/eval/rerank.

## Success Semantics

`bridge_transport_ok is not runtime_probe_ok`. Transport success only means the
controlled bridge delivered a generated command and collected child status.
Child exit success only means the child process exited zero. Neither one proves
that Step3 ran a batch.

`child_exit_ok is not evidence_complete`. Runtime success requires real
evidence rows.

`runtime_verified=false must fail`. A report with `runtime_verified=false` must
produce `success=false`, `runtime_probe_ok=false`, and non-zero exit. The
combination `success=true` plus `runtime_verified=false` is forbidden.

`metrics all null must fail`. Empty timing rows, empty memory rows, null
prefetch metrics, missing grad metrics, or missing DDP/gather metrics must set
`evidence_complete=false`.

`status-only/plan-only is not performance-probe`. Plan artifacts may be named
`step3-performance-plan` or `step3-probe-plan`; they must not feed Stage2
runtime verdicts or candidate selection.

## Required Runtime Path

`step3-performance-probe` must execute a bounded Step3 validation window through
the live Step3 path:

- One-Control resolver payload
- pre-DDP cache readiness
- dataloader builder
- model builder
- loss composer
- DDP initialization
- optimizer builder
- forward, loss, backward, grad check, optimizer or explicit skip
- diagnostics evidence writers

The validation namespace forbids formal latest updates, formal checkpoints,
formal run-summary overwrite, preprocess rerun, Step4/Step5/eval/rerank, and
permanent model save.

## Evidence Rows

A passing runtime probe writes:

- `timing_breakdown.csv/jsonl`
- `memory_phase_summary.csv/jsonl`
- `prefetch_overlap_summary.json`
- `grad_monitor_validation.json`
- `ddp_gather_sync_summary.json`
- `run_summary_validation.json`
- `report.json`

Required rows missing, required fields null/NA, timing rows fewer than measured
steps, missing memory phases, or missing probe-specific metrics are hard FAIL.

## Stage2 Gate

Stage2 candidate selection may proceed only after prerequisite runtime probes
pass with `runtime_verified=true` and `evidence_complete=true`. G1-M/G2-C/G3
must be `skipped_by_gate` until timing, memory, prefetch, grad, and DDP evidence
all pass.

Stage2 is runtime-first: post-edit full is not a GPU prerequisite, and a
post-edit `exit -9` / resource kill does not erase or block runtime evidence.
Fast sanity plus fresh current-pane validation is the GPU preflight. Semantic
P0 findings can still block a future formal candidate.

## Tombstone

Retired false-positive behaviors:

- `odcr_step3_performance_probe.py` as a status writer
- `runtime_verified=false` with `success=true`
- child process completion counted as runtime success
- null summary files counted as evidence
- tests that only proved a mode existed
