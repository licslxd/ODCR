# CSB-ODCR Step3 Formal Rerun Startup Contract

This contract is the startup gate for the next task2 Step3 formal rerun.

## Current Formal Run State

- Deleted or archived historical runs are not part of the active startup path.
- A fresh CSB-native Step3 task2 run must start from the One-Control resolver,
  not from historical latest pointers or checkpoint compatibility paths.
- Failed, partial, or non-evaluated historical runs must not be marked
  successful or consumed by Step4, Step5, eval, or rerank.

## Formal Profile Binding

- Stage2 selected the task2 G1S candidate for the next formal rerun.
- The live formal default must resolve to `task2_strong_forward_g1s`.
- G1 is backup-only and requires explicit future rollback evidence.
- G1-M and G2-C remain probe-only; G2-C is not formal-ready.
- Startup may assert the profile with `--expect-profile
  task2_strong_forward_g1s`; mismatch must fail before launch.

## Tokenizer Cache Gate

Step3 tokenizer cache compatibility is split:

- `tokenization_compat_hash` is the hard reuse gate.
- `run_lineage_hash` is record-only lineage.

`full_run_config_hash`, resolved config full hash, source table full hash,
training runtime hash, profile id, batch size, micro batch size, learning rate,
optimizer, scheduler, checkpoint policy, logging paths, and run id must not
force tokenizer cache rebuild when tokenizer/data/length inputs are unchanged.

Before formal rerun, run the read-only cache-check. It reports whether the
completed cache would be reused, the selected formal profile, record-only
mismatches, and the num_proc value that would be used only for a cold rebuild.

## Hardware Num Proc

Cold pre-DDP tokenization uses resolver-selected auto num_proc:

- `max_parallel_cpu=12`
- `reserved_cpu=2`
- `max_num_proc=8`
- selected cold tokenization num_proc is `8`

Warm cache hits report that tokenization workers are not used, while preserving
`selected_num_proc_if_rebuild=8`.

## Checkpoint Write Preflight

Before a formal rerun, run the checkpoint-write preflight. It must create only
validation evidence under `AI_analysis/06_probe_evidence`, write a fake
checkpoint sidecar, call `checkpoint_event_from_sidecar` with explicit
`reason` and `replaced_previous`, write a temporary checkpoint lineage event,
and verify the required event fields. It must not write formal latest pointers
or formal checkpoints.

## Handoff Rule

Do not use historical downstream evidence. The next formal rerun must use a
fresh run id, must resolve to `task2_strong_forward_g1s`, and must pass
cache-check and checkpoint-write preflight before launch.
