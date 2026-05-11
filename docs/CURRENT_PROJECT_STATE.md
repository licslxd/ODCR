# ODCR Current Project State

This is the only active human-readable state entry for ODCR. Historical docs and AI_analysis are not live truth; they are background evidence only.

Live run truth is resolved from the pointer plus a strict verifier:

1. `runs/{stage}/task{task_id}/latest.json`
2. `runs/{stage}/task{task_id}/{run_id}/meta/stage_status.json`
3. Read-time artifact revalidation by `odcr_core.stage_status_validator`

Downstream stages must use `odcr_core.upstream_resolver.resolve_upstream(...)`.
They must not infer current state from historical notes, old AI_analysis
reports, `completed.stamp` files, or Step3 `quality_audit.json`.
`latest.json` is pointer-only for formal eligibility; any legacy
`latest_status` field is display/deprecation residue and is ignored by the
formal gate after `stage_status.json` and referenced artifacts are verified.
Historical docs and AI_analysis are not live truth.

## Active Batch Semantics

ODCR active training uses `batch_semantics_version=odcr_no_accum/1`.
`per_gpu_batch_size` is the tuning knob for each rank forward/backward pass,
and `global_batch_size = per_gpu_batch_size * ddp_world_size`.
`micro_batch_size` is only a display alias for `per_gpu_batch_size`.
`grad_accum`, `gradient_accumulation_steps`, and
`accumulate_grad_batches` are retired historical concepts and fail fast on
config, CLI, and env inputs. Step3 structured representation losses rely on
same-forward sample relationships; gradient accumulation would not enlarge
that pool and would confuse optimizer, scheduler, logging, checkpoint, and
throughput semantics.

## Task2 Step3 Active Upstream

The active Step3 run for task2 is resolved by:

```text
runs/step3/task2/latest.json -> runs/step3/task2/2/meta/stage_status.json
```

Current accepted status:

- stage: `step3`
- task: `2`
- active run: `2`
- final status: `completed_with_eval_handoff`
- downstream ready: `true`
- ready for: `step4`
- selected checkpoint: `runs/step3/task2/2/model/best_observed.pth`
- eval handoff: `runs/step3/task2/2/meta/eval_handoff.json`
- old post-train eval failure: preserved in `failure_history`, not an active
  downstream block

`runs/step3/task2/2/meta/quality_audit.json` is a preserved training
diagnostic. It has a superseded sidecar and must not be used as the final
downstream gate.

`stage_status.json` is not accepted as a self-proving declaration. For Step3 ->
Step4 formal upstream, the resolver recomputes the checkpoint SHA256 and
revalidates `eval_handoff.json`, `state/checkpoint_lineage.json`,
`meta/run_summary.json`, `meta/source_table.json`, and
`meta/resolved_config.json` on every read. Quality-audit true/false claims
cannot override strict stage status truth.

`AI_analysis/05_final_reports/stage_truth_antiforgery_machine_verdict.json`
is the machine-readable verdict file for the Stage Truth Anti-Forgery rebuild.
The human final report must follow that verdict. Dry-run, guardrail, doctor,
and `completed.stamp` evidence are validation signals only; they are not formal
experiment results.

## Formal Upstream Rule

Default formal Step4 uses the active latest Step3 run:

```text
./odcr step4 --task 2 --dry-run
```

Manual formal selection uses the same gate:

```text
./odcr step4 --task 2 --from-step3-run 2 --dry-run
```

Failed, running, partial, quality-blocked, or superseded runs are not eligible
formal upstreams. A non-latest but otherwise eligible run must be promoted
before formal downstream use:

```text
./odcr promote-upstream --stage step3 --task 2 --run-id <RUN_ID>
```

Promotion validates the target run through the same strict resolver and then
atomically switches only `latest.json`. It does not rewrite older run
`stage_status.json` files into `superseded`; old run statuses are immutable
historical evidence, and active identity is decided by the latest pointer.

## Step4 Evidence-Level Warning

CPU preview is `E1_schema_preview`. It is a schema/contract preview that uses
proxy diagnostics such as fixed `rating_delta=0`,
`rating_stability_score=1`, `shared_latent_similarity=1`, and
`specific_latent_shift=0.5`. CPU preview is not tuning evidence.

CPU preview cannot be used for Step4 RCR candidate ranking, cannot produce
`best_candidate.yaml`, cannot produce a formal patch suggestion, cannot support
machine verdict A, and cannot support a formal Step4 prompt. CUDA availability
or a tmux bridge probe is only `E3_gpu_transport`; it proves transport, not
Step4 posterior runtime evidence.

Only `E4_gpu_shard_forward_bounded` or `E5_formal_full_run` evidence may enter
Step4 RCR candidate ranking, best-candidate selection, patch suggestion, or
machine-verdict A. The old `C9_balanced_quantile` and
`C9_bucket_balanced` CPU-preview best candidates are superseded by the real GPU
diagnosis and must not be used as formal candidates.

Current formal Step4 remains blocked until a real GPU
`E4_gpu_shard_forward_bounded` candidate completes the required bounded
validation. The next allowed phase is real-GPU RCR candidate completion, not
formal Step4.
