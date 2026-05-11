# [SUPERSEDED / HISTORICAL ONLY]

This file preserves the Step3 eval/effectiveness contract history. It is not a
live project-state source. Use `docs/CURRENT_PROJECT_STATE.md` and
`runs/*/meta/stage_status.json` for current run status.

# ODCR Step3 Eval And Effectiveness Contract

This contract locks the Step3 paper-compatible eval surface, two-phase eval
runtime, explicit LR damping semantics, and training-effectiveness evidence.

## Eval Protocol Layers

Step3 eval protocols are mutually labeled and must not be mixed.

- `minimal_eval`: post-train default. It runs checkpoint/rating sanity only,
  skips BLEU/ROUGE/METEOR/DIST, skips BERTScore, and is not paper-comparable.
- `odcr_step3_diagnostic`: internal ODCR diagnosis on `merged/<task>/aug_valid.csv`.
  It uses merged auxiliary+target data, 48-token text, may write samples and
  collapse stats, and must record `diagnostic_only=true` plus
  `not_paper_comparable=true`.
- `paper_target_only_eval`: D4C paper-compatible Step3 eval. It reads target-only
  `data/<target>/{valid,test}.csv`, uses 25-token reference/decode, records
  `paper_comparable=true`, and computes only MAE/RMSE, ROUGE-1/ROUGE-L,
  BLEU-1/2/3/4, DIST-1/2, and METEOR.
- `full_pipeline_final_eval`: reserved interface for Step4/Step5 final ODCR
  paper metrics. Step3 diagnostic output is never a substitute for this layer.

## Paper Metric Lock

Formal paper comparison excludes BERTScore. Code1 utilities may contain
BERTScore helpers, but ODCR `paper_target_only_eval` keeps
`bertscore_enabled=false` and must not gate quality or registry rows on
BERTScore resources. METEOR failures must surface as eval errors or explicit
offline fallback status; they must not silently become normal zero scores.

## Two-Phase Runtime

Step3 eval is split into:

- GPU inference phase: multi-rank DDP, `torch.inference_mode`, bf16 autocast,
  TF32 backend already applied by the runtime, per-rank prediction shards, then
  immediate `destroy_process_group`.
- CPU metric phase: single process, no NCCL, reads prediction shards, sorts by
  stable `sample_id`, checks missing/duplicate/count alignment, then computes
  CPU-heavy text metrics and writes eval artifacts.

Prediction shards live under `meta/eval_<protocol>_<split>/prediction_shards/`
and must include `sample_id`, `row_id`, `split`, `domain`, `user_id`,
`item_id`, `rating_gold`, `rating_pred`, `pred_text`, `ref_text`,
`decode_status`, `source_row_index`, and `rank`.

## Batch Invariance

Eval batch size may scale from 1536 to 3072 and 6144 only when prediction rows
and metrics are invariant after sorting by `sample_id`. Batch scaling changes
speed, not protocol or metrics. A metric/sample mismatch is a P0 eval
correctness bug and blocks paper-comparable output.

## Safe Damping

Step3 scheduler modes are mutually exclusive:

- `warmup_cosine`: pure scheduler, `damping_enabled=false`, no damping events,
  and current LR must not fall below `base_min_lr`.
- `safe_damping_v2`: probe-only scheduler, `damping_enabled=true`,
  `base_scheduler=warmup_cosine`, with capped events, real cooldown,
  `effective_min_lr`, `damping_factor_cumulative`, and structured
  `damping_events.jsonl`.

Current LR below the original cosine floor is allowed only when the effective
floor is logged and explained by safe damping.

## Training Effectiveness

Every epoch records `training_effectiveness.jsonl` with valid-loss movement,
best gap, recent delta, base/effective LR, damping events, checkpoint
improvement, gradient status, and action gates such as
`stop_and_select_candidate`, `run_paper_target_only_eval`, or
`review_loss_rebalance`. Loss components are
summarized in:

- `loss_component_epoch_summary.csv`
- `loss_component_trends.json`
- `component_contribution_summary.md`

These artifacts explain whether post-epoch gains are optimizer-limited,
component-saturated, protocol-misaligned, or likely in need of loss rebalance.

## Run2 Semantics

Run2 training completed, but post-train eval failed. The run summary must split
this as `train_status=completed`, `eval_status=failed`,
`quality_status=not_evaluated`, `downstream_ready=false`, and
`failure_phase=post_train_eval`. The checkpoint remains eligible for
eval-only validation, but Step4/Step5 downstream handoff remains blocked until
the required eval protocol passes.
