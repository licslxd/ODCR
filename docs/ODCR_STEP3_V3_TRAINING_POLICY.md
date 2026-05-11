# ODCR Step3 V3 Training Policy

This document is the active Step3 V3 policy handoff. It does not authorize a
formal run by itself.

## Objective Drift

Step3 V3 treats a large validation gap plus component drift as an actionable
state, not a scheduler hint. The epoch log writes `objective_drift_status`
using `none`, `warning`, `objective_drift`, or `severe_objective_drift`.

## Recovery

Severe drift plans recovery from `best_observed`, never from `latest`. The
controller saves a drift diagnostic checkpoint, rolls back to `best_observed`,
uses a short cosine restart, and disables damping during recovery.

## Phase-Wise

Loss weights are scheduled by phase: `alignment_warmup`, `task_refinement`, and
`light_regularization`. Late phases reduce heavy structure losses so rating and
explanation objectives are not dragged by fixed all-run weights.

## Conflict Audit

Gradient conflict auditing is real-data only. Synthetic benchmarks are
forbidden. The probe reports loss-group norms, cosine matrix, conflict rate,
and recommendation without writing formal checkpoints.

## Paper-Aware

Downstream candidate selection is paper-aware. It keeps cheap valid-loss
retention for `best_observed`, top-k, milestones, and recovery candidates, then
selects scorer/explainer checkpoints from paper metrics.

## DIST

The explainer downstream choice has a DIST guard. A checkpoint with strong
MAE/RMSE but collapsed DIST is not automatically selected as the explainer
downstream checkpoint.

