# CSB-ODCR Step3 V3 Training Policy

This document is the active Step3 V3 policy handoff for CSB-ODCR. It does not
authorize a formal run by itself.

## Objective Drift

Step3 V3 treats a large validation gap plus component drift as an actionable
state, not a scheduler hint. The epoch log writes `objective_drift_status`
using `none`, `warning`, `objective_drift`, or `severe_objective_drift`.

## Recovery

Severe drift plans recovery from `best_observed`, never from `latest`. The
controller saves a drift diagnostic checkpoint, rolls back to `best_observed`,
uses a short cosine restart, and disables damping during recovery.

## Phase-Wise

Loss weights are scheduled by phase: `primary_fit`,
`csb_alignment_controlled_injection`, and `recovery_pareto_candidate`.
EASD/HSS/geometry are CSB training signals, not separate hard-loss
contributions that directly pull the primary rating path.

## Conflict Audit

Gradient conflict auditing is real-data only. Synthetic benchmarks are
forbidden. The probe reports loss-group norms, cosine matrix, conflict rate,
and recommendation without writing formal checkpoints.

Formal CSB-ODCR conflict control uses rating-anchor / paper-anchor routing.
`L_rating_shared` protects the primary scorer path, `L_light_explainer`
anchors the explanation path, and DIST guards explainer candidate selection.

## Paper-Aware

Downstream candidate selection is paper-aware. It keeps cheap valid-loss
retention for `best_observed`, top-k, milestones, and recovery candidates, then
selects scorer/explainer checkpoints from paper metrics.

## DIST

The explainer downstream choice has a DIST guard. A checkpoint with strong
MAE/RMSE but collapsed DIST is not automatically selected as the explainer
downstream checkpoint.
