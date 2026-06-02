# Weak Cross-Platform Ablation Plan

## Scope

This plan covers only task8 (`TripAdvisor -> Yelp`) and task7 (`Yelp -> TripAdvisor`).

## Variants

| Variant | Purpose | Status |
|---|---|---|
| `wo_rcr` | Disable RCR route/pool weighting and use flat eligible-pool sampling. | planned skeleton only |
| `wo_cf` | Disable CF, aux-CF, and auxiliary cross-domain samples; keep target gold only. | planned skeleton only |
| `wo_ccv_fca` | Disable CCV/FCA explanation consistency constraints while preserving the rest of Step5_e. | planned skeleton only |

## Non-Goals

- No task2/task5 ablations.
- No 5-seed runs.
- No longest-reference rebuild.
- No CIER/MAPLE/ELIXIR baseline adaptation.
- No Step5A restoration or Step5A-vs-Step5_e result.
- No greedy-vs-sampling official ablation.

## Paper Boundary

All planned ablations remain paper-ineligible until formal train/eval artifacts exist, result snapshots are extracted, and manual review clears `requires_manual_review`.
