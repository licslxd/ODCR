# Launch Spec: task7 wo_rcr

## Status

planned_skeleton_no_training

## Boundary

This spec is dry-run/smoke only. Do not start formal Step5 ablation training or eval in this infrastructure phase.

## Validation

```bash
./odcr ablation validate --task 7 --variant wo_rcr
./odcr ablation dry-run --task 7 --variant wo_rcr
```

## Intended Run Namespace

`runs/step5/task7/ablation_wo_rcr_1`

## Variant Semantics

Disable RCR route/pool weighting and use flat/uniform eligible-pool sampling.

## Safety

This run must not update `runs/step5/task7/latest.json`.
