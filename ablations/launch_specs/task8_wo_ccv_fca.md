# Launch Spec: task8 wo_ccv_fca

## Status

planned_skeleton_no_training

## Boundary

This spec is dry-run/smoke only. Do not start formal Step5 ablation training or eval in this infrastructure phase.

## Validation

```bash
./odcr ablation validate --task 8 --variant wo_ccv_fca
./odcr ablation dry-run --task 8 --variant wo_ccv_fca
```

## Intended Run Namespace

`runs/step5/task8/ablation_wo_ccv_fca_1`

## Variant Semantics

Disable CCV/FCA explanation consistency constraints while preserving the rest of Step5_e.

## Safety

This run must not update `runs/step5/task8/latest.json`.
