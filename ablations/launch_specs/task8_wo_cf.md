# Launch Spec: task8 wo_cf

## Status

planned_skeleton_no_training

## Boundary

This spec is dry-run/smoke only. Do not start formal Step5 ablation training or eval in this infrastructure phase.

## Validation

```bash
./odcr ablation validate --task 8 --variant wo_cf
./odcr ablation dry-run --task 8 --variant wo_cf
```

## Intended Run Namespace

`runs/step5/task8/ablation_wo_cf_1`

## Variant Semantics

Disable CF, aux-CF, and auxiliary cross-domain samples; keep target gold only.

## Safety

This run must not update `runs/step5/task8/latest.json`.
