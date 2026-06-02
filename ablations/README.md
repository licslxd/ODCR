# ODCR Weak Cross-Platform Ablations

This directory is the control console for task7/task8 weak cross-platform ablations.

It owns registry entries, relative override specs, manifest schemas, evidence indexes, result snapshot skeletons, and paper-table templates. Real run artifacts belong under `runs/step5/task7|task8/ablation_*`.

This phase is infrastructure-only:

- no formal Step5 ablation training has been started;
- no formal ablation eval has been started;
- no ablation result is paper-eligible;
- no ablation run may update `runs/step5/task7/latest.json` or `runs/step5/task8/latest.json`.

Use:

```bash
./odcr ablation show --task 8 --variant wo_rcr
./odcr ablation validate --task 8 --variant wo_rcr
./odcr ablation dry-run --task 8 --variant wo_rcr
```

The variants in scope are only `wo_rcr`, `wo_cf`, and `wo_ccv_fca` for task7 and task8.
