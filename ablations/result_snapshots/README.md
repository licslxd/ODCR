# Ablation Result Snapshots

This directory stores small structured snapshots extracted from formal ablation eval artifacts.

Current status: planned skeleton only. Missing-artifact snapshots are allowed here, but they are not numeric paper evidence.

The extractor expects no-reference official eval artifacts only:

- `runs/step5/task7|task8/ablation_*/post_train_eval_no_ref/valid/eval_metrics.json`
- `runs/step5/task7|task8/ablation_*/post_train_eval_no_ref/test/eval_metrics.json`
- `runs/step5/task7|task8/ablation_*/post_train_eval_no_ref/valid/official_eval_report.json`
- `runs/step5/task7|task8/ablation_*/post_train_eval_no_ref/test/official_eval_report.json`

Legacy oracle-content `post_train_eval/` artifacts are excluded from paper/candidate tables.

Use `./odcr ablation snapshot --task 8 --variant wo_rcr` for a no-write view, or `./odcr ablation snapshot --task 8 --variant wo_rcr --write` to refresh all skeleton snapshots.
