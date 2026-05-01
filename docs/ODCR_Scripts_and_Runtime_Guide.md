# ODCR Runtime Guide

The user-visible runtime surface is intentionally small:

```bash
./odcr <command> [options]
python code/odcr.py <command> [options]
```

The only main configuration file is `configs/odcr.yaml`.

Common commands:

```bash
./odcr doctor
./odcr show --stage step3 --task 4
./odcr step3 --task 4 --dry-run
./odcr step5 --task 4 --dry-run
./odcr eval --task 4 --dry-run
./odcr step3 --task 4 --set step3.train.batch_size=512 --dry-run
```

Run artifacts go to `runs/task{task}/{stage}/{run_id}/`.
Run metadata and logs go under each run's `meta/` directory.
Data artifacts stay in `data/` and `merged/`.
