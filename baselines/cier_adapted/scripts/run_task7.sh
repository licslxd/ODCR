#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK=7
MODE="${CIER_MODE:-source_to_target}"
RUN_MODE="${1:-dry-run}"
RUN_ID="${CIER_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

case "$RUN_MODE" in
  dry-run)
    python "$ROOT/baselines/cier_adapted/adapter/build_cier_dataset.py" --task "$TASK" --mode "$MODE" --dry-run
    python "$ROOT/baselines/cier_adapted/adapter/train_cier_odcr.py" --task "$TASK" --mode "$MODE" --dry-run
    python "$ROOT/baselines/cier_adapted/adapter/export_predictions.py" --task "$TASK" --mode "$MODE" --run-id smoke --dry-run
    python "$ROOT/baselines/cier_adapted/adapter/eval_with_odcr_metrics.py" --task "$TASK" --mode "$MODE" --run-id smoke --split valid --dry-run
    ;;
  smoke)
    RUN_ID="${CIER_RUN_ID:-smoke}"
    python "$ROOT/baselines/cier_adapted/adapter/build_cier_dataset.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --smoke
    python "$ROOT/baselines/cier_adapted/adapter/train_cier_odcr.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --smoke --max-steps "${CIER_SMOKE_STEPS:-1}"
    python "$ROOT/baselines/cier_adapted/adapter/export_predictions.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID"
    python "$ROOT/baselines/cier_adapted/adapter/eval_with_odcr_metrics.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --split valid
    python "$ROOT/baselines/cier_adapted/adapter/eval_with_odcr_metrics.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --split test
    ;;
  full)
    python "$ROOT/baselines/cier_adapted/adapter/build_cier_dataset.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID"
    python "$ROOT/baselines/cier_adapted/adapter/train_cier_odcr.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID"
    python "$ROOT/baselines/cier_adapted/adapter/export_predictions.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID"
    python "$ROOT/baselines/cier_adapted/adapter/eval_with_odcr_metrics.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --split valid
    python "$ROOT/baselines/cier_adapted/adapter/eval_with_odcr_metrics.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --split test
    ;;
  *)
    echo "Usage: $0 [dry-run|smoke|full]" >&2
    exit 2
    ;;
esac

