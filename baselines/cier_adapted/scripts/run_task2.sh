#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK=2
MODE="${CIER_MODE:-source_to_target}"
RUN_MODE="${1:-dry-run}"
RUN_ID="${CIER_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

TRAIN_ARGS=()
[[ -n "${CIER_MODEL_NAME:-}" ]] && TRAIN_ARGS+=(--model-name "$CIER_MODEL_NAME")
[[ -n "${CIER_BATCH_SIZE:-}" ]] && TRAIN_ARGS+=(--batch-size "$CIER_BATCH_SIZE")
[[ -n "${CIER_EVAL_BATCH_SIZE:-}" ]] && TRAIN_ARGS+=(--eval-batch-size "$CIER_EVAL_BATCH_SIZE")
[[ -n "${CIER_LEARNING_RATE:-}" ]] && TRAIN_ARGS+=(--learning-rate "$CIER_LEARNING_RATE")
[[ -n "${CIER_ACCUMULATION_STEPS:-}" ]] && TRAIN_ARGS+=(--accumulation-steps "$CIER_ACCUMULATION_STEPS")
[[ -n "${CIER_NUM_WORKERS:-}" ]] && TRAIN_ARGS+=(--num-workers "$CIER_NUM_WORKERS")
[[ -n "${CIER_EVAL_NUM_WORKERS:-}" ]] && TRAIN_ARGS+=(--eval-num-workers "$CIER_EVAL_NUM_WORKERS")
[[ -n "${CIER_PREFETCH_FACTOR:-}" ]] && TRAIN_ARGS+=(--prefetch-factor "$CIER_PREFETCH_FACTOR")
[[ -n "${CIER_PRECISION:-}" ]] && TRAIN_ARGS+=(--precision "$CIER_PRECISION")
[[ -n "${CIER_DEVICE_INDEX:-}" ]] && TRAIN_ARGS+=(--device-index "$CIER_DEVICE_INDEX")
[[ -n "${CIER_SOURCE_EPOCHS:-}" ]] && TRAIN_ARGS+=(--source-epochs "$CIER_SOURCE_EPOCHS")
[[ -n "${CIER_TARGET_EPOCHS:-}" ]] && TRAIN_ARGS+=(--target-epochs "$CIER_TARGET_EPOCHS")
[[ -n "${CIER_SHOW_TRAIN_LOSS_STEPS:-}" ]] && TRAIN_ARGS+=(--show-train-loss-steps "$CIER_SHOW_TRAIN_LOSS_STEPS")
[[ -n "${CIER_LORA_R:-}" ]] && TRAIN_ARGS+=(--lora-r "$CIER_LORA_R")
[[ -n "${CIER_LORA_ALPHA:-}" ]] && TRAIN_ARGS+=(--lora-alpha "$CIER_LORA_ALPHA")
[[ -n "${CIER_LORA_DROPOUT:-}" ]] && TRAIN_ARGS+=(--lora-dropout "$CIER_LORA_DROPOUT")
case "${CIER_PIN_MEMORY:-}" in
  true|1|yes) TRAIN_ARGS+=(--pin-memory) ;;
  false|0|no) TRAIN_ARGS+=(--no-pin-memory) ;;
esac
case "${CIER_PERSISTENT_WORKERS:-}" in
  true|1|yes) TRAIN_ARGS+=(--persistent-workers) ;;
  false|0|no) TRAIN_ARGS+=(--no-persistent-workers) ;;
esac
case "${CIER_TF32:-}" in
  true|1|yes) TRAIN_ARGS+=(--tf32) ;;
  false|0|no) TRAIN_ARGS+=(--no-tf32) ;;
esac

case "$RUN_MODE" in
  dry-run)
    python "$ROOT/baselines/cier_adapted/adapter/build_cier_dataset.py" --task "$TASK" --mode "$MODE" --dry-run
    python "$ROOT/baselines/cier_adapted/adapter/train_cier_odcr.py" --task "$TASK" --mode "$MODE" --dry-run "${TRAIN_ARGS[@]}"
    python "$ROOT/baselines/cier_adapted/adapter/export_predictions.py" --task "$TASK" --mode "$MODE" --run-id smoke --dry-run
    python "$ROOT/baselines/cier_adapted/adapter/eval_with_odcr_metrics.py" --task "$TASK" --mode "$MODE" --run-id smoke --split valid --dry-run
    ;;
  smoke)
    RUN_ID="${CIER_RUN_ID:-smoke}"
    python "$ROOT/baselines/cier_adapted/adapter/build_cier_dataset.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --smoke
    python "$ROOT/baselines/cier_adapted/adapter/train_cier_odcr.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --smoke --max-steps "${CIER_SMOKE_STEPS:-1}" "${TRAIN_ARGS[@]}"
    python "$ROOT/baselines/cier_adapted/adapter/export_predictions.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID"
    python "$ROOT/baselines/cier_adapted/adapter/eval_with_odcr_metrics.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --split valid
    python "$ROOT/baselines/cier_adapted/adapter/eval_with_odcr_metrics.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --split test
    ;;
  full)
    python "$ROOT/baselines/cier_adapted/adapter/build_cier_dataset.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID"
    python "$ROOT/baselines/cier_adapted/adapter/train_cier_odcr.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" "${TRAIN_ARGS[@]}"
    python "$ROOT/baselines/cier_adapted/adapter/export_predictions.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID"
    python "$ROOT/baselines/cier_adapted/adapter/eval_with_odcr_metrics.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --split valid
    python "$ROOT/baselines/cier_adapted/adapter/eval_with_odcr_metrics.py" --task "$TASK" --mode "$MODE" --run-id "$RUN_ID" --split test
    ;;
  *)
    echo "Usage: $0 [dry-run|smoke|full]" >&2
    exit 2
    ;;
esac
