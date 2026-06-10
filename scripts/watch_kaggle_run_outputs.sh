#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-a4_convnext_tiny_full_finetune_320}"
RESULTS_DIR="${2:-results/accuracy_phase1/$RUN_ID}"
CHECK_SECONDS="${3:-60}"
MAX_ATTEMPTS="${4:-0}"

MODE="${WATCH_MODE:-local}"
KERNEL_SLUG="${KAGGLE_KERNEL_SLUG:-}"
KAGGLE_WORK_DIR="${KAGGLE_WORK_DIR:-./.kaggle_watch}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/watch_kaggle_run_outputs.sh [run_id] [results_dir] [check_seconds] [max_attempts]

By default, the script checks local result files.

Optional env vars:
  WATCH_MODE=local|kaggle
  KAGGLE_KERNEL_SLUG=tuannm3823/<kernel-name>
  KAGGLE_WORK_DIR=./.kaggle_watch

Examples:
  # Local watcher
  ./scripts/watch_kaggle_run_outputs.sh a4_convnext_tiny_full_finetune_320 results/accuracy_phase1/a4_convnext_tiny_full_finetune_320 120 0

  # Kaggle watcher (polls kernel output)
  WATCH_MODE=kaggle \
  KAGGLE_KERNEL_SLUG=tuannm3823/foodlens-a4-convnext-tiny-ft-hr-320 \
  KAGGLE_WORK_DIR=/tmp/kaggle_a4_check \
  ./scripts/watch_kaggle_run_outputs.sh a4_convnext_tiny_full_finetune_320 /tmp/kaggle_a4_check/accuracy_phase1/a4_convnext_tiny_full_finetune_320 120 0
EOF
}

REQUIRED_FILES=(
  "test_predictions.csv"
  "test_class_report.csv"
  "test_confusion_pairs.csv"
)

attempt=0
while true; do
  attempt=$((attempt + 1))
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$timestamp] check #$attempt: $RESULTS_DIR"

  if [[ "$MODE" == "kaggle" ]]; then
    if ! command -v kaggle >/dev/null 2>&1; then
      echo "Error: kaggle CLI is not available in PATH. Install kaggle package and configure credentials."
      exit 1
    fi
    if [[ -z "$KERNEL_SLUG" ]]; then
      echo "Error: KAGGLE_KERNEL_SLUG is required for WATCH_MODE=kaggle."
      echo "Example: KAGGLE_KERNEL_SLUG=tuannm3823/foodlens-a4-convnext-tiny-ft-hr-320"
      usage
      exit 1
    fi

    mkdir -p "$KAGGLE_WORK_DIR"
    echo "Syncing Kaggle output for ${KERNEL_SLUG} -> ${KAGGLE_WORK_DIR}"
    kaggle kernels output "$KERNEL_SLUG" -p "$KAGGLE_WORK_DIR"
  fi

  missing=()
  for filename in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$RESULTS_DIR/$filename" ]]; then
      missing+=("$filename")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "Base experiment artifacts found."
  else
    echo "Missing base files: ${missing[*]}"
  fi

  decision_candidates=(
    "$RESULTS_DIR/test_decision_layer"
    "$RESULTS_DIR/a4_decision_layer"
    "$RESULTS_DIR/a3b_decision_layer"
  )

  decision_found="none"
  for candidate in "${decision_candidates[@]}"; do
    if [[ -f "$candidate/decision_policy.json" && -f "$candidate/test_predictions_with_decisions.csv" ]]; then
      decision_found="$candidate"
      break
    fi
  done

  if [[ "$decision_found" != "none" ]]; then
    echo "Decision-layer outputs found at: $decision_found"
    break
  fi

  if [[ $MAX_ATTEMPTS -ne 0 && $attempt -ge $MAX_ATTEMPTS ]]; then
    echo "Stopped after $attempt checks (max attempts reached)."
    exit 1
  fi

  echo "Not ready yet. Waiting ${CHECK_SECONDS}s..."
  sleep "$CHECK_SECONDS"
done

echo "Monitor complete."
