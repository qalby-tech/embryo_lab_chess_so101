#!/bin/bash
# Train on whatever machine this runs on. Resumes automatically from the last
# complete checkpoint, so re-running after any interruption continues the run.
set -e
cd "$(dirname "$0")/.."
source training/settings.env
PY=${PYTHON:-$(command -v python || command -v python3)}   # a venv has "python"; a bare host may not

[ -f "$DATASET_ROOT/.export-complete" ] || {
  echo "dataset $DATASET_ROOT is not finished - run training/build_dataset.sh first"
  echo "(info.json exists from the moment an export starts, so it does not prove completion)"
  exit 1; }

# Video dataloading opens more descriptors than the usual 1024 default allows.
ulimit -n 65536 2>/dev/null || true

exec $PY -u examples/train_policy.py \
  --root "$DATASET_ROOT" --repo-id "$REPO_ID" --out "$OUTPUT_DIR" \
  --total-steps "$TOTAL_STEPS" --block "$BLOCK" \
  --batch-size "$BATCH_SIZE" --grad-accum "${GRAD_ACCUM:-1}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --eval-episodes "$EVAL_EPISODES" --eval-captures "${EVAL_CAPTURES:-0}" \
  --eval-max-steps "$EVAL_MAX_STEPS" \
  $([ "${INTERPOLATE:-1}" = "1" ] || echo --no-interpolate) \
  "$@"
