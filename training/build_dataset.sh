#!/bin/bash
# Record the three instruction families and export them as one LeRobot dataset.
# Skip the collection steps if datasets/chess_* already hold the recordings.
set -e
cd "$(dirname "$0")/.."
source training/settings.env
PY=${PYTHON:-$(command -v python || command -v python3)}   # a venv has "python"; a bare host may not
EPISODES_MOVE=${EPISODES_MOVE:-5000}
EPISODES_CAPTURE=${EPISODES_CAPTURE:-1200}
EPISODES_RESTORE=${EPISODES_RESTORE:-1200}
WORKERS=${WORKERS:-4}

# One scene per worker process and a fresh process per chunk: every scene
# rebuild leaks about a gigabyte, and long-lived workers end up swapping.
# That fix took collection from 650 to 928 episodes/hour.
collect () {   # task, episodes, out
  [ -d "$3" ] && [ "$(find "$3" -name meta.json | wc -l)" -ge "$2" ] && { echo "$3 already has $2 episodes"; return; }
  $PY -u examples/collect_dataset.py --episodes "$2" --workers "$WORKERS" --chunk 25 \
      --randomize --task "$1" --out "$3"
}
collect move    "$EPISODES_MOVE"    datasets/chess_vla2
collect capture "$EPISODES_CAPTURE" datasets/chess_capture
collect restore "$EPISODES_RESTORE" datasets/chess_restore

# Only verified successes are exported; ~99% of recordings qualify.
rm -rf "$DATASET_ROOT"
$PY -u examples/export_lerobot.py \
    --in datasets/chess_vla2 datasets/chess_capture datasets/chess_restore \
    --repo-id "$REPO_ID" --root "$DATASET_ROOT" \
    --stride "$STRIDE" --encoder-threads 8
touch "$DATASET_ROOT/.export-complete"
echo "dataset built at $DATASET_ROOT"
