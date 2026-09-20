#!/bin/bash
# Record both instruction families and export them as one LeRobot dataset.
# Skip the collection steps if datasets/chess_* already hold the recordings.
set -e
cd "$(dirname "$0")/.."
source training/settings.env
PY=${PYTHON:-$(command -v python || command -v python3)}   # a venv has "python"; a bare host may not
EPISODES_MOVE=${EPISODES_MOVE:-5000}
EPISODES_CAPTURE=${EPISODES_CAPTURE:-1200}
WORKERS=${WORKERS:-4}

# One scene per worker process and a fresh process per chunk: every scene
# rebuild leaks about a gigabyte, and long-lived workers end up swapping.
# That fix took collection from 650 to 928 episodes/hour.
collect () {   # task, episodes, out
  have=$( [ -d "$3" ] && find "$3" -name meta.json | wc -l || echo 0 )
  [ "$have" -ge "$2" ] && { echo "$3 already has $have episodes"; return; }
  # An interrupted collection resumes: new shards get new seeds, so only the
  # shortfall is recorded, not the whole count on top of what is there.
  $PY -u examples/collect_demonstrations.py --episodes $(( $2 - have )) --workers "$WORKERS" --chunk 25 \
      --randomize --task "$1" --out "$3"
}
collect move    "$EPISODES_MOVE"    "$MOVE_RECORDINGS"
collect capture "$EPISODES_CAPTURE" "$CAPTURE_RECORDINGS"

# Only verified successes are exported; ~99% of recordings qualify.
rm -rf "$DATASET_ROOT"
$PY -u examples/export_dataset.py \
    --in "$MOVE_RECORDINGS" "$CAPTURE_RECORDINGS" \
    --repo-id "$REPO_ID" --root "$DATASET_ROOT" \
    --stride "$STRIDE" --encoder-threads 8
touch "$DATASET_ROOT/.export-complete"
echo "dataset built at $DATASET_ROOT"
