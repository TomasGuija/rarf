#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

IMAGE_NAME="${IMAGE_NAME:-brats-rarf-submission}"
LOCAL_INPUT="${LOCAL_INPUT:?Set LOCAL_INPUT to the challenge input directory.}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:?Set LOCAL_OUTPUT to an empty output directory.}"
SAMPLE_STEPS="${SAMPLE_STEPS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MERGE_STRATEGY="${MERGE_STRATEGY:-mean}"
SEED="${SEED:-0}"

if [[ ! -d "$LOCAL_INPUT" ]]; then
  echo "Input directory does not exist: $LOCAL_INPUT" >&2
  exit 1
fi

mkdir -p "$LOCAL_OUTPUT"

docker run --rm --gpus all \
  -e SAMPLE_STEPS="$SAMPLE_STEPS" \
  -e NUM_SAMPLES="$NUM_SAMPLES" \
  -e MERGE_STRATEGY="$MERGE_STRATEGY" \
  -e SEED="$SEED" \
  -v "$LOCAL_INPUT:/input:ro" \
  -v "$LOCAL_OUTPUT:/output" \
  "$IMAGE_NAME"

echo "Submission inference completed successfully: $LOCAL_OUTPUT"
