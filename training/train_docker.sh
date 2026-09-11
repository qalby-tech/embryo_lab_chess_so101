#!/bin/bash
# Same run inside the container built from training/Dockerfile.
#
# --ipc=host is not optional: PyTorch dataloader workers deadlock without it,
# which looks like a hang with the GPU holding memory at 2% utilisation.
# On WSL, /dev/dxg and /usr/lib/wsl let the simulator render during evaluation;
# both are harmless elsewhere and can be dropped on a normal Linux host.
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)
IMAGE=${IMAGE:-chess-train:cu128}
WSL_ARGS=""
[ -e /dev/dxg ] && WSL_ARGS="--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -e GALLIUM_DRIVER=d3d12"

exec docker run --rm --gpus all --ipc=host --shm-size=8g \
  --ulimit nofile=65536:65536 $WSL_ARGS \
  -v "$REPO":/workspace \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  -w /workspace \
  -e HF_HOME=/root/.cache/huggingface \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name chess-train \
  "$IMAGE" bash training/train.sh "$@"
