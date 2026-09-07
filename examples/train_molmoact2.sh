#!/usr/bin/env bash
# Fine-tune MolmoAct2 on a chess_so101 LeRobot dataset, on one GPU.
#
#   examples/train_molmoact2.sh datasets/lerobot/chess_so101 outputs/molmoact2_chess
#
# Runs in the VLA environment (Python 3.12 with lerobot installed from git main;
# the PyPI 0.6.1 release cannot train MolmoAct2, see huggingface/lerobot#3998),
# NOT in the simulator environment.
#
# Starts from allenai/MolmoAct2-SO100_101, which was fine-tuned on 1,659 public
# SO-100/101 LeRobot datasets and shares this action space exactly: 6-D absolute
# joint pose at 30 Hz with a 30-step chunk. LoRA on the VLM with a fully
# trainable action expert is the recipe the model card recommends under ~200
# demonstrations, and is what fits in 32 GB (~20 GiB at batch 8 per the
# published memory table; full fine-tuning needs ~48 GiB).
set -euo pipefail

DATA_ROOT=${1:-datasets/lerobot/chess_so101}
OUT_DIR=${2:-outputs/molmoact2_chess}
REPO_ID=${REPO_ID:-local/chess_so101}
STEPS=${STEPS:-6000}
BATCH=${BATCH:-8}

accelerate launch --num_processes=1 --mixed_precision=bf16 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATA_ROOT" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --policy.type=molmoact2 \
    --policy.checkpoint_path=allenai/MolmoAct2-SO100_101 \
    --policy.device=cuda \
    --policy.action_mode=both \
    --policy.train_mode_vlm=lora \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.setup_type='single so100/so101 robotic arm in molmoact2' \
    --policy.control_mode='absolute joint pose' \
    --policy.image_keys='["observation.images.top","observation.images.wrist"]' \
    --policy.gradient_checkpointing=true \
    --policy.normalize_gripper=true \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --save_freq=1000 \
    --env_eval_freq=-1 \
    --output_dir="$OUT_DIR" \
    "${@:3}"
