"""Where the published policy, dataset and showcase videos live."""
from __future__ import annotations

MODEL_REPO = "XvKuoMing/so101_chess"      # MolmoAct2 + LoRA, 70,000 steps
DATASET_REPO = "XvKuoMing/so101_chess"    # the demonstrations it was trained on
MODEL_COLLECTION = "XvKuoMing/models-so101-6aae68c3343d3bfa98934a11"
DATASET_COLLECTION = "XvKuoMing/so101-datasets-6a8dd2002c9ff7c5a51dcc04"

MODEL_URL = f"https://huggingface.co/{MODEL_REPO}"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO}"
# showcase reels on the model page: the policy driving the arm, unedited
SHOWCASE = {"moves": f"{MODEL_URL}/resolve/main/media/reel_moves.mp4",
            "captures": f"{MODEL_URL}/resolve/main/media/reel_captures.mp4"}
EXPERIMENT_RECORD = f"{MODEL_URL}/blob/main/docs/EXPERIMENTS.md"
