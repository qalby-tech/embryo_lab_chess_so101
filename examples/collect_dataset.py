"""Collect a chess-move dataset with the scripted expert.

    python examples/collect_dataset.py --episodes 100 --out datasets/chess

Every episode is written with its verdict in meta.json / manifest.jsonl; use
only verified successes for training.
"""
import argparse

from play_random_moves import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="datasets/chess")
    args = ap.parse_args()
    run(args.episodes, args.seed, args.out)
