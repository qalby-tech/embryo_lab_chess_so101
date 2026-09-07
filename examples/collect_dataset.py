"""Collect a chess-move dataset with the scripted expert, in parallel.

    python examples/collect_dataset.py --episodes 500 --workers 4 --randomize \
        --out datasets/chess_vla

Each episode is one pick-and-place move on a random sparse position, recorded
from the robot's two cameras (see `chess_sim.scene.ROBOT_CAMERAS`) with its
verdict in meta.json / manifest.jsonl; train only on verified successes.
`--randomize` gives every episode a different piece size, colors and lighting.

Workers each build their own scene and write into `<out>/shard_XX`; the
LeRobot exporter (`examples/export_lerobot.py`) reads all shards.
On WSL2 export GALLIUM_DRIVER=d3d12 first or rendering runs on the CPU.
"""
import argparse
import multiprocessing as mp
import os
import random
import time

import chess
import numpy as np

from chess_sim import Appearance, ChessSimEnv, EpisodeRecorder
from play_random_moves import REBUILD_EVERY, describe, random_position


def collect(out: str, episodes: int, seed: int, randomize: bool, image_size) -> tuple[int, int]:
    """Record `episodes` moves into `out`; returns (successes, episodes)."""
    rng = random.Random(seed)
    looks = np.random.default_rng(seed)
    env = ChessSimEnv(appearance=Appearance.random(looks) if randomize else Appearance(),
                      image_size=image_size)
    recorder = EpisodeRecorder(env, out)
    successes = 0
    for i in range(episodes):
        if randomize and i > 0:
            if i % REBUILD_EVERY == 0:
                env.close()
                env = ChessSimEnv(appearance=Appearance.random(looks), image_size=image_size)
                recorder = EpisodeRecorder(env, out)
            else:
                env.recolor(Appearance.random(looks))
        while True:
            board = random_position(rng, rng.randint(2, 8))
            env.reset(board.board_fen())
            candidates = env.executable_moves()
            if candidates:
                break
        mv = rng.choice(candidates)
        recorder.begin(describe(env.board, mv), fen=env.board.fen(), move=mv.uci(),
                       task="chess_move", appearance=env.appearance.__dict__ if randomize else None)
        result = env.move(chess.square_name(mv.from_square), chess.square_name(mv.to_square),
                          on_step=recorder.on_step)
        recorder.end(result.success, placement_error=result.placement_error,
                     disturbed=result.disturbed, reason=result.reason)
        successes += result.success
    env.close()
    return successes, episodes


def _worker(args):
    out, episodes, seed, randomize, image_size = args
    return collect(out, episodes, seed, randomize, image_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="random appearance per episode")
    ap.add_argument("--image-size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    ap.add_argument("--out", default="datasets/chess")
    args = ap.parse_args()

    if args.workers <= 1:
        started = time.time()
        ok, n = collect(args.out, args.episodes, args.seed, args.randomize, tuple(args.image_size))
    else:
        share, extra = divmod(args.episodes, args.workers)
        jobs = [(os.path.join(args.out, f"shard_{w:02d}"), share + (w < extra),
                 args.seed + 1000 * w, args.randomize, tuple(args.image_size))
                for w in range(args.workers)]
        started = time.time()
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_worker, jobs)
        ok, n = (sum(v) for v in zip(*results))
    elapsed = time.time() - started
    print(f"{ok}/{n} verified successes in {elapsed / 60:.1f} min "
          f"({elapsed / max(n, 1):.1f} s/episode, {args.workers} workers) -> {args.out}")


if __name__ == "__main__":
    main()
