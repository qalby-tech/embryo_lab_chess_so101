"""Record expert demonstrations, in parallel.

    python examples/collect_demonstrations.py --episodes 500 --workers 4 --randomize \
        --out datasets/chess_moves

Each episode is one pick-and-place on a random sparse position, recorded from
the robot's two cameras with its verdict in meta.json / manifest.jsonl; train
only on verified successes. `--randomize` gives every episode a different piece
size, colors, lighting, board position and arm start pose.

Work is split into chunks of `--chunk` episodes, each run in a fresh process
writing its own `<out>/shard_XXXX`; the exporter reads all shards. Recycling the
process per chunk matters: a worker's memory grows with every scene rebuild, and
on a 23 GB box four long-lived workers end up swapping and halve the throughput.
On WSL2 export GALLIUM_DRIVER=d3d12 first or rendering runs on the CPU.
"""
import argparse
import multiprocessing as mp
import os
import random
import time

import numpy as np

from chess_sim import (AppearanceConfig, CaptureSampler, ChessSimEnv, ControlConfig, EnvConfig,
                       EpisodeRecorder, MoveSampler, RecorderConfig, TaskFamily, demonstrate)

SHARD_PREFIX = "shard_"
SEED_STRIDE = 1000      # keeps shards' random streams apart


def collect(out: str, episodes: int, seed: int, randomize: bool, image_size: tuple[int, int],
            moves: tuple[str, ...], family: TaskFamily) -> tuple[int, int]:
    """Record `episodes` demonstrations into `out`; returns (successes, episodes)."""
    rng, looks = random.Random(seed), np.random.default_rng(seed)
    # One scene per process: piece size and the board texture are baked into the
    # compiled model, and every rebuild leaks about a gigabyte of driver memory.
    # Size and board vary across chunks instead, colors and lighting per episode.
    config = EnvConfig(appearance=AppearanceConfig.sample(looks) if randomize else AppearanceConfig(),
                       control=ControlConfig(image_size=image_size))
    env = ChessSimEnv(config)
    recorder = EpisodeRecorder(env, RecorderConfig(root=out))
    sampler = (CaptureSampler(randomize_layout=randomize) if family is TaskFamily.CAPTURE
               else MoveSampler(moves=moves, randomize_layout=randomize))

    def new_look(index: int) -> None:
        if randomize and index > 0:
            env.recolor(AppearanceConfig.sample(looks))

    results = demonstrate(env, sampler, episodes, rng, recorder=recorder,
                          record_appearance=randomize, on_episode_start=new_look)
    env.close()
    return sum(r.success for r in results), episodes


def _worker(job):
    return collect(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true",
                    help="random appearance, board placement and arm start pose per episode")
    ap.add_argument("--image-size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    ap.add_argument("--chunk", type=int, default=25, help="episodes per worker process")
    ap.add_argument("--task", type=TaskFamily, choices=list(TaskFamily), default=TaskFamily.MOVE,
                    help="ordinary moves, or pieces taken off the board")
    ap.add_argument("--moves", default=None,
                    help="comma-separated UCI moves to record instead of random ones, e.g. e2e4,d7d5")
    ap.add_argument("--out", default="datasets/chess")
    args = ap.parse_args()

    moves = tuple(m.strip() for m in args.moves.split(",")) if args.moves else ()
    existing = len([d for d in os.listdir(args.out)
                    if d.startswith(SHARD_PREFIX)]) if os.path.isdir(args.out) else 0
    jobs, remaining, index = [], args.episodes, existing
    while remaining > 0:
        count = min(args.chunk, remaining)
        jobs.append((os.path.join(args.out, f"{SHARD_PREFIX}{index:04d}"), count,
                     args.seed + SEED_STRIDE * index, args.randomize, tuple(args.image_size),
                     moves, args.task))
        remaining -= count
        index += 1

    started = time.time()
    if args.workers <= 1:
        results = [_worker(job) for job in jobs]
    else:
        # one process per chunk: the simulator's memory grows with every scene
        # rebuild, so long-lived workers eventually swap
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            results = list(pool.imap_unordered(_worker, jobs))
    successes, episodes = (sum(v) for v in zip(*results))
    elapsed = time.time() - started
    print(f"{successes}/{episodes} verified successes in {elapsed / 60:.1f} min "
          f"({elapsed / max(episodes, 1):.1f} s/episode, {args.workers} workers) -> {args.out}")


if __name__ == "__main__":
    main()
