"""Collect a chess-move dataset with the scripted expert, in parallel.

    python examples/collect_dataset.py --episodes 500 --workers 4 --randomize \
        --out datasets/chess_vla

Each episode is one pick-and-place move on a random sparse position, recorded
from the robot's two cameras (see `chess_sim.scene.ROBOT_CAMERAS`) with its
verdict in meta.json / manifest.jsonl; train only on verified successes.
`--randomize` gives every episode a different piece size, colors and lighting.

Work is split into chunks of `--chunk` episodes, each run in a fresh process
writing its own `<out>/shard_XXXX`; the LeRobot exporter
(`examples/export_lerobot.py`) reads all shards. Recycling the process per
chunk matters: a worker's memory grows with every scene rebuild, and on a
23 GB box four long-lived workers end up swapping and halve the throughput.
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
from play_random_moves import describe, describe_capture, position_with_move, random_position


def collect(out: str, episodes: int, seed: int, randomize: bool, image_size,
            moves: tuple[str, ...] = (), task: str = "move") -> tuple[int, int]:
    """Record `episodes` moves into `out`; returns (successes, episodes)."""
    rng = random.Random(seed)
    looks = np.random.default_rng(seed)
    # One scene per process: piece size and the board texture are baked into the
    # compiled model, and every rebuild leaks about a gigabyte of driver memory.
    # Size and board vary across chunks instead, colours and lighting per episode.
    env = ChessSimEnv(appearance=Appearance.random(looks) if randomize else Appearance(),
                      image_size=image_size)
    recorder = EpisodeRecorder(env, out)
    successes = 0
    for i in range(episodes):
        if randomize and i > 0:
            env.recolor(Appearance.random(looks))
        if task == "capture":
            while True:
                board = random_position(rng, rng.randint(2, 8))
                env.reset(board.board_fen(), rng=rng if randomize else None)
                targets = env.executable_captures()
                if targets:
                    break
            square = chess.square_name(rng.choice(targets))
            recorder.begin(describe_capture(square), fen=env.board.fen(), layout=env.layout, move=f"x{square}",
                           task="capture",
                           appearance=env.appearance.__dict__ if randomize else None)
            result = env.capture(square, on_step=recorder.on_step)
            recorder.end(result.success, placement_error=result.placement_error,
                         disturbed=result.disturbed, reason=result.reason)
            successes += result.success
            continue
        while True:
            if moves:
                # narrow task: the move is fixed, the rest of the board is not
                mv = chess.Move.from_uci(rng.choice(moves))
                board = position_with_move(rng, rng.randint(2, 8), mv)
                if board is None:
                    continue
                env.reset(board.board_fen(), rng=rng if randomize else None)
                if mv in env.executable_moves():
                    break
            else:
                board = random_position(rng, rng.randint(2, 8))
                env.reset(board.board_fen(), rng=rng if randomize else None)
                candidates = env.executable_moves()
                if candidates:
                    mv = rng.choice(candidates)
                    break
        recorder.begin(describe(env.board, mv), fen=env.board.fen(), layout=env.layout, move=mv.uci(),
                       task="chess_move", appearance=env.appearance.__dict__ if randomize else None)
        result = env.move(chess.square_name(mv.from_square), chess.square_name(mv.to_square),
                          on_step=recorder.on_step)
        recorder.end(result.success, placement_error=result.placement_error,
                     disturbed=result.disturbed, reason=result.reason)
        successes += result.success
    env.close()
    return successes, episodes


def _worker(args):
    out, episodes, seed, randomize, image_size, moves, task = args
    return collect(out, episodes, seed, randomize, image_size, moves, task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="random appearance, board placement and arm start pose per episode")
    ap.add_argument("--image-size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    ap.add_argument("--chunk", type=int, default=25, help="episodes per worker process")
    ap.add_argument("--task", choices=["move", "capture"], default="move",
                    help="ordinary moves, or pieces taken off the board")
    ap.add_argument("--moves", default=None,
                    help="comma-separated UCI moves to record instead of random ones, e.g. e2e4,d7d5")
    ap.add_argument("--out", default="datasets/chess")
    args = ap.parse_args()

    moves = tuple(m.strip() for m in args.moves.split(",")) if args.moves else ()
    existing = len([d for d in os.listdir(args.out) if d.startswith("shard_")]) if os.path.isdir(args.out) else 0
    jobs, remaining, index = [], args.episodes, existing
    while remaining > 0:
        count = min(args.chunk, remaining)
        jobs.append((os.path.join(args.out, f"shard_{index:04d}"), count,
                     args.seed + 1000 * index, args.randomize, tuple(args.image_size),
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
    ok, n = (sum(v) for v in zip(*results))
    elapsed = time.time() - started
    print(f"{ok}/{n} verified successes in {elapsed / 60:.1f} min "
          f"({elapsed / max(n, 1):.1f} s/episode, {args.workers} workers) -> {args.out}")


if __name__ == "__main__":
    main()
