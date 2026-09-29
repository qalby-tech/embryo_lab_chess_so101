"""Let the scripted expert play every imported piece set; keep the ones it can handle.

    python tools/check_piece_sets.py                  # every set not checked yet
    python tools/check_piece_sets.py --recheck        # all of them again

A generated piece can look right and still defeat the gripper: a neck too thin to
pinch, a crown wider than the jaw opens, a base that rocks. The expert is the only
source of demonstrations, so a set it fails on would only teach failures. Each set
plays 12 moves and 6 captures on the start position family the collector uses; a
set passes at `--min-rate` or better. Results go to piece_sets/verified.json, which
`available_piece_sets(verified_only=True)` reads.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from chess_sim import AppearanceConfig, CaptureSampler, ChessSimEnv, EnvConfig, MoveSampler
from chess_sim.assets import DEFAULT_SET, PIECE_SETS_DIR, VERIFIED_FILE, available_piece_sets


def check(piece_set: str, moves: int, captures: int, seed: int) -> dict:
    env = ChessSimEnv(EnvConfig(appearance=AppearanceConfig(piece_set=piece_set)))
    rng, started = random.Random(seed), time.time()
    outcomes = {"move": [], "capture": []}
    for family, sampler, count in (("move", MoveSampler(), moves), ("capture", CaptureSampler(), captures)):
        for index in range(count):
            task = sampler.sample(env, rng, index)
            result = env.execute(task)
            outcomes[family].append(None if result.success else str(result.reason))
    env.close()
    failures = [r for rs in outcomes.values() for r in rs if r is not None]
    total = moves + captures
    return {"successes": total - len(failures), "episodes": total,
            "failures": sorted(set(failures)), "seconds": round(time.time() - started)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", type=int, default=12)
    ap.add_argument("--captures", type=int, default=6)
    ap.add_argument("--min-rate", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--recheck", action="store_true")
    args = ap.parse_args()

    path = os.path.join(PIECE_SETS_DIR, VERIFIED_FILE)
    results = {}
    if os.path.isfile(path) and not args.recheck:
        with open(path) as f:
            results = json.load(f)
    for piece_set in available_piece_sets():
        if piece_set == DEFAULT_SET or (piece_set in results and not args.recheck):
            continue
        outcome = check(piece_set, args.moves, args.captures, args.seed)
        outcome["passed"] = outcome["successes"] >= args.min_rate * outcome["episodes"]
        results[piece_set] = outcome
        print(f"{piece_set:22s} {outcome['successes']}/{outcome['episodes']} "
              f"{'pass' if outcome['passed'] else 'FAIL'} {outcome['failures'] or ''} ({outcome['seconds']} s)",
              flush=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
    passed = sorted(name for name, r in results.items() if r["passed"])
    print(f"\n{len(passed)} of {len(results)} sets pass: {', '.join(passed)}")


if __name__ == "__main__":
    main()
