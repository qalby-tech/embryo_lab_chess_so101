"""Compare inference settings on identical positions, in one process.

    python tools/sweep_inference.py --positions 100 \
        --checkpoint outputs/molmoact2_mc/checkpoints/070000/pretrained_model \
        --dataset-root datasets/lerobot/chess_mc --out outputs/sweeps/inference.jsonl

Every arm faces the same drawn position, the same board shift and the same arm
start pose, because each episode re-seeds the sampler from the position index.
Paired like that, a difference shows up with a fraction of the episodes two
independent runs would need - which matters when a model call costs seconds.

Results are appended as they happen, so a run that is interrupted still leaves
everything it measured behind, and `--resume` picks up where it stopped.
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

from chess_sim import (CaptureSampler, ChessSimEnv, Config, ControlConfig, EnvConfig, LeRobotPolicy,
                       LeRobotPolicyConfig, MoveSampler, RolloutConfig, TaskFamily, run_episode,
                       wilson_interval)

MOVES_PER_CAPTURE = 2      # the published protocol scores 64 moves against 32 captures


class Arm(Config):
    """One inference setting to score."""

    label: str
    n_action_steps: int
    num_inference_steps: int | None = None


DEFAULT_ARMS = [
    Arm(label="horizon5", n_action_steps=5),                              # the measured default
    Arm(label="horizon30", n_action_steps=30),                            # what the checkpoint ships
    Arm(label="horizon5_flow20", n_action_steps=5, num_inference_steps=20),
    Arm(label="horizon5_flow40", n_action_steps=5, num_inference_steps=40),
]


def mcnemar(pairs: list[tuple[bool, bool]]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired outcomes; returns (b, c, p).

    Only the positions where the two arms disagree carry information: b is where
    the first won, c where the second did."""
    from math import comb

    b = sum(1 for first, second in pairs if first and not second)
    c = sum(1 for first, second in pairs if second and not first)
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--positions", type=int, default=100, help="positions, each scored by every arm")
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/sweeps/inference.jsonl")
    ap.add_argument("--resume", action="store_true", help="skip positions already in --out")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="only these arms by label, for extending one comparison cheaply")
    args = ap.parse_args()

    arms = DEFAULT_ARMS if not args.arms else [a for a in DEFAULT_ARMS if a.label in args.arms]
    if not arms:
        raise SystemExit(f"no arms match {args.arms}; known: {[a.label for a in DEFAULT_ARMS]}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done: set[tuple[int, str]] = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            done = {(json.loads(line)["position"], json.loads(line)["arm"]) for line in f if line.strip()}
        print(f"resuming: {len(done)} episodes already scored")

    control = ControlConfig()
    base = LeRobotPolicyConfig.for_dataset(args.checkpoint, args.dataset_root, device=args.device,
                                           control_hz=control.hz)
    loaded = time.perf_counter()
    policy = LeRobotPolicy.load(base)
    print(f"loaded {policy.policy_type} in {time.perf_counter() - loaded:.0f} s; "
          f"chunk {policy.chunk_size}, targets held {base.hold} control steps", flush=True)
    env = ChessSimEnv(EnvConfig(control=control.model_copy(update={"cameras": policy.cameras})))
    rollout = RolloutConfig(max_steps=args.max_steps)

    outcomes: dict[str, dict[int, bool]] = defaultdict(dict)
    for line in open(args.out) if os.path.exists(args.out) else []:
        if line.strip():
            record = json.loads(line)
            outcomes[record["arm"]][record["position"]] = record["success"]

    started = time.time()
    for index in range(args.positions):
        family = TaskFamily.CAPTURE if index % (MOVES_PER_CAPTURE + 1) == MOVES_PER_CAPTURE \
            else TaskFamily.MOVE
        sampler = CaptureSampler() if family is TaskFamily.CAPTURE else MoveSampler()
        for arm in arms:
            if (index, arm.label) in done:
                continue
            # same seed per position, so every arm sees the same board, shift and start pose
            task = sampler.sample(env, random.Random(args.seed + index))
            adapter = LeRobotPolicy(base.model_copy(update={
                "n_action_steps": arm.n_action_steps,
                "num_inference_steps": arm.num_inference_steps}),
                policy.policy, (policy.preprocess, policy.postprocess), policy.features)
            adapter.policy.config.n_action_steps = arm.n_action_steps
            result = run_episode(env, adapter, task, rollout)
            outcomes[arm.label][index] = result.success
            with open(args.out, "a") as f:
                f.write(json.dumps({"position": index, "arm": arm.label, "family": str(family),
                                    "task": task.label, "success": result.success,
                                    "placement_error": result.placement_error,
                                    "right_piece": result.right_piece,
                                    "reason": str(result.reason) if result.reason else None}) + "\n")
        scored = {label: sum(v.values()) for label, v in outcomes.items()}
        elapsed = (time.time() - started) / 60
        print(f"[{index + 1}/{args.positions}] {scored} ({elapsed:.0f} min)", flush=True)

    print("\nper arm:")
    for arm in arms:
        results = outcomes[arm.label]
        low, high = wilson_interval(sum(results.values()), len(results))
        print(f"  {arm.label:18s} {sum(results.values())}/{len(results)} "
              f"({100 * sum(results.values()) / max(len(results), 1):.0f}%, "
              f"95% CI {low * 100:.0f}-{high * 100:.0f}%)")
    baseline = arms[0]
    print(f"\npaired against {baseline.label}:")
    for arm in arms[1:]:
        shared = sorted(set(outcomes[baseline.label]) & set(outcomes[arm.label]))
        pairs = [(outcomes[baseline.label][i], outcomes[arm.label][i]) for i in shared]
        b, c, p = mcnemar(pairs)
        print(f"  {arm.label:18s} {len(shared)} pairs, {baseline.label} won {b}, {arm.label} won {c}"
              f", p={p:.3f}")
    env.close()


if __name__ == "__main__":
    main()
