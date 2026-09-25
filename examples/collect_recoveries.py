"""Collect recovery demonstrations: let the policy fail, then record the expert's fix.

    python examples/collect_recoveries.py --episodes 200 --task capture \
        --checkpoint outputs/molmoact2_mc/checkpoints/070000/pretrained_model \
        --dataset-root datasets/lerobot/chess_mc --out datasets/chess_recoveries

Ordinary demonstrations only ever show the arm doing things right, so a policy
never learns its way out of the messes it makes. Here the policy drives until it
is demonstrably going wrong - wrong piece engaged, a neighbour knocked, or the
named piece still sitting where it started past most of the budget - and the
scripted expert takes over from that exact state. Only the expert's half is
recorded, tagged in meta.json with what went wrong.

Captures are the default because that is where the shipped 30-action horizon
fails: 68% against 85% at a fifth of the chunk (docs/EXPERIMENTS.md 5.7).

One model load, one episode at a time: a 12 GB policy per worker does not fit
this box twice. Export the shards alongside the ordinary demonstrations with
examples/export_dataset.py.
"""
import argparse
import collections
import random

from chess_sim import (CaptureSampler, ChessSimEnv, ControlConfig, DaggerConfig, EnvConfig,
                       EpisodeRecorder, LeRobotPolicy, LeRobotPolicyConfig, MoveSampler,
                       RecorderConfig, RolloutConfig, TaskFamily, recover)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="the policy whose failures to correct")
    ap.add_argument("--dataset-root", default=None, help="read for the rate it emits targets at")
    ap.add_argument("--episodes", type=int, default=100, help="episodes to attempt at most")
    ap.add_argument("--corrections", type=int, default=None,
                    help="stop once this many usable corrections are recorded")
    ap.add_argument("--task", type=TaskFamily, choices=list(TaskFamily), default=TaskFamily.CAPTURE)
    ap.add_argument("--out", default="datasets/chess_recoveries")
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=450, help="policy budget before the episode ends")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="the horizon to collect failures at; default is the checkpoint's own")
    ap.add_argument("--min-prefix", type=int, default=30,
                    help="control steps before anything counts as going wrong")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    control = ControlConfig()
    policy = LeRobotPolicy.load(LeRobotPolicyConfig.for_dataset(
        args.checkpoint, args.dataset_root, device=args.device, control_hz=control.hz,
        n_action_steps=args.n_action_steps))
    env = ChessSimEnv(EnvConfig(control=control.model_copy(update={"cameras": policy.cameras})))
    recorder = EpisodeRecorder(env, RecorderConfig(root=args.out))
    sampler = CaptureSampler() if args.task is TaskFamily.CAPTURE else MoveSampler()
    rollout = RolloutConfig(max_steps=args.max_steps)
    dagger = DaggerConfig(min_prefix=args.min_prefix)
    print(f"{policy.policy_type} at {policy.action_steps} actions per call, correcting "
          f"{args.task} episodes into {args.out}", flush=True)

    rng = random.Random(args.seed)
    triggers: collections.Counter = collections.Counter()
    recovered = policy_ok = 0
    for index in range(args.episodes):
        task = sampler.sample(env, rng, index)
        outcome = recover(env, policy, task, recorder, rollout, dagger)
        if outcome.trigger is None:
            policy_ok += outcome.policy_success
            triggers["policy finished it"] += 1
        else:
            triggers[str(outcome.trigger)] += 1
            recovered += bool(outcome.result and outcome.result.success)
        print(f"[{index + 1}/{args.episodes}] {recovered} usable | {task.label}: "
              f"{outcome.trigger or 'no correction needed'} after {outcome.prefix_steps} steps"
              + (f" -> expert {'fixed it' if outcome.result.success else 'failed too'}"
                 if outcome.result else ""), flush=True)
        if args.corrections and recovered >= args.corrections:
            print(f"reached {recovered} corrections", flush=True)
            break

    print(f"\n{recovered} usable corrections from {args.episodes} episodes "
          f"({policy_ok} the policy got right on its own)")
    for reason, count in triggers.most_common():
        print(f"  {reason}: {count}")
    print(f"export with: python examples/export_dataset.py --in {args.out} ... --stride 3")
    env.close()


if __name__ == "__main__":
    main()
