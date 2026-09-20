"""Score a trained policy closed-loop in the simulator.

    python examples/evaluate_policy.py --checkpoint outputs/molmoact2_mc/checkpoints/070000/pretrained_model \
        --moves 64 --captures 32 --video sim/out/eval.mp4

Each episode resets to a random sparse position, draws a task the arm could
execute there and hands the policy its instruction. The policy then drives the
arm at the control rate with no scripted help, and the attempt is scored by the
same rule the expert is held to.

Runs in the VLA environment (Python 3.12 with lerobot), with the simulator
package importable: PYTHONPATH=~/chess_so101.
"""
import argparse

import numpy as np

from chess_sim import (Camera, CaptureSampler, ChessSimEnv, ControlConfig, EnvConfig, LeRobotPolicy,
                       LeRobotPolicyConfig, MoveSampler, RolloutConfig, evaluate)
from chess_sim.policies import DEFAULT_ACTION_STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="trained policy directory")
    ap.add_argument("--moves", type=int, default=20, help="move episodes to score")
    ap.add_argument("--captures", type=int, default=0, help="capture episodes to score")
    ap.add_argument("--only-moves", default=None,
                    help="comma-separated UCI moves to score instead of random ones")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=450, help="control steps per episode")
    ap.add_argument("--dataset-root", default=None,
                    help="training dataset, read for the rate the policy emits targets at")
    ap.add_argument("--no-interpolate", action="store_true",
                    help="step to each emitted target instead of ramping between them")
    ap.add_argument("--nominal-layout", action="store_true",
                    help="board and arm start exactly in place, instead of shifted as in training")
    ap.add_argument("--n-action-steps", type=int, default=DEFAULT_ACTION_STEPS,
                    help="actions executed per model call before it looks again; "
                         "0 keeps the checkpoint's own value (30)")
    ap.add_argument("--num-inference-steps", type=int, default=None,
                    help="flow-matching steps per action chunk; more costs time and cuts sampling noise")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video", default=None, help="record the episodes to this mp4")
    ap.add_argument("--report", default=None, help="write the full report as JSON to this path")
    args = ap.parse_args()

    control = ControlConfig()
    policy_config = LeRobotPolicyConfig.for_dataset(
        args.checkpoint, args.dataset_root, device=args.device, control_hz=control.hz,
        interpolate=not args.no_interpolate, n_action_steps=args.n_action_steps or None,
        num_inference_steps=args.num_inference_steps)
    policy = LeRobotPolicy.load(policy_config)
    # the env renders exactly what this checkpoint asks for, plus the side view for the video
    cameras = policy.cameras + (Camera.EXTERNAL,) if args.video else policy.cameras
    env = ChessSimEnv(EnvConfig(control=control.model_copy(update={"cameras": cameras})))
    print(f"{policy.policy_type}: executing {policy.action_steps} of {policy.chunk_size} actions "
          f"per model call; targets held for {policy_config.hold} control steps"
          + (f"; {args.num_inference_steps} flow steps" if args.num_inference_steps else ""))

    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.video, fps=env.control_hz, macro_block_size=1)

    def record(_action):
        if writer is not None:
            writer.append_data(np.concatenate([env.render(Camera.EXTERNAL),
                                               env.render(Camera.TOP)], axis=1))

    only = tuple(m.strip() for m in args.only_moves.split(",")) if args.only_moves else ()
    randomize = not args.nominal_layout
    schedule = []
    if args.moves:
        schedule.append((MoveSampler(moves=only, randomize_layout=randomize), args.moves))
    if args.captures:
        schedule.append((CaptureSampler(randomize_layout=randomize), args.captures))

    index = 0

    def report(result):
        nonlocal index
        print(f"[{index}] {result.instruction}: {'OK' if result.success else 'FAIL'} "
              f"({result.placement_error * 1000:.1f} mm"
              f"{', fell' if not result.upright else ''}"
              f"{', disturbed ' + ','.join(result.disturbed) if result.disturbed else ''})",
              flush=True)
        index += 1

    report_card = evaluate(env, policy, schedule, RolloutConfig(max_steps=args.max_steps, seed=args.seed),
                           on_step=record, on_episode=report)
    print(report_card.summary())
    if only:
        for uci in only:
            group = [r for r in report_card.results if r.task.label == uci]
            if group:
                print(f"  {uci}: {sum(r.success for r in group)}/{len(group)} success, "
                      f"{sum(r.right_piece for r in group)}/{len(group)} right piece")
    if args.report:
        with open(args.report, "w") as f:
            f.write(report_card.model_dump_json(indent=2))
        print("wrote", args.report)
    if writer is not None:
        writer.close()
        print("wrote", args.video)
    env.close()


if __name__ == "__main__":
    main()
