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

from chess_sim.assets import available_piece_sets
from chess_sim.config import BOARD_FINISHES, PER_EPISODE_PLACEMENT
from chess_sim.rollout import EvaluationReport
from chess_sim import (AppearanceConfig, BoardConfig, Camera, CaptureSampler, RandomizationConfig, ChessSimEnv, ControlConfig, EnvConfig, LeRobotPolicy,
                       LeRobotPolicyConfig, MoveSampler, RolloutConfig, evaluate)
from chess_sim.policies import DEFAULT_ACTION_STEPS


def sample_scenes(args) -> list:
    """`--scenes` boards and looks, drawn as the wide collection draws them, from the
    allowed piece sets and board materials."""
    sets = [s for s in (args.piece_sets or available_piece_sets(verified_only=True))
            if s not in args.exclude_piece_sets]
    finishes = [f for f in (args.board_finishes or BOARD_FINISHES) if f not in args.exclude_board_finishes]
    rng = np.random.default_rng(args.scene_seed)
    scenes = []
    while len(scenes) < args.scenes:
        board, look = BoardConfig.sample(rng, per_episode_placement=True), AppearanceConfig.sample(rng, piece_sets=sets)
        if look.board_finish in finishes:
            scenes.append((board, look))
    return scenes


def describe(board, look) -> str:
    if look.camera_height is None:
        camera = "rig camera"
    elif look.camera_offset is None:
        camera = f"mast-line camera {look.camera_height * 100:.0f} cm up"
    else:
        out = float(np.hypot(*look.camera_offset))
        camera = f"{'side' if out > 0.27 else 'boom'} camera {out * 100:.0f} cm out, {look.camera_height * 100:.0f} cm up"
    return (f"{look.piece_set}, {look.board_finish} board {board.square * 1000:.0f} mm, turned "
            f"{np.degrees(board.yaw):+.0f} deg, {camera}, {look.table_surface} table")


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
                    help="actions executed per model call before it looks again; the default 30 "
                         "is the trained chunk, 5 is worth +17 points on captures")
    ap.add_argument("--num-inference-steps", type=int, default=None,
                    help="flow-matching steps per action chunk; more costs time and cuts sampling noise")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video", default=None, help="record the episodes to this mp4")
    ap.add_argument("--video-cameras", nargs="+", type=Camera, choices=list(Camera),
                    default=[Camera.EXTERNAL, Camera.TOP],
                    help="views tiled left to right; 'top wrist' is exactly what the policy sees")
    ap.add_argument("--report", default=None, help="write the full report as JSON to this path")
    ap.add_argument("--scenes", type=int, default=0,
                    help="score on this many sampled scenes - board, piece set, camera, lighting, "
                         "table - with the episodes split evenly between them; 0 is the published scene")
    ap.add_argument("--scene-seed", type=int, default=7)
    ap.add_argument("--piece-sets", nargs="*", default=None,
                    help="draw scenes only from these sets; default is every set the expert passed")
    ap.add_argument("--exclude-piece-sets", nargs="*", default=[])
    ap.add_argument("--board-finishes", nargs="*", default=None, help="draw scenes only on these board materials")
    ap.add_argument("--exclude-board-finishes", nargs="*", default=[])
    args = ap.parse_args()

    control = ControlConfig()
    policy_config = LeRobotPolicyConfig.for_dataset(
        args.checkpoint, args.dataset_root, device=args.device, control_hz=control.hz,
        interpolate=not args.no_interpolate, n_action_steps=args.n_action_steps or None,
        num_inference_steps=args.num_inference_steps)
    policy = LeRobotPolicy.load(policy_config)
    # the env renders exactly what this checkpoint asks for, plus any other view the video wants
    extra = tuple(c for c in (args.video_cameras if args.video else []) if c not in policy.cameras)
    cameras = policy.cameras + extra
    control = control.model_copy(update={"cameras": cameras})
    scenes = sample_scenes(args) if args.scenes else [(BoardConfig(), AppearanceConfig())]
    # sampled scenes also move the board and the camera every episode, as the wide collection does
    randomization = RandomizationConfig(**PER_EPISODE_PLACEMENT) if args.scenes else RandomizationConfig()
    env = ChessSimEnv(EnvConfig(board=scenes[0][0], appearance=scenes[0][1], control=control,
                                randomization=randomization))
    print(f"{policy.policy_type}: executing {policy.action_steps} of {policy.chunk_size} actions "
          f"per model call; targets held for {policy_config.hold} control steps"
          + (f"; {args.num_inference_steps} flow steps" if args.num_inference_steps else ""))

    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.video, fps=env.control_hz, macro_block_size=1)

    def record(_action):
        if writer is not None:
            writer.append_data(np.concatenate([env.render(cam) for cam in args.video_cameras], axis=1))

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

    results = []
    for number, (board, appearance) in enumerate(scenes):
        if number:                      # a scene is compiled into the model: a new one is a new env
            env.close()
            env = ChessSimEnv(EnvConfig(board=board, appearance=appearance, control=control,
                                        randomization=randomization))
        share = [(sampler, count // len(scenes) + (number < count % len(scenes)))
                 for sampler, count in schedule]
        card = evaluate(env, policy, [(sampler, count) for sampler, count in share if count],
                        RolloutConfig(max_steps=args.max_steps, seed=args.seed + number),
                        on_step=record, on_episode=report)
        results += card.results
        if args.scenes:
            print(f"scene {number}: {card.successes}/{card.episodes} | {describe(board, appearance)}", flush=True)
    report_card = EvaluationReport(results=results)
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
