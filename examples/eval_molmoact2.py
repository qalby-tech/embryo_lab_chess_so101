"""Run a trained MolmoAct2 policy closed-loop in the simulator and score it.

    python examples/eval_molmoact2.py --checkpoint outputs/molmoact2_chess/checkpoints/last/pretrained_model \
        --episodes 20 --video sim/out/molmoact2_eval.mp4

For each episode the simulator is reset to a random sparse position, one legal
move is chosen, and its natural-language instruction is given to the policy.
The policy then drives the arm at the control rate with no scripted help; the
move counts as a success under the same rule the scripted expert is held to
(piece within 11 mm of the target square, upright, nothing else displaced).

Runs in the VLA environment (Python 3.12 with lerobot), with the simulator
package importable: PYTHONPATH=~/chess_so101:~/chess_so101/examples.
"""
import argparse
import json
import os
import random

import chess
import numpy as np
import torch

from chess_sim import ChessSimEnv
from chess_sim.env import CAPTURE_TOLERANCE, DISTURB_TOLERANCE, PLACEMENT_TOLERANCE, UPRIGHT_MIN
from export_lerobot import CAMERAS, JOINT_OFFSETS, JOINT_SIGNS, to_so101_degrees
from play_random_moves import describe, describe_capture, position_with_move, random_position


def to_radians(degrees: np.ndarray) -> np.ndarray:
    """Inverse of `export_lerobot.to_so101_degrees`."""
    return np.radians((np.asarray(degrees, dtype=float) - JOINT_OFFSETS) / JOINT_SIGNS)


def observation(env: ChessSimEnv, instruction: str, device: str) -> dict:
    batch = {"task": [instruction],
             "observation.state": torch.from_numpy(
                 to_so101_degrees(env.arm_joint_positions()).astype(np.float32))[None].to(device)}
    for cam in CAMERAS:
        frame = torch.from_numpy(env.render(cam).copy()).permute(2, 0, 1).float() / 255.0
        batch[f"observation.images.{cam}"] = frame[None].to(device)
    return batch


def dataset_fps(root: str | None, default: int = 30) -> int:
    """The rate the policy was trained at. A policy trained on a strided export
    emits targets slower than the control loop runs, and each one has to be held
    for the difference or the arm races through the trajectory."""
    if not root:
        return default
    try:
        with open(os.path.join(root, "meta", "info.json")) as f:
            return int(json.load(f)["fps"])
    except (OSError, ValueError, KeyError):
        return default


def run_episode(env, policy, processors, rng, max_steps: int, device: str, on_step=None,
                hold: int = 1, moves: tuple[str, ...] = (), interpolate: bool = False,
                index: int = 0, task: str = "move", jitter: bool = True) -> dict:
    """One instructed move under policy control; returns the outcome."""
    source = None
    if task == "capture":
        # the named piece has to leave the board for the discard tray
        while True:
            board = random_position(rng, rng.randint(2, 8))
            env.reset(board.board_fen(), rng=rng if jitter else None)
            targets = env.executable_captures()
            if targets:
                source = rng.choice(targets)
                break
        name = chess.square_name(source)
        instruction = describe_capture(name)
        slot = env.slot_at(name)
        target = np.array(env.board_spec.capture_slot(0))
        label = f"x{name}"
    else:
        while True:
            if moves:   # score the narrow task the policy was trained on, each move equally often
                move = chess.Move.from_uci(moves[index % len(moves)])
                board = position_with_move(rng, rng.randint(2, 8), move)
                if board is None:
                    continue
                env.reset(board.board_fen(), rng=rng if jitter else None)
                if move in env.executable_moves():
                    break
            else:
                board = random_position(rng, rng.randint(2, 8))
                env.reset(board.board_fen(), rng=rng if jitter else None)
                candidates = env.executable_moves()
                if candidates:
                    move = rng.choice(candidates)
                    break
        instruction = describe(env.board, move)
        source = move.from_square
        slot = env.slot_at(chess.square_name(source))
        target = np.array(env.board_spec.square_center(move.to_square))
        label = move.uci()
    before = {sq: env.piece_position(sl)[:2].copy() for sq, sl in env._square_slot.items()}

    preprocess, postprocess = processors
    policy.reset()
    previous = env.arm_joint_positions().copy()
    steps = 0
    while steps < max_steps:
        with torch.inference_mode():
            batch = preprocess(observation(env, instruction, device))
            action = postprocess(policy.select_action(batch))
        command = to_radians(action[0].float().cpu().numpy())
        for k in range(hold):
            # A policy trained at a reduced rate emits one target per `hold`
            # control periods. Holding it as a step asks the servo for the whole
            # jump at once, which is what knocks pieces over; interpolating asks
            # for the same motion spread across the period.
            if interpolate:
                a = (k + 1) / hold
                env.apply_action(previous * (1 - a) + command * a, on_step)
            else:
                env.apply_action(command, on_step)
            steps += 1
        previous = command

    position = env.piece_position(slot)
    error = float(np.linalg.norm(position[:2] - target))
    upright = env.piece_upright(slot) > UPRIGHT_MIN
    disturbed = [chess.square_name(sq) for sq, xy in before.items()
                 if sq != source
                 and np.linalg.norm(env.piece_position(env._square_slot[sq])[:2] - xy) > DISTURB_TOLERANCE]
    if task == "capture":
        # the scripted rule: off the board, in its tray slot, standing, nothing else touched
        off_board = not env.board_spec.on_field(position[:2])
        success = off_board and error < CAPTURE_TOLERANCE and upright and not disturbed
        right_piece = float(np.linalg.norm(position[:2] - before[source])) > DISTURB_TOLERANCE
        picked = chess.square_name(source) if right_piece else None
    else:
        # Which piece actually moved. With several moves in play, going to the wrong
        # square is a failure of grounding and needs telling apart from a clumsy
        # grasp of the right one.
        shifted = {sq: float(np.linalg.norm(env.piece_position(env._square_slot[sq])[:2] - xy))
                   for sq, xy in before.items()}
        moved = max(shifted, key=shifted.get) if shifted else None
        if moved is None or shifted[moved] < DISTURB_TOLERANCE:
            moved = None
        right_piece = moved == source
        picked = chess.square_name(moved) if moved is not None else None
        success = error < PLACEMENT_TOLERANCE and upright and not disturbed
    return {"move": label, "task": task, "instruction": instruction, "placement_error": error,
            "upright": upright, "disturbed": disturbed, "picked": picked,
            "right_piece": bool(right_piece), "success": bool(success)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="trained policy directory")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=400, help="control steps per episode")
    ap.add_argument("--dataset-root", default=None,
                    help="training dataset, read for the rate the policy emits targets at")
    ap.add_argument("--task", choices=["move", "capture"], default="move",
                    help="which instruction family to score")
    ap.add_argument("--moves", default=None,
                    help="comma-separated UCI moves to score instead of random ones")
    ap.add_argument("--interpolate", action="store_true",
                    help="ramp between the policy's targets instead of stepping to each")
    ap.add_argument("--nominal-layout", action="store_true",
                    help="board and arm start exactly in place, instead of shifted as in training")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video", default=None, help="record the episodes to this mp4")
    args = ap.parse_args()

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    policy = MolmoAct2Policy.from_pretrained(args.checkpoint).to(args.device).eval()
    # the checkpoint carries its own normalization pipelines; the policy sees
    # normalized inputs and returns actions in dataset units through them
    processors = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.checkpoint)
    env = ChessSimEnv(cameras=tuple(CAMERAS) + ("external",), image_size=(640, 480))
    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.video, fps=env.control_hz, macro_block_size=1)

    def record(_action):
        if writer is not None:
            writer.append_data(np.concatenate([env.render("external"), env.render("top")], axis=1))

    hold = max(1, round(env.control_hz / dataset_fps(args.dataset_root)))
    if hold > 1:
        print(f"policy emits targets at {env.control_hz // hold} Hz; "
              f"holding each for {hold} control steps")
    moves = tuple(m.strip() for m in args.moves.split(",")) if args.moves else ()
    rng = random.Random(args.seed)
    successes, errors, outcomes = 0, [], []
    for i in range(args.episodes):
        outcome = run_episode(env, policy, processors, rng, args.max_steps, args.device,
                              record, hold, moves, args.interpolate, i, args.task,
                              not args.nominal_layout)
        successes += outcome["success"]
        errors.append(outcome["placement_error"])
        outcomes.append(outcome)
        print(f"[{i}] {outcome['instruction']}: {'OK' if outcome['success'] else 'FAIL'} "
              f"({outcome['placement_error'] * 1000:.1f} mm"
              f"{', fell' if not outcome['upright'] else ''}"
              f"{', disturbed ' + ','.join(outcome['disturbed']) if outcome['disturbed'] else ''})",
              flush=True)
    print(f"{successes}/{args.episodes} successes; median placement error "
          f"{np.median(errors) * 1000:.1f} mm")
    if moves or args.task != "move":
        right = sum(o["right_piece"] for o in outcomes)
        print(f"picked the named piece in {right}/{len(outcomes)} episodes")
        for uci in moves:
            group = [o for o in outcomes if o["move"] == uci]
            if group:
                print(f"  {uci}: {sum(o['success'] for o in group)}/{len(group)} success, "
                      f"{sum(o['right_piece'] for o in group)}/{len(group)} right piece")
    if writer is not None:
        writer.close()
        print("wrote", args.video)
    env.close()


if __name__ == "__main__":
    main()
