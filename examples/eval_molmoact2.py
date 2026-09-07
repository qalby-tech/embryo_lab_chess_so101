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
import random

import chess
import numpy as np
import torch

from chess_sim import ChessSimEnv
from chess_sim.env import DISTURB_TOLERANCE, PLACEMENT_TOLERANCE, UPRIGHT_MIN
from export_lerobot import CAMERAS, JOINT_OFFSETS, JOINT_SIGNS, to_so101_degrees
from play_random_moves import describe, random_position


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


def run_episode(env, policy, rng, max_steps: int, device: str, on_step=None) -> dict:
    """One instructed move under policy control; returns the outcome."""
    while True:
        board = random_position(rng, rng.randint(2, 8))
        env.reset(board.board_fen())
        candidates = env.executable_moves()
        if candidates:
            break
    move = rng.choice(candidates)
    instruction = describe(env.board, move)
    slot = env.slot_at(chess.square_name(move.from_square))
    before = {sq: env.piece_position(sl)[:2].copy() for sq, sl in env._square_slot.items()}
    target = np.array(env.board_spec.square_center(move.to_square))

    policy.reset()
    for _ in range(max_steps):
        with torch.inference_mode():
            action = policy.select_action(observation(env, instruction, device))
        env.apply_action(to_radians(action[0].float().cpu().numpy()), on_step)

    position = env.piece_position(slot)
    error = float(np.linalg.norm(position[:2] - target))
    upright = env.piece_upright(slot) > UPRIGHT_MIN
    disturbed = [chess.square_name(sq) for sq, xy in before.items()
                 if sq != move.from_square
                 and np.linalg.norm(env.piece_position(env._square_slot[sq])[:2] - xy) > DISTURB_TOLERANCE]
    return {"move": move.uci(), "instruction": instruction, "placement_error": error,
            "upright": upright, "disturbed": disturbed,
            "success": bool(error < PLACEMENT_TOLERANCE and upright and not disturbed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="trained policy directory")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=300, help="control steps per episode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video", default=None, help="record the episodes to this mp4")
    args = ap.parse_args()

    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    policy = MolmoAct2Policy.from_pretrained(args.checkpoint).to(args.device).eval()
    env = ChessSimEnv(cameras=tuple(CAMERAS) + ("external",), image_size=(640, 480))
    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.video, fps=env.control_hz, macro_block_size=1)

    def record(_action):
        if writer is not None:
            writer.append_data(np.concatenate([env.render("external"), env.render("top")], axis=1))

    rng = random.Random(args.seed)
    successes, errors = 0, []
    for i in range(args.episodes):
        outcome = run_episode(env, policy, rng, args.max_steps, args.device, record)
        successes += outcome["success"]
        errors.append(outcome["placement_error"])
        print(f"[{i}] {outcome['instruction']}: {'OK' if outcome['success'] else 'FAIL'} "
              f"({outcome['placement_error'] * 1000:.1f} mm"
              f"{', fell' if not outcome['upright'] else ''}"
              f"{', disturbed ' + ','.join(outcome['disturbed']) if outcome['disturbed'] else ''})",
              flush=True)
    print(f"{successes}/{args.episodes} successes; median placement error "
          f"{np.median(errors) * 1000:.1f} mm")
    if writer is not None:
        writer.close()
        print("wrote", args.video)
    env.close()


if __name__ == "__main__":
    main()
