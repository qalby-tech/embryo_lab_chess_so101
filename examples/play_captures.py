"""Demonstrate taking pieces off the board.

    python examples/play_captures.py --viewer            # live, real time
    python examples/play_captures.py                     # writes sim/out/captures.mp4

A capture is two pick-and-places: the piece standing on the target square is
lifted into the discard tray on the arm's left, and only then does the capturing
piece move onto the square it just vacated. The same scripted expert and the
same mink IK solve both halves - a capture is an ordinary pick-and-place whose
destination is off the board.

`--viewer` needs a display (WSLg on WSL2 is fine).
"""
import argparse
import time

import chess
import mujoco
import numpy as np

from chess_sim import ChessSimEnv
from chess_sim.board import parse_square

DEMO_CAMERAS = ("external",)

# black pieces sitting where white wants to go: each capture clears one, then
# the white piece takes the square.
POSITION = "4k3/8/2n2q2/3p4/2B1P3/8/5R2/4K3"
CAPTURES = [("d5", "e4"),      # the pawn on d5 goes, the e4 pawn takes it
            ("f6", "f2")]      # the queen on f6 goes, the f2 rook takes it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default=POSITION, help="starting position (board field)")
    ap.add_argument("--cameras", nargs="+", default=list(DEMO_CAMERAS),
                    choices=["external", "top", "wrist"])
    ap.add_argument("--out", default=None, help="video path (default sim/out/captures.mp4)")
    ap.add_argument("--viewer", action="store_true", help="play live in the MuJoCo viewer")
    args = ap.parse_args()
    if args.out is None and not args.viewer:
        args.out = "sim/out/captures.mp4"

    import os
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    env = ChessSimEnv(cameras=tuple(args.cameras))
    env.reset(args.fen)

    writer = None
    if args.out:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.out, fps=env.control_hz, macro_block_size=1)
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data,
                                              show_left_ui=False, show_right_ui=False)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                                  args.cameras[0])
        viewer.sync()
    period = 1.0 / env.control_hz
    next_tick = time.perf_counter()

    def record(_action=None):
        nonlocal next_tick
        if writer is not None:
            writer.append_data(np.concatenate([env.render(c) for c in args.cameras], axis=1))
        if viewer is not None:
            if not viewer.is_running():
                raise SystemExit("viewer closed")
            viewer.sync()
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    def pause(ticks=15):
        for _ in range(ticks):
            record()

    done = 0
    for target, attacker in CAPTURES:
        taken = env.capture(target, on_step=record)
        print(f"take {target} off the board: {'ok' if taken.success else 'FAILED'} "
              f"({taken.placement_error * 1000:.1f} mm"
              f"{', ' + taken.reason if taken.reason else ''})")
        pause()
        moved = env.move(attacker, target, on_step=record)
        print(f"{attacker}->{target}:                  {'ok' if moved.success else 'FAILED'} "
              f"({moved.placement_error * 1000:.1f} mm"
              f"{', ' + moved.reason if moved.reason else ''})")
        pause()
        done += taken.success and moved.success

    print(f"{done}/{len(CAPTURES)} sequences completed; final position: {env.board.board_fen()}")
    if writer is not None:
        writer.close()
        print("wrote", args.out)
    if viewer is not None:
        while viewer.is_running():
            viewer.sync()
    env.close()


if __name__ == "__main__":
    main()
