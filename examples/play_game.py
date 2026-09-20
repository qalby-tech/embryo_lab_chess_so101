"""Play a game from the initial position: live in the MuJoCo viewer, or to a video.

    python examples/play_game.py --viewer              # live, real time, interactive camera
    python examples/play_game.py                       # 20-ply Giuoco Pianissimo, side-view video
    python examples/play_game.py --cameras external top wrist   # all views, side by side
    python examples/play_game.py --fen "4k3/8/2n2q2/3p4/2B1P3/8/5R2/4K3" --moves e4d5 f2f6

The arm plays both sides. Moves come from a fixed line here; swap in a chess
engine (python-chess + Stockfish) to play a real game. A move onto an occupied
square is executed as a real capture is: the piece standing there is lifted into
the discard tray first, and only then does the capturing piece move in. Castling,
promotion and en passant are not implemented.

`--viewer` needs a display (WSLg on WSL2 is fine); it opens on the first
`--cameras` view and the mouse takes over from there.
"""
import argparse
import time

import chess
import imageio.v2 as imageio
import mujoco
import numpy as np

from chess_sim import (START_FEN, Camera, CaptureTask, ChessSimEnv, ControlConfig, EnvConfig,
                       MoveTask)

DEMO_CAMERAS = (Camera.EXTERNAL,)     # the side view is easiest to follow; datasets never use it
IMAGE_SIZE = (960, 540)
PAUSE_TICKS = 15                      # control steps of stillness between moves

# Giuoco Pianissimo: 20 plies without a capture, castling or promotion
GIUOCO_PIANISSIMO = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6",
                     "d2d3", "d7d6", "b1d2", "a7a6", "c4b3", "c5a7", "h2h3", "h7h6",
                     "d2f1", "c8e6", "f1g3", "d8d7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", nargs="*", default=GIUOCO_PIANISSIMO, help="UCI moves")
    ap.add_argument("--fen", default=START_FEN, help="starting position (board field)")
    ap.add_argument("--cameras", nargs="+", type=Camera, choices=list(Camera),
                    default=list(DEMO_CAMERAS), help="cameras to record, tiled left to right")
    ap.add_argument("--out", default=None, help="video path (default sim/out/game.mp4 unless --viewer)")
    ap.add_argument("--viewer", action="store_true", help="play live in the MuJoCo viewer, in real time")
    args = ap.parse_args()
    if args.out is None and not args.viewer:
        args.out = "sim/out/game.mp4"

    env = ChessSimEnv(EnvConfig(control=ControlConfig(cameras=tuple(args.cameras),
                                                      image_size=IMAGE_SIZE)))
    env.reset(args.fen)
    writer = imageio.get_writer(args.out, fps=env.control_hz, macro_block_size=1) if args.out else None
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False,
                                              show_right_ui=False)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                                  str(args.cameras[0]))
        viewer.sync()
    period, next_tick = 1.0 / env.control_hz, time.perf_counter()

    def record(_action=None):
        nonlocal next_tick
        if writer is not None:
            writer.append_data(np.concatenate([env.render(cam) for cam in args.cameras], axis=1))
        if viewer is not None:
            if not viewer.is_running():
                raise SystemExit("viewer closed")
            viewer.sync()
            next_tick += period                   # pace the physics to real time
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    played = 0
    for uci in args.moves:
        move = chess.Move.from_uci(uci)
        san = env.position.san(move)
        task = MoveTask.from_move(move)
        if env.slot_at(task.to_square) is not None:
            # a capture: clear the square into the tray, then move in
            taken = env.execute(CaptureTask(square=task.to_square), on_step=record)
            print(f"{'take ' + task.to_square:6s} {'ok' if taken.success else 'FAILED'} "
                  f"({taken.placement_error * 1000:.1f} mm"
                  f"{', ' + taken.reason if taken.reason else ''})")
            for _ in range(PAUSE_TICKS):
                record()
        result = env.execute(task, on_step=record)
        played += result.success
        print(f"{san:6s} {'ok' if result.success else 'FAILED'} "
              f"({result.placement_error * 1000:.1f} mm"
              f"{', ' + result.reason if result.reason else ''})")
        for _ in range(PAUSE_TICKS):
            record()

    print(f"{played}/{len(args.moves)} moves executed; final position: {env.position.fen()}")
    if writer is not None:
        writer.close()
        print("wrote", args.out)
    if viewer is not None:                        # leave the final position on screen
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        viewer.close()
    env.close()


if __name__ == "__main__":
    main()
