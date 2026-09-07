"""Play a game from the initial position: live in the MuJoCo viewer, or to a video.

    python examples/play_game.py --viewer              # live, real time, interactive camera
    python examples/play_game.py                       # 20-ply Giuoco Pianissimo, side-view video
    python examples/play_game.py --cameras external top wrist   # all views, side by side
    python examples/play_game.py --moves e2e4 e7e5 --cameras top --out sim/out/e4e5.mp4
    python examples/play_game.py --fen "4k3/8/4K3/8/8/8/1Q6/8" \
        --moves b2b7 e8f8 e6f6 f8e8 b7e7 --out sim/out/ladder_mate.mp4

`--viewer` needs a display (WSLg on WSL2 is fine); it opens on the first
`--cameras` view and the mouse takes over from there. The window stays open
after the last move until closed.
Cameras: `external` (side view, easiest to follow), `top` (overhead mast
camera) and `wrist` (gripper camera); the last two are what the real robot
records. Any subset in any order is tiled left to right in the video.

The arm plays both sides. Moves come from a fixed line here; swap in a chess
engine (python-chess + Stockfish) to play a real game. The expert executes
quiet moves only: captures, castling, promotion and en passant are not
implemented yet. Measured on the default layout with the physical grasp: the
20-ply line below executes 20/20 at 0.3-4.0 mm placement.
"""
import argparse
import time

import chess
import imageio.v2 as imageio
import mujoco
import numpy as np

from chess_sim import ChessSimEnv, START_FEN

DEMO_CAMERAS = ("external",)   # the side view is the easiest to follow; datasets never use it

# Giuoco Pianissimo: 20 plies without a capture, castling or promotion
GIUOCO_PIANISSIMO = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6",
                     "d2d3", "d7d6", "b1d2", "a7a6", "c4b3", "c5a7", "h2h3", "h7h6",
                     "d2f1", "c8e6", "f1g3", "d8d7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", nargs="*", default=GIUOCO_PIANISSIMO, help="UCI moves")
    ap.add_argument("--fen", default=START_FEN, help="starting position (board field)")
    ap.add_argument("--cameras", nargs="+", default=list(DEMO_CAMERAS),
                    choices=["external", "top", "wrist"],
                    help="cameras to record, tiled left to right (default: external)")
    ap.add_argument("--out", default=None, help="video path (default sim/out/game.mp4 unless --viewer)")
    ap.add_argument("--viewer", action="store_true", help="play live in the MuJoCo viewer, in real time")
    args = ap.parse_args()
    if args.out is None and not args.viewer:
        args.out = "sim/out/game.mp4"

    env = ChessSimEnv(cameras=tuple(args.cameras), image_size=(960, 540))
    env.reset(args.fen)
    writer = imageio.get_writer(args.out, fps=30, macro_block_size=1) if args.out else None
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.cameras[0])
        viewer.sync()
    period = 1.0 / env.control_hz
    next_tick = time.perf_counter()

    def record(action):
        nonlocal next_tick
        if writer is not None:
            writer.append_data(np.concatenate([env.render(cam) for cam in args.cameras], axis=1))
        if viewer is not None:
            if not viewer.is_running():
                raise SystemExit("viewer closed")
            viewer.sync()
            next_tick += period                       # pace the physics to real time
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    played = 0
    for uci in args.moves:
        mv = chess.Move.from_uci(uci)
        san = env.board.san(mv)
        result = env.move(chess.square_name(mv.from_square), chess.square_name(mv.to_square),
                          on_step=record)
        played += result.success
        print(f"{san:6s} {'ok' if result.success else 'FAILED'} "
              f"({result.placement_error * 1000:.1f} mm{', ' + result.reason if result.reason else ''})")
        for _ in range(15):                       # a short pause between moves
            record(None)
    print(f"{played}/{len(args.moves)} moves executed; final position: {env.board.fen()}")
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
