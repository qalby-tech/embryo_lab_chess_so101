"""Watch the scripted expert work, live or to a video.

    python examples/play_expert.py --episodes 5                     # random moves
    python examples/play_expert.py --task capture --episodes 3      # pieces taken off the board
    python examples/play_expert.py --episodes 3 --out sim/out/expert.mp4 --cameras external top
    python examples/play_expert.py --viewer                         # live, real time

The expert plans against simulator state, so this is the bar a learned policy is
measured against, and the source of every demonstration in the dataset.
`--viewer` needs a display (WSLg on WSL2 is fine).
"""
import argparse
import random
import time

import numpy as np

from chess_sim import (AppearanceConfig, Camera, CaptureSampler, ChessSimEnv, ControlConfig,
                       EnvConfig, MoveSampler, TaskFamily, demonstrate)

PAUSE_TICKS = 15      # control steps of stillness between episodes, so the video reads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--task", type=TaskFamily, choices=list(TaskFamily), default=TaskFamily.MOVE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true",
                    help="a different look, board position and arm start pose per episode")
    ap.add_argument("--cameras", nargs="+", type=Camera, choices=list(Camera),
                    default=[Camera.EXTERNAL], help="views to record, tiled left to right")
    ap.add_argument("--out", default=None, help="video path")
    ap.add_argument("--viewer", action="store_true", help="play live in the MuJoCo viewer")
    args = ap.parse_args()

    rng, looks = random.Random(args.seed), np.random.default_rng(args.seed)
    env = ChessSimEnv(EnvConfig(
        appearance=AppearanceConfig.sample(looks) if args.randomize else AppearanceConfig(),
        control=ControlConfig(cameras=tuple(args.cameras))))

    writer = None
    if args.out:
        import os
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        writer = imageio.get_writer(args.out, fps=env.control_hz, macro_block_size=1)
    viewer = None
    if args.viewer:
        import mujoco
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
            next_tick += period               # pace the physics to real time
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    def episode_done(result):
        print(f"{result.instruction}: {'OK' if result.success else 'FAILED'} "
              f"({result.placement_error * 1000:.1f} mm"
              f"{', ' + result.reason if result.reason else ''})", flush=True)
        for _ in range(PAUSE_TICKS):
            record()

    def new_look(index):
        if args.randomize and index > 0:
            env.recolor(AppearanceConfig.sample(looks))

    sampler = (CaptureSampler(randomize_layout=args.randomize) if args.task is TaskFamily.CAPTURE
               else MoveSampler(randomize_layout=args.randomize))
    # the recorder writes datasets; here the callback just draws the frames
    results = demonstrate(env, sampler, args.episodes, rng, on_step=record,
                          on_episode=episode_done, on_episode_start=new_look)
    print(f"{sum(r.success for r in results)}/{len(results)} verified successes")
    if writer is not None:
        writer.close()
        print("wrote", args.out)
    if viewer is not None:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        viewer.close()
    env.close()


if __name__ == "__main__":
    main()
