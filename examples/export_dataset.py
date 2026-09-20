"""Convert recorded episodes into a LeRobot v3.0 dataset for VLA training.

    python examples/export_dataset.py --in datasets/chess_moves datasets/chess_captures \
        --repo-id XvKuoMing/so101_chess --root datasets/lerobot/chess_mc --stride 3

Reads the episode directories written by `EpisodeRecorder` (data.npz, one mp4
per robot camera, meta.json), keeps the verified successes and writes the layout
MolmoAct2 and the other LeRobot policies consume: `observation.state` and
`action` (the 6 SO-101 joints), one image stream per robot camera, and one
language instruction per frame.

Runs in the VLA environment (Python 3.12 with `lerobot` installed).
"""
import argparse
import json
import os
import time

import imageio.v2 as imageio
import numpy as np

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from chess_sim import JOINTS, ROBOT_CAMERAS
from chess_sim.conventions import CONVENTIONS, SO101_DEGREES
from chess_sim.hub import DATASET_REPO

RETRIES = 3           # opening a video can fail transiently while the encoder is loaded
ROBOT_TYPE = "so101"


def read_meta(path: str) -> dict | None:
    """An episode's metadata, or None if it was never written completely (a
    recording interrupted mid-write leaves an empty meta.json)."""
    try:
        with open(os.path.join(path, "meta.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def episode_dirs(root: str) -> list[str]:
    """Episode directories under `root`, including worker shards, in order."""
    out = []
    for base, dirs, files in os.walk(root):
        if "meta.json" in files and "data.npz" in files:
            out.append(base)
        dirs.sort()
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="source", required=True, nargs="+",
                    help="recorded dataset directories; several are merged into one dataset")
    ap.add_argument("--repo-id", default=DATASET_REPO)
    ap.add_argument("--root", required=True, help="output directory for the LeRobot dataset")
    ap.add_argument("--convention", choices=list(CONVENTIONS), default=SO101_DEGREES.name,
                    help="joint units: the public SO-101 degrees-with-offset, or raw radians")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--keep-failures", action="store_true", help="also export unsuccessful attempts")
    # LeRobot's default (libsvtav1, staged PNGs) encodes at about a minute per
    # episode here; h264 with streaming encoding is ~20x faster and the frames
    # are re-encoded from h264 recordings anyway.
    ap.add_argument("--vcodec", default="h264",
                    choices=["h264", "h264_nvenc", "hevc", "libsvtav1", "auto"])
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--encoder-threads", type=int, default=4)
    # Keeping every frame makes the action a copy of the next observed state (the
    # servo tracks its target within a degree), so a policy scores well by echoing
    # its input and never has to ground the instruction. `--stride 3` keeps every
    # third frame: consecutive targets are then several degrees apart, an action
    # chunk spans the whole approach instead of one second, and the decisive
    # opening frames make up three times as much of the data.
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    args = ap.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")

    convention = CONVENTIONS[args.convention]
    sources = [d for root in args.source for d in episode_dirs(root)]
    if not sources:
        raise SystemExit(f"no recorded episodes under {args.source}")
    first = next((m for m in map(read_meta, sources) if m), None)
    if first is None:
        raise SystemExit(f"no readable episode metadata under {args.source}")
    recorded_fps = int(first["fps"])
    if recorded_fps % args.stride:
        raise SystemExit(f"--stride {args.stride} does not divide the recorded {recorded_fps} fps")
    fps = recorded_fps // args.stride
    cameras = first.get("cameras", [str(c) for c in ROBOT_CAMERAS])
    probe = imageio.get_reader(os.path.join(sources[0], f"{cameras[0]}.mp4"))
    height, width = probe.get_data(0).shape[:2]
    probe.close()

    joints = [str(j) for j in JOINTS]
    features = {"observation.state": {"dtype": "float32", "shape": (len(joints),), "names": joints},
                "action": {"dtype": "float32", "shape": (len(joints),), "names": joints}}
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=features, root=args.root, robot_type=ROBOT_TYPE,
        rgb_encoder=VideoEncoderConfig(vcodec=args.vcodec, crf=args.crf, g=2, pix_fmt="yuv420p"),
        streaming_encoding=True, encoder_threads=args.encoder_threads)
    exported = skipped = damaged = 0
    for path in sources:
        if args.max_episodes is not None and exported >= args.max_episodes:
            break
        meta = read_meta(path)
        if meta is None:
            damaged += 1
            continue
        if not meta.get("success") and not args.keep_failures:
            skipped += 1
            continue
        data = np.load(os.path.join(path, "data.npz"))
        states, actions = data["observation_state"], data["action"]
        # A failure to read is retried before the episode is written off: silently
        # dropping sound recordings would thin the data unnoticed.
        for attempt in range(RETRIES):
            readers = {}
            try:
                for cam in cameras:
                    readers[cam] = imageio.get_reader(os.path.join(path, f"{cam}.mp4"))
                frames = min(len(states), len(actions), *(r.count_frames() for r in readers.values()))
                for i in range(0, frames, args.stride):
                    frame = {"observation.state": convention.from_radians(states[i]).astype(np.float32),
                             "action": convention.from_radians(actions[i]).astype(np.float32),
                             "task": meta["instruction"]}
                    for cam, reader in readers.items():
                        frame[f"observation.images.{cam}"] = reader.get_data(i)
                    dataset.add_frame(frame)
                break
            except (OSError, ValueError, RuntimeError, IndexError) as err:
                dataset.clear_episode_buffer()
                if attempt + 1 < RETRIES:
                    print(f"  retrying {path}: {err}", flush=True)
                    time.sleep(2)
                else:
                    print(f"  skipping {path}: {err}", flush=True)
            finally:
                for reader in readers.values():
                    reader.close()
        else:
            damaged += 1
            continue
        dataset.save_episode()
        exported += 1
        if exported % 25 == 0:
            print(f"  {exported} episodes exported", flush=True)
    dataset.finalize()
    print(f"exported {exported} episodes ({skipped} unsuccessful, {damaged} unreadable "
          f"recordings skipped) to {args.root}")
    for root in args.source:
        print(f"  from {root}: {sum(1 for d in sources if d.startswith(root))} recordings")
    print(f"cameras {cameras} at {width}x{height}, {fps} fps "
          f"(recorded {recorded_fps}, stride {args.stride}), convention {convention.name}")


if __name__ == "__main__":
    main()
