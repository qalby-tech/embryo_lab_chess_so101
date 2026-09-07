"""Convert recorded episodes into a LeRobot v3.0 dataset for VLA training.

    python examples/export_lerobot.py --in datasets/chess_vla \
        --repo-id local/chess_so101 --root datasets/lerobot/chess_so101

Reads the episode directories written by `EpisodeRecorder` (data.npz, one mp4
per robot camera, meta.json), keeps the verified successes and writes the
layout MolmoAct2 and the other LeRobot policies consume: `observation.state`
and `action` (6-D, the SO-101 joints), `observation.images.top` and
`observation.images.wrist`, one language instruction per frame.

Runs in the VLA environment (Python 3.12 with `lerobot` installed), NOT in the
simulator environment.

Joint convention: the simulator records MuJoCo joint targets in radians, whose
zero is mid-range like LeRobot's post-0.5.0 convention. `--convention so101`
(the default) maps them to the degrees-with-offset convention the public
SO-100/101 datasets use, so the values line up with a checkpoint pretrained on
them (shoulder_lift and elbow_flex are mirrored about 90 degrees). Use
`--convention radians` to export the raw simulator values instead.
"""
import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERAS = ["top", "wrist"]
# radians -> SO-100/101 dataset degrees: value * sign * 180/pi + offset
JOINT_SIGNS = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
JOINT_OFFSETS = np.array([0.0, 90.0, 90.0, 0.0, 0.0, 0.0])


def to_so101_degrees(values: np.ndarray) -> np.ndarray:
    return np.degrees(values) * JOINT_SIGNS + JOINT_OFFSETS


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
    ap.add_argument("--in", dest="source", required=True, help="recorded dataset directory")
    ap.add_argument("--repo-id", default="local/chess_so101")
    ap.add_argument("--root", required=True, help="output directory for the LeRobot dataset")
    ap.add_argument("--convention", choices=["so101", "radians"], default="so101")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--keep-failures", action="store_true", help="also export unsuccessful moves")
    # LeRobot's default (libsvtav1, staged PNGs) encodes at about a minute per
    # episode here; h264 with streaming encoding is ~20x faster and the frames
    # are re-encoded from h264 recordings anyway.
    ap.add_argument("--vcodec", default="h264",
                    choices=["h264", "h264_nvenc", "hevc", "libsvtav1", "auto"])
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--encoder-threads", type=int, default=4)
    args = ap.parse_args()

    convert = to_so101_degrees if args.convention == "so101" else (lambda v: v)
    sources = episode_dirs(args.source)
    if not sources:
        raise SystemExit(f"no recorded episodes under {args.source}")

    first = json.load(open(os.path.join(sources[0], "meta.json")))
    fps = int(first.get("fps", 30))
    probe = imageio.get_reader(os.path.join(sources[0], f"{CAMERAS[0]}.mp4"))
    height, width = probe.get_data(0).shape[:2]
    probe.close()

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(JOINTS),), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (len(JOINTS),), "names": JOINTS},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=features, root=args.root, robot_type="so101",
        rgb_encoder=VideoEncoderConfig(vcodec=args.vcodec, crf=args.crf, g=2, pix_fmt="yuv420p"),
        streaming_encoding=True, encoder_threads=args.encoder_threads)
    exported = skipped = 0
    for path in sources:
        if args.max_episodes is not None and exported >= args.max_episodes:
            break
        meta = json.load(open(os.path.join(path, "meta.json")))
        if not meta.get("success") and not args.keep_failures:
            skipped += 1
            continue
        data = np.load(os.path.join(path, "data.npz"))
        states, actions = data["observation_state"], data["action"]
        readers = {cam: imageio.get_reader(os.path.join(path, f"{cam}.mp4")) for cam in CAMERAS}
        frames = min(len(states), len(actions), *(r.count_frames() for r in readers.values()))
        for i in range(frames):
            frame = {"observation.state": convert(states[i]).astype(np.float32),
                     "action": convert(actions[i]).astype(np.float32),
                     "task": meta["instruction"]}
            for cam, reader in readers.items():
                frame[f"observation.images.{cam}"] = reader.get_data(i)
            dataset.add_frame(frame)
        for reader in readers.values():
            reader.close()
        dataset.save_episode()
        exported += 1
        if exported % 25 == 0:
            print(f"  {exported} episodes exported", flush=True)
    dataset.finalize()
    print(f"exported {exported} episodes ({skipped} unsuccessful skipped) to {args.root}")
    print(f"cameras {CAMERAS} at {width}x{height}, {fps} fps, convention {args.convention}")


if __name__ == "__main__":
    main()
