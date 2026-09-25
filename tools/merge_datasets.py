"""Merge exported LeRobot datasets into one, without re-encoding anything.

    python tools/merge_datasets.py --out datasets/lerobot/chess_mc_v2 \
        datasets/lerobot/chess_mc datasets/lerobot/chess_mc_new

Re-exporting every recording to add a couple of thousand episodes costs hours of
video encoding; merging copies the finished files and rewrites the indices. The
sources must share frame rate, robot type and features - LeRobot checks.

The result is verified before it is marked complete: episode and frame counts
must add up, and the first and last frames must decode.
"""
import argparse
import json
import os
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

COMPLETE = ".export-complete"    # what training/train.sh waits for


def info(root: Path) -> dict:
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", type=Path, help="exported dataset roots, in order")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo-id", default="XvKuoMing/so101_chess")
    ap.add_argument("--repeat", nargs="*", default=[], metavar="ROOT=N",
                    help="include a source N times, so a small set of corrections weighs "
                         "more than its share of frames without touching the sampler")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; remove it first or choose another --out")

    repeats = {Path(spec.split("=")[0]): int(spec.split("=")[1]) for spec in args.repeat}
    unknown = set(repeats) - set(args.sources)
    if unknown:
        raise SystemExit(f"--repeat names sources that are not listed: {sorted(map(str, unknown))}")
    roots = [root for root in args.sources for _ in range(repeats.get(root, 1))]

    expected_episodes = sum(info(root)["total_episodes"] for root in roots)
    expected_frames = sum(info(root)["total_frames"] for root in roots)
    for root in args.sources:
        meta = info(root)
        times = f" x{repeats[root]}" if repeats.get(root, 1) > 1 else ""
        print(f"  {root}{times}: {meta['total_episodes']} episodes, {meta['total_frames']} frames")

    aggregate_datasets(repo_ids=[info(root).get("repo_id", args.repo_id) or args.repo_id
                                 for root in roots],
                       aggr_repo_id=args.repo_id, roots=roots, aggr_root=args.out)

    merged = info(args.out)
    if (merged["total_episodes"], merged["total_frames"]) != (expected_episodes, expected_frames):
        raise SystemExit(f"counts do not add up: got {merged['total_episodes']} episodes / "
                         f"{merged['total_frames']} frames, expected {expected_episodes} / "
                         f"{expected_frames}")
    dataset = LeRobotDataset(args.repo_id, root=args.out, video_backend="pyav")
    for index in (0, len(dataset) - 1):
        frame = dataset[index]
        assert frame["observation.state"].shape[-1] == 6, "state is not six joints"
    tasks = len(dataset.meta.tasks)
    (args.out / COMPLETE).touch()
    print(f"merged into {args.out}: {merged['total_episodes']} episodes, "
          f"{merged['total_frames']} frames, {tasks} distinct instructions; "
          f"first and last frames decode")


if __name__ == "__main__":
    main()
