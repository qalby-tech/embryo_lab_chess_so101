"""Write the scene behind every exported episode as a sidecar, `scenes.jsonl`.

    python tools/scene_index.py --in datasets/chess_wide/captures datasets/chess_wide/moves \
        --dataset datasets/lerobot/chess_wide \
        --hold-out-piece-sets marble token_disc_big --hold-out-board-finishes cardboard

A LeRobot export keeps frames, joints and the instruction; what the recording knew about
its scene - the piece set, the board's size and material, where the camera stood, the
light - stays behind in the raw episodes' meta.json. This replays the exporter's own
selection (same sources, same order, same filters), so line N describes episode N, and
checks that against the export: the episode count and every instruction must agree.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import pandas as pd


def read_meta(path: str) -> dict | None:
    try:
        with open(os.path.join(path, "meta.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def episode_dirs(root: str) -> list[str]:
    out = []
    for base, dirs, files in os.walk(root):
        if "meta.json" in files and "data.npz" in files:
            out.append(base)
        dirs.sort()
    return sorted(out)


def camera(look: dict) -> dict:
    if look.get("camera_height") is None:
        return {"placement": "mast"}
    offset = look.get("camera_offset")
    if offset is None:
        placement = "mast line"
    else:
        placement = "side" if math.hypot(*offset) > 0.27 else "boom"
    return {"placement": placement, "height_m": round(look["camera_height"], 3),
            "offset_m": None if offset is None else [round(v, 3) for v in offset],
            "fovy_deg": round(look["camera_fovy"], 1), "roll_deg": round(look.get("camera_roll", 0.0), 1)}


def scene(meta: dict) -> dict:
    look, board = meta.get("appearance") or {}, meta.get("board") or {}
    row = {"task": meta["task"], "move": meta["move"], "instruction": meta["instruction"],
           "recovery": meta.get("recovery")}
    if look:
        row |= {"piece_set": look.get("piece_set", "default"), "piece_scale": round(look["piece_scale"], 3),
                "board_finish": look.get("board_finish", "wood"), "board_labels": look.get("board_labels", False),
                "table_surface": look.get("table_surface", "plain"), "camera": camera(look),
                "light": {"intensity": round(look["light_intensity"], 2),
                          "color": [round(c, 2) for c in look.get("light_color", (1, 1, 1))],
                          "fill": round(look.get("fill_intensity", 1.0), 2),
                          "ambient": round(look.get("ambient", 0.0), 2),
                          "shadow_softness_m": round(look.get("shadow_softness", 0.02), 2)}}
    if board:
        row["board"] = {"square_mm": round(board["square"] * 1000, 1), "border_mm": round(board["border"] * 1000, 1),
                        "thickness_mm": round(board["thickness"] * 1000, 1),
                        "yaw_deg": round(math.degrees(board.get("yaw", 0.0)), 1),
                        "arm_gap_mm": round(board["arm_gap"] * 1000), "arm_riser_mm": round(board["arm_riser"] * 1000)}
    layout = meta.get("layout") or {}
    if layout.get("board_origin") is not None:
        row["board_origin_mm"] = [round(v * 1000, 1) for v in layout["board_origin"]]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="source", required=True, nargs="+")
    ap.add_argument("--dataset", required=True, help="the exported LeRobot dataset the sidecar belongs to")
    ap.add_argument("--hold-out-piece-sets", nargs="*", default=[])
    ap.add_argument("--hold-out-board-finishes", nargs="*", default=[])
    args = ap.parse_args()

    rows = []
    for path in (d for root in args.source for d in episode_dirs(root)):
        meta = read_meta(path)
        if meta is None or not meta.get("success"):
            continue
        look = meta.get("appearance") or {}
        if (look.get("piece_set", "default") in args.hold_out_piece_sets
                or look.get("board_finish", "wood") in args.hold_out_board_finishes):
            continue
        rows.append({"episode_index": len(rows), **scene(meta)})

    with open(os.path.join(args.dataset, "meta", "info.json")) as f:
        total = json.load(f)["total_episodes"]
    if len(rows) != total:
        raise SystemExit(f"selection gives {len(rows)} episodes, the export has {total}: sources changed since")
    episodes = pd.concat(pd.read_parquet(os.path.join(base, f), columns=["episode_index", "tasks"])
                         for base, _, files in os.walk(os.path.join(args.dataset, "meta", "episodes"))
                         for f in sorted(files) if f.endswith(".parquet")).sort_values("episode_index")
    for row, tasks in zip(rows, episodes["tasks"]):
        if row["instruction"] not in list(tasks):
            raise SystemExit(f"episode {row['episode_index']}: recorded '{row['instruction']}', exported {list(tasks)}")
    out = os.path.join(args.dataset, "scenes.jsonl")
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} scenes -> {out}; every instruction matches the export")


if __name__ == "__main__":
    main()
