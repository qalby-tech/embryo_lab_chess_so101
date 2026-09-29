"""Bring EmbodiedGen-generated chess sets into the scene's asset tree.

    python tools/import_piece_sets.py                         # every finished set
    python tools/import_piece_sets.py --sets staunton_maple marble

Reads ~/EmbodiedGen/outputs/piece_sets/<set>/asset3d/<piece>/result/mesh/<piece>.obj
(y-up, metres, one texture) and writes chess_sim/assets/piece_sets/<set>/<piece>/
<piece>.obj + material_0.png, z-up like the vendored set, base at z = 0 and axis on
the origin. Scale does not matter - the scene rescales every piece to the board -
but proportions do, so each set is checked against what the gripper can hold and a
set that fails is skipped with the reason.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import trimesh

from chess_sim.assets import ASSET_DIR, PIECE_SETS_DIR

SOURCE = os.path.expanduser("~/EmbodiedGen/outputs/piece_sets")
PIECES = ("pawn", "rook", "knight", "bishop", "queen", "king")
Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
# a piece far wider than tall is a failed generation (a board, a plate, a figurine
# lying down); Staunton pieces run 0.35-0.6 wide per unit height
MAX_WIDTH_PER_HEIGHT = 0.8
# the image model turns a rook into a slim king unless told otherwise; a Staunton
# rook stands about 1.4 times its width (the vendored one 1.41); the king-shaped rooks
# it draws measure 1.8, generated kings 2.0-2.3
MAX_HEIGHT_PER_WIDTH = {"rook": 1.65}


def import_piece(src: str, dst_dir: str) -> str | None:
    """Convert one piece; returns a reason when it is unusable."""
    mesh = trimesh.load(src, force="mesh")
    lo, hi = mesh.bounds
    if int(np.argmax(hi - lo)) != 1:
        return f"tallest axis is {'xyz'[int(np.argmax(hi - lo))]}, not y"
    mesh.apply_transform(Y_UP_TO_Z_UP)
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    extent = mesh.bounds[1] - mesh.bounds[0]
    if max(extent[0], extent[1]) > MAX_WIDTH_PER_HEIGHT * extent[2]:
        return f"{max(extent[:2]) / extent[2]:.2f} wide per unit height"
    os.makedirs(dst_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(src))[0]
    mesh.export(os.path.join(dst_dir, f"{name}.obj"))
    # trimesh writes the texture as material_0.png next to the OBJ; keep the
    # generator's own file in case the exporter re-encoded it
    texture = os.path.join(os.path.dirname(src), "material_0.png")
    if os.path.isfile(texture):
        shutil.copyfile(texture, os.path.join(dst_dir, "material_0.png"))
    return None


def problem(src: str) -> str | None:
    """Why a generated piece cannot be used, without importing it."""
    mesh = trimesh.load(src, force="mesh")
    extent = mesh.bounds[1] - mesh.bounds[0]
    if int(np.argmax(extent)) != 1:
        return f"tallest axis is {'xyz'[int(np.argmax(extent))]}, not y"
    width = max(extent[0], extent[2])
    if width > MAX_WIDTH_PER_HEIGHT * extent[1]:
        return f"{width / extent[1]:.2f} wide per unit height"
    piece = os.path.splitext(os.path.basename(src))[0]
    if piece in MAX_HEIGHT_PER_WIDTH and extent[1] > MAX_HEIGHT_PER_WIDTH[piece] * width:
        return f"{extent[1] / width:.2f} tall per unit width, too slim for a {piece}"
    return None


def audit(source: str) -> None:
    """Move every malformed piece's output to <set>/rejected/ so a retry regenerates it."""
    for name in sorted(os.listdir(source)):
        for piece in PIECES:
            src = os.path.join(source, name, "asset3d", piece, "result", "mesh", f"{piece}.obj")
            if not os.path.isfile(src):
                continue
            why = problem(src)
            if why:
                target = os.path.join(source, name, "rejected", f"{piece}_{len(os.listdir(os.path.join(source, name)))}")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.move(os.path.join(source, name, "asset3d", piece), target)
                image = os.path.join(source, name, "images", f"{piece}.png")
                if os.path.isfile(image):
                    shutil.move(image, target + ".png")
                print(f"{name}/{piece}: rejected - {why}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=None, help="default: every finished set")
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--max-fallbacks", type=int, default=1,
                    help="pieces a set may borrow from the vendored set before it is left out")
    ap.add_argument("--audit-only", action="store_true",
                    help="move malformed generated pieces aside (so a retry regenerates them) and stop")
    args = ap.parse_args()
    if args.audit_only:
        audit(args.source)
        return
    sets = args.sets or sorted(os.listdir(args.source))
    imported = []
    for name in sets:
        mesh_dir = os.path.join(args.source, name, "asset3d")
        sources = {p: os.path.join(mesh_dir, p, "result", "mesh", f"{p}.obj") for p in PIECES}
        missing = [p for p, f in sources.items() if not os.path.isfile(f) or problem(f)]
        if len(missing) > args.max_fallbacks:
            print(f"{name}: not finished (missing {', '.join(missing)})")
            continue
        target = os.path.join(PIECE_SETS_DIR, name)
        shutil.rmtree(target, ignore_errors=True)
        problems = {p: why for p, f in sources.items()
                    if p not in missing and (why := import_piece(f, os.path.join(target, p)))}
        for p in missing:
            # the image model rarely draws a rook; after the retries a set borrows the
            # vendored shape for what it could not generate, so no set is left unusable
            vendored = os.path.join(ASSET_DIR, "pieces", f"white_{p}")
            os.makedirs(os.path.join(target, p), exist_ok=True)
            shutil.copyfile(os.path.join(vendored, f"white_{p}.obj"), os.path.join(target, p, f"{p}.obj"))
            for extra in ("material_0.png", "material.mtl"):
                if os.path.isfile(os.path.join(vendored, extra)):
                    shutil.copyfile(os.path.join(vendored, extra), os.path.join(target, p, extra))
        if missing:
            with open(os.path.join(target, "fallbacks.json"), "w") as f:
                json.dump(missing, f)
        if problems:
            shutil.rmtree(target, ignore_errors=True)
            print(f"{name}: skipped - " + "; ".join(f"{p}: {why}" for p, why in problems.items()))
            continue
        imported.append(name)
        print(f"{name}: imported" + (f" (vendored {', '.join(missing)})" if missing else ""))
    print(f"\n{len(imported)} sets in {PIECE_SETS_DIR}")


if __name__ == "__main__":
    main()
