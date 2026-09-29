"""Coin-style chess sets: flat discs and short cylinders with the piece's symbol on top.

    python tools/make_token_sets.py

Travel and magnetic sets often use tokens instead of turned pieces - a disc with a
printed or embossed symbol. Every token of a set has the same size, so the sets are
written at their true size in metres (`set.json`: `"scale": "absolute"`) and bring
their own black pieces: a dark token with a light symbol, not a tinted light one.
Written to chess_sim/assets/piece_sets/<name>/, then checked by the scripted expert
like any other set (tools/check_piece_sets.py): a token too thin for the jaws to
close on is dropped there, not here.
"""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from chess_sim.assets import PIECE_SETS_DIR

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
GLYPHS = {"king": "♚", "queen": "♛", "rook": "♜",
          "bishop": "♝", "knight": "♞", "pawn": "♟"}
# name: (diameter m, height m, light face RGB, dark face RGB, edge bevel fraction)
TOKEN_SETS = {
    "token_coin":     (0.022, 0.006, (236, 228, 208), (48, 40, 36), 0.10),
    "token_checker":  (0.024, 0.009, (226, 204, 160), (92, 30, 24), 0.18),
    "token_magnet":   (0.020, 0.007, (242, 242, 238), (30, 30, 34), 0.08),
    "token_tall":     (0.018, 0.016, (214, 190, 150), (60, 44, 32), 0.12),
    "token_disc_big": (0.025, 0.011, (250, 236, 200), (20, 60, 44), 0.15),
}
FACE_PX = 512


def face_texture(glyph: str, face: tuple[int, int, int], ink: tuple[int, int, int]) -> Image.Image:
    """Top half: the face with the symbol; bottom half: the plain body colour."""
    image = Image.new("RGB", (FACE_PX, 2 * FACE_PX), face)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, int(FACE_PX * 0.62))
    draw.ellipse([FACE_PX * 0.06, FACE_PX * 0.06, FACE_PX * 0.94, FACE_PX * 0.94], outline=ink,
                 width=int(FACE_PX * 0.025))
    draw.text((FACE_PX / 2, FACE_PX / 2), glyph, fill=ink, font=font, anchor="mm")
    grain = np.random.default_rng(0).normal(0, 4, (2 * FACE_PX, FACE_PX, 1))
    return Image.fromarray(np.clip(np.asarray(image, float) + grain, 0, 255).astype(np.uint8))


def token_mesh(diameter: float, height: float, bevel: float, segments: int = 64) -> trimesh.Trimesh:
    """A bevelled cylinder, base at z = 0, UVs: top cap onto the face, the rest onto the body."""
    r, b = diameter / 2, min(bevel * diameter / 2, height / 3)
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    # rings from the bottom up: base edge, side, bevel start, top edge
    levels = [(r - b, 0.0), (r, b), (r, height - b), (r - b, height)]
    vertices, uv = [], []
    for radius, z in levels:
        for c, s in ring:
            vertices.append((radius * c, radius * s, z)); uv.append((0.5, 0.25))
    bottom_centre, top_centre = len(vertices), len(vertices) + 1
    vertices += [(0, 0, 0), (0, 0, height)]; uv += [(0.5, 0.25), (0.5, 0.75)]
    # the top cap gets its own ring so it can carry face UVs
    cap = len(vertices)
    for c, s in ring:
        vertices.append(((r - b) * c, (r - b) * s, height))
        uv.append((0.5 + 0.47 * c, 0.75 + 0.235 * s))
    faces = []
    n = segments
    for level in range(len(levels) - 1):
        for i in range(n):
            a, a2 = level * n + i, level * n + (i + 1) % n
            faces += [(a, a2, a2 + n), (a, a2 + n, a + n)]
    for i in range(n):
        faces.append((bottom_centre, (i + 1) % n, i))                         # base
        faces.append((top_centre, cap + i, cap + (i + 1) % n))                # face
    mesh = trimesh.Trimesh(np.array(vertices), np.array(faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(uv))
    return mesh


def main():
    for name, (diameter, height, light, dark, bevel) in TOKEN_SETS.items():
        root = os.path.join(PIECE_SETS_DIR, name)
        for piece, glyph in GLYPHS.items():
            for colour, face, ink in (("white", light, dark), ("black", dark, light)):
                folder = os.path.join(root, f"{colour}_{piece}")
                os.makedirs(folder, exist_ok=True)
                mesh = token_mesh(diameter, height, bevel)
                mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv,
                                                            image=face_texture(glyph, face, ink))
                mesh.export(os.path.join(folder, f"{colour}_{piece}.obj"))
        with open(os.path.join(root, "set.json"), "w") as f:
            json.dump({"scale": "absolute", "kind": "token",
                       "diameter_mm": round(diameter * 1000, 1), "height_mm": round(height * 1000, 1)}, f)
        print(f"{name}: {diameter * 1000:.0f} mm x {height * 1000:.0f} mm")


if __name__ == "__main__":
    main()
