"""Asset registry: sim-ready SO-101 and the 12 chess piece meshes.

Pieces are EmbodiedGen-generated OBJ meshes with a single texture each. The
registry resolves paths and caches mesh bounds, which drive the per-piece scale
(height target, capped so the footprint fits inside a square).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import chess
import numpy as np
import trimesh

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SO101_XML = os.path.join(ASSET_DIR, "so101", "so101.xml")
BACKDROP_OBJ = os.path.join(ASSET_DIR, "scene", "pano_cylinder.obj")
BACKDROP_IMAGE = os.path.join(ASSET_DIR, "scene", "pano_image.png")
BOARD_TEXTURE = os.path.join(ASSET_DIR, "scene", "board_texture.png")

# Staunton-proportioned heights at a 3.5 cm square; scaled with the board.
_REFERENCE_SQUARE = 0.035
_PIECE_HEIGHT = {
    chess.PAWN: 0.034, chess.ROOK: 0.038, chess.KNIGHT: 0.042,
    chess.BISHOP: 0.046, chess.QUEEN: 0.052, chess.KING: 0.058,
}
_TYPE_NAME = {
    chess.PAWN: "pawn", chess.ROOK: "rook", chess.KNIGHT: "knight",
    chess.BISHOP: "bishop", chess.QUEEN: "queen", chess.KING: "king",
}
# A full set: how many bodies of each (color, type) the scene always contains.
SET_COUNTS = {chess.PAWN: 8, chess.ROOK: 2, chess.KNIGHT: 2,
              chess.BISHOP: 2, chess.QUEEN: 1, chess.KING: 1}


def piece_asset_name(piece: chess.Piece) -> str:
    """'white_knight', 'black_pawn', ... — the asset directory name."""
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color}_{_TYPE_NAME[piece.piece_type]}"


def piece_obj_path(piece: chess.Piece) -> str:
    name = piece_asset_name(piece)
    return os.path.join(ASSET_DIR, "pieces", name, f"{name}.obj")


def piece_texture_path(piece: chess.Piece) -> str:
    return os.path.join(ASSET_DIR, "pieces", piece_asset_name(piece), "material_0.png")


# Collider profile: (bottom, top) height fractions of each cylinder segment and
# the height fraction at which its radius is sampled from the mesh.
PROFILE_SEGMENTS = ((0.00, 0.25, 0.10), (0.25, 0.45, 0.35), (0.45, 0.65, 0.55), (0.65, 1.00, 0.80))
WAIST_FRACTION = 0.55


@dataclass(frozen=True)
class PieceGeometry:
    """Scaled dimensions of a piece as it appears in the scene."""

    scale: float
    height: float
    width: float          # max horizontal extent
    bottom_offset: float  # mesh-frame z of the base (scaled), usually negative
    center: np.ndarray    # scaled mesh AABB center (mesh frame)
    profile: tuple        # ((z_bottom, z_top, radius), ...) in scaled mesh frame
    waist_radius: float   # mesh radius at the grasp height

    @property
    def waist(self) -> float:
        """Grasp height above the base: the center of the widest collider
        segment above the base. Straight prongs cannot pinch a narrow neck
        below a wider crown, so a king is held by its crown, a pawn by its head."""
        z0, z1, _ = self._grasp_segment
        return 0.5 * (z0 + z1) - self.bottom_offset

    @property
    def grasp_radius(self) -> float:
        """Piece radius at the grasp height (the jaws close to this)."""
        return self._grasp_segment[2]

    @property
    def _grasp_segment(self):
        return max(self.profile[1:], key=lambda seg: seg[2])

    @property
    def collider_radius(self) -> float:
        """Footprint radius (the base segment of the collider profile)."""
        return self.profile[0][2]

    @property
    def crown_radius(self) -> float:
        """Widest radius at or above the grasp height (what open prongs must
        clear while descending onto the piece and retreating from it)."""
        grasp_z = self.bottom_offset + self.waist
        return max(r for z0, z1, r in self.profile if z1 > grasp_z)


@lru_cache(maxsize=None)
def _mesh(path: str) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh")


def _bounds(path: str) -> np.ndarray:
    """Body-frame AABB of a piece mesh (the vendored OBJs are z-up)."""
    return _mesh(path).bounds.copy()


@lru_cache(maxsize=None)
def _radius_at(path: str, fraction: float) -> float:
    """Largest horizontal radius of the mesh cross-section at a height fraction."""
    mesh = _mesh(path)
    lo, hi = mesh.bounds
    z = lo[2] + fraction * (hi[2] - lo[2])
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return 0.5 * float(max(hi[0] - lo[0], hi[1] - lo[1]))
    pts = section.vertices[:, :2]
    axis = 0.5 * (lo[:2] + hi[:2])
    return float(np.linalg.norm(pts - axis, axis=1).max())


def piece_geometry(piece: chess.Piece, square: float) -> PieceGeometry:
    """Scale a piece to the board: target height, footprint capped to the square."""
    path = piece_obj_path(piece)
    lo, hi = _bounds(path)
    extent = hi - lo
    target_h = _PIECE_HEIGHT[piece.piece_type] * (square / _REFERENCE_SQUARE)
    scale = min(target_h / max(extent[2], 1e-6),
                0.88 * square / max(extent[0], extent[1], 1e-6))
    profile = tuple(
        (float(lo[2] + b * extent[2]) * scale, float(lo[2] + t * extent[2]) * scale,
         (_radius_at(path, f) + 0.0004) * scale)
        for b, t, f in PROFILE_SEGMENTS)
    return PieceGeometry(
        scale=scale,
        height=float(extent[2] * scale),
        width=float(max(extent[0], extent[1]) * scale),
        bottom_offset=float(lo[2] * scale),
        center=(lo + hi) * 0.5 * scale,
        profile=profile,
        waist_radius=_radius_at(path, WAIST_FRACTION) * scale,
    )


# -- procedural scene assets (generated once, then reused) --------------------

def ensure_scene_assets(board) -> None:
    """Create the board texture and backdrop mesh if they are not vendored yet."""
    if not os.path.exists(BOARD_TEXTURE):
        _write_board_texture(BOARD_TEXTURE, board)
    if not os.path.exists(BACKDROP_OBJ):
        _write_backdrop(BACKDROP_OBJ, BACKDROP_IMAGE)


def _write_board_texture(path: str, board, px_per_square: int = 128) -> None:
    from PIL import Image

    rng = np.random.default_rng(0)
    border_px = int(px_per_square * board.border / board.square)
    size = 8 * px_per_square + 2 * border_px
    img = np.zeros((size, size, 3), np.float32)

    def wood(shape, base, variation=14.0):
        grain = np.cumsum(rng.normal(0, 1, shape), axis=1)
        grain = (grain - grain.min()) / (np.ptp(grain) + 1e-6) - 0.5
        rows = rng.normal(0, 1, (shape[0], 1)) * 0.35
        return np.clip(base + (grain + rows) * variation, 0, 255)

    for c, v in enumerate((92, 58, 32)):
        img[..., c] = wood((size, size), v)
    for i in range(8):
        for j in range(8):
            rgb = (214, 178, 132) if (i + j) % 2 else (99, 64, 40)
            y0, x0 = border_px + i * px_per_square, border_px + j * px_per_square
            for c in range(3):
                img[y0:y0 + px_per_square, x0:x0 + px_per_square, c] = wood(
                    (px_per_square, px_per_square), rgb[c])
    Image.fromarray(img.astype(np.uint8)).save(path)


def _write_backdrop(obj_path: str, image_path: str,
                    radius: float = 3.5, height: float = 3.4, segments: int = 96) -> None:
    """Inward-facing cylinder UV-mapped to the room panorama."""
    from PIL import Image

    ang = np.linspace(0, 2 * np.pi, segments + 1)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * radius
    verts = np.vstack([np.column_stack([ring, np.zeros(segments + 1)]),
                       np.column_stack([ring, np.full(segments + 1, height)])])
    uv = [[1.0 - i / segments, 0.0] for i in range(segments + 1)] + \
         [[1.0 - i / segments, 1.0] for i in range(segments + 1)]
    faces = []
    for i in range(segments):
        a, b, c, d = i, i + 1, segments + 1 + i, segments + 2 + i
        faces += [[a, c, b], [b, c, d]]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(uv),
                                                image=Image.open(image_path).convert("RGB"))
    mesh.export(obj_path)
