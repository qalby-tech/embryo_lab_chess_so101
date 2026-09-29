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


# Piece sets. "default" is the vendored set, one textured mesh per colour and type.
# Every other set lives in piece_sets/<name>/<type>/<type>.obj: one light-coloured
# mesh per type, imported from EmbodiedGen by tools/import_piece_sets.py, and the
# black side is the same mesh under a dark tint.
DEFAULT_SET = "default"
PIECE_SETS_DIR = os.path.join(ASSET_DIR, "piece_sets")
DARK_SIDE = 0.22          # how far a single-colour set's black pieces are darkened
VERIFIED_FILE = "verified.json"   # tools/check_piece_sets.py: which sets the expert handles


def piece_asset_name(piece: chess.Piece) -> str:
    """'white_knight', 'black_pawn', ... — the body and material name in the scene."""
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color}_{_TYPE_NAME[piece.piece_type]}"


def available_piece_sets(verified_only: bool = False) -> list[str]:
    """Every set the scene can build: the vendored one first, then the imported ones.
    `verified_only` keeps the imported sets the scripted expert passed on."""
    def complete(d: str) -> bool:        # a set is found by its king, one- or two-coloured
        return (os.path.isfile(os.path.join(PIECE_SETS_DIR, d, "king", "king.obj"))
                or os.path.isfile(os.path.join(PIECE_SETS_DIR, d, "white_king", "white_king.obj")))
    imported = sorted(d for d in os.listdir(PIECE_SETS_DIR) if complete(d)) \
        if os.path.isdir(PIECE_SETS_DIR) else []
    if verified_only:
        path = os.path.join(PIECE_SETS_DIR, VERIFIED_FILE)
        verdicts = {}
        if os.path.isfile(path):
            import json
            with open(path) as f:
                verdicts = json.load(f)
        imported = [name for name in imported if verdicts.get(name, {}).get("passed")]
    return [DEFAULT_SET, *imported]


def single_colour(piece_set: str) -> bool:
    """One light mesh per type, the black side tinted - unless the set brings its
    own black pieces (black_<type>/), as the vendored set and the token sets do."""
    if piece_set == DEFAULT_SET:
        return False
    return not os.path.isdir(os.path.join(PIECE_SETS_DIR, piece_set, "black_king"))


@lru_cache(maxsize=None)
def set_properties(piece_set: str) -> dict:
    """piece_sets/<set>/set.json, if any. `"scale": "absolute"` means the meshes are
    already in metres at their true size - tokens, not Staunton pieces - and are
    not stretched to chess heights."""
    path = os.path.join(PIECE_SETS_DIR, piece_set, "set.json")
    if piece_set == DEFAULT_SET or not os.path.isfile(path):
        return {}
    import json
    with open(path) as f:
        return json.load(f)


def _piece_dir(piece: chess.Piece, piece_set: str) -> tuple[str, str]:
    if piece_set == DEFAULT_SET:
        name = piece_asset_name(piece)
        return os.path.join(ASSET_DIR, "pieces", name), name
    if not single_colour(piece_set):
        name = piece_asset_name(piece)
        return os.path.join(PIECE_SETS_DIR, piece_set, name), name
    name = _TYPE_NAME[piece.piece_type]
    return os.path.join(PIECE_SETS_DIR, piece_set, name), name


def piece_obj_path(piece: chess.Piece, piece_set: str = DEFAULT_SET) -> str:
    folder, name = _piece_dir(piece, piece_set)
    return os.path.join(folder, f"{name}.obj")


def piece_texture_path(piece: chess.Piece, piece_set: str = DEFAULT_SET) -> str:
    return os.path.join(_piece_dir(piece, piece_set)[0], "material_0.png")


def piece_tint(piece: chess.Piece, white_rgba, black_rgba, piece_set: str = DEFAULT_SET):
    """The material colour over a piece's texture. The vendored set has dark
    textures for black; a single-colour set darkens the tint instead."""
    if piece.color == chess.WHITE:
        return tuple(white_rgba)
    if not single_colour(piece_set):
        return tuple(black_rgba)
    return (*(c * DARK_SIDE for c in black_rgba[:3]), black_rgba[3])


# Collider profile: (bottom, top) height fractions of each cylinder segment and
# the height fraction at which its radius is sampled from the mesh.
PROFILE_SEGMENTS = ((0.00, 0.25, 0.10), (0.25, 0.45, 0.35), (0.45, 0.65, 0.55), (0.65, 1.00, 0.80))


@dataclass(frozen=True)
class PieceGeometry:
    """Scaled dimensions of a piece as it appears in the scene."""

    scale: float
    height: float
    width: float          # max horizontal extent
    bottom_offset: float  # mesh-frame z of the base (scaled), usually negative
    center: np.ndarray    # scaled mesh AABB center (mesh frame)
    profile: tuple        # ((z_bottom, z_top, radius), ...) in scaled mesh frame

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
    def flange_top(self) -> float:
        """Height above the base of the top of the highest segment below the
        grasp segment that is wider than it (0 if none): pads clamping the
        grasp segment must not reach down into it."""
        z0, _, radius = self._grasp_segment
        tops = [z1 - self.bottom_offset for zb, z1, r in self.profile if z1 <= z0 and r > radius]
        return max(tops, default=0.0)

    @property
    def collider_radius(self) -> float:
        """Footprint radius (the base segment of the collider profile)."""
        return self.profile[0][2]

    def radius_between(self, lo: float, hi: float) -> float:
        """Widest collider radius between heights `lo` and `hi` above the base
        (what open prongs spanning that band must clear on the way in and out)."""
        z_lo, z_hi = self.bottom_offset + lo, self.bottom_offset + hi
        return max(r for z0, z1, r in self.profile if z1 > z_lo and z0 < z_hi)


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


def piece_geometry(piece: chess.Piece, square: float, piece_scale: float = 1.0,
                   piece_set: str = DEFAULT_SET) -> PieceGeometry:
    """Scale a piece to the board: target height, footprint capped to the square;
    `piece_scale` multiplies both (see `AppearanceConfig`)."""
    path = piece_obj_path(piece, piece_set)
    lo, hi = _bounds(path)
    extent = hi - lo
    fit = 0.88 * square * piece_scale / max(extent[0], extent[1], 1e-6)   # footprint inside a square
    if set_properties(piece_set).get("scale") == "absolute":
        scale = min(piece_scale, fit)
    else:
        target_h = _PIECE_HEIGHT[piece.piece_type] * (square / _REFERENCE_SQUARE) * piece_scale
        scale = min(target_h / max(extent[2], 1e-6), fit)
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
    )


# -- procedural scene assets (generated once, then reused) --------------------

GENERATED_DIR = os.path.join(ASSET_DIR, "scene", "generated")   # per-appearance textures, not vendored


def ensure_scene_assets(board, appearance=None) -> str:
    """Create the backdrop mesh and the board texture for `appearance` if they
    do not exist yet; returns the texture path relative to the asset dir."""
    if not os.path.exists(BACKDROP_OBJ):
        _write_backdrop(BACKDROP_OBJ, BACKDROP_IMAGE)
    if appearance is None or appearance.board_key == _DEFAULT_BOARD_KEY:
        path = BOARD_TEXTURE
        colors = None
    else:
        os.makedirs(GENERATED_DIR, exist_ok=True)
        path = os.path.join(GENERATED_DIR, f"board_{appearance.board_key}.png")
        colors = (appearance.light_square, appearance.dark_square, appearance.border)
    if not os.path.exists(path):
        _write_board_texture(path, board, colors,
                             labels=bool(appearance is not None and appearance.board_labels),
                             finish=appearance.board_finish if appearance is not None else "wood")
    return os.path.relpath(path, ASSET_DIR)


BOARD_FINISH_SPECKLE = {"plastic": 2.0, "vinyl": 5.0, "magnetic": 1.5, "cardboard": 4.0}
FOLDING_FINISHES = ("magnetic", "cardboard")
_DEFAULT_BOARD_COLORS = ((214, 178, 132), (99, 64, 40), (92, 58, 32))
_DEFAULT_BOARD_KEY = "-".join(f"{c:02x}" for rgb in _DEFAULT_BOARD_COLORS for c in rgb)


def _write_board_texture(path: str, board, colors=None, px_per_square: int = 128,
                         labels: bool = False, finish: str = "wood") -> None:
    from PIL import Image, ImageDraw, ImageFont

    light, dark, border = colors or _DEFAULT_BOARD_COLORS
    rng = np.random.default_rng(0)
    border_px = int(px_per_square * board.border / board.square)
    size = 8 * px_per_square + 2 * border_px
    img = np.zeros((size, size, 3), np.float32)

    def wood(shape, base, variation=14.0):
        if finish != "wood":
            # printed or moulded: flat colour with a finish-dependent speckle, no grain
            return np.clip(base + rng.normal(0, BOARD_FINISH_SPECKLE[finish], shape), 0, 255)
        grain = np.cumsum(rng.normal(0, 1, shape), axis=1)
        grain = (grain - grain.min()) / (np.ptp(grain) + 1e-6) - 0.5
        rows = rng.normal(0, 1, (shape[0], 1)) * 0.35
        return np.clip(base + (grain + rows) * variation, 0, 255)

    for c, v in enumerate(border):
        img[..., c] = wood((size, size), v)
    for i in range(8):
        for j in range(8):
            rgb = light if (i + j) % 2 else dark
            y0, x0 = border_px + i * px_per_square, border_px + j * px_per_square
            for c in range(3):
                img[y0:y0 + px_per_square, x0:x0 + px_per_square, c] = wood(
                    (px_per_square, px_per_square), rgb[c])
    if finish in FOLDING_FINISHES:
        # a folding board's hinge: a thin darker line across the middle, between ranks 4 and 5
        mid, half = size // 2, max(1, px_per_square // 64)
        img[mid - half:mid + half + 1, :, :] *= 0.55
    image = Image.fromarray(img.astype(np.uint8))
    if labels and border_px >= 24:
        # files along the near and far borders, ranks along the sides, as printed boards
        # have them; the image's row 0 is the far (rank 8) edge
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=int(border_px * 0.6))
        ink = tuple(int(v) for v in (np.array(light) * 0.9))
        for i in range(8):
            file_ = "abcdefgh"[i]
            x = border_px + (i + 0.5) * px_per_square
            for y in (border_px / 2, size - border_px / 2):
                draw.text((x, y), file_, fill=ink, font=font, anchor="mm")
            rank = str(8 - i)
            y = border_px + (i + 0.5) * px_per_square
            for x in (border_px / 2, size - border_px / 2):
                draw.text((x, y), rank, fill=ink, font=font, anchor="mm")
    image.save(path)


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


def ensure_table_texture(kind: str) -> str:
    """A light, mostly grey surface texture the table colour tints: wood grain,
    woven cloth, veined marble or speckled laminate. Written once; returns the
    path relative to the asset dir."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    path = os.path.join(GENERATED_DIR, f"table_{kind}.png")
    if not os.path.exists(path):
        from PIL import Image
        rng = np.random.default_rng(7)
        n = 512
        y, x = np.mgrid[0:n, 0:n] / n
        if kind == "wood":
            # long streaks across the plank, gently wandering, with fine fibre noise
            streaks = np.sin(y * 140 + 5 * np.sin(x * 3.1) + 2 * np.sin(x * 11 + y * 4))
            fibre = np.repeat(rng.normal(0, 1, (n, 1)), n, axis=1) * 0.04
            img = 0.8 + 0.06 * streaks + fibre + rng.normal(0, 0.02, (n, n))
        elif kind == "cloth":
            weave = 0.5 * (np.sin(x * n * 0.9) + np.sin(y * n * 0.9))
            img = 0.8 + 0.05 * weave + rng.normal(0, 0.05, (n, n))
        elif kind == "marble":
            veins = np.abs(np.sin((x + y) * 7 + 3 * np.sin(x * 5) * np.cos(y * 4)))
            img = 0.92 - 0.35 * np.exp(-veins * 18) + rng.normal(0, 0.015, (n, n))
        else:                                   # laminate
            img = 0.86 + rng.normal(0, 0.025, (n, n)) + 0.03 * (rng.random((n, n)) > 0.995)
        rgb = np.clip(np.stack([img] * 3, axis=-1), 0, 1)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
    return os.path.relpath(path, ASSET_DIR)
