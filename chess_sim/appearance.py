"""Visual variety of the scene: piece size and colors, board and table colors, lighting.

    env = ChessSimEnv(appearance=Appearance(piece_scale=1.1, white_rgba=(0.95, 0.9, 0.75, 1)))
    env = ChessSimEnv(appearance=Appearance.random(np.random.default_rng(3)))
    env.recolor(Appearance.random(rng))      # colors and lights only: no recompile

Piece size and the board texture are baked into the compiled scene; every
other attribute can also be changed at run time with `ChessSimEnv.recolor`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

RGB = tuple[float, float, float]
RGBA = tuple[float, float, float, float]


@dataclass(frozen=True)
class Appearance:
    piece_scale: float = 1.0                 # multiplies piece heights (and the footprint cap); 0.7-1.1
    white_rgba: RGBA = (1.0, 1.0, 1.0, 1.0)  # tint multiplied over the white pieces' texture
    black_rgba: RGBA = (1.0, 1.0, 1.0, 1.0)  # same for the black pieces
    light_square: RGB = (214, 178, 132)      # board colors, 0-255
    dark_square: RGB = (99, 64, 40)
    border: RGB = (92, 58, 32)
    table_rgb: RGB = (0.42, 0.28, 0.17)      # 0-1
    light_intensity: float = 1.0             # key light brightness multiplier
    light_dir: RGB = (-0.3, 0.3, -1.0)       # key light direction (world)

    def __post_init__(self):
        if not 0.7 <= self.piece_scale <= 1.1:
            raise ValueError("piece_scale must be within [0.7, 1.1] (pieces must fit their squares)")

    @classmethod
    def random(cls, rng: np.random.Generator, piece_scale_range=(0.85, 1.1)) -> "Appearance":
        """A plausible random look: wood or painted boards, ivory-to-cream white
        sets, black-to-dark-colored black sets, varied table and lighting."""
        u = rng.uniform
        light = tuple(int(v) for v in rng.uniform((160, 130, 90), (240, 225, 200)))
        dark = tuple(int(v) for v in rng.uniform((30, 20, 15), (130, 95, 70)))
        if rng.random() < 0.3:                          # painted (green/blue/red) dark squares
            hue = rng.integers(3)
            dark = tuple(int(v) for v in (rng.uniform(20, 60) + (70 if hue == c else 0) for c in range(3)))
        border = tuple(int(v) for v in rng.uniform((40, 25, 15), (150, 110, 80)))
        white = (u(0.8, 1.0), u(0.75, 1.0), u(0.6, 1.0), 1.0)
        black = (u(0.5, 1.0), u(0.5, 1.0), u(0.5, 1.0), 1.0)
        azimuth, elevation = u(0, 2 * np.pi), np.radians(u(40, 75))
        light_dir = (float(np.cos(azimuth) * np.cos(elevation)), float(np.sin(azimuth) * np.cos(elevation)),
                     -float(np.sin(elevation)))
        return cls(piece_scale=float(u(*piece_scale_range)), white_rgba=white, black_rgba=black,
                   light_square=light, dark_square=dark, border=border,
                   table_rgb=tuple(float(v) for v in u((0.2, 0.15, 0.1), (0.75, 0.65, 0.55))),
                   light_intensity=float(u(0.6, 1.4)), light_dir=light_dir)

    def with_(self, **changes) -> "Appearance":
        return replace(self, **changes)

    @property
    def board_key(self) -> str:
        """Identifies the board texture (its colors) for caching."""
        return "-".join(f"{c:02x}" for rgb in (self.light_square, self.dark_square, self.border) for c in rgb)
