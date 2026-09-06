"""Board geometry and square/FEN helpers.

The board is procedural: a bordered slab with an 8x8 checker texture, so every
square center is an exact analytic coordinate in the world frame. The board
center is the world origin in x/y; the arm sits on the -y side (white's side).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import chess

START_FEN = chess.STARTING_BOARD_FEN
FILES = "abcdefgh"


@dataclass(frozen=True)
class BoardSpec:
    """Physical layout of the board and the arm mount, in meters."""

    square: float = 0.028          # square edge (mini set: SO-101 reach is ~30 cm)
    border: float = 0.015          # wood border around the 8x8 field
    thickness: float = 0.012       # board slab thickness
    table_top: float = 0.43        # table height above the floor
    arm_gap: float = 0.04          # distance from board edge to arm base center
    arm_riser: float = 0.14        # pedestal height under the arm base

    @property
    def field(self) -> float:
        return 8 * self.square

    @property
    def width(self) -> float:
        return self.field + 2 * self.border

    @property
    def top(self) -> float:
        """World z of the board's playing surface."""
        return self.table_top + self.thickness

    @property
    def arm_base(self) -> tuple[float, float, float]:
        """World position of the SO-101 base frame."""
        return (0.0, -(self.width / 2 + self.arm_gap), self.table_top + self.arm_riser)

    def square_center(self, square: int) -> tuple[float, float]:
        """World x/y of a python-chess square index (0 = a1 ... 63 = h8)."""
        f, r = chess.square_file(square), chess.square_rank(square)
        return (f - 3.5) * self.square, (r - 3.5) * self.square

    def square_at(self, x: float, y: float) -> int | None:
        """Square index under world x/y, or None off the playing field."""
        f, r = math.floor(x / self.square + 4), math.floor(y / self.square + 4)
        return chess.square(f, r) if 0 <= f < 8 and 0 <= r < 8 else None

    def graveyard_slot(self, index: int) -> tuple[float, float]:
        """Off-board parking spot for pieces absent from the position."""
        col, row = divmod(index, 4)[1], divmod(index, 4)[0]
        x = self.width / 2 + 0.06 + col * 0.035
        y = -self.field / 2 + row * 0.035
        return x, y


def parse_square(name: str) -> int:
    """'e4' -> python-chess square index."""
    return chess.parse_square(name)


def square_name(square: int) -> str:
    return chess.square_name(square)
