"""Which squares the arm can actually work on, and which moves are executable.

Eligibility combines three things: IK solvability with the tool within a tilt
cone, an *empirical* executed-accuracy map (servo tracking measured per square,
see examples/calibrate_reach.py), and a near-field exclusion right in front of
the base where the servos cannot track under load.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import chess
import numpy as np

from .board import BoardSpec

CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib")
EXECUTED_REACH_FILE = os.path.join(CALIB_DIR, "executed_reach.json")

MAX_TILT = np.radians(30)
MAX_IK_ERROR = 0.008
MAX_EXECUTED_ERROR = 0.005
NEAR_FIELD_RADIUS = 0.14
# candidate jaw-span directions (the moving jaw opens along one board axis)
SPAN_DIRECTIONS = (np.array([0.0, 1.0]), np.array([0.0, -1.0]),
                   np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
MAX_SPAN_ERROR = np.radians(20)   # a span the wrist roll cannot turn this close to is unusable


def load_executed_reach(path: str = EXECUTED_REACH_FILE) -> dict[int, float]:
    """Per-square executed fingertip error (meters), keyed by chess square."""
    if not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    out = {}
    for key, err in raw.items():
        f, r = (int(v) for v in key.split(","))
        out[chess.square(f, r)] = float(err)
    return out


OVERHANG_SQUARES = 2   # the tilted wrist body extends this far past the fingertips


@dataclass
class ReachMap:
    """Squares the scripted controller can grasp/place on, and how the wrist
    leans over the board there (to keep it clear of neighboring pieces)."""

    board: BoardSpec
    grasp_height: float
    tool_query: Callable = field(repr=False, compare=False)
    squares: set[int] = field(default_factory=set)
    lean: dict[int, np.ndarray] = field(default_factory=dict)   # wrist overhang direction
    _spans: dict[int, list[np.ndarray]] = field(default_factory=dict, repr=False)

    @classmethod
    def compute(cls, board: BoardSpec, tool_query, grasp_height: float,
                executed: dict[int, float] | None = None) -> "ReachMap":
        """`tool_query(target, span=None) -> (IkResult, tool_rotation)` for a
        world target; the rotation's columns are the tool x (approach) and z
        (jaw span) axes."""
        executed = load_executed_reach() if executed is None else executed
        ax, ay, _ = board.arm_base
        squares, lean = set(), {}
        for sq in chess.SQUARES:
            x, y = board.square_center(sq)
            res, mat = tool_query(np.array([x, y, board.top + grasp_height]))
            if res.position_error > MAX_IK_ERROR or res.tilt > MAX_TILT:
                continue
            if executed.get(sq, 0.0) > MAX_EXECUTED_ERROR:
                continue
            if np.hypot(x - ax, y - ay) < NEAR_FIELD_RADIUS:
                continue
            squares.add(sq)
            # the gripper body lies along -approach from the fingertips
            h = -mat[:2, 0]
            lean[sq] = h / (np.linalg.norm(h) + 1e-9)
        return cls(board=board, grasp_height=grasp_height, tool_query=tool_query,
                   squares=squares, lean=lean)

    def __contains__(self, square: int) -> bool:
        return square in self.squares

    def feasible_spans(self, square: int) -> list[np.ndarray]:
        """Jaw-span directions the wrist roll can realize at `square`, in
        SPAN_DIRECTIONS order (solved on first use, then cached)."""
        if square not in self._spans:
            x, y = self.board.square_center(square)
            target = np.array([x, y, self.board.top + self.grasp_height])
            feasible = []
            for d in SPAN_DIRECTIONS:
                res, mat = self.tool_query(target, span=d)
                z = mat[:2, 2] / (np.linalg.norm(mat[:2, 2]) + 1e-9)
                if (res.position_error <= MAX_IK_ERROR and res.tilt <= MAX_TILT
                        and np.arccos(np.clip(np.dot(z, d), -1.0, 1.0)) <= MAX_SPAN_ERROR):
                    feasible.append(d)
            self._spans[square] = feasible
        return self._spans[square]

    def overhang_squares(self, square: int) -> list[int]:
        """Squares under the leaning wrist body when working on `square`
        (jaw-span clearance is judged geometrically by ChessSimEnv)."""
        out = []
        h = self.lean.get(square)
        if h is not None:
            for k in range(1, OVERHANG_SQUARES + 1):
                out += self._offset_square(square, k * h)
        return out

    @staticmethod
    def _offset_square(square: int, v: np.ndarray) -> list[int]:
        f = chess.square_file(square) + int(round(v[0]))
        r = chess.square_rank(square) + int(round(v[1]))
        return [chess.square(f, r)] if 0 <= f < 8 and 0 <= r < 8 and (f, r) != (
            chess.square_file(square), chess.square_rank(square)) else []

    def executable_moves(self, position: chess.Board) -> list[chess.Move]:
        """Quiet legal moves whose endpoints are reachable with clear overhang."""
        moves = []
        for mv in position.legal_moves:
            if position.is_capture(mv) or mv.promotion or position.is_castling(mv):
                continue
            if mv.from_square not in self or mv.to_square not in self:
                continue
            blocked = [sq for sq in self.overhang_squares(mv.from_square) if position.piece_at(sq)]
            blocked += [sq for sq in self.overhang_squares(mv.to_square)
                        if sq != mv.from_square and position.piece_at(sq)]
            if not blocked:
                moves.append(mv)
        return moves
