"""Random chess positions to practise on.

The scene owns exactly one set of pieces, so a position can only ask for pieces
the set still has: `PositionConfig` draws kings plus a handful of others without
replacement.
"""
from __future__ import annotations

import random

import chess
from pydantic import Field

from .assets import SET_COUNTS
from .config import Config

PROMOTION_RANKS = (0, 7)        # a pawn cannot stand on the back ranks
PLACEMENT_ATTEMPTS = 60         # tries before giving up on a position with a required move

# every non-king piece of a full set, drawn without replacement
SET_POOL = [chess.Piece(piece_type, color)
            for color in (chess.WHITE, chess.BLACK)
            for piece_type, count in SET_COUNTS.items() if piece_type != chess.KING
            for _ in range(count)]


class PositionConfig(Config):
    """How crowded the practice positions are: two kings plus this many pieces."""

    min_extra_pieces: int = Field(2, ge=0)
    max_extra_pieces: int = Field(8, ge=0)

    def sample(self, rng: random.Random) -> chess.Board:
        """Kings plus a few random pieces from one set; legal, white to move."""
        extra = rng.randint(self.min_extra_pieces, self.max_extra_pieces)
        while True:
            board = chess.Board(None)
            squares = rng.sample(chess.SQUARES, 2 + extra)
            board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
            for square, piece in zip(squares[2:], rng.sample(SET_POOL, extra)):
                if piece.piece_type == chess.PAWN and chess.square_rank(square) in PROMOTION_RANKS:
                    continue
                board.set_piece_at(square, piece)
            if board.is_valid():
                return board

    def sample_with_move(self, rng: random.Random, move: chess.Move) -> chess.Board | None:
        """A random position in which `move` is legal for the side to move: a
        piece to pick on the source square, nothing on the target. Which piece
        goes there is chosen so the move is legal - a pawn cannot move backwards,
        so a far-rank move needs a piece that can make it."""
        for _ in range(PLACEMENT_ATTEMPTS):
            board = self.sample(rng)
            if any(board.king(color) in (move.from_square, move.to_square)
                   for color in (chess.WHITE, chess.BLACK)):
                continue
            board.remove_piece_at(move.to_square)
            board.remove_piece_at(move.from_square)
            board.turn = chess.WHITE
            spare = [p for p in spare_pieces(board, move.from_square) if p.color == chess.WHITE]
            rng.shuffle(spare)
            for piece in spare:
                board.set_piece_at(move.from_square, piece)
                if board.is_valid() and move in board.legal_moves:
                    return board
                board.remove_piece_at(move.from_square)
        return None


def spare_pieces(board: chess.Board, square: int) -> list[chess.Piece]:
    """Pieces a set still has left over once `board` is accounted for."""
    used: dict[tuple[int, bool], int] = {}
    for piece in board.piece_map().values():
        used[(piece.piece_type, piece.color)] = used.get((piece.piece_type, piece.color), 0) + 1
    spare = []
    for piece_type, count in SET_COUNTS.items():
        if piece_type == chess.KING:
            continue
        if piece_type == chess.PAWN and chess.square_rank(square) in PROMOTION_RANKS:
            continue
        for color in (chess.WHITE, chess.BLACK):
            spare += [chess.Piece(piece_type, color)] * (count - used.get((piece_type, color), 0))
    return spare
