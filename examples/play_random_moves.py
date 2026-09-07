"""Play random executable moves from random sparse positions and report success.

    python examples/play_random_moves.py --moves 10 --record datasets/chess
    python examples/play_random_moves.py --moves 10 --randomize      # random look per episode
"""
import argparse
import random

import chess
import numpy as np

from chess_sim import Appearance, ChessSimEnv, EpisodeRecorder
from chess_sim.assets import SET_COUNTS

REBUILD_EVERY = 5   # with --randomize: new piece size / board every N episodes, new colors every episode

# every non-king piece of a full set, drawn without replacement
_SET_POOL = [chess.Piece(ptype, color)
             for color in (chess.WHITE, chess.BLACK)
             for ptype, count in SET_COUNTS.items() if ptype != chess.KING
             for _ in range(count)]


def random_position(rng: random.Random, extra_pieces: int) -> chess.Board:
    """Kings plus a few random pieces from one set; legal, white to move."""
    while True:
        board = chess.Board(None)
        squares = rng.sample(chess.SQUARES, 2 + extra_pieces)
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        for sq, piece in zip(squares[2:], rng.sample(_SET_POOL, extra_pieces)):
            if piece.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            board.set_piece_at(sq, piece)
        if board.is_valid():
            return board


def describe(board: chess.Board, move: chess.Move) -> str:
    piece = board.piece_at(move.from_square)
    color = "white" if piece.color else "black"
    return (f"move the {color} {chess.piece_name(piece.piece_type)} "
            f"from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}")


def run(moves: int, seed: int = 0, record: str | None = None, randomize: bool = False) -> int:
    """Play `moves` random moves; returns the number of verified successes.
    With `randomize`, every episode gets a random look (`Appearance.random`):
    colors and lighting change per episode, piece size and board texture
    every REBUILD_EVERY episodes (those need a rebuilt scene)."""
    rng = random.Random(seed)
    looks = np.random.default_rng(seed)
    env = ChessSimEnv(appearance=Appearance.random(looks) if randomize else Appearance())
    recorder = EpisodeRecorder(env, record) if record else None
    successes = 0
    for i in range(moves):
        if randomize and i > 0:
            if i % REBUILD_EVERY == 0:
                env.close()
                env = ChessSimEnv(appearance=Appearance.random(looks))
                if recorder:
                    recorder = EpisodeRecorder(env, record)
            else:
                env.recolor(Appearance.random(looks))
        while True:
            board = random_position(rng, rng.randint(2, 8))
            env.reset(board.board_fen())
            candidates = env.executable_moves()
            if candidates:
                break
        mv = rng.choice(candidates)
        instruction = describe(env.board, mv)
        if recorder:
            recorder.begin(instruction, fen=env.board.fen(), move=mv.uci(), task="chess_move",
                           appearance=env.appearance.__dict__ if randomize else None)
        result = env.move(chess.square_name(mv.from_square), chess.square_name(mv.to_square),
                          on_step=recorder.on_step if recorder else None)
        if recorder:
            recorder.end(result.success, placement_error=result.placement_error,
                         disturbed=result.disturbed, reason=result.reason)
        successes += result.success
        print(f"[{i}] {instruction}: {'OK' if result.success else 'FAIL'} "
              f"({result.placement_error * 1000:.1f} mm{', ' + result.reason if result.reason else ''})")
    print(f"{successes}/{moves} verified successes")
    env.close()
    return successes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", default=None, help="dataset directory (optional)")
    ap.add_argument("--randomize", action="store_true", help="random appearance per episode")
    args = ap.parse_args()
    run(args.moves, args.seed, args.record, args.randomize)


if __name__ == "__main__":
    main()
