"""chess_sim: a MuJoCo framework for an SO-101 arm playing chess.

    from chess_sim import ChessSimEnv, START_FEN

    env = ChessSimEnv()
    env.reset(START_FEN)
    result = env.move("e2", "e4")     # scripted expert executes the move
    print(result.success, env.board.fen())
"""
from .board import START_FEN, BoardSpec
from .env import ChessSimEnv, MoveResult, Observation
from .recorder import EpisodeRecorder

__all__ = ["ChessSimEnv", "MoveResult", "Observation", "EpisodeRecorder",
           "BoardSpec", "START_FEN"]
