"""Measure the fingertip accuracy the arm actually achieves at every square.

Writes chess_sim/calib/executed_reach.json, which ReachMap uses to exclude
squares the servos cannot track. Re-run after changing the board layout.
"""
import json

import chess
import numpy as np

from chess_sim import ChessSimEnv
from chess_sim.reach import EXECUTED_REACH_FILE


def main():
    env = ChessSimEnv(cameras=())
    env.reset("8/8/8/8/8/8/8/8")
    board = env.board_spec
    result, grid = {}, np.zeros((8, 8))
    for sq in chess.SQUARES:
        x, y = board.square_center(sq)
        target = np.array([x, y, board.top + 0.016])
        env.controller._go(target + [0, 0, 0.05], 0.03, np.zeros(3), 20, None)
        env.controller._go(target, 0.03, np.zeros(3), 12, None, precise=True)
        pos, _ = env.ik.tool_pose(env.data, np.zeros(3))
        err = float(np.linalg.norm(pos - target))
        result[f"{chess.square_file(sq)},{chess.square_rank(sq)}"] = err
        grid[chess.square_rank(sq), chess.square_file(sq)] = err * 1000
        env.controller._go(target + [0, 0, 0.05], 0.03, np.zeros(3), 12, None)
    for r in range(7, -1, -1):
        print(f"rank {r + 1}: " + " ".join(f"{grid[r, f]:4.0f}" for f in range(8)))
    with open(EXECUTED_REACH_FILE, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", EXECUTED_REACH_FILE)


if __name__ == "__main__":
    main()
