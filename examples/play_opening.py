"""Play a scripted move sequence from a position and record one video.

    python examples/play_opening.py --out sim/out/opening.mp4
    python examples/play_opening.py --fen "4k3/8/4K3/8/8/8/1Q6/8" \
        --moves b2b7 e8f8 e6f6 f8e8 b7e7 --out sim/out/ladder_mate.mp4

The arm plays both sides. Moves come from a fixed line here; swap in a chess
engine (python-chess + Stockfish) to play a real game.
"""
import argparse

import chess
import imageio.v2 as imageio

from chess_sim import ChessSimEnv, START_FEN

ITALIAN_GAME = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", nargs="*", default=ITALIAN_GAME, help="UCI moves")
    ap.add_argument("--fen", default=START_FEN, help="starting position (board field)")
    ap.add_argument("--out", default="sim/out/opening.mp4")
    args = ap.parse_args()

    env = ChessSimEnv(cameras=("external",), image_size=(960, 540))
    env.reset(args.fen)
    frames = []

    def record(action):
        frames.append(env.render("external"))

    for uci in args.moves:
        mv = chess.Move.from_uci(uci)
        san = env.board.san(mv)
        result = env.move(chess.square_name(mv.from_square), chess.square_name(mv.to_square),
                          on_step=record)
        print(f"{san:6s} {'ok' if result.success else 'FAILED'} "
              f"({result.placement_error * 1000:.1f} mm{', ' + result.reason if result.reason else ''})")
        for _ in range(15):                       # a short pause between moves
            record(None)
    imageio.mimsave(args.out, frames, fps=30)
    print("final position:", env.board.fen())
    print("wrote", args.out)
    env.close()


if __name__ == "__main__":
    main()
