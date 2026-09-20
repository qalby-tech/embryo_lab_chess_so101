"""What the arm is asked to do, and how episodes of it are drawn.

A task carries its own instruction, so the text a policy is trained on and the
text it is evaluated on can never drift apart:

    task = MoveTask(from_square="g1", to_square="f3")
    task.instruction        # 'pick up the piece on g1 and place it on f3'
    result = env.execute(task)

Instructions are purely spatial: executing one needs no chess knowledge and no
piece recognition. Which move to play is the caller's business - a sampler here,
an engine in a real game.
"""
from __future__ import annotations

import random
from enum import StrEnum
from typing import Annotated, Literal, Protocol

import chess
from pydantic import Field

from .config import Config, Square, square_at
from .positions import PositionConfig

MOVE_INSTRUCTION = "pick up the piece on {source} and place it on {target}"
CAPTURE_INSTRUCTION = "take the piece on {square} off the board"


class TaskFamily(StrEnum):
    MOVE = "move"        # carry a piece to another square
    CAPTURE = "capture"  # take a piece off the board, into the discard tray


class Task(Config):
    """Base class: a task knows its instruction and where the piece must end up."""

    family: TaskFamily

    @property
    def instruction(self) -> str:
        raise NotImplementedError

    @property
    def source(self) -> Square:
        """The square the piece to pick up stands on."""
        raise NotImplementedError

    def target_xy(self, env) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def label(self) -> str:
        """Short identifier for logs and dataset metadata."""
        raise NotImplementedError


class MoveTask(Task):
    family: Literal[TaskFamily.MOVE] = TaskFamily.MOVE
    from_square: Square
    to_square: Square

    @classmethod
    def from_move(cls, move: chess.Move) -> "MoveTask":
        return cls(from_square=square_at(move.from_square), to_square=square_at(move.to_square))

    @property
    def instruction(self) -> str:
        return MOVE_INSTRUCTION.format(source=self.from_square, target=self.to_square)

    @property
    def source(self) -> Square:
        return self.from_square

    def target_xy(self, env) -> tuple[float, float]:
        # env.board, not the configured one: the board may have been shifted on reset
        return env.board.square_center(self.to_square)

    @property
    def label(self) -> str:
        return f"{self.from_square}{self.to_square}"


class CaptureTask(Task):
    family: Literal[TaskFamily.CAPTURE] = TaskFamily.CAPTURE
    square: Square

    @property
    def instruction(self) -> str:
        return CAPTURE_INSTRUCTION.format(square=self.square)

    @property
    def source(self) -> Square:
        return self.square

    def target_xy(self, env) -> tuple[float, float]:
        """The tray slot the piece is headed for - the next free one."""
        return env.free_tray_slot() or env.board.tray_slot(0)

    @property
    def label(self) -> str:
        return f"x{self.square}"


# `family` tells the two apart, so a stored result reads back as the task it was
AnyTask = Annotated[MoveTask | CaptureTask, Field(discriminator="family")]


class TaskSampler(Protocol):
    """Draws an episode: resets the env to a fresh position and returns a task
    the arm can physically attempt there."""

    family: TaskFamily

    def sample(self, env, rng: random.Random, index: int = 0) -> Task: ...


class MoveSampler(Config):
    """Random executable moves, or a fixed repertoire drawn in turn.

    With `moves` set, each UCI move is drawn equally often (that is what makes
    per-move scores comparable) and the rest of the board is randomized around it.
    """

    family: TaskFamily = TaskFamily.MOVE
    position: PositionConfig = PositionConfig()
    moves: tuple[str, ...] = ()
    randomize_layout: bool = True

    def sample(self, env, rng: random.Random, index: int = 0) -> MoveTask:
        while True:
            draw = rng if self.randomize_layout else None
            if self.moves:
                move = chess.Move.from_uci(self.moves[index % len(self.moves)])
                board = self.position.sample_with_move(rng, move)
                if board is None:
                    continue
                env.reset(board.board_fen(), rng=draw)
                if move in env.executable_moves():
                    return MoveTask.from_move(move)
            else:
                board = self.position.sample(rng)
                env.reset(board.board_fen(), rng=draw)
                candidates = env.executable_moves()
                if candidates:
                    return MoveTask.from_move(rng.choice(candidates))


class CaptureSampler(Config):
    """Random pieces the arm can lift off the board into the tray."""

    family: TaskFamily = TaskFamily.CAPTURE
    position: PositionConfig = PositionConfig()
    randomize_layout: bool = True

    def sample(self, env, rng: random.Random, index: int = 0) -> CaptureTask:
        while True:
            board = self.position.sample(rng)
            env.reset(board.board_fen(), rng=rng if self.randomize_layout else None)
            targets = env.executable_captures()
            if targets:
                return CaptureTask(square=square_at(rng.choice(targets)))
