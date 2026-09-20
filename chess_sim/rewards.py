"""Rewards for reinforcement learning on top of a trained policy.

The success rule lives in `env.evaluate`; this module turns it into numbers and
adds shaping dense enough for a policy to improve between successes:

    config = RewardConfig()
    before = env.piece_snapshot()
    ...                                       # step the env with your policy
    result = env.evaluate(task, before)
    total = terminal_reward(result, config) + shaping_reward(env, task, before, config)

Shaping has two terms because a policy that has not grasped anything yet moves
no piece, and piece progress alone would hand it a flat zero: `reach_reward`
rewards bringing the tool to the piece, `progress_reward` rewards carrying the
piece to its target.

These weights are a starting point, not a tuned recipe.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .env import ChessSimEnv, PieceSnapshot, TaskResult
from .tasks import Task

TOOL_POINT = np.zeros(3)      # the pinch pocket itself, no offset


class RewardConfig(Config):
    success: float = 1.0          # the episode passed the same rule the expert is held to
    right_piece: float = 0.2      # engaged the piece the instruction named
    progress: float = 1.0         # scaled by the fraction of the piece's journey covered
    reach: float = 0.3            # scaled by how close the tool is to that piece
    reach_scale: float = 0.25     # m: distance at which the reach term has decayed to nothing
    disturbance: float = -0.5     # per other piece moved
    drop: float = -0.5            # the piece ended up lying down
    step_cost: float = -0.001     # per control step, to favour shorter trajectories


def terminal_reward(result: TaskResult, config: RewardConfig = RewardConfig()) -> float:
    """What the episode was worth once it is over."""
    return (config.success * result.success
            + config.right_piece * result.right_piece
            + config.disturbance * len(result.disturbed)
            + config.drop * (not result.upright)
            + config.step_cost * result.steps)


def reach_reward(env: ChessSimEnv, task: Task, config: RewardConfig = RewardConfig()) -> float:
    """How close the gripper has come to the piece named in the instruction:
    `reach` at the piece, nothing at `reach_scale` away."""
    slot = env.slot_at(task.source)
    if slot is None:
        return 0.0
    tool, _ = env.ik.tool_pose(env.data, TOOL_POINT)
    distance = float(np.linalg.norm(tool - env.piece_position(slot)))
    return config.reach * max(0.0, 1.0 - distance / config.reach_scale)


def progress_reward(env: ChessSimEnv, task: Task, before: PieceSnapshot,
                    config: RewardConfig = RewardConfig()) -> float:
    """How much of the way to its target the named piece has come, in [0, 1],
    scaled by `progress`. Zero at the start, negative if it moved away."""
    slot = env.slot_at(task.source)
    if slot is None or task.source not in before:
        return 0.0
    target = np.asarray(task.target_xy(env))
    initial = float(np.linalg.norm(np.asarray(before[task.source]) - target))
    if initial <= 0:
        return 0.0
    now = float(np.linalg.norm(env.piece_position(slot)[:2] - target))
    return config.progress * (initial - now) / initial


def shaping_reward(env: ChessSimEnv, task: Task, before: PieceSnapshot,
                   config: RewardConfig = RewardConfig()) -> float:
    """Both shaping terms: get to the piece, then carry it."""
    return reach_reward(env, task, config) + progress_reward(env, task, before, config)
