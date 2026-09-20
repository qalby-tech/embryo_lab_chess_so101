"""Policies that drive the arm.

A policy answers one question - what joint targets to command next:

    policy = LeRobotPolicy.load(LeRobotPolicyConfig(checkpoint=path))
    action = policy.select_action(observation, task)

`LeRobotPolicy` takes any checkpoint in LeRobot's registry - the type is read
from the checkpoint itself, and the observation is assembled from the features
that checkpoint declares, so act, diffusion, pi0, smolvla and molmoact2 all load
the same way. It has been exercised here on molmoact2.

The scripted expert is not one of these: it plans against simulator state, so it
lives on the env as `env.execute(task)` and is used for demonstrations, not for
inference.
"""
from __future__ import annotations

import inspect
import json
import os
from typing import Protocol, runtime_checkable

import numpy as np

from .config import Camera, Config
from .conventions import SO101_DEGREES, JointConvention
from .env import Observation
from .tasks import Task

# Measured (docs/EXPERIMENTS.md 5.6): executing 5 actions of the 30-action chunk
# scores 82% against 75% for the whole chunk, with half the dropped pieces
# (128 paired positions, McNemar p=0.043). The published checkpoint still ships 30.
DEFAULT_ACTION_STEPS = 5
DEFAULT_TARGET_HZ = 10          # the strided export's frame rate
PIXEL_MAX = 255.0
STATE_FEATURE = "observation.state"
IMAGE_PREFIX = "observation.images."
# where a policy keeps its open-loop horizon, in the order we prefer to read it
HORIZON_FIELDS = ("n_action_steps", "horizon", "chunk_size")


@runtime_checkable
class Policy(Protocol):
    """What a rollout needs from a policy."""

    def reset(self) -> None:
        """Forget anything carried over from the last episode."""

    def needs_images(self) -> bool:
        """False while the policy still has actions queued - rendering can be skipped."""

    def select_action(self, observation: Observation, task: Task) -> np.ndarray:
        """Joint targets in radians for the next control period."""


class LeRobotPolicyConfig(Config):
    """How a trained LeRobot checkpoint is run in this simulator.

    A policy trained on a strided export emits targets slower than the control
    loop runs (`target_hz` against `control_hz`), so each one is held for the
    difference. Stepping straight to a held target is what knocks pieces over;
    interpolating asks for the same motion spread across the period.
    """

    checkpoint: str                                     # local directory or a Hub repo id
    device: str = "cuda"
    convention: JointConvention = SO101_DEGREES
    n_action_steps: int | None = DEFAULT_ACTION_STEPS   # None keeps the checkpoint's own value
    num_inference_steps: int | None = None              # flow-matching steps, for policies that take them
    control_hz: int = 30
    target_hz: int = DEFAULT_TARGET_HZ
    interpolate: bool = True

    @property
    def hold(self) -> int:
        """Control periods each emitted target is held for."""
        return max(1, round(self.control_hz / self.target_hz))

    @classmethod
    def for_dataset(cls, checkpoint: str, dataset_root: str | None = None,
                    **kwargs) -> "LeRobotPolicyConfig":
        """Read the target rate off the dataset the policy was trained on."""
        return cls(checkpoint=checkpoint, target_hz=dataset_fps(dataset_root), **kwargs)


def dataset_fps(root: str | None, default: int = DEFAULT_TARGET_HZ) -> int:
    """The frame rate a LeRobot dataset was written at."""
    if not root:
        return default
    try:
        with open(os.path.join(root, "meta", "info.json")) as f:
            return int(json.load(f)["fps"])
    except (OSError, ValueError, KeyError):
        return default


class LeRobotPolicy:
    """Adapter around a LeRobot checkpoint: builds the observation the checkpoint
    asks for, and expands each emitted target into control-rate commands.

    lerobot and torch are imported on load, so importing `chess_sim` does not
    need them.
    """

    def __init__(self, config: LeRobotPolicyConfig, policy, processors, features: list[str]):
        self.config = config
        self.policy = policy
        self.features = features
        self.preprocess, self.postprocess = processors
        self._takes_inference_steps = _accepts_kwarg(policy.select_action, "num_steps")
        self._queue: list[np.ndarray] = []
        self._previous: np.ndarray | None = None

    @classmethod
    def load(cls, config: LeRobotPolicyConfig) -> "LeRobotPolicy":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        # the checkpoint names its own type and the inputs it was trained on
        pretrained = PreTrainedConfig.from_pretrained(config.checkpoint)
        policy = get_policy_class(pretrained.type).from_pretrained(config.checkpoint)
        policy = policy.to(config.device).eval()
        # the checkpoint carries its own normalization pipelines; the policy sees
        # normalized inputs and returns actions in dataset units through them
        processors = make_pre_post_processors(policy_cfg=policy.config,
                                              pretrained_path=config.checkpoint)
        if config.n_action_steps is not None and hasattr(policy.config, "n_action_steps"):
            policy.config.n_action_steps = config.n_action_steps   # reset() rebuilds the queue
        return cls(config, policy, processors, list(policy.config.input_features))

    @property
    def policy_type(self) -> str:
        return str(self.policy.config.type)

    @property
    def cameras(self) -> tuple[Camera, ...]:
        """The cameras this checkpoint expects - configure the env to render these."""
        return tuple(Camera(name[len(IMAGE_PREFIX):]) for name in self.features
                     if name.startswith(IMAGE_PREFIX))

    @property
    def chunk_size(self) -> int | None:
        return getattr(self.policy.config, "chunk_size", None)

    @property
    def action_steps(self) -> int | None:
        """Actions executed per model call - the open-loop horizon."""
        for field in HORIZON_FIELDS:
            value = getattr(self.policy.config, field, None)
            if value is not None:
                return int(value)
        return None

    def reset(self) -> None:
        self.policy.reset()
        self._queue = []
        self._previous = None

    def needs_images(self) -> bool:
        return not self._queue

    def select_action(self, observation: Observation, task: Task) -> np.ndarray:
        if not self._queue:
            self._queue = self._next_targets(observation, task)
        return self._queue.pop(0)

    def _next_targets(self, observation: Observation, task: Task) -> list[np.ndarray]:
        """Ask the policy for one target and spread it over `hold` control periods."""
        import torch

        kwargs = {}
        if self.config.num_inference_steps and self._takes_inference_steps:
            kwargs["num_steps"] = self.config.num_inference_steps
        with torch.inference_mode():
            batch = self.preprocess(self._batch(observation, task))
            action = self.postprocess(self.policy.select_action(batch, **kwargs))
        command = self.config.convention.to_radians(action[0].float().cpu().numpy())
        previous = self._previous if self._previous is not None else observation.joint_pos
        self._previous = command
        hold = self.config.hold
        if not self.config.interpolate:
            return [command] * hold
        return [previous + (command - previous) * (k + 1) / hold for k in range(hold)]

    def _batch(self, observation: Observation, task: Task) -> dict:
        """One observation in the shape this checkpoint declared."""
        import torch

        device = self.config.device
        batch: dict = {"task": [task.instruction]}
        for name in self.features:
            if name == STATE_FEATURE:
                state = self.config.convention.from_radians(observation.joint_pos)
                batch[name] = torch.from_numpy(state.astype(np.float32))[None].to(device)
            elif name.startswith(IMAGE_PREFIX):
                camera = Camera(name[len(IMAGE_PREFIX):])
                if camera not in observation.images:
                    raise KeyError(f"the checkpoint needs camera {camera}; the env renders "
                                   f"{sorted(observation.images)}")
                frame = torch.from_numpy(observation.images[camera].copy())
                batch[name] = (frame.permute(2, 0, 1).float() / PIXEL_MAX)[None].to(device)
        return batch


def _accepts_kwarg(function, name: str) -> bool:
    """Whether `function` takes this keyword, directly or through **kwargs."""
    parameters = inspect.signature(function).parameters
    return name in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD
                                     for p in parameters.values())
