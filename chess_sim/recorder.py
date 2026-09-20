"""Episode recording in a LeRobot-compatible layout.

    recorder = EpisodeRecorder(env, RecorderConfig(root="datasets/chess"))
    recorder.begin(task)
    result = env.execute(task, on_step=recorder.on_step)
    recorder.end(result)

Each episode directory holds data.npz (observation_state and action at the
control rate), one mp4 per camera and meta.json; a manifest.jsonl indexes them.
Only the cameras the real robot has are recorded, whatever else the env renders
for viewing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import imageio.v2 as imageio
import numpy as np

from .config import JOINTS, ROBOT_CAMERAS, AppearanceConfig, Camera, Config
from .env import ChessSimEnv, Layout, Record, TaskResult
from .tasks import Task, TaskFamily

EPISODE_PREFIX = "episode_"
MANIFEST = "manifest.jsonl"


class RecorderConfig(Config):
    root: str
    cameras: tuple[Camera, ...] = ROBOT_CAMERAS


class EpisodeMeta(Record):
    """What meta.json holds - the exporter and the dataset card read it."""

    episode: int
    instruction: str
    task: TaskFamily
    move: str                       # the task's label: 'g1f3' or 'xe4'
    fen: str
    layout: Layout
    appearance: AppearanceConfig | None
    fps: int
    joints: list[str]
    cameras: list[str]
    success: bool = False
    steps: int = 0
    placement_error: float = 0.0
    disturbed: list[str] = []
    reason: str = ""


@dataclass
class _Buffer:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    frames: dict[Camera, list] = field(default_factory=dict)


class EpisodeRecorder:
    def __init__(self, env: ChessSimEnv, config: RecorderConfig):
        missing = [cam for cam in config.cameras if cam not in env.cameras]
        if missing:
            raise ValueError(f"env must render {list(config.cameras)}; missing {missing}")
        self.env = env
        self.config = config
        os.makedirs(config.root, exist_ok=True)
        self._index = self._next_index()
        self._meta: EpisodeMeta | None = None
        self._buf = _Buffer()

    def begin(self, task: Task, record_appearance: bool = False) -> None:
        """Start an episode: the position and layout are read off the env."""
        self._buf = _Buffer(frames={cam: [] for cam in self.config.cameras})
        self._meta = EpisodeMeta(
            episode=self._index, instruction=task.instruction, task=task.family, move=task.label,
            fen=self.env.position.fen(), layout=self.env.layout,
            appearance=self.env.config.appearance if record_appearance else None,
            fps=self.env.control_hz, joints=[str(j) for j in JOINTS],
            cameras=[str(c) for c in self.config.cameras])

    def on_step(self, action: np.ndarray) -> None:
        """Pass as `on_step` to env.execute()/env.step(); samples before each action."""
        self._buf.states.append(self.env.arm_joint_positions().copy())
        self._buf.actions.append(np.array(action, dtype=np.float32))
        for cam in self.config.cameras:
            self._buf.frames[cam].append(self.env.render(cam))

    def end(self, result: TaskResult) -> str:
        """Write the episode and its verdict; returns the directory."""
        if self._meta is None:
            raise RuntimeError("call begin() before end()")
        meta = self._meta.model_copy(update={
            "success": result.success, "steps": len(self._buf.actions),
            "placement_error": result.placement_error, "disturbed": result.disturbed,
            "reason": result.reason})
        path = os.path.join(self.config.root, f"{EPISODE_PREFIX}{self._index:04d}")
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(os.path.join(path, "data.npz"),
                            observation_state=np.array(self._buf.states, dtype=np.float32),
                            action=np.array(self._buf.actions, dtype=np.float32))
        for cam, frames in self._buf.frames.items():
            if frames:
                imageio.mimsave(os.path.join(path, f"{cam}.mp4"), frames, fps=self.env.control_hz)
        payload = meta.model_dump(mode="json")
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(payload, f, indent=2)
        with open(os.path.join(self.config.root, MANIFEST), "a") as f:
            f.write(json.dumps(payload) + "\n")
        self._index += 1
        return path

    def _next_index(self) -> int:
        existing = [d for d in os.listdir(self.config.root) if d.startswith(EPISODE_PREFIX)]
        return max((int(d.split("_")[1]) for d in existing), default=-1) + 1
