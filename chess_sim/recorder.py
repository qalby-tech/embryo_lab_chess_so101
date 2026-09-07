"""Episode recording in a LeRobot-compatible layout.

    rec = EpisodeRecorder(env, "datasets/chess")
    rec.begin("move the white knight from g1 to f3", fen=env.board.fen(), move="g1f3")
    result = env.move("g1", "f3", on_step=rec.on_step)
    rec.end(result.success, placement_error=result.placement_error)

Each episode directory holds data.npz (observation_state, action at the control
rate), one mp4 per camera, and meta.json; a manifest.jsonl indexes them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import imageio.v2 as imageio
import numpy as np

from .env import JOINTS
from .scene import ROBOT_CAMERAS


@dataclass
class _Buffer:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    frames: dict[str, list] = field(default_factory=dict)


class EpisodeRecorder:
    """Records exactly the cameras the real robot has (`ROBOT_CAMERAS`: the
    overhead and wrist cameras), whatever else the env renders for viewing."""

    def __init__(self, env, root: str):
        missing = [cam for cam in ROBOT_CAMERAS if cam not in env.cameras]
        if missing:
            raise ValueError(f"env must render the robot cameras {ROBOT_CAMERAS}; missing {missing}")
        self.env = env
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._index = self._next_index()
        self._meta: dict = {}
        self._buf = _Buffer()

    def begin(self, instruction: str, **meta) -> None:
        self._buf = _Buffer(frames={cam: [] for cam in ROBOT_CAMERAS})
        self._meta = {"instruction": instruction, "fps": self.env.control_hz,
                      "joints": list(JOINTS), "cameras": list(ROBOT_CAMERAS), **meta}

    def on_step(self, action: np.ndarray) -> None:
        """Pass as `on_step` to env.move()/apply_action(); samples before each action."""
        self._buf.states.append(self.env.arm_joint_positions().copy())
        self._buf.actions.append(np.array(action, dtype=np.float32))
        for cam in ROBOT_CAMERAS:
            self._buf.frames[cam].append(self.env.render(cam))

    def end(self, success: bool, **extra) -> str:
        ep_dir = os.path.join(self.root, f"episode_{self._index:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        np.savez_compressed(os.path.join(ep_dir, "data.npz"),
                            observation_state=np.array(self._buf.states, dtype=np.float32),
                            action=np.array(self._buf.actions, dtype=np.float32))
        for cam, frames in self._buf.frames.items():
            if frames:
                imageio.mimsave(os.path.join(ep_dir, f"{cam}.mp4"), frames, fps=self.env.control_hz)
        meta = {**self._meta, **extra, "success": bool(success),
                "steps": len(self._buf.actions), "episode": self._index}
        with open(os.path.join(ep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(self.root, "manifest.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")
        self._index += 1
        return ep_dir

    def _next_index(self) -> int:
        existing = [d for d in os.listdir(self.root) if d.startswith("episode_")]
        return max((int(d.split("_")[1]) for d in existing), default=-1) + 1
