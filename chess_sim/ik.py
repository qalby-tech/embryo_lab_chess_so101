"""Damped-least-squares inverse kinematics for the SO-101's five arm joints.

Solves for a tool point (an offset from the `gripperframe` site, e.g. the pinch
pocket) to reach a world position, with the approach axis (site +x) kept close
to vertical. The rest-posture attraction is nullspace-projected and disabled
for the final iterations: with a damped pseudo-inverse the projector leaks and
would otherwise stall convergence about a centimeter short.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
_DOWN = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class IkResult:
    q: np.ndarray            # the five arm joint angles
    position_error: float    # meters, tool point vs target
    tilt: float              # radians between the approach axis and straight down


class So101Ik:
    def __init__(self, model: mujoco.MjModel, prefix: str = "",
                 site: str = "gripperframe",
                 rest: tuple[float, ...] = (0.0, -0.6, 0.9, 0.7, 0.0)):
        self.model = model
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + site)
        if self.site < 0:
            raise ValueError(f"site {prefix + site!r} not found")
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)
                  for n in ARM_JOINTS]
        if min(joints) < 0:
            raise ValueError("SO-101 arm joints not found (check prefix)")
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joints])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in joints])
        self.lower = model.jnt_range[joints, 0]
        self.upper = model.jnt_range[joints, 1]
        self.rest = np.asarray(rest, dtype=float)
        # the wrist roll is left free of the posture attraction: the jaw-span
        # objective owns it, and pulling it toward zero fights 180-degree spans
        self._posture_mask = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        self._scratch = mujoco.MjData(model)

    # -- kinematics helpers -------------------------------------------------

    def tool_pose(self, data: mujoco.MjData, offset: np.ndarray):
        """World position of `offset` (site frame) and the site rotation matrix."""
        mat = data.site_xmat[self.site].reshape(3, 3)
        return data.site_xpos[self.site] + mat @ offset, mat

    def forward(self, q: np.ndarray, data: mujoco.MjData, offset: np.ndarray):
        """Tool pose the arm would have at joint angles `q` (other DOFs from `data`)."""
        s = self._scratch
        s.qpos[:] = data.qpos
        s.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, s)
        return self.tool_pose(s, offset)

    # -- solver -------------------------------------------------------------

    def solve(self, data: mujoco.MjData, target: np.ndarray,
              offset: np.ndarray | None = None, keep_vertical: bool = True,
              span: np.ndarray | None = None, iterations: int = 120,
              tolerance: float = 1.5e-3, damping: float = 5e-3) -> IkResult:
        """Joint angles placing the tool point at `target` (world, meters).

        `span`, if given, is a horizontal unit vector the jaw-span axis (tool z,
        the direction the moving jaw opens toward) should align with; the wrist
        roll provides that freedom without disturbing the approach direction.
        A span opposite to the current roll is a saddle for the local solver,
        so the roll is multi-started (current, current + pi). A span beyond the
        roll's range is realized as far as the limit allows; the position and
        tilt objectives are never traded for it.
        """
        offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=float)
        target = np.asarray(target, dtype=float)
        span = None if span is None else np.asarray(span, dtype=float)
        q0 = data.qpos[self.qpos_adr].copy()
        seeds = [q0]
        if span is not None:
            flipped = q0.copy()
            flipped[4] = self._wrap_roll(q0[4] + np.pi)
            seeds.append(flipped)
        # keep the current roll branch unless the flipped one is clearly better:
        # flipping between consecutive waypoints would twist the wrist mid-motion
        best = None
        for i, seed in enumerate(seeds):
            res = self._iterate(data, seed, target, offset, keep_vertical, span,
                                iterations, tolerance, damping)
            if best is None or res[1] < (0.5 if i else 1.0) * best[1]:
                best = res
        q, _ = best
        pos, mat = self.forward(q, data, offset)
        tilt = float(np.arccos(np.clip(-mat[2, 0], -1.0, 1.0)))
        return IkResult(q=q, position_error=float(np.linalg.norm(target - pos)), tilt=tilt)

    def _wrap_roll(self, roll: float) -> float:
        lo, hi = self.lower[4], self.upper[4]
        while roll > hi:
            roll -= 2 * np.pi
        while roll < lo:
            roll += 2 * np.pi
        return float(np.clip(roll, lo, hi))

    def _iterate(self, data, q, target, offset, keep_vertical, span,
                 iterations, tolerance, damping):
        """Damped least squares from seed `q`; returns (q, objective)."""
        s = self._scratch
        s.qpos[:] = data.qpos
        body = self.model.site_bodyid[self.site]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        posture_iters = iterations - 30
        residual = np.zeros(3)

        for it in range(iterations):
            s.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(self.model, s)
            mujoco.mj_comPos(self.model, s)
            pos, mat = self.tool_pose(s, offset)
            approach = mat[:, 0]
            err = target - pos
            rows = [err]
            if keep_vertical:
                rows.append(0.5 * np.cross(approach, _DOWN)[:2])
            if span is not None:
                rows.append(0.6 * (span - mat[:2, 2]))   # unique minimum at z == span
            residual = np.concatenate(rows)
            if np.linalg.norm(err) < tolerance and (
                    not keep_vertical or np.linalg.norm(rows[1]) < 0.36) and (
                    span is None or np.linalg.norm(rows[-1]) < 0.06):
                break

            mujoco.mj_jac(self.model, s, jacp, jacr, pos, body)
            J = jacp[:, self.dof_adr]
            if keep_vertical:
                d_approach = np.cross(jacr[:, self.dof_adr].T, approach).T
                J_tilt = np.cross(d_approach.T, _DOWN).T * -0.5
                J = np.vstack([J, J_tilt[:2]])
            if span is not None:
                d_z = np.cross(jacr[:, self.dof_adr].T, mat[:, 2]).T   # d(tool z)/dq
                J_span = 0.6 * d_z[:2]
                # the span is the roll's job alone: when the roll saturates at
                # its limit the other joints must not bend the arm to serve it
                J_span[:, :4] = 0.0
                J = np.vstack([J, J_span])
            JT = J.T
            inv = np.linalg.inv(J @ JT + damping * np.eye(J.shape[0]))
            dq = JT @ (inv @ residual)
            if it < posture_iters:
                nullspace = np.eye(len(q)) - JT @ (inv @ J)
                dq += nullspace @ (0.15 * self._posture_mask * (self.rest - q))
            q = np.clip(q + dq, self.lower, self.upper)
        return q, float(np.linalg.norm(residual))
