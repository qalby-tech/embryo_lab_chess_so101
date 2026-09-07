"""Inverse kinematics for the SO-101's five arm joints, solved with mink.

A solve is a short sequence of quadratic programs over the arm-only kinematic
model (see `scene.build_arm`): a tool point (an offset from the `gripperframe`
site, e.g. the pinch pocket) is driven to a world position while the approach
axis (site +x) is kept close to vertical, the jaw-span axis (site +z) is turned
toward a requested horizontal direction by the wrist roll alone, and a weak
posture term regularizes the remaining redundancy. Joint limits, a step bound
and self-collision avoidance between the hand and the arm's links are hard
inequality constraints of each QP.
"""
from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
_DOWN = np.array([0.0, 0.0, -1.0])

POSITION_COST = 1.0    # tool-point error, per meter
TILT_COST = 0.5        # approach-axis direction error (unitless)
SPAN_COST = 0.6        # jaw-span direction error, horizontal components (unitless)
POSTURE_COST = 0.003   # tie-breaker toward the rest posture, per radian (roll excluded)
ITERATIONS = 60
TOLERANCE = 1.5e-3     # meters; convergence test for the tool point
TILT_EXIT = 0.7        # axis-align error chord 2*sin(theta/2): about 41 degrees
SPAN_EXIT = 0.1        # jaw-span error, horizontal chord: about 6 degrees at zero tilt
STALL = 1e-6           # radians; a QP step this small means a KKT point (e.g. roll at its limit)
DAMPING = 1e-6         # Levenberg-Marquardt damping on every QP
STEP_LIMIT = 0.5       # radians per joint per iteration: bounds the Gauss-Newton step
                       # through singular poses (the zero pose is a straight vertical stack)
ROLL_STAY = np.pi / 2  # a roll branch that moves less than this keeps the "no wrist twist" bonus
ROLL_MARGIN = 0.15     # radians kept between planned roll angles and the joint limits
# self-collision avoidance: the hand must keep clear of the shoulder and upper
# arm (a tight fold over the near squares otherwise drives the gripper into the
# shoulder). The lower arm is left out: at full wrist flex, needed for the far
# corners, the hand legitimately sits within millimeters of it.
HAND_BODIES = ("gripper", "moving_jaw_so101_v1")
ARM_BODIES = ("shoulder", "upper_arm")
COLLISION_MARGIN = 0.005   # meters kept between hand and arm geoms
COLLISION_DETECT = 0.03    # meters at which the avoidance constraint activates
SOLVER = "daqp"


@dataclass(frozen=True)
class IkResult:
    q: np.ndarray            # the five arm joint angles
    position_error: float    # meters, tool point vs target
    tilt: float              # radians between the approach axis and straight down


class _PointTask(mink.Task):
    """World position of a point fixed in a site frame."""

    k = 3

    def __init__(self, model: mujoco.MjModel, site: int, cost: float):
        super().__init__(cost=np.full(self.k, cost))
        self.site = site
        self.body = int(model.site_bodyid[site])
        self.offset = np.zeros(3)
        self.target = np.zeros(3)
        self._jacp = np.zeros((3, model.nv))

    def point(self, configuration: mink.Configuration) -> np.ndarray:
        data = configuration.data
        return data.site_xpos[self.site] + data.site_xmat[self.site].reshape(3, 3) @ self.offset

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        return self.point(configuration) - self.target

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        mujoco.mj_jac(configuration.model, configuration.data, self._jacp, None,
                      self.point(configuration), self.body)
        return self._jacp.copy()


class _SpanTask(mink.Task):
    """Horizontal direction of a site's z axis, served by one joint only."""

    k = 2

    def __init__(self, model: mujoco.MjModel, site: int, dof: int, cost: float):
        super().__init__(cost=np.full(self.k, cost))
        self.site = site
        self.dof = dof
        self.target = np.zeros(2)
        self._jacr = np.zeros((3, model.nv))

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        z = configuration.data.site_xmat[self.site].reshape(3, 3)[:, 2]
        return z[:2] - self.target

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        mujoco.mj_jacSite(configuration.model, configuration.data, None, self._jacr, self.site)
        z = configuration.data.site_xmat[self.site].reshape(3, 3)[:, 2]
        jac = np.zeros((self.k, self._jacr.shape[1]))
        jac[:, self.dof] = np.cross(self._jacr[:, self.dof], z)[:2]   # d z / d q_dof
        return jac


class So101Ik:
    """`model` is the scene (read joint angles from its data); `arm_model` is
    the arm alone (`scene.build_arm`), mounted identically, that the QPs run on.
    Joint and site names carry the same `prefix` in both."""

    def __init__(self, model: mujoco.MjModel, arm_model: mujoco.MjModel, prefix: str = "",
                 site: str = "gripperframe",
                 rest: tuple[float, ...] = (0.0, -0.6, 0.9, 0.7, 0.0)):
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + site)
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n) for n in ARM_JOINTS]
        if self.site < 0 or min(joints) < 0:
            raise ValueError("SO-101 site or arm joints not found in the scene (check prefix)")
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joints])

        self.arm = arm_model
        self.configuration = mink.Configuration(arm_model)
        arm_site = mujoco.mj_name2id(arm_model, mujoco.mjtObj.mjOBJ_SITE, prefix + site)
        arm_joints = [mujoco.mj_name2id(arm_model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n) for n in ARM_JOINTS]
        if arm_site < 0 or min(arm_joints) < 0:
            raise ValueError("SO-101 site or arm joints not found in the arm model (check prefix)")
        self._arm_site = arm_site
        self._arm_qpos = np.array([arm_model.jnt_qposadr[j] for j in arm_joints])
        self._arm_dof = np.array([arm_model.jnt_dofadr[j] for j in arm_joints])
        # plan the roll a little inside its range: a target exactly on the joint
        # limit is one the position servo cannot settle on, and a span in the
        # roll's unreachable arc would otherwise be planned right onto the limit
        arm_model.jnt_range[arm_joints[4], 0] += ROLL_MARGIN
        arm_model.jnt_range[arm_joints[4], 1] -= ROLL_MARGIN
        self.lower = arm_model.jnt_range[arm_joints, 0].copy()
        self.upper = arm_model.jnt_range[arm_joints, 1].copy()

        self._point = _PointTask(arm_model, arm_site, POSITION_COST)
        self._tilt = mink.AxisAlignTask(arm_site, "site", axis=(1.0, 0.0, 0.0),
                                        cost=TILT_COST, lm_damping=0.1)
        self._tilt.set_target(_DOWN)
        self._span = _SpanTask(arm_model, arm_site, int(self._arm_dof[4]), SPAN_COST)
        # the wrist roll is left free of the posture pull: the span objective
        # owns it, and pulling it toward zero fights 180-degree spans
        posture_cost = np.zeros(arm_model.nv)
        posture_cost[self._arm_dof[:4]] = POSTURE_COST
        self._posture = mink.PostureTask(arm_model, cost=posture_cost)
        q_rest = self.configuration.q
        q_rest[self._arm_qpos] = rest
        self._posture.set_target(q_rest)
        hand = [g for b in HAND_BODIES for g in mink.get_body_geom_ids(arm_model, arm_model.body(prefix + b).id)]
        arm = [g for b in ARM_BODIES for g in mink.get_body_geom_ids(arm_model, arm_model.body(prefix + b).id)]
        self._limits = [mink.ConfigurationLimit(arm_model),
                        mink.VelocityLimit(arm_model, {prefix + n: STEP_LIMIT for n in ARM_JOINTS}),
                        mink.CollisionAvoidanceLimit(arm_model, [(hand, arm)],
                                                     minimum_distance_from_collisions=COLLISION_MARGIN,
                                                     collision_detection_distance=COLLISION_DETECT)]

    # -- kinematics helpers -------------------------------------------------

    def tool_pose(self, data: mujoco.MjData, offset: np.ndarray):
        """World position of `offset` (site frame) and the site rotation matrix,
        from the scene's data."""
        mat = data.site_xmat[self.site].reshape(3, 3)
        return data.site_xpos[self.site] + mat @ offset, mat

    def forward(self, q: np.ndarray, offset: np.ndarray):
        """Tool pose the arm would have at joint angles `q`."""
        self._set_arm(q)
        data = self.configuration.data
        mat = data.site_xmat[self._arm_site].reshape(3, 3)
        return data.site_xpos[self._arm_site] + mat @ offset, mat

    def _set_arm(self, q: np.ndarray) -> None:
        full = self.arm.qpos0.copy()     # every other dof (the gripper) at its default
        full[self._arm_qpos] = q
        self.configuration.update(full)

    # -- solver -------------------------------------------------------------

    def solve(self, data: mujoco.MjData, target: np.ndarray,
              offset: np.ndarray | None = None, span: np.ndarray | None = None) -> IkResult:
        """Joint angles placing the tool point at `target` (world, meters),
        starting from the arm's current angles in `data`.

        `span`, if given, is the horizontal direction (x, y) the jaw-span axis
        (tool z, the direction the moving jaw opens toward) should align with;
        the wrist roll provides that freedom without disturbing the approach
        direction. A span opposite to the current roll is a saddle for the
        local solver, so the roll is multi-started (current, current + pi). The
        span objective acts on the roll alone: a span beyond the roll's range
        is realized as far as the limit allows (the 320-degree range puts every
        direction within ~20 degrees of a reachable roll, a little more when
        staying on the current roll branch avoids a wrist flip mid-carry), and
        the other joints are never bent to serve it.
        """
        offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=float)
        target = np.asarray(target, dtype=float)
        if span is not None:
            span = np.asarray(span, dtype=float)[:2]
            span = span / np.linalg.norm(span)
        if not (np.all(np.isfinite(target)) and np.all(np.isfinite(offset))
                and (span is None or np.all(np.isfinite(span)))):
            raise ValueError("IK target, offset and span must be finite")
        q0 = data.qpos[self.qpos_adr].copy()
        best = self._iterate(q0, target, offset, span)
        if span is not None:
            flipped = q0.copy()
            flipped[4] = self._wrap_roll(q0[4] + np.pi)
            other = self._iterate(flipped, target, offset, span)
            # keep the current roll branch unless the flipped one is clearly
            # better: flipping between consecutive waypoints would twist the
            # wrist mid-motion. When the current branch itself had to travel far
            # (both branches pinned at a limit across the unreachable arc) there
            # is no twist to avoid, and the better residual simply wins.
            stays = abs(best[0][4] - q0[4]) < ROLL_STAY
            if other[1] < (0.5 if stays else 1.0) * best[1]:
                best = other
        q, _ = best
        pos, mat = self.forward(q, offset)
        tilt = float(np.arccos(np.clip(-mat[2, 0], -1.0, 1.0)))
        return IkResult(q=q, position_error=float(np.linalg.norm(target - pos)), tilt=tilt)

    def _wrap_roll(self, roll: float) -> float:
        """Equivalent angle nearest the roll range's center, clipped into range:
        an angle in the unreachable arc lands on the nearer limit."""
        lo, hi = self.lower[4], self.upper[4]
        mid = 0.5 * (lo + hi)
        roll = mid + (roll - mid + np.pi) % (2 * np.pi) - np.pi
        return float(np.clip(roll, lo, hi))

    def _iterate(self, seed, target, offset, span):
        """QP iterations from `seed`; returns (q, weighted residual norm)."""
        cfg = self.configuration
        self._set_arm(seed)
        self._point.offset = offset
        self._point.target = target
        tasks = [self._point, self._tilt, self._posture]
        if span is not None:
            self._span.target = span
            tasks.append(self._span)
        for _ in range(ITERATIONS):
            if (np.linalg.norm(self._point.compute_error(cfg)) < TOLERANCE
                    and np.linalg.norm(self._tilt.compute_error(cfg)) < TILT_EXIT
                    and (span is None or np.linalg.norm(self._span.compute_error(cfg)) < SPAN_EXIT)):
                break
            # dt = 1: the QP solves for the joint displacement of this iteration
            v = mink.solve_ik(cfg, tasks, dt=1.0, solver=SOLVER, damping=DAMPING, limits=self._limits)
            if np.abs(v).max() < STALL:
                break
            cfg.integrate_inplace(v, 1.0)
        residual = [POSITION_COST * self._point.compute_error(cfg), TILT_COST * self._tilt.compute_error(cfg)]
        if span is not None:
            residual.append(SPAN_COST * self._span.compute_error(cfg))
        return cfg.q[self._arm_qpos].copy(), float(np.linalg.norm(np.concatenate(residual)))
