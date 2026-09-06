"""SO-101 gripper calibration.

The SO-101 pinches tip-first: the prongs converge at the fingertips, so the
pinch pocket sits at the tips, offset half a jaw gap along the site z-axis.
Constants below were measured from the prong meshes (see `measure_pocket`).
Frame: the `gripperframe` site; +x is the approach axis, +z spans the jaw gap.
"""
from __future__ import annotations

import numpy as np

# Jaw gap at the fingertips as a function of the gripper joint angle (rad):
# gap ≈ GAP_AT_ZERO + GAP_PER_RAD * angle, valid over the closing range.
GAP_AT_ZERO = 0.0158
GAP_PER_RAD = 0.074
JOINT_MIN, JOINT_MAX = -0.17, 1.74

POCKET_X = -0.004    # pocket center along the approach axis: the fingertip pads' center
TIP_X = 0.008        # furthest fingertip point along the approach axis
TIP_CLEARANCE = 0.003  # keep fingertips at least this far above the board


# Fingertip contact pads (body frame of each jaw, measured at joint angle 0):
# thin boxes on the inner faces of the prong tips. The stock finger meshes
# collide as convex hulls, which fill the pinch pocket and make friction
# meaningless; the pads replace them for contact.
FINGER_PADS = {
    "gripper": {"pos": (-0.0099, -0.0002, -0.0941), "quat": (0.7071, 0.0, 0.7071, 0.0),
                "size": (0.010, 0.008, 0.002)},
    "moving_jaw_so101_v1": {"pos": (-0.0101, -0.0707, 0.019), "quat": (-0.5, 0.5, -0.5, 0.5),
                            "size": (0.010, 0.008, 0.002)},
}
FINGER_HULL_MESHES = ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1")
# Contact parameters that make a friction grasp hold: MuJoCo's default soft
# contact is mass-normalized and clamps a 4 g piece with ~0.1 N; the pads get
# an explicit stiffness (negative solref = N/m, N*s/m), full 6-D friction and
# priority over the piece's contact parameters.
PAD_FRICTION = (2.0, 0.05, 0.01)
PAD_SOLREF = (-50000.0, -500.0)
PAD_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2)
PAD_CONDIM = 6
PAD_PRIORITY = 2
SQUEEZE = 0.0025     # jaw closure past the piece surface for a physical grasp
# The moving jaw swings about its hinge, so its pad face tilts as it closes and
# the two pads form a wedge that ejects the piece. The moving pad is mounted
# rotated by the nominal grasp angle so the faces are parallel when closed.
NOMINAL_GRASP_ANGLE = -0.075   # rad, gripper joint angle at a typical pinch


def gap_to_angle(gap: float) -> float:
    """Gripper joint angle that yields a fingertip gap of `gap` meters."""
    return float(np.clip((gap - GAP_AT_ZERO) / GAP_PER_RAD, JOINT_MIN, JOINT_MAX))


def angle_to_gap(angle: float) -> float:
    return GAP_AT_ZERO + GAP_PER_RAD * angle


def pocket_offset(gap: float) -> np.ndarray:
    """Pinch-pocket center in the site frame for a given jaw gap."""
    return np.array([POCKET_X, 0.0, 0.5 * gap])


def measure_pocket(model, data, gripper_joint: str = "gripper",
                   site: str = "gripperframe") -> list[tuple[float, float, np.ndarray]]:
    """Re-derive the calibration from prong meshes: (angle, gap, pocket) rows.

    Closest point pair between the fixed and moving prongs, within the front
    4 cm of the fingers, expressed in the site frame.
    """
    import mujoco
    from scipy.spatial import cKDTree

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gripper_joint)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    fixed = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    moving = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")

    def prong_points(body_id, smat, spos):
        pts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body_id or model.geom_dataid[g] < 0:
                continue
            mid = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            pts.append((data.geom_xmat[g].reshape(3, 3) @ v.T).T + data.geom_xpos[g])
        local = (np.vstack(pts) - spos) @ smat
        return local[local[:, 0] > -0.04]

    rows = []
    for angle in np.linspace(0.3, JOINT_MIN, 8):
        data.qpos[model.jnt_qposadr[jid]] = angle
        mujoco.mj_kinematics(model, data)
        smat = data.site_xmat[sid].reshape(3, 3)
        spos = data.site_xpos[sid]
        fp, mp = prong_points(fixed, smat, spos), prong_points(moving, smat, spos)
        dist, idx = cKDTree(fp).query(mp)
        k = int(np.argmin(dist))
        rows.append((float(angle), float(dist[k]), 0.5 * (mp[k] + fp[idx[k]])))
    return rows
