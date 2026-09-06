"""Damped-least-squares IK for the SO-101 in MuJoCo.

Solves for the 5 arm joints (gripper excluded) to place the `gripperframe`
site at a world position with the tool axis pointing down (top-down grasp).
Works on any model that contains the (possibly prefixed) SO-101; the same
joint-space waypoints are executed in Genesis, which loads the same MJCF.
"""
import mujoco
import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP_LEN = 0.005  # gripperframe -> pinch point (measured from jaw mesh extents)


class So101Ik:
    def __init__(self, model: mujoco.MjModel, prefix: str = ""):
        self.m = model
        self.prefix = prefix
        self.site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, prefix + "gripperframe"
        )
        assert self.site >= 0, "gripperframe site not found"
        self.jids = []
        for n in ARM_JOINTS:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)
            assert j >= 0, f"joint {prefix + n} not found"
            self.jids.append(j)
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = model.jnt_range[self.jids, 0]
        self.hi = model.jnt_range[self.jids, 1]
        self.rest = np.array([0.0, -0.6, 0.9, 0.7, 0.0])

    def solve(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        down_axis: bool = True,
        iters: int = 120,
        tol: float = 1.5e-3,
        damping: float = 5e-3,
        z_off: float = 0.0,
    ):
        """Returns (q, err) — joint values for the 5 arm joints."""
        q = data.qpos[self.qadr].copy()
        scratch = mujoco.MjData(self.m)
        scratch.qpos[:] = data.qpos
        for it in range(iters):
            scratch.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, scratch)
            mujoco.mj_comPos(self.m, scratch)
            mat = scratch.site_xmat[self.site].reshape(3, 3)
            local_off = np.array([TIP_LEN, 0.0, z_off])
            pos = scratch.site_xpos[self.site] + mat @ local_off
            e_pos = target_pos - pos
            err_list = [e_pos]
            # tool x-axis is the approach axis: it should point down (world -z)
            if down_axis:
                z_tool = mat[:, 0]
                e_rot = np.cross(z_tool, np.array([0.0, 0.0, -1.0]))
                err_list.append(0.5 * e_rot[:2])  # xy components suffice
            err = np.concatenate(err_list)
            if np.linalg.norm(e_pos) < tol and (
                not down_axis or np.linalg.norm(err_list[1]) < 0.36
            ):
                break
            jacp = np.zeros((3, self.m.nv))
            jacr = np.zeros((3, self.m.nv))
            body_id = self.m.site_bodyid[self.site]
            mujoco.mj_jac(self.m, scratch, jacp, jacr, pos, body_id)
            J = jacp[:, self.vadr]
            if down_axis:
                # d(cross(x_tool,-z))/dq — use rotational jac rows
                zt = mat[:, 0]
                Jz = np.cross(jacr[:, self.vadr].T, zt).T  # d z_tool/dq
                Jrot = np.cross(Jz.T, np.array([0.0, 0.0, -1.0])).T * -0.5
                J = np.vstack([J, Jrot[:2]])
            JT = J.T
            JJT_inv = np.linalg.inv(J @ JT + damping * np.eye(J.shape[0]))
            q_delta = JT @ (JJT_inv @ err)
            # rest-posture attraction, nullspace-projected, and OFF for the final
            # refinement iterations: with a damped pseudo-inverse the projector
            # leaks into task space, which otherwise stalls convergence ~10mm out
            if it < iters - 30:
                N = np.eye(len(q)) - JT @ (JJT_inv @ J)
                q_delta += N @ (0.15 * (self.rest - q))
            q = np.clip(q + q_delta, self.lo, self.hi)
        scratch.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, scratch)
        mat = scratch.site_xmat[self.site].reshape(3, 3)
        tip = scratch.site_xpos[self.site] + mat @ np.array([TIP_LEN, 0.0, z_off])
        final_err = float(np.linalg.norm(target_pos - tip))
        return q, final_err
