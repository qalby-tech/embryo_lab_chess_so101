"""Empirical reach calibration: what fingertip accuracy does the arm ACHIEVE
at each square center (servo tracking included), not just what IK can solve.

Writes out/executed_reach.json: {"f,r": executed_err_m} and a text map.
The collectors use it to sample only squares the arm can really hit.
"""
import json

import mujoco
import numpy as np

import build_chess_scene as bc
from so101_ik import TIP_LEN, So101Ik

GRASP_H = 0.016
TOL_MM = 5.0


def main():
    spec = bc.build("8/8/8/8/8/8/8/8")
    model = spec.compile()
    data = mujoco.MjData(model)
    ik = So101Ik(model, prefix="so101:")
    board_top = bc.TABLE_TOP + bc.BOARD_T
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "so101:" + n)] for n in joints])
    aids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "so101:" + n) for n in joints]
    rest = np.array([0.0, -1.0, 1.3, 0.4, 0.0])
    loc = np.array([TIP_LEN, 0.0, 0.0])

    def go(q_goal, grip=0.3, n=40):
        """Command with the same integral-bias hold the collector uses."""
        bias = np.zeros(5)
        for round_i in range(12):
            full = np.concatenate([q_goal + bias, [grip]])
            for aid, t in zip(aids, full):
                data.ctrl[aid] = t
            for _ in range(8 * 16):
                mujoco.mj_step(model, data)
            resid = q_goal - data.qpos[qadr][:5]
            if np.max(np.abs(resid)) < 0.004:
                break
            bias = np.clip(bias + 0.6 * resid, -0.45, 0.45)

    # park
    data.qpos[qadr[:5]] = rest
    mujoco.mj_forward(model, data)
    go(rest)

    result, grid = {}, np.zeros((8, 8))
    for f in range(8):
        for r in range(8):
            x, y = bc.square_center(f, r)
            target = np.array([x, y, board_top + GRASP_H])
            q_hover, _ = ik.solve(data, target + [0, 0, 0.05])
            go(q_hover)
            q_goal, ik_err = ik.solve(data, target)
            go(q_goal)
            sm = data.site_xmat[ik.site].reshape(3, 3)
            tip = data.site_xpos[ik.site] + sm @ loc
            err = float(np.linalg.norm(tip - target))
            result[f"{f},{r}"] = err
            grid[r, f] = err * 1000
            go(q_hover)
    go(rest)

    ok = grid < TOL_MM
    print(f"executed-reachable (<{TOL_MM}mm): {int(ok.sum())}/64")
    for r in range(7, -1, -1):
        print(f"rank {r+1}: " + " ".join(f"{grid[r,f]:4.0f}" for f in range(8)))
    with open(f"{bc.OUT}/executed_reach.json", "w") as fp:
        json.dump(result, fp, indent=1)
    print("wrote", f"{bc.OUT}/executed_reach.json")


if __name__ == "__main__":
    main()
