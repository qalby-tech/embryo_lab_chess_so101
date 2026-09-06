"""Reachability audit: IK over all 64 square centers of the chess board."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import build_chess_scene as bc
from so101_ik import So101Ik

GRASP_H = 0.020  # pinch height above board top


def audit(square, arm_y, arm_z, tilt_ok=0.35):
    """Count reachable squares for a config. tilt_ok: max |cross(z_tool,-z)|."""
    bc.SQUARE = square
    bc.FIELD = 8 * square
    bc.BOARD_W = bc.FIELD + 2 * bc.BORDER
    spec = bc.build("8/8/8/8/8/8/8/8")
    # move the arm frame (last frame added is the arm's)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ik = So101Ik(model, prefix="so101:")
    board_top = bc.TABLE_TOP + bc.BOARD_T
    ok = 0
    for f in range(8):
        for r in range(8):
            x, y = bc.square_center(f, r)
            q, err = ik.solve(data, np.array([x, y, board_top + GRASP_H]))
            if err < 0.008:
                ok += 1
    return ok


def sweep():
    import itertools
    best = []
    for square, dy, dz in itertools.product(
        [0.022, 0.025, 0.028], [0.0, 0.02, 0.04, 0.08], [0.10, 0.14, 0.18]
    ):
        bc.ARM_OFFSET_Y = dy
        bc.ARM_OFFSET_Z = dz
        n = audit(square, dy, dz)
        best.append((n, square, dy, dz))
        print(f"square={square*100:.1f}cm arm_gap={dy*100:.0f}cm riser={dz*100:.0f}cm -> {n}/64")
    best.sort(reverse=True)
    print("BEST:", best[:5])


def main():
    spec = bc.build("8/8/8/8/8/8/8/8")  # empty board
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ik = So101Ik(model, prefix="so101:")

    board_top = bc.TABLE_TOP + bc.BOARD_T
    errs = np.zeros((8, 8))
    for f in range(8):
        for r in range(8):
            x, y = bc.square_center(f, r)
            _, err = ik.solve(data, np.array([x, y, board_top + GRASP_H]))
            errs[r, f] = err
    ok = errs < 0.008
    print("reachable squares:", int(ok.sum()), "/ 64")
    for r in range(7, -1, -1):
        print(f"rank {r+1}: " + " ".join("O" if ok[r, f] else "." for f in range(8)))

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(errs * 1000, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=30)
    ax.set_xticks(range(8), list("abcdefgh"))
    ax.set_yticks(range(8), [str(i + 1) for i in range(8)])
    fig.colorbar(im, label="IK position error (mm)")
    ax.set_title("SO-101 reachability (arm at white edge)")
    fig.savefig(f"{bc.OUT}/reach_map.png", dpi=120, bbox_inches="tight")
    print("wrote", f"{bc.OUT}/reach_map.png")


if __name__ == "__main__":
    import sys
    sweep() if "--sweep" in sys.argv else main()
