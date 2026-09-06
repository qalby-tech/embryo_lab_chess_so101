"""Scripted-expert episode collector: SO-101 puts the sock in the box (Genesis).

Sock = PBD cloth (EmbodiedGen-generated mesh). Grasp = pinch + pinning the
nearest cloth particles to the gripper link; release drops them. Joint-space
waypoints come from the shared MuJoCo IK (same MJCF → same kinematics).
Records the same npz/mp4/meta format as collect_chess.py.

Usage: python collect_sock.py --episodes 3 [--out ~/chess_so101/datasets/sock]
"""
import argparse
import json
import math
import os
import random

import genesis as gs
import imageio.v2 as imageio
import mujoco
import numpy as np

from so101_ik import So101Ik, TIP_LEN

SO101_XML = "/home/kamil/chess_so101/so_arm_repo/Simulation/SO101/so101_new_calib.xml"
SOCK_OBJ = "/home/kamil/EmbodiedGen/outputs/sock_asset/asset3d/sock/result/mesh/sock.obj"
CTRL_HZ = 30
STEPS_PER_CTRL = 16  # dt=2e-3 -> ~30Hz control
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIP_OPEN, GRIP_CLOSED = 1.2, 0.0
ARM_BASE = np.array([0.0, 0.0, 0.10])  # riser, mirrors chess layout
BOX_POS = (0.10, -0.17)
BOX_IN, BOX_WALL, BOX_H = 0.07, 0.008, 0.055


def build_scene(sock_pos, sock_yaw, sock_scale):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=8),
        pbd_options=gs.options.PBDOptions(particle_size=7e-3),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    for dx, dy, sx, sy in [
        (0, 0, BOX_IN, BOX_IN),
        (BOX_IN + BOX_WALL / 2, 0, BOX_WALL / 2, BOX_IN + BOX_WALL),
        (-(BOX_IN + BOX_WALL / 2), 0, BOX_WALL / 2, BOX_IN + BOX_WALL),
        (0, BOX_IN + BOX_WALL / 2, BOX_IN + BOX_WALL, BOX_WALL / 2),
        (0, -(BOX_IN + BOX_WALL / 2), BOX_IN + BOX_WALL, BOX_WALL / 2),
    ]:
        bottom = sx == BOX_IN and sy == BOX_IN
        h = BOX_WALL / 2 if bottom else BOX_H / 2
        scene.add_entity(
            gs.morphs.Box(pos=(BOX_POS[0] + dx, BOX_POS[1] + dy, h),
                          size=(2 * sx, 2 * sy, 2 * h), fixed=True),
            surface=gs.surfaces.Default(color=(0.70, 0.53, 0.34, 1.0)),
        )
    sock = scene.add_entity(
        material=gs.materials.PBD.Cloth(),
        morph=gs.morphs.Mesh(file=SOCK_OBJ, scale=sock_scale, pos=sock_pos,
                             euler=(0, 0, math.degrees(sock_yaw))),
        surface=gs.surfaces.Default(),
    )
    scene.add_entity(
        gs.morphs.Box(pos=(0, 0, ARM_BASE[2] / 2), size=(0.11, 0.11, ARM_BASE[2]),
                      fixed=True),
        surface=gs.surfaces.Default(color=(0.25, 0.25, 0.28, 1.0)),
    )
    arm = scene.add_entity(gs.morphs.MJCF(file=SO101_XML, pos=tuple(ARM_BASE)))
    cam = scene.add_camera(res=(640, 480), pos=(0.42, 0.40, 0.35),
                           lookat=(0.13, -0.04, 0.05), fov=42)
    scene.build()
    return scene, arm, sock, cam


def run_episode(ep_idx, rng, out_dir):
    # MuJoCo twin for IK
    mj_model = mujoco.MjModel.from_xml_path(SO101_XML)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    ik = So101Ik(mj_model, prefix="")

    # sample a sock pose that the arm can actually reach
    for _ in range(40):
        ang = rng.uniform(0.35, 1.15)  # polar angle from +x toward +y
        rad = rng.uniform(0.17, 0.25)
        sock_pos = (rad * math.cos(ang - 0.5), rad * math.sin(ang - 0.5), 0.015)
        q_test, err = ik.solve(mj_data, np.array([sock_pos[0], sock_pos[1], 0.012]) - ARM_BASE)
        if err < 0.008:
            break
    else:
        return None
    sock_yaw = rng.uniform(0, 2 * math.pi)

    import trimesh
    scale = 0.16 / float(max(trimesh.load(SOCK_OBJ, force="mesh").extents))
    scene, arm, sock, cam = build_scene(sock_pos, sock_yaw, scale)
    dofs = [arm.get_joint(n).dof_idx_local for n in JOINTS]
    gripper_link_idx = arm.get_link("gripper").idx

    # settle cloth
    arm.control_dofs_position(np.array([0.0, -1.0, 1.3, 0.4, 0.0, GRIP_OPEN]), dofs)
    for _ in range(150):
        scene.step()

    centroid = sock.get_particles_pos().cpu().numpy().mean(axis=0)
    gx, gy = float(centroid[0]), float(centroid[1])
    zg, zh = 0.012, 0.12
    box_hover = np.array([BOX_POS[0], BOX_POS[1], 0.16])

    plan = []  # (mujoco target, grip, n_ctrl, action)
    plan.append((np.array([gx, gy, zh]), GRIP_OPEN, 25, None))
    plan.append((np.array([gx, gy, zg]), GRIP_OPEN, 20, None))
    plan.append((np.array([gx, gy, zg]), GRIP_CLOSED, 8, "pin"))
    plan.append((np.array([gx, gy, zh]), GRIP_CLOSED, 18, None))
    plan.append((box_hover, GRIP_CLOSED, 25, None))
    plan.append((box_hover, GRIP_OPEN, 8, "release"))
    plan.append((np.array([box_hover[0] - 0.06, box_hover[1] + 0.10, 0.20]), GRIP_OPEN, 14, "retreat"))

    obs_state, actions, frames = [], [], []
    q_now = np.array([0.0, -1.0, 1.3, 0.4, 0.0])
    pinned = None
    ok = True
    for target, grip, n_ctrl, act in plan:
        mj_data.qpos[ik.qadr] = q_now
        q_goal, err = ik.solve(mj_data, target - ARM_BASE,
                               down_axis=(act != "retreat"))
        # only the pinch approach needs metric precision; transit waypoints are
        # verified by the final in-box outcome check instead
        if err > 0.012 and abs(float(target[2]) - zg) < 1e-9:
            ok = False
        if act == "pin":
            tip = target  # commanded pinch point
            pos_t = sock.get_particles_pos().cpu().numpy()
            d = np.linalg.norm(pos_t - np.array(tip), axis=1)
            pinned = np.argsort(d)[:6].tolist()
            sock.fix_particles_to_link(gripper_link_idx, particles_idx_local=pinned)
        for i in range(1, n_ctrl + 1):
            q = q_now + (q_goal - q_now) * (i / n_ctrl)
            full = np.concatenate([q, [grip]])
            state = arm.get_dofs_position(dofs).cpu().numpy()
            obs_state.append(state)
            actions.append(full.copy())
            arm.control_dofs_position(full, dofs)
            for _ in range(STEPS_PER_CTRL):
                scene.step()
            frames.append(cam.render()[0])
        if act == "release" and pinned is not None:
            sock.release_particle(particles_idx_local=pinned)
            pinned = None
            for _ in range(45):  # let it fall
                scene.step()
                frames.append(cam.render()[0]) if len(frames) % 1 == 0 else None
        q_now = q_goal.copy()

    final = sock.get_particles_pos().cpu().numpy().mean(axis=0)
    in_box = (abs(final[0] - BOX_POS[0]) < BOX_IN and
              abs(final[1] - BOX_POS[1]) < BOX_IN and final[2] < BOX_H + 0.02)
    success = bool(ok and in_box)

    ep_dir = os.path.join(out_dir, f"episode_{ep_idx:04d}")
    os.makedirs(ep_dir, exist_ok=True)
    np.savez_compressed(os.path.join(ep_dir, "data.npz"),
                        observation_state=np.array(obs_state, dtype=np.float32),
                        action=np.array(actions, dtype=np.float32))
    imageio.mimsave(os.path.join(ep_dir, "external.mp4"), frames, fps=CTRL_HZ)
    meta = {"task": "sock_in_box", "instruction": "pick up the sock and put it in the box",
            "sock_pos": [round(v, 4) for v in sock_pos], "sock_yaw": round(sock_yaw, 3),
            "box_pos": list(BOX_POS), "success": success,
            "final_centroid": [round(float(v), 4) for v in final],
            "fps": CTRL_HZ, "n_steps": len(actions), "joints": JOINTS,
            "grasp_mode": "particle_pin"}
    with open(os.path.join(ep_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="/home/kamil/chess_so101/datasets/sock")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gs.init(backend=gs.gpu, logging_level="warning")
    rng = random.Random(args.seed)
    manifest = open(os.path.join(args.out, "manifest.jsonl"), "a")
    n_ok = 0
    for i in range(args.episodes):
        meta = run_episode(i, rng, args.out)
        if meta is None:
            print(f"ep {i}: no reachable sock pose, skipped")
            continue
        manifest.write(json.dumps({"episode": i, **meta}) + "\n")
        manifest.flush()
        n_ok += meta["success"]
        print(f"ep {i}: success={meta['success']} final={meta['final_centroid']}")
    print(f"done: {n_ok}/{args.episodes} verified successes")


if __name__ == "__main__":
    main()
