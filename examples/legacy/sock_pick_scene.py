"""Genesis scene: SO-101 picks a sock (cloth) off a table and drops it in a box.

The sock is a deformable PBD cloth entity. Until the EmbodiedGen-generated sock
asset lands, --placeholder builds a sock-shaped shell mesh procedurally so the
whole pipeline (cloth sim, arm control, rendering) can run end to end.

Run: MUJOCO_GL=egl not needed; Genesis renders with its own rasterizer.
    python sock_pick_scene.py [--placeholder] [--steps N]
"""
import argparse
import os

import genesis as gs
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SO101_XML = "/home/kamil/chess_so101/so_arm_repo/Simulation/SO101/so101_new_calib.xml"
SOCK_RESULT = "/home/kamil/EmbodiedGen/outputs/sock_asset/asset3d/sock/result"

TABLE_H = 0.0          # arm and objects sit on the ground plane (table frame)
SOCK_POS = (0.18, 0.05, 0.02)
BOX_POS = (0.12, -0.16, 0.0)
BOX_IN = 0.07          # inner half-extent of the open box
BOX_WALL = 0.008
BOX_H = 0.06


def make_placeholder_sock(path: str) -> str:
    """Sock-ish open shell: bent tube (leg + foot) as a thin cloth surface."""
    leg = trimesh.creation.cylinder(radius=0.022, height=0.09, sections=24)
    leg.apply_translation([0, 0, 0.045])
    foot = trimesh.creation.capsule(radius=0.022, height=0.07, count=[12, 24])
    foot.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    )
    foot.apply_translation([0.035, 0, 0.022])
    sock = trimesh.util.concatenate([leg, foot])
    sock.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    )  # lay flat
    sock.apply_scale(1.0)
    sock.export(path)
    return path


def build_scene(sock_mesh: str, sock_scale: float):
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=8),
        pbd_options=gs.options.PBDOptions(particle_size=6e-3),
        viewer_options=gs.options.ViewerOptions(),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    # open-top box from 5 static boxes
    for dx, dy, sx, sy in [
        (0, 0, BOX_IN, BOX_IN),                                  # bottom
        (BOX_IN + BOX_WALL / 2, 0, BOX_WALL / 2, BOX_IN + BOX_WALL),
        (-(BOX_IN + BOX_WALL / 2), 0, BOX_WALL / 2, BOX_IN + BOX_WALL),
        (0, BOX_IN + BOX_WALL / 2, BOX_IN + BOX_WALL, BOX_WALL / 2),
        (0, -(BOX_IN + BOX_WALL / 2), BOX_IN + BOX_WALL, BOX_WALL / 2),
    ]:
        is_bottom = sx == BOX_IN and sy == BOX_IN
        h = BOX_WALL / 2 if is_bottom else BOX_H / 2
        z = BOX_WALL / 2 if is_bottom else BOX_H / 2
        scene.add_entity(
            gs.morphs.Box(
                pos=(BOX_POS[0] + dx, BOX_POS[1] + dy, z),
                size=(2 * sx, 2 * sy, 2 * h),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.72, 0.55, 0.35, 1.0)),
        )

    sock = scene.add_entity(
        material=gs.materials.PBD.Cloth(),
        morph=gs.morphs.Mesh(
            file=sock_mesh,
            scale=sock_scale,
            pos=SOCK_POS,
        ),
        surface=gs.surfaces.Default(),
    )

    arm = scene.add_entity(gs.morphs.MJCF(file=SO101_XML))

    cam = scene.add_camera(
        res=(1280, 720), pos=(0.55, 0.55, 0.45), lookat=(0.1, -0.02, 0.1), fov=42
    )
    scene.build()
    return scene, arm, sock, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placeholder", action="store_true")
    ap.add_argument("--steps", type=int, default=900)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.placeholder or not os.path.exists(f"{SOCK_RESULT}/mesh/sock.obj"):
        sock_mesh = make_placeholder_sock(os.path.join(OUT, "sock_placeholder.obj"))
        scale = 1.0
        print("using placeholder sock")
    else:
        sock_mesh = f"{SOCK_RESULT}/mesh/sock.obj"
        m = trimesh.load(sock_mesh, force="mesh")
        scale = 0.16 / float(max(m.extents))  # ~16cm sock
        print("using generated sock asset, scale", scale)

    scene, arm, sock, cam = build_scene(sock_mesh, scale)

    # simple scripted reach: hover over sock, descend, close, lift, move over box, open
    jnames = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
              "wrist_roll", "gripper"]
    dofs = [arm.get_joint(n).dof_idx_local for n in jnames]
    key_frames = [
        (0.00, [0.28, -0.10, 0.50, 0.90, 0.0, 1.5]),   # hover above sock, open
        (0.25, [0.28, 0.42, 0.85, 0.95, 0.0, 1.5]),    # descend
        (0.40, [0.28, 0.42, 0.85, 0.95, 0.0, -0.1]),   # close jaw
        (0.55, [0.28, -0.35, 0.45, 0.80, 0.0, -0.1]),  # lift
        (0.75, [-0.85, -0.25, 0.55, 0.85, 0.0, -0.1]), # rotate over box
        (0.88, [-0.85, -0.25, 0.55, 0.85, 0.0, 1.5]),  # open, drop
        (1.00, [-0.85, -0.60, 0.30, 0.60, 0.0, 1.5]),  # retreat
    ]

    frames = []
    n = args.steps
    for i in range(n):
        t = i / n
        # piecewise-linear joint targets
        for (t0, q0), (t1, q1) in zip(key_frames[:-1], key_frames[1:]):
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0 + 1e-9)
                q = [x + a * (y - x) for x, y in zip(q0, q1)]
                arm.control_dofs_position(np.array(q), dofs)
                break
        scene.step()
        if i % 6 == 0:
            rgb = cam.render()[0]
            frames.append(rgb)

    import imageio.v2 as imageio
    imageio.mimsave(os.path.join(OUT, "sock_pick.mp4"), frames, fps=25)
    imageio.imwrite(os.path.join(OUT, "sock_pick.png"), frames[len(frames) // 3])
    print("wrote", os.path.join(OUT, "sock_pick.mp4"))


if __name__ == "__main__":
    main()
