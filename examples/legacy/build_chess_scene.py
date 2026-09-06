"""Deterministic chess scene for the SO-101 in MuJoCo.

Builds a procedurally exact 8x8 board (known square coordinates, FEN-driven piece
placement) using EmbodiedGen-generated piece assets, the generated room panorama
as a surrounding backdrop, and the official SO-101 model. Run inside the
`embodiedgen` conda env with MUJOCO_GL=egl.

Usage:
    python build_chess_scene.py [--fen "<FEN board field>"] [--settle_s 2.0]
"""
import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TASK = "/home/kamil/EmbodiedGen/outputs/chess_layout/task_0000"
CHESS_SET = "/home/kamil/EmbodiedGen/outputs/chess_set/asset3d"
SO101_XML = "/home/kamil/chess_so101/so_arm_repo/Simulation/SO101/so101_new_calib.xml"
PANO = f"{TASK}/background/pano_image.png"

# --- geometry constants (meters); mini board sized for SO-101's ~30cm reach ---
SQUARE = 0.028
BORDER = 0.015
BOARD_T = 0.012
FIELD = 8 * SQUARE                      # 0.28
BOARD_W = FIELD + 2 * BORDER            # 0.31
TABLE_TOP = 0.43                        # table height we build (box table)
ARM_OFFSET_Y = 0.04                     # gap between board edge and arm base
ARM_OFFSET_Z = 0.14                     # arm base riser height
# Staunton-ish piece heights for a mini set, keyed by FEN letter (lowercase)
PIECE_H = {"p": 0.034, "r": 0.038, "n": 0.042, "b": 0.046, "q": 0.052, "k": 0.058}

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

# FEN letter -> generated asset directory (filled progressively as assets land)
ASSET_DIRS = {
    "P": f"{CHESS_SET}/white_pawn/result",
    "R": f"{CHESS_SET}/white_rook/result",
    "N": f"{TASK}/asset3d/white_knight_chess_piece/result",
    "B": f"{CHESS_SET}/white_bishop/result",
    "Q": f"{CHESS_SET}/white_queen/result",
    "K": f"{CHESS_SET}/white_king/result",
    "p": f"{CHESS_SET}/black_pawn/result",
    "r": f"{CHESS_SET}/black_rook/result",
    "n": f"{CHESS_SET}/black_knight/result",
    "b": f"{CHESS_SET}/black_bishop/result",
    "q": f"{CHESS_SET}/black_queen/result",
    "k": f"{TASK}/asset3d/black_king_chess_piece/result",
}


def square_center(file_idx: int, rank_idx: int) -> tuple[float, float]:
    """Board-frame xy of a square center. file a..h -> 0..7, rank 1..8 -> 0..7."""
    x = (file_idx - 3.5) * SQUARE
    y = (rank_idx - 3.5) * SQUARE
    return x, y


def parse_fen_board(fen: str) -> list[tuple[str, int, int]]:
    """Yields (piece_letter, file_idx, rank_idx) from the FEN board field."""
    out = []
    for r, row in enumerate(fen.split()[0].split("/")):
        rank_idx = 7 - r
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                out.append((ch, f, rank_idx))
                f += 1
    return out


def make_board_texture(path: str, px_per_square: int = 128) -> None:
    """Wood checkerboard texture with border, light/dark squares."""
    rng = np.random.default_rng(0)
    n = 8 * px_per_square
    b = int(px_per_square * BORDER / SQUARE)
    size = n + 2 * b
    img = np.zeros((size, size, 3), np.float32)

    def wood(shape, base, var=14.0):
        h, w = shape
        grain = np.cumsum(rng.normal(0, 1, (h, w)), axis=1)
        grain = (grain - grain.min()) / (grain.ptp() + 1e-6) - 0.5
        rows = rng.normal(0, 1, (h, 1)) * 0.35
        return np.clip(base + (grain + rows) * var, 0, 255)

    border_rgb = (92, 58, 32)
    light_rgb = (214, 178, 132)
    dark_rgb = (99, 64, 40)
    for c in range(3):
        img[..., c] = wood((size, size), border_rgb[c])
    for i in range(8):
        for j in range(8):
            base = light_rgb if (i + j) % 2 else dark_rgb
            y0, x0 = b + i * px_per_square, b + j * px_per_square
            for c in range(3):
                img[y0:y0 + px_per_square, x0:x0 + px_per_square, c] = wood(
                    (px_per_square, px_per_square), base[c]
                )
    Image.fromarray(img.astype(np.uint8)).save(path)


def make_pano_cylinder(obj_path: str, radius: float = 3.5, height: float = 3.4) -> None:
    """Inward-facing cylinder around the scene, UV-mapped to the room panorama."""
    seg = 96
    ang = np.linspace(0, 2 * np.pi, seg + 1)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * radius
    v_bot = np.column_stack([ring, np.zeros(seg + 1)])
    v_top = np.column_stack([ring, np.full(seg + 1, height)])
    verts = np.vstack([v_bot, v_top])
    faces, uv = [], []
    for i in range(seg + 1):
        uv.append([1.0 - i / seg, 0.0])
    for i in range(seg + 1):
        uv.append([1.0 - i / seg, 1.0])
    for i in range(seg):
        a, b_, c, d = i, i + 1, seg + 1 + i, seg + 1 + i + 1
        faces += [[a, c, b_], [b_, c, d]]  # inward winding
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.array(uv), image=Image.open(PANO).convert("RGB")
    )
    mesh.export(obj_path)


_BOUNDS_CACHE = {}


def asset_bounds(adir, name):
    """Mesh AABB of a piece asset, cached (OBJ loads are the slow part of a scene build)."""
    key = os.path.join(adir, "mesh", f"{name}.obj")
    if key not in _BOUNDS_CACHE:
        _BOUNDS_CACHE[key] = trimesh.load(key, force="mesh").bounds.copy()
    return _BOUNDS_CACHE[key]


def add_piece(spec, letter, file_idx, rank_idx, board_top_z, idx):
    adir = ASSET_DIRS[letter]
    name = os.path.basename(os.path.dirname(adir))  # e.g. white_pawn
    mjcf = os.path.join(os.path.dirname(adir), "mjcf", f"{name}.xml")
    if not os.path.exists(mjcf):
        return False
    child = mujoco.MjSpec.from_file(mjcf)
    # rescale meshes so piece height matches Staunton target
    bounds = asset_bounds(adir, name)
    cur_h = float(bounds[1][2] - bounds[0][2])
    cur_w = float(max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]))
    target_h = PIECE_H[letter.lower()] * (SQUARE / 0.035)
    s = min(target_h / max(cur_h, 1e-6),
            0.88 * SQUARE / max(cur_w, 1e-6))
    for m in child.meshes:
        m.scale = [s, s, s]
    # Generated convex hulls rest on a tiny rounded nub (the knight creeps 35mm
    # on an idle board). Use a flat-bottomed cylinder collider instead: stable
    # settling, exact footprint, cheap contacts. Visual mesh is unchanged.
    pbody = child.body(name)
    for g in pbody.geoms:
        g.contype = 0
        g.conaffinity = 0
    ext = (bounds[1] - bounds[0]) * s
    ctr = (bounds[1] + bounds[0]) * 0.5 * s
    pbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.5 * max(ext[0], ext[1]) * 0.92, 0.5 * ext[2], 0],
        pos=[ctr[0], ctr[1], ctr[2]],
        rgba=[1, 1, 1, 0], contype=1, conaffinity=1,
    )
    x, y = square_center(file_idx, rank_idx)
    z = board_top_z - float(bounds[0][2]) * s + 0.0015
    frame = spec.worldbody.add_frame(pos=[x, y, z])
    body = frame.attach_body(pbody, f"pc{idx}_", "")
    body.add_freejoint()
    return True


def build(fen: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.shadowsize = 8192

    os.makedirs(OUT, exist_ok=True)
    board_tex = os.path.join(OUT, "board_texture.png")
    if not os.path.exists(board_tex):
        make_board_texture(board_tex)
    pano_obj = os.path.join(OUT, "pano_cylinder.obj")
    if not os.path.exists(pano_obj):
        make_pano_cylinder(pano_obj)  # exports the full panorama; cache it

    # room backdrop + wood floor
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.1],
        rgba=[0.5, 0.42, 0.34, 1],
    )
    spec.add_mesh(name="pano_mesh", file=pano_obj)
    spec.add_texture(name="pano_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=PANO, width=0, height=0)
    spec.add_material(name="pano_mat", emission=0.55
                      ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "pano_tex"
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH, meshname="pano_mesh", material="pano_mat",
        contype=0, conaffinity=0, group=1,
    )

    # lighting
    spec.worldbody.add_light(pos=[0.8, -0.8, 2.2], dir=[-0.3, 0.3, -1],
                             diffuse=[0.85, 0.82, 0.75], specular=[0.3, 0.3, 0.3],
                             castshadow=True)
    spec.worldbody.add_light(pos=[-1.2, 1.0, 1.8], dir=[0.5, -0.4, -1],
                             diffuse=[0.35, 0.36, 0.4], castshadow=False)

    # simple sturdy table under the board (exact, no floating)
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.35, 0.28, TABLE_TOP / 2],
        pos=[0, 0, TABLE_TOP / 2], rgba=[0.42, 0.28, 0.17, 1],
    )

    # board: bordered box + checkered top plate
    board_z = TABLE_TOP + BOARD_T / 2
    spec.add_texture(name="board_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=board_tex, width=0, height=0)
    spec.add_material(name="board_mat", reflectance=0.04
                      ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "board_tex"
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BOARD_W / 2, BOARD_W / 2, BOARD_T / 2],
        pos=[0, 0, board_z], material="board_mat",
    )
    board_top = TABLE_TOP + BOARD_T

    # pieces per FEN
    placed = skipped = 0
    for idx, (letter, f, r) in enumerate(parse_fen_board(fen)):
        if add_piece(spec, letter, f, r, board_top, idx):
            placed += 1
        else:
            skipped += 1
    print(f"pieces placed: {placed}, assets not ready: {skipped}")

    # SO-101 at the white side edge, facing the board
    arm = mujoco.MjSpec.from_file(SO101_XML)
    if ARM_OFFSET_Z > 0:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.055, 0.055, ARM_OFFSET_Z / 2],
            pos=[0, -(BOARD_W / 2 + ARM_OFFSET_Y), TABLE_TOP + ARM_OFFSET_Z / 2],
            rgba=[0.25, 0.25, 0.28, 1],
        )
    frame = spec.worldbody.add_frame(
        pos=[0, -(BOARD_W / 2 + ARM_OFFSET_Y), TABLE_TOP + ARM_OFFSET_Z],
        quat=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)],  # face +y (the board)
    )
    frame.attach_body(arm.body("base"), "so101:", "")
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default=START_FEN)
    ap.add_argument("--settle_s", type=float, default=2.0)
    args = ap.parse_args()

    spec = build(args.fen)
    model = spec.compile()
    data = mujoco.MjData(model)
    rest = {"so101:shoulder_pan": 0.0, "so101:shoulder_lift": -1.0,
            "so101:elbow_flex": 1.3, "so101:wrist_flex": 0.4,
            "so101:wrist_roll": 0.0, "so101:gripper": 0.3}
    for jname, val in rest.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = val
    for aname, val in rest.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if aid >= 0:
            data.ctrl[aid] = val
    print("bodies:", model.nbody, "| geoms:", model.ngeom)

    cam = mujoco.MjvCamera()
    cam.lookat = [0, 0, TABLE_TOP + 0.03]
    cam.distance, cam.azimuth, cam.elevation = 0.8, 205, -22

    frames = []
    steps = int(args.settle_s / model.opt.timestep)
    with mujoco.Renderer(model, height=1080, width=1920) as ren:
        for i in range(steps):
            mujoco.mj_step(model, data)
            if i % 33 == 0:
                ren.update_scene(data, camera=cam)
                frames.append(ren.render())
        ren.update_scene(data, camera=cam)
        final = ren.render()

    imageio.mimsave(os.path.join(OUT, "chess_scene.mp4"), frames, fps=15)
    imageio.imwrite(os.path.join(OUT, "chess_scene.png"), final)
    print("wrote", os.path.join(OUT, "chess_scene.mp4"), "and chess_scene.png")


if __name__ == "__main__":
    main()
