"""Scene assembly: table, board, backdrop, a full 32-piece set, the SO-101, cameras.

The scene has a *fixed topology*: all 32 pieces always exist as free bodies.
A position is applied by moving pieces to their squares (or to off-board
graveyard slots), so one compiled model serves every episode and the scene can
be exported as a standalone MuJoCo XML.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import chess
import mujoco
import numpy as np

from . import assets, gripper
from .assets import (ASSET_DIR, DEFAULT_SET, SET_COUNTS, PieceGeometry, piece_asset_name,
                     piece_geometry, piece_obj_path, piece_texture_path, piece_tint)
from .config import BOARD_FINISHES, AppearanceConfig, BoardConfig, Camera, Config, JointPose

ARM_PREFIX = "so101:"
# MuJoCo caps offscreen rendering at these; the default 640x480 is too small for a
# printable figure, and a bigger buffer costs nothing until something asks for it
OFFSCREEN_SIZE = (1280, 960)
KEY_LIGHT_DIFFUSE = np.array([0.85, 0.82, 0.75])
FILL_LIGHT_DIFFUSE = np.array([0.35, 0.36, 0.4])


def key_diffuse(appearance) -> np.ndarray:
    return KEY_LIGHT_DIFFUSE * appearance.light_intensity * np.asarray(appearance.light_color)
class CameraMount(Config):
    """Where a camera sits and what it looks at, in the frame it is mounted in."""

    pos: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fovy: float = 42


# wrist camera in the gripperframe site frame (+x approach, +z jaw span):
# beside the wrist-roll motor, looking at the fingertips, wide lens
WRIST_CAMERA = CameraMount(pos=(-0.075, 0.045, 0.012), target=(0.0, 0.0, 0.012),
                           up=(-1.0, 0.0, 0.0), fovy=75)
ARM_REST = JointPose(shoulder_pan=0.0, shoulder_lift=-1.0, elbow_flex=1.3,
                     wrist_flex=0.4, wrist_roll=0.0, gripper=0.6)


@dataclass(frozen=True)
class PieceSlot:
    """One of the 32 piece bodies in the scene."""

    body: str
    piece: chess.Piece
    geometry: PieceGeometry


# Contact impedance of the surfaces pieces stand on. MuJoCo's soft contact
# leaves a (1 - impedance) share of the pushing acceleration unopposed, and with
# the default 0.9-0.95 a 1 g piece under a 10 N press sinks 40 mm: through the
# board, into the table (a 10 mm gripper press alone drove a rook 18 mm down).
# Near-unit impedance on the board and table, which outrank the pieces so their
# setting is used, holds the same press to 0.6 mm. Explicit stiffness (negative
# solref) is unusable at this mass - it flings every piece - and a shorter time
# constant on every geom tunnels at 10 N.
SURFACE_SOLIMP = (0.999, 0.9999, 0.001, 0.5, 2.0)
SURFACE_PRIORITY = 1


def piece_slots(board: BoardConfig, piece_scale: float = 1.0,
                piece_set: str = DEFAULT_SET) -> list[PieceSlot]:
    """The fixed set of piece bodies, in a stable order."""
    slots = []
    for color in (chess.WHITE, chess.BLACK):
        for ptype, count in SET_COUNTS.items():
            piece = chess.Piece(ptype, color)
            for i in range(1, count + 1):
                slots.append(PieceSlot(
                    body=f"{piece_asset_name(piece)}{i}",
                    piece=piece,
                    geometry=piece_geometry(piece, board.square, piece_scale, piece_set),
                ))
    return slots


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)):
    """Camera x/y axes (MuJoCo convention: -z is the view direction); `up` is
    the world direction that should point up in the image."""
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.concatenate([right, up])


def _add_camera(spec, name: Camera, pos, target, fovy=42, up=(0.0, 0.0, 1.0)):
    spec.worldbody.add_camera(name=str(name), pos=list(pos),
                              xyaxes=list(lookat_xyaxes(pos, target, up)), fovy=fovy)


def build_scene(board: BoardConfig = BoardConfig(), appearance: AppearanceConfig = AppearanceConfig()) -> mujoco.MjSpec:
    """Assemble the MjSpec. Mesh/texture paths are relative to the assets dir."""
    board_texture = assets.ensure_scene_assets(board, appearance)
    spec = mujoco.MjSpec()
    spec.modelname = "chess_so101"
    spec.meshdir = assets.ASSET_DIR
    spec.texturedir = assets.ASSET_DIR
    spec.option.timestep = 0.002
    # Elliptic friction cones with a high impedance ratio: with the default
    # pyramidal cones a friction-held piece creeps through the pads by ~1 cm
    # over a two-second carry and slides off its widest segment.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.quality.shadowsize = 4096
    # near clipping plane is znear * extent: the backdrop would push it past the
    # fingertips 8 cm in front of the wrist camera, so pin the extent
    spec.stat.extent = 1.0
    spec.visual.map.znear = 0.004

    _add_environment(spec, board, appearance)
    _add_board(spec, board, board_texture, appearance)
    _add_pieces(spec, board, appearance)
    _add_arm(spec, board)
    camera_pos = _add_camera_mast(spec, board)

    _add_camera(spec, Camera.EXTERNAL, (-0.31, -0.16, board.top + 0.26), (0.0, 0.02, board.top + 0.04))
    # the overhead image is upright along the files (white at the bottom) and
    # framed on the board: at this distance the board fills ~86% of the frame
    # height, which is what a policy needs to tell one square from another
    pos, target, fovy, up = overhead_camera(board, appearance, camera_pos)
    _add_camera(spec, Camera.TOP, pos, target, fovy=fovy, up=up)
    # last, because merging the arm's MJCF brings that model's visual settings with it
    spec.visual.global_.offwidth, spec.visual.global_.offheight = OFFSCREEN_SIZE
    return spec


def overhead_camera(board, appearance, mast_lens):
    """Where the overhead camera is, what it looks at, its field of view and its up
    direction - the published rig unless the appearance moves it."""
    cx, cy = board.origin
    if appearance.camera_height is None:
        return mast_lens, (cx, cy, board.top), 24.0, (0.0, 1.0, 0.0)
    if appearance.camera_offset is not None:
        x, y = cx + appearance.camera_offset[0], cy + appearance.camera_offset[1]
    else:
        t = appearance.camera_over_board
        x = (1 - t) * mast_lens[0] + t * cx
        y = (1 - t) * mast_lens[1] + t * cy
    target = (cx + appearance.camera_aim[0], cy + appearance.camera_aim[1], board.top)
    pos = (x, y, board.top + appearance.camera_height)
    view = np.subtract(target, pos)
    view /= np.linalg.norm(view)
    roll = np.radians(appearance.camera_roll)
    if np.degrees(np.arcsin(-view[2])) > LOOKING_DOWN_DEG:
        # looking down on the board: files run up the image, as on the published rig
        up = (float(np.sin(roll)), float(np.cos(roll)), 0.0)
    else:
        # looking across the table: gravity is up, rolled about the line of sight
        z = np.array([0.0, 0.0, 1.0])
        up = z * np.cos(roll) + np.cross(view, z) * np.sin(roll) + view * np.dot(view, z) * (1 - np.cos(roll))
        up = tuple(float(v) for v in up)
    return pos, target, appearance.camera_fovy, up


LOOKING_DOWN_DEG = 52.0    # steeper than this the image is oriented along the files, not by gravity


MAST_SIDE = 0.16      # mast axis this far beside the arm axis (+x: the arm's right)
MAST_HEIGHT = 0.60    # pole height above the base plate
MAST_WIDTH = 0.03     # square-tube side


def _add_camera_mast(spec, board) -> tuple[float, float, float]:
    """Overhead-camera mast as on the real rig: a plate under the arm base
    carries a square tube beside the arm with the workspace camera on top,
    pointed down at the board. Returns the camera position."""
    ax, ay, az = board.arm_base
    plate = MAST_PLATE
    spec.worldbody.add_geom(name="mast_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[0.5 * MAST_SIDE + 0.06, 0.07, plate / 2],
                            pos=[ax + 0.5 * MAST_SIDE, ay, az + plate / 2],
                            rgba=[0.78, 0.86, 0.20, 1])
    mx, lower = ax + MAST_SIDE, 0.20
    spec.worldbody.add_geom(name="mast_lower", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[MAST_WIDTH / 2, MAST_WIDTH / 2, lower / 2],
                            pos=[mx, ay, az + plate + lower / 2], rgba=[0.78, 0.86, 0.20, 1])
    spec.worldbody.add_geom(name="mast_upper", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[MAST_WIDTH / 2, MAST_WIDTH / 2, (MAST_HEIGHT - lower) / 2],
                            pos=[mx, ay, az + plate + lower + (MAST_HEIGHT - lower) / 2],
                            rgba=[0.08, 0.08, 0.08, 1])
    cam_z = az + plate + MAST_HEIGHT + 0.02
    spec.worldbody.add_geom(name="mast_camera", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=list(MAST_HOUSING), pos=[mx, ay + 0.02, cam_z],
                            rgba=[0.08, 0.08, 0.08, 1], contype=0, conaffinity=0)
    return mast_lens(board)


MAST_PLATE = 0.010
MAST_HOUSING = (0.015, 0.03, 0.02)


def mast_lens(board) -> tuple[float, float, float]:
    """Where the lens of the published rig's camera sits: just ahead of the housing
    on top of the mast beside the arm."""
    ax, ay, az = board.arm_base
    return (ax + MAST_SIDE, ay + 0.02 + MAST_HOUSING[1] + 0.005, az + MAST_PLATE + MAST_HEIGHT + 0.02)


def _add_environment(spec, board, appearance):
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.1],
                            rgba=[0.5, 0.42, 0.34, 1])
    spec.add_mesh(name="backdrop", file="scene/pano_cylinder.obj")
    spec.add_texture(name="backdrop_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file="scene/pano_image.png")
    mat = spec.add_material(name="backdrop_mat", emission=appearance.backdrop_brightness,
                            rgba=[*appearance.backdrop_tint, 1.0])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "backdrop_tex"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="backdrop",
                            material="backdrop_mat", contype=0, conaffinity=0, group=1)
    key_dir = np.asarray(appearance.light_dir, dtype=float)
    key_dir /= np.linalg.norm(key_dir)
    key = spec.worldbody.add_light(name="key", pos=list(-2.4 * key_dir + [0, 0, 0.4]), dir=list(key_dir),
                                   diffuse=list(key_diffuse(appearance)), ambient=[appearance.ambient] * 3,
                                   specular=[0.3, 0.3, 0.3], castshadow=True)
    key.bulbradius = appearance.shadow_softness
    spec.worldbody.add_light(name="fill", pos=[-1.2, 1.0, 1.8], dir=[0.5, -0.4, -1],
                             diffuse=list(FILL_LIGHT_DIFFUSE * appearance.fill_intensity), castshadow=False)
    # table and arm pedestal
    table_material = ""
    if appearance.table_surface != "plain":
        spec.add_texture(name="table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         file=assets.ensure_table_texture(appearance.table_surface))
        tmat = spec.add_material(name="table_mat", rgba=[*appearance.table_rgb, 1],
                                 texuniform=True, texrepeat=[3, 3])
        tmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "table_tex"
        table_material = "table_mat"
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[0.36, 0.30, board.table_top / 2], material=table_material,
                            pos=[0, 0, board.table_top / 2], rgba=[*appearance.table_rgb, 1],
                            solimp=list(SURFACE_SOLIMP), priority=SURFACE_PRIORITY)
    ax, ay, az = board.arm_base
    if board.arm_riser > 0:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[0.055, 0.055, board.arm_riser / 2],
                                pos=[ax, ay, board.table_top + board.arm_riser / 2],
                                rgba=[0.25, 0.25, 0.28, 1])


def _add_board(spec, board, texture_file, appearance):
    spec.add_texture(name="board_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=texture_file)
    reflectance, grip = BOARD_FINISHES[appearance.board_finish]
    mat = spec.add_material(name="board_mat", reflectance=reflectance)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "board_tex"
    # A mocap body, not a world geom, so the board can be shifted between
    # episodes. A static world geom's collision bounds are fixed at compile
    # time: moving one at runtime moved its picture while pieces fell through
    # the spot it now covered.
    body = spec.worldbody.add_body(name="board", mocap=True,
                                   pos=[board.origin[0], board.origin[1],
                                        board.table_top + board.thickness / 2],
                                   quat=[float(np.cos(board.yaw / 2)), 0.0, 0.0, float(np.sin(board.yaw / 2))])
    body.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[board.width / 2, board.width / 2, board.thickness / 2],
                  material="board_mat", friction=[grip, 0.005, 0.0001],
                  solimp=list(SURFACE_SOLIMP), priority=SURFACE_PRIORITY)


def _add_pieces(spec, board, appearance):
    kinds_done = set()
    piece_set = appearance.piece_set
    for i, slot in enumerate(piece_slots(board, appearance.piece_scale, piece_set)):
        kind = piece_asset_name(slot.piece)
        g = slot.geometry
        if kind not in kinds_done:
            spec.add_mesh(name=f"mesh_{kind}", scale=[g.scale] * 3,
                          file=os.path.relpath(piece_obj_path(slot.piece, piece_set), ASSET_DIR))
            spec.add_texture(name=f"tex_{kind}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                             file=os.path.relpath(piece_texture_path(slot.piece, piece_set), ASSET_DIR))
            tint = piece_tint(slot.piece, appearance.white_rgba, appearance.black_rgba, piece_set)
            mat = spec.add_material(name=f"mat_{kind}", reflectance=0.2, rgba=list(tint))
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = f"tex_{kind}"
            kinds_done.add(kind)
        gx, gy = board.graveyard_slot(i)
        body = spec.worldbody.add_body(name=slot.body,
                                       pos=[gx, gy, board.table_top - g.bottom_offset])
        body.add_freejoint(name=f"{slot.body}_joint")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"mesh_{kind}",
                      material=f"mat_{kind}", contype=0, conaffinity=0, density=0)
        # Profiled collider: a short stack of cylinders following the mesh
        # (flat base for stable settling, true radii above it so the jaws close
        # on the widest segment). Generated hulls rest on a rounded nub and
        # creep across the board.
        for z0, z1, radius in g.profile:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          size=[radius, 0.5 * (z1 - z0), 0],
                          pos=[g.center[0], g.center[1], 0.5 * (z0 + z1)],
                          rgba=[1, 1, 1, 0], density=600)


def build_arm(board: BoardConfig = BoardConfig()) -> mujoco.MjSpec:
    """The SO-101 alone, mounted exactly as in the scene (same names): the
    kinematic model the IK solves on, 33x fewer degrees of freedom."""
    spec = mujoco.MjSpec()
    spec.modelname = "so101_arm"
    spec.meshdir = assets.ASSET_DIR
    _add_arm(spec, board)
    return spec


def _add_arm(spec, board):
    arm = mujoco.MjSpec.from_file(assets.SO101_XML)
    arm.meshdir = assets.ASSET_DIR  # merged assets resolve against the shared tree
    for mesh in arm.meshes:
        mesh.file = os.path.join("so101", os.path.basename(mesh.file))
    # fingertip contact pads replace the finger hulls for contact (see gripper.py)
    hinge_axis = np.array(arm.joint("gripper").axis, dtype=float)
    for body_name, pad in gripper.FINGER_PADS.items():
        body = arm.body(body_name)
        for geom in body.geoms:
            if geom.meshname in gripper.FINGER_HULL_MESHES and geom.contype != 0:
                geom.contype = 0
                geom.conaffinity = 0
        quat = np.array(pad.quat, dtype=float)
        if body_name == "moving_jaw_so101_v1":
            # undo the hinge rotation the pad will have at the nominal grasp angle
            comp = np.zeros(4)
            mujoco.mju_axisAngle2Quat(comp, hinge_axis, -gripper.NOMINAL_GRASP_ANGLE)
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, comp, quat)
            quat = out
        body.add_geom(name=f"{body_name}_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=list(pad.pos), quat=list(quat), size=list(pad.size),
                      friction=list(gripper.PAD_FRICTION), solref=list(gripper.PAD_SOLREF),
                      solimp=list(gripper.PAD_SOLIMP), condim=gripper.PAD_CONDIM,
                      priority=gripper.PAD_PRIORITY, rgba=[0.1, 0.1, 0.1, 1], group=3)
    ax, ay, az = board.arm_base
    frame = spec.worldbody.add_frame(pos=[ax, ay, az],
                                     quat=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    frame.attach_body(arm.body("base"), ARM_PREFIX, "")
    _add_wrist_camera(spec)


def _add_wrist_camera(spec):
    """Camera on the gripper body, placed relative to the gripperframe site."""
    site = spec.site(ARM_PREFIX + "gripperframe")
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(site.quat, dtype=float))
    rot = rot.reshape(3, 3)                      # site axes in the gripper body frame
    cam = WRIST_CAMERA
    pos = np.asarray(site.pos, dtype=float) + rot @ np.asarray(cam.pos, dtype=float)
    axes = lookat_xyaxes(cam.pos, cam.target, cam.up)
    xyaxes = np.concatenate([rot @ axes[:3], rot @ axes[3:]])
    gripper = spec.body(ARM_PREFIX + "gripper")
    gripper.add_camera(name=str(Camera.WRIST), pos=list(pos), xyaxes=list(xyaxes), fovy=cam.fovy)
    gripper.add_geom(name="wrist_camera", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.012, 0.008],
                     pos=list(pos - rot @ np.array([0.01, 0.0, 0.0])),
                     rgba=[0.08, 0.08, 0.08, 1], contype=0, conaffinity=0, group=1, density=0)


def arm_rest_pose() -> JointPose:
    """Parked joint targets (name -> radians), keys without the scene prefix."""
    return ARM_REST


def export_xml(spec: mujoco.MjSpec, path: str, model: mujoco.MjModel | None = None,
               data: mujoco.MjData | None = None) -> str:
    """Write a standalone MuJoCo XML for the scene; returns the path.

    With `model`/`data`, the pieces' body poses are written as they stand now
    and a keyframe `position` holds the full state (arm included), so the file
    opens in the MuJoCo viewer showing the current position."""
    if data is not None:
        for body in spec.worldbody.bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.name)
            if bid >= 0 and model.body_jntnum[bid] == 1 and \
                    model.jnt_type[model.body_jntadr[bid]] == mujoco.mjtJoint.mjJNT_FREE:
                adr = model.jnt_qposadr[model.body_jntadr[bid]]
                body.pos = data.qpos[adr:adr + 3]
                body.quat = data.qpos[adr + 3:adr + 7]
        key = next((k for k in spec.keys if k.name == "position"), None) or spec.add_key()
        key.name = "position"
        key.qpos = data.qpos.copy()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(spec.to_xml())
    return path
