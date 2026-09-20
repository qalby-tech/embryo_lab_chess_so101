"""Derive the physical build sheet from the simulated rig.

    python hardware/generate.py

Writes dimensions.json, printable STLs and a 1:1 board artwork, all computed
from `chess_sim` - so the parts you build match the scene the policy trained in.
Re-run it after changing any dimension in `chess_sim/config.py` or `scene.py`.
"""
from __future__ import annotations

import argparse
import os

import chess
import numpy as np
import trimesh

from chess_sim import BoardConfig, Camera, ChessSimEnv, Config, ControlConfig, EnvConfig, START_FEN
from diagrams import write_elevation, write_plan
from chess_sim.assets import ASSET_DIR, SET_COUNTS, piece_geometry
from chess_sim.scene import (MAST_HEIGHT, MAST_SIDE, MAST_WIDTH, WRIST_CAMERA)

HERE = os.path.dirname(os.path.abspath(__file__))
STL_DIR = os.path.join(HERE, "stl")
MEDIA_DIR = os.path.join(os.path.dirname(HERE), "docs", "media")
MM = 1000.0
TOP_CAMERA_FOVY = 24          # degrees, as built in scene.py
MAST_PLATE_THICKNESS = 0.010  # scene.py: the plate under the arm base
MAST_PLATE_MARGIN = 0.06      # how far the plate reaches past the mast axis
MAST_PLATE_DEPTH = 0.14       # full depth (scene uses half-size 0.07)
TRAY_WALL = 0.003             # printed tray: wall thickness
TRAY_DEPTH = 0.012            # printed tray: pocket depth
PRINT_DPI = 300


class PieceRow(Config):
    piece: str
    per_side: int
    height_mm: float
    widest_diameter_mm: float
    grasp_diameter_mm: float
    grasp_height_mm: float


class BoardSheet(Config):
    square_mm: float
    border_mm: float
    thickness_mm: float
    playing_field_mm: float
    overall_mm: float


class WorkstationSheet(Config):
    table_top_mm: float
    arm_riser_mm: float
    board_edge_to_arm_base_mm: float
    arm_base_from_board_centre_mm: tuple[float, float]
    arm_base_footprint_mm: tuple[float, float]


class TraySheet(Config):
    first_slot_from_board_centre_mm: tuple[float, float]
    slot_pitch_mm: float
    slots: int


class CameraSheet(Config):
    mast_offset_from_arm_axis_mm: float
    mast_height_above_plate_mm: float
    mast_tube_mm: float
    lens_height_above_table_mm: float
    lens_height_above_board_mm: float
    simulated_vertical_fov_deg: float
    vertical_footprint_at_board_mm: float
    min_fov_to_see_whole_board_deg: float
    wrist_camera_offset_mm: tuple[float, float, float]
    wrist_camera_fov_deg: float


class BuildSheet(Config):
    """Everything needed to build the rig, in millimetres."""

    figures: list[str] = []

    board: BoardSheet
    workstation: WorkstationSheet
    capture_tray: TraySheet
    cameras: CameraSheet
    pieces: list[PieceRow]
    printable: list[str] = []
    parametric: str = ""
    board_artwork: str = ""


def piece_table(board: BoardConfig) -> list[PieceRow]:
    """Height and diameter of every piece type, at this board's square size."""
    rows = []
    for piece_type, count in SET_COUNTS.items():
        geometry = piece_geometry(chess.Piece(piece_type, chess.WHITE), board.square)
        rows.append(PieceRow(piece=chess.piece_name(piece_type), per_side=count,
                             height_mm=round(geometry.height * MM, 1),
                             widest_diameter_mm=round(2 * geometry.collider_radius * MM, 1),
                             grasp_diameter_mm=round(2 * geometry.grasp_radius * MM, 1),
                             grasp_height_mm=round(geometry.waist * MM, 1)))
    return rows


def camera_geometry(board: BoardConfig) -> CameraSheet:
    """Where the overhead camera sits and what it must see."""
    _, arm_y, arm_z = board.arm_base
    lens_z = arm_z + MAST_PLATE_THICKNESS + MAST_HEIGHT + 0.02
    above_board = lens_z - board.top
    footprint = 2 * above_board * np.tan(np.radians(TOP_CAMERA_FOVY) / 2)
    diagonal = np.degrees(2 * np.arctan(board.width / 2 / above_board))
    return CameraSheet(mast_offset_from_arm_axis_mm=round(MAST_SIDE * MM, 1),
                       mast_height_above_plate_mm=round(MAST_HEIGHT * MM, 1),
                       mast_tube_mm=round(MAST_WIDTH * MM, 1),
                       lens_height_above_table_mm=round((lens_z - board.table_top) * MM, 1),
                       lens_height_above_board_mm=round(above_board * MM, 1),
                       simulated_vertical_fov_deg=TOP_CAMERA_FOVY,
                       vertical_footprint_at_board_mm=round(footprint * MM, 1),
                       min_fov_to_see_whole_board_deg=round(diagonal, 1),
                       wrist_camera_offset_mm=tuple(round(v * MM, 1) for v in WRIST_CAMERA.pos),
                       wrist_camera_fov_deg=WRIST_CAMERA.fovy)


def arm_base_footprint() -> tuple[float, float]:
    """Footprint of the SO-101 base, measured off its own mesh."""
    mesh = trimesh.load(os.path.join(ASSET_DIR, "so101", "base_so101_v2.stl"), force="mesh")
    extents = mesh.extents / MM if mesh.extents.max() > 1 else mesh.extents   # STLs are in mm
    return float(extents[0]), float(extents[1])


def write_stls(board: BoardConfig, base_x: float, base_y: float) -> list[str]:
    """Printable parts as plain boxes - the shapes the scene actually contains."""
    os.makedirs(STL_DIR, exist_ok=True)
    plate_x = 2 * (0.5 * MAST_SIDE + MAST_PLATE_MARGIN)
    parts = {
        # pedestal under the arm base: lifts the arm to the height the scene uses
        "arm_riser": (base_x, base_y, board.arm_riser),
        # plate that carries both the arm and the mast
        "mast_plate": (plate_x, MAST_PLATE_DEPTH, MAST_PLATE_THICKNESS),
        # the tube itself, printable in two halves if the bed is short
        "mast_tube": (MAST_WIDTH, MAST_WIDTH, MAST_HEIGHT / 2),
        # discard tray: four pockets at the pitch the expert drops pieces into
        "capture_tray": (board.tray_pitch + 2 * TRAY_WALL,
                         board.tray_slots * board.tray_pitch + 2 * TRAY_WALL,
                         TRAY_DEPTH + TRAY_WALL),
    }
    written = []
    for name, size in parts.items():
        path = os.path.join(STL_DIR, f"{name}.stl")
        trimesh.creation.box(extents=np.array(size) * MM).export(path)   # STL convention: mm
        written.append(os.path.relpath(path, HERE))
    return written


def write_board_artwork(board: BoardConfig) -> str:
    """The board texture at 1:1, ready to print and mount."""
    from PIL import Image

    from chess_sim.assets import BOARD_TEXTURE, ensure_scene_assets

    ensure_scene_assets(board)
    width_mm = board.width * MM
    pixels = int(round(width_mm / 25.4 * PRINT_DPI))
    image = Image.open(BOARD_TEXTURE).convert("RGB").resize((pixels, pixels), Image.LANCZOS)
    path = os.path.join(HERE, f"board_{int(round(width_mm))}mm.png")
    image.save(path, dpi=(PRINT_DPI, PRINT_DPI))
    return os.path.relpath(path, HERE)


SCAD_TEMPLATE = """// Printable parts of the chess rig, parametric.
// Generated by hardware/generate.py from chess_sim - edit there, not here.
// All dimensions in millimetres.

board_square      = {square};
board_overall     = {overall};
arm_riser_height  = {riser};
arm_base_x        = {base_x};
arm_base_y        = {base_y};
mast_plate        = [{plate_x}, {plate_y}, {plate_z}];
mast_tube         = {tube};
mast_height       = {mast_height};
tray_slots        = {tray_slots};
tray_pitch        = {tray_pitch};
tray_wall         = {tray_wall};
tray_depth        = {tray_depth};

module arm_riser() {{ cube([arm_base_x, arm_base_y, arm_riser_height], center = true); }}

module mast_plate() {{ cube(mast_plate, center = true); }}

// print in two halves if the bed is shorter than the mast
module mast_tube(length = mast_height / 2, wall = 2) {{
    difference() {{
        cube([mast_tube, mast_tube, length], center = true);
        cube([mast_tube - 2 * wall, mast_tube - 2 * wall, length + 1], center = true);
    }}
}}

module capture_tray() {{
    pocket = tray_pitch - tray_wall;
    difference() {{
        cube([tray_pitch + 2 * tray_wall,
              tray_slots * tray_pitch + 2 * tray_wall,
              tray_depth + tray_wall], center = true);
        for (i = [0 : tray_slots - 1])
            translate([0,
                       (i - (tray_slots - 1) / 2) * tray_pitch,
                       tray_wall])
                cube([pocket, pocket, tray_depth + 1], center = true);
    }}
}}

arm_riser();
translate([0, 120, 0]) mast_plate();
translate([0, 220, 0]) mast_tube();
translate([0, 340, 0]) capture_tray();
"""


def write_scad(board: BoardConfig, base_x: float, base_y: float) -> str:
    """The same parts, parametric, for anyone who wants to change a dimension."""
    path = os.path.join(HERE, "parts.scad")
    with open(path, "w") as f:
        f.write(SCAD_TEMPLATE.format(
            square=round(board.square * MM, 1), overall=round(board.width * MM, 1),
            riser=round(board.arm_riser * MM, 1), base_x=round(base_x * MM, 1),
            base_y=round(base_y * MM, 1),
            plate_x=round(2 * (0.5 * MAST_SIDE + MAST_PLATE_MARGIN) * MM, 1),
            plate_y=round(MAST_PLATE_DEPTH * MM, 1),
            plate_z=round(MAST_PLATE_THICKNESS * MM, 1),
            tube=round(MAST_WIDTH * MM, 1), mast_height=round(MAST_HEIGHT * MM, 1),
            tray_slots=board.tray_slots, tray_pitch=round(board.tray_pitch * MM, 1),
            tray_wall=round(TRAY_WALL * MM, 1), tray_depth=round(TRAY_DEPTH * MM, 1)))
    return os.path.relpath(path, HERE)


def write_figures(board: BoardConfig, render: bool) -> list[str]:
    """Scale drawings, and photographs of the scene they describe."""
    os.makedirs(MEDIA_DIR, exist_ok=True)
    written = []
    for name, draw in (("rig_plan.svg", write_plan), ("rig_elevation.svg", write_elevation)):
        draw(board, os.path.join(MEDIA_DIR, name))
        written.append(f"docs/media/{name}")
    if render:
        import imageio.v2 as imageio
        import mujoco

        env = ChessSimEnv(EnvConfig(control=ControlConfig(
            cameras=(Camera.TOP,), image_size=(1100, 800))))
        env.reset(START_FEN)
        # a free camera set back far enough to take in the table, the arm and the mast,
        # which the scene's own demo view is too close for
        wide = mujoco.MjvCamera()
        wide.type = mujoco.mjtCamera.mjCAMERA_FREE
        wide.lookat[:] = [0.02, -0.10, board.table_top + 0.26]
        wide.distance, wide.azimuth, wide.elevation = 1.35, -128, -12
        env.render(Camera.TOP)                      # builds the renderer
        env._renderer.update_scene(env.data, camera=wide)
        imageio.imwrite(os.path.join(MEDIA_DIR, "rig_overview.png"), env._renderer.render())
        written.append("docs/media/rig_overview.png")
        imageio.imwrite(os.path.join(MEDIA_DIR, "rig_top.png"), env.render(Camera.TOP))
        written.append("docs/media/rig_top.png")
        env.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-render", action="store_true",
                        help="skip the simulator photographs (the drawings still get written)")
    args = parser.parse_args()
    board = BoardConfig()
    base_x, base_y = arm_base_footprint()
    sheet = BuildSheet(
        board=BoardSheet(square_mm=round(board.square * MM, 1),
                         border_mm=round(board.border * MM, 1),
                         thickness_mm=round(board.thickness * MM, 1),
                         playing_field_mm=round(board.field * MM, 1),
                         overall_mm=round(board.width * MM, 1)),
        workstation=WorkstationSheet(
            table_top_mm=round(board.table_top * MM, 1),
            arm_riser_mm=round(board.arm_riser * MM, 1),
            board_edge_to_arm_base_mm=round(board.arm_gap * MM, 1),
            arm_base_from_board_centre_mm=tuple(round(v * MM, 1) for v in board.arm_base[:2]),
            arm_base_footprint_mm=(round(base_x * MM, 1), round(base_y * MM, 1))),
        capture_tray=TraySheet(
            first_slot_from_board_centre_mm=(round(board.tray_x * MM, 1),
                                             round(board.tray_y0 * MM, 1)),
            slot_pitch_mm=round(board.tray_pitch * MM, 1), slots=board.tray_slots),
        cameras=camera_geometry(board),
        pieces=piece_table(board),
        printable=write_stls(board, base_x, base_y),
        parametric=write_scad(board, base_x, base_y),
        board_artwork=write_board_artwork(board),
        figures=write_figures(board, render=not args.no_render))
    with open(os.path.join(HERE, "dimensions.json"), "w") as f:
        f.write(sheet.model_dump_json(indent=2))
    print(sheet.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
