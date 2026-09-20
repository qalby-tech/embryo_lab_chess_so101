"""Scale drawings of the rig, generated from the board configuration.

Two views, both in millimetres and both dimensioned: a plan looking down at the
table, and an elevation looking along it. Written as SVG so they stay legible
at any zoom and never go stale - `hardware/generate.py` rewrites them.
"""
from __future__ import annotations

from chess_sim import BoardConfig
from chess_sim.scene import MAST_HEIGHT, MAST_SIDE, MAST_WIDTH

MM = 1000.0
SCALE = 1.6                 # svg pixels per millimetre
MARGIN = 60                 # mm of white space around the drawing
PLATE = 10.0                # mast plate thickness, mm (scene.py)
LENS_RISE = 20.0            # camera above the mast top, mm (scene.py)

BOARD_FILL = "#e8d5b0"
DARK_SQUARE = "#8a5a3b"
PART_FILL = "#d8dee6"
ACCENT = "#2f6f4f"
LINE = "#44484d"
TEXT = "#22252a"
PAPER = "#ffffff"


def _svg(body: str, x0: float, y0: float, width: float, height: float) -> str:
    """Wrap drawing commands in an SVG with a white card behind them, so the
    figure reads on a light or a dark page."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * SCALE:.0f}" '
            f'height="{height * SCALE:.0f}" viewBox="{x0:.1f} {y0:.1f} {width:.1f} {height:.1f}" '
            f'font-family="DejaVu Sans, Arial, sans-serif">\n'
            f'  <rect x="{x0:.1f}" y="{y0:.1f}" width="{width:.1f}" height="{height:.1f}" '
            f'fill="{PAPER}"/>\n{body}</svg>\n')


def _label(x: float, y: float, text: str, size: float = 9, anchor: str = "middle",
           fill: str = TEXT) -> str:
    return (f'  <text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}">{text}</text>\n')


def _dimension(x1: float, y1: float, x2: float, y2: float, text: str, offset: float = 0) -> str:
    """A thin measure line with its value above the middle."""
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{ACCENT}" '
            f'stroke-width="0.8" stroke-dasharray="4 3"/>\n'
            + _label(mid_x, mid_y - 3 + offset, text, 9, "middle", ACCENT))


def write_plan(board: BoardConfig, path: str) -> None:
    """Looking down at the table. World +x is the arm's right, +y is away from it."""
    half, field = board.width / 2 * MM, board.field / 2 * MM
    square = board.square * MM
    arm_x, arm_y = 0.0, board.arm_base[1] * MM          # negative: the arm's side
    mast_x = MAST_SIDE * MM
    tray_x, tray_y0 = board.tray_x * MM, board.tray_y0 * MM
    pitch, slots = board.tray_pitch * MM, board.tray_slots

    def px(x: float) -> float:
        return x

    def py(y: float) -> float:
        return -y                                        # svg y grows downward

    body = ""
    # board: border slab, then the checkered field
    body += (f'  <rect x="{px(-half):.1f}" y="{py(half):.1f}" width="{2 * half:.1f}" '
             f'height="{2 * half:.1f}" rx="3" fill="{BOARD_FILL}" stroke="{LINE}"/>\n')
    for file in range(8):
        for rank in range(8):
            if (file + rank) % 2:
                continue
            x = -field + file * square
            y = field - (rank + 1) * square
            body += (f'  <rect x="{px(x):.1f}" y="{py(y + square):.1f}" width="{square:.1f}" '
                     f'height="{square:.1f}" fill="{DARK_SQUARE}" opacity="0.75"/>\n')
    body += _label(0, py(0) + 3, "board", 11)
    body += _label(0, py(field) - 6, "black's side", 8)
    body += _label(0, py(-field) + 12, "white's side - the arm works from here", 8)

    # arm base and the plate that carries it, with the mast beside it
    base_w, base_d = 111.0, 72.0                         # measured from the arm's own mesh
    body += (f'  <rect x="{px(arm_x - base_w / 2):.1f}" y="{py(arm_y + base_d / 2):.1f}" '
             f'width="{base_w:.1f}" height="{base_d:.1f}" rx="4" fill="{PART_FILL}" '
             f'stroke="{LINE}"/>\n')
    body += _label(arm_x, py(arm_y) + 3, "SO-101 base", 9)
    body += (f'  <rect x="{px(mast_x - MAST_WIDTH * MM / 2):.1f}" '
             f'y="{py(arm_y + MAST_WIDTH * MM / 2):.1f}" width="{MAST_WIDTH * MM:.1f}" '
             f'height="{MAST_WIDTH * MM:.1f}" fill="{LINE}"/>\n')
    body += _label(mast_x + 34, py(arm_y) + 3, "mast", 9, "start")

    # discard tray
    for slot in range(slots):
        y = tray_y0 + slot * pitch
        body += (f'  <rect x="{px(tray_x - pitch / 2):.1f}" y="{py(y + pitch / 2):.1f}" '
                 f'width="{pitch:.1f}" height="{pitch:.1f}" fill="none" stroke="{LINE}" '
                 f'stroke-dasharray="3 2"/>\n')
    body += _label(tray_x, py(tray_y0 + slots * pitch) - 4, "discard tray", 9)

    # parked pieces the position does not use
    park_x = board.width / 2 * MM + board.graveyard_gap * MM
    park_pitch = board.graveyard_pitch * MM
    body += (f'  <rect x="{px(park_x - park_pitch / 2):.1f}" '
             f'y="{py(-field + 8 * park_pitch):.1f}" width="{4 * park_pitch:.1f}" '
             f'height="{8 * park_pitch:.1f}" fill="none" stroke="{LINE}" stroke-dasharray="2 3"/>\n')
    body += _label(park_x + 2 * park_pitch, py(-field + 8 * park_pitch) - 4, "parked pieces", 9)

    # dimensions
    body += _dimension(px(-half), py(half) - 14, px(half), py(half) - 14,
                       f"{board.width * MM:.0f} board")
    body += _dimension(px(-half - 26), py(-half), px(-half - 26), py(arm_y),
                       f"{abs(arm_y) - half:.0f} gap")
    body += _dimension(px(0), py(0), px(0), py(arm_y), f"{abs(arm_y):.0f} to arm axis", offset=-6)
    body += _dimension(px(0), py(arm_y) + 26, px(mast_x), py(arm_y) + 26, f"{mast_x:.0f} mast")
    body += _dimension(px(tray_x), py(tray_y0) + 26, px(0), py(tray_y0) + 26, f"{abs(tray_x):.0f}")

    x0 = tray_x - pitch / 2 - MARGIN
    y0 = py(half) - MARGIN
    width = (park_x + 4 * park_pitch + MARGIN) - x0
    height = (py(arm_y) + 40 + MARGIN) - y0
    with open(path, "w") as f:
        f.write(_svg(body, x0, y0, width, height))


def write_elevation(board: BoardConfig, path: str) -> None:
    """Looking along the table from white's left: heights, and what stands where."""
    table = board.table_top * MM
    board_top = board.top * MM
    arm_z = board.arm_base[2] * MM
    lens = arm_z + PLATE + MAST_HEIGHT * MM + LENS_RISE
    half = board.width / 2 * MM
    arm_y = board.arm_base[1] * MM
    mast_y = arm_y                                        # the mast stands beside the arm
    floor, top = 0.0, lens + 60

    def py(z: float) -> float:
        return top - z                                    # draw with z upward

    body = ""
    body += (f'  <line x1="{arm_y - 120:.1f}" y1="{py(floor):.1f}" x2="{half + 90:.1f}" '
             f'y2="{py(floor):.1f}" stroke="{LINE}" stroke-width="1.5"/>\n')
    body += _label(arm_y - 100, py(floor) + 14, "floor", 9, "start")
    # table
    body += (f'  <rect x="{arm_y - 90:.1f}" y="{py(table):.1f}" width="{half + 150:.1f}" '
             f'height="18" fill="{PART_FILL}" stroke="{LINE}"/>\n')
    body += _label(half + 40, py(table) + 30, f"table top {table:.0f}", 9, "middle")
    # board slab
    body += (f'  <rect x="{-half:.1f}" y="{py(board_top):.1f}" width="{2 * half:.1f}" '
             f'height="{board.thickness * MM:.1f}" fill="{BOARD_FILL}" stroke="{LINE}"/>\n')
    body += _label(0, py(board_top) - 6, f"board, {board.thickness * MM:.0f} thick", 9)
    # a king, to scale
    king = 46.4
    body += (f'  <rect x="{-10:.1f}" y="{py(board_top + king):.1f}" width="20" '
             f'height="{king:.1f}" rx="6" fill="#f3f0ea" stroke="{LINE}"/>\n')
    body += _label(40, py(board_top + king) + 10, f"king {king:.0f}", 8, "start")
    # riser and arm
    riser = board.arm_riser * MM
    body += (f'  <rect x="{arm_y - 36:.1f}" y="{py(arm_z):.1f}" width="72" height="{riser:.1f}" '
             f'fill="{PART_FILL}" stroke="{LINE}"/>\n')
    body += _label(arm_y - 44, py(table + riser / 2), f"riser {riser:.0f}", 9, "end")
    body += (f'  <circle cx="{arm_y:.1f}" cy="{py(arm_z + 18):.1f}" r="18" fill="{ACCENT}" '
             f'opacity="0.85"/>\n')
    body += _label(arm_y, py(arm_z + 60), "SO-101", 9)
    # mast and camera
    body += (f'  <rect x="{mast_y + 44:.1f}" y="{py(lens - LENS_RISE):.1f}" width="{MAST_WIDTH * MM:.1f}" '
             f'height="{MAST_HEIGHT * MM:.1f}" fill="{LINE}"/>\n')
    body += (f'  <rect x="{mast_y + 30:.1f}" y="{py(lens + 16):.1f}" width="58" height="26" rx="4" '
             f'fill="{ACCENT}"/>\n')
    body += _label(mast_y + 100, py(lens), "overhead camera", 9, "start")
    body += (f'  <line x1="{mast_y + 59:.1f}" y1="{py(lens):.1f}" x2="{-half:.1f}" '
             f'y2="{py(board_top):.1f}" stroke="{ACCENT}" stroke-width="0.7" stroke-dasharray="5 4"/>\n')
    body += (f'  <line x1="{mast_y + 59:.1f}" y1="{py(lens):.1f}" x2="{half:.1f}" '
             f'y2="{py(board_top):.1f}" stroke="{ACCENT}" stroke-width="0.7" stroke-dasharray="5 4"/>\n')
    body += _dimension(mast_y - 60, py(table), mast_y - 60, py(lens),
                       f"{lens - table:.0f} lens above table")
    body += _dimension(half + 60, py(table), half + 60, py(board_top),
                       f"{board_top - table:.0f}")

    x0 = arm_y - 170
    y0 = py(top)
    with open(path, "w") as f:
        f.write(_svg(body, x0, y0, (half + 230) - x0, top + 40))
