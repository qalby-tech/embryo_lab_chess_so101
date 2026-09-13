"""ChessSimEnv: the user-facing simulation of an SO-101 playing chess.

    env = ChessSimEnv()
    env.reset("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    env.move("g1", "f3")            # scripted expert; env.board mirrors the result
    obs = env.step(action)          # or drive the joints yourself (6 targets, rad)
    frame = env.render("external")
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import chess
import mujoco
import numpy as np

from .appearance import Appearance
from .assets import piece_asset_name
from .board import START_FEN, BoardSpec, parse_square
from .controller import PickPlaceController, grasp_plan
from .ik import So101Ik
from .reach import MAX_IK_ERROR, MAX_TILT, ReachMap
from .scene import (ARM_PREFIX, KEY_LIGHT_DIFFUSE, ROBOT_CAMERAS, PieceSlot, arm_rest_pose, build_arm,
                    build_scene, export_xml, piece_slots)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
PLACEMENT_TOLERANCE = 0.011   # piece axis vs square center
CAPTURE_TOLERANCE = 0.030     # a discarded piece only has to land in its tray slot
DISTURB_TOLERANCE = 0.010     # any other piece moving more than this fails a move
UPRIGHT_MIN = 0.85            # z-axis alignment of a standing piece
# free space the open jaw needs beyond the grasped piece's surface
MOVING_JAW_ROOM = 0.020       # moving prong swings out this far past the piece
FIXED_PRONG_ROOM = 0.009      # fixed prong: ~3 mm slack + prong thickness
# candidate jaw-span directions: the moving jaw opens along one board axis
SPAN_DIRECTIONS = (np.array([0.0, 1.0]), np.array([0.0, -1.0]),
                   np.array([1.0, 0.0]), np.array([-1.0, 0.0]))


@dataclass
class Observation:
    time: float
    joint_pos: np.ndarray                 # 6 joints, radians
    joint_vel: np.ndarray
    images: dict[str, np.ndarray] = field(default_factory=dict)
    fen: str = ""


@dataclass
class MoveResult:
    from_square: str
    to_square: str
    success: bool                         # verified outcome (placement, upright, no disturbance)
    placement_error: float                # meters from the target square center
    disturbed: list[str]                  # squares whose pieces moved
    steps: int
    reason: str = ""
    waypoints_precise: bool = True        # diagnostic: grasp/place waypoints hit tolerance


# Layout randomization for reset(rng=...). A real board is never set down in
# exactly the same spot and a real arm is never parked in exactly the same pose;
# a policy that only ever saw one of each can key on it instead of looking.
BOARD_SHIFT = 0.010         # m, each axis - about a third of a square
ARM_START_JITTER = 0.10     # rad, each joint


class ChessSimEnv:
    def __init__(self, board_spec: BoardSpec = BoardSpec(), appearance: Appearance = Appearance(),
                 cameras: tuple[str, ...] = ROBOT_CAMERAS,
                 image_size: tuple[int, int] = (640, 480),
                 control_hz: int = 30):
        self.board_spec = board_spec
        self._nominal_origin = board_spec.origin
        self.appearance = appearance
        self.spec = build_scene(board_spec, appearance)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self._board_mocap = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "board")]
        self.layout = {"board_origin": list(board_spec.origin), "arm_start": arm_rest_pose()}
        self.control_hz = control_hz
        self.substeps = max(1, int(round(1.0 / (control_hz * self.model.opt.timestep))))
        self.board = chess.Board(None)

        self.slots = piece_slots(board_spec, appearance.piece_scale)
        self._slot_body = {s.body: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, s.body)
                           for s in self.slots}
        self._slot_qpos = {s.body: self.model.jnt_qposadr[self.model.body_jntadr[self._slot_body[s.body]]]
                           for s in self.slots}
        self._slot_dof = {s.body: self.model.jnt_dofadr[self.model.body_jntadr[self._slot_body[s.body]]]
                          for s in self.slots}
        self._square_slot: dict[int, PieceSlot] = {}
        self._captured: list[PieceSlot] = []

        self.ik = So101Ik(self.model, build_arm(board_spec).compile(), prefix=ARM_PREFIX)
        self._joint_qpos = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_PREFIX + n)] for n in JOINTS])
        self._joint_dof = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_PREFIX + n)] for n in JOINTS])
        self._actuators = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_PREFIX + n)
                           for n in JOINTS]

        self.cameras = tuple(cameras)
        self._image_size = image_size
        self._renderer: mujoco.Renderer | None = None
        self.controller = PickPlaceController(self)
        self._step_count = 0
        self.reach = self._compute_reach()

    def _compute_reach(self) -> ReachMap:
        return ReachMap.compute(self.board_spec, self._tool_query, grasp_height=0.016)

    def _move_board(self, origin) -> None:
        """Put the board's centre at `origin` (world x/y). Pieces are not moved;
        reset() lays them out afterwards."""
        origin = (float(origin[0]), float(origin[1]))
        self.data.mocap_pos[self._board_mocap][:2] = origin
        if origin != self.board_spec.origin:
            self.board_spec = replace(self.board_spec, origin=origin)
            # which squares the scripted grasp can work on depends on where they are
            self.reach = self._compute_reach()

    def _tool_query(self, target: np.ndarray):
        res = self.ik.solve(self.data, target)
        _, mat = self.ik.forward(res.q, np.zeros(3))
        return res, mat

    # -- lifecycle -----------------------------------------------------------

    def reset(self, fen: str = START_FEN, settle_seconds: float = 0.8, rng=None) -> Observation:
        """Lay out a position (FEN board field or full FEN) and park the arm.

        With `rng` (anything with `uniform`, e.g. random.Random) the board is
        shifted up to BOARD_SHIFT from its nominal spot and the arm starts up to
        ARM_START_JITTER from its parked pose; without it both are nominal. The
        draw is kept in `layout`."""
        if rng is None:
            origin = self._nominal_origin
        else:
            origin = (self._nominal_origin[0] + rng.uniform(-BOARD_SHIFT, BOARD_SHIFT),
                      self._nominal_origin[1] + rng.uniform(-BOARD_SHIFT, BOARD_SHIFT))
        self._move_board(origin)
        position = chess.Board(fen if " " in fen else f"{fen} w - - 0 1")
        self.board = position
        self._square_slot = {}
        self._captured = []
        pool = {s.body: s for s in self.slots}
        for square, piece in position.piece_map().items():
            slot = next((s for s in pool.values() if s.piece == piece), None)
            if slot is None:
                raise ValueError(f"position needs more {piece} than the set contains")
            del pool[slot.body]
            self._square_slot[square] = slot
            x, y = self.board_spec.square_center(square)
            self._place(slot, x, y, self.board_spec.top)
        for i, slot in enumerate(pool.values()):
            x, y = self.board_spec.graveyard_slot(i)
            self._place(slot, x, y, self.board_spec.table_top)
        rest = arm_rest_pose()
        if rng is not None:
            ranges = self.model.actuator_ctrlrange[self._actuators]
            rest = {n: float(np.clip(v + rng.uniform(-ARM_START_JITTER, ARM_START_JITTER),
                                     *ranges[JOINTS.index(n)]))
                    for n, v in rest.items()}
        self.layout = {"board_origin": list(self.board_spec.origin), "arm_start": rest}
        for name, value in rest.items():
            adr = self._joint_qpos[JOINTS.index(name)]
            self.data.qpos[adr] = value
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        action = np.array([rest[n] for n in JOINTS])
        for _ in range(int(settle_seconds * self.control_hz)):
            self.apply_action(action)
        self._step_count = 0
        return self.observe()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def recolor(self, appearance: Appearance) -> None:
        """Apply `appearance`'s colors and lighting to the compiled scene without
        recompiling. Piece size and the board texture stay as built."""
        m = self.model
        for slot in self.slots:
            mat = m.material(f"mat_{piece_asset_name(slot.piece)}")
            mat.rgba[:] = appearance.white_rgba if slot.piece.color == chess.WHITE else appearance.black_rgba
        m.geom("table").rgba[:] = [*appearance.table_rgb, 1.0]
        key = m.light("key")
        direction = np.asarray(appearance.light_dir, dtype=float)
        direction /= np.linalg.norm(direction)
        key.dir[:] = direction
        key.pos[:] = -2.4 * direction + [0, 0, 0.4]
        key.diffuse[:] = KEY_LIGHT_DIFFUSE * appearance.light_intensity
        self.appearance = self.appearance.with_(
            white_rgba=appearance.white_rgba, black_rgba=appearance.black_rgba,
            table_rgb=appearance.table_rgb, light_intensity=appearance.light_intensity,
            light_dir=appearance.light_dir)

    # -- control -------------------------------------------------------------

    def apply_action(self, action: np.ndarray,
                     on_step: Callable[[np.ndarray], None] | None = None) -> None:
        """Command joint targets (6, radians) for one control period."""
        action = np.asarray(action, dtype=float)
        if on_step is not None:
            on_step(action)
        for aid, target in zip(self._actuators, action):
            self.data.ctrl[aid] = target
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

    def step(self, action: np.ndarray) -> Observation:
        self.apply_action(action)
        return self.observe()

    def move(self, from_square: str, to_square: str,
             on_step: Callable[[np.ndarray], None] | None = None) -> MoveResult:
        """Execute a move with the scripted expert and verify the outcome."""
        src, dst = parse_square(from_square), parse_square(to_square)
        slot = self._square_slot.get(src)
        if slot is None:
            return MoveResult(from_square, to_square, False, 0.0, [], 0, "no piece on source square")
        if dst in self._square_slot:
            return MoveResult(from_square, to_square, False, 0.0, [], 0, "target square occupied")
        before = {s: self.piece_position(sl)[:2] for s, sl in self._square_slot.items()}
        start_steps = self._step_count

        executed = self.controller.pick_place(slot, self.board_spec.square_center(dst), on_step)
        for _ in range(int(0.3 * self.control_hz)):   # let the released piece settle
            self.apply_action(self._current_action(), on_step)

        pos = self.piece_position(slot)
        tx, ty = self.board_spec.square_center(dst)
        error = float(np.hypot(pos[0] - tx, pos[1] - ty))
        upright = self.piece_upright(slot) > UPRIGHT_MIN
        disturbed = [chess.square_name(s) for s, xy in before.items()
                     if s != src and np.linalg.norm(self.piece_position(self._square_slot[s])[:2] - xy)
                     > DISTURB_TOLERANCE]
        success = error < PLACEMENT_TOLERANCE and upright and not disturbed
        reason = "" if success else (
            "placement error" if error >= PLACEMENT_TOLERANCE else
            "piece fell" if not upright else "disturbed other pieces")
        if error < PLACEMENT_TOLERANCE and upright:
            # the piece physically stands on the target: keep the mirror honest
            # even when the move is flagged for disturbing a neighbor
            self._square_slot[dst] = self._square_slot.pop(src)
            mv = chess.Move(src, dst)
            if mv in self.board.legal_moves:
                self.board.push(mv)
            else:  # keep the mirror consistent even for non-legal training moves
                piece = self.board.remove_piece_at(src)
                self.board.set_piece_at(dst, piece)
        return MoveResult(from_square, to_square, bool(success), error, disturbed,
                          self._step_count - start_steps, reason, waypoints_precise=bool(executed))

    def active_slots(self) -> list[PieceSlot]:
        """Every piece the arm has to work around: those on squares and
        those already in the discard tray."""
        return list(self._square_slot.values()) + self._captured

    def executable_captures(self) -> list[int]:
        """Squares whose piece the arm can lift and set down in the tray."""
        if self.free_capture_slot() is None:
            return []
        tray = np.array(self.free_capture_slot())
        out = []
        for square, slot in self._square_slot.items():
            if square not in self.reach.squares:
                continue
            plan = grasp_plan(slot.geometry, self.board_spec)
            src = np.array([*self.board_spec.square_center(square), plan.z_grasp])
            dst = np.array([*tray, plan.z_grasp])
            if (self.grasp_span(src, plan.off_open, exclude=slot) is not None
                    and self.grasp_span(dst, plan.off_hold, exclude=slot) is not None):
                out.append(square)
        return out

    def free_capture_slot(self) -> tuple[float, float] | None:
        """The next unused discard position, or None once the tray is full."""
        if len(self._captured) >= self.board_spec.capture_slots:
            return None
        return self.board_spec.capture_slot(len(self._captured))

    def capture(self, square: str,
                on_step: Callable[[np.ndarray], None] | None = None) -> MoveResult:
        """Take the piece on `square` off the board and set it down in the tray.

        This is the first half of a capture: the piece standing on the target
        square is removed before the capturing piece moves onto it. Where in the
        tray it lands hardly matters, so the placement rule is looser than a
        move's - what matters is that it leaves the board, stays upright and
        disturbs nothing still in play."""
        src = parse_square(square)
        slot = self._square_slot.get(src)
        if slot is None:
            return MoveResult(square, "off board", False, 0.0, [], 0, "no piece on source square")
        target = self.free_capture_slot()
        if target is None:
            return MoveResult(square, "off board", False, 0.0, [], 0, "discard tray is full")
        before = {s: self.piece_position(sl)[:2] for s, sl in self._square_slot.items()}
        start_steps = self._step_count

        executed = self.controller.pick_place(slot, target, on_step,
                                              target_z=self.board_spec.table_top)
        for _ in range(int(0.3 * self.control_hz)):
            self.apply_action(self._current_action(), on_step)

        pos = self.piece_position(slot)
        error = float(np.hypot(pos[0] - target[0], pos[1] - target[1]))
        off_board = not self.board_spec.on_field(pos[:2])
        upright = self.piece_upright(slot) > UPRIGHT_MIN
        disturbed = [chess.square_name(s) for s, xy in before.items()
                     if s != src and np.linalg.norm(self.piece_position(self._square_slot[s])[:2] - xy)
                     > DISTURB_TOLERANCE]
        success = off_board and error < CAPTURE_TOLERANCE and upright and not disturbed
        reason = "" if success else (
            "still on the board" if not off_board else
            "missed the tray" if error >= CAPTURE_TOLERANCE else
            "piece fell" if not upright else "disturbed other pieces")
        if off_board and upright and error < CAPTURE_TOLERANCE:
            self._captured.append(self._square_slot.pop(src))
            self.board.remove_piece_at(src)
        return MoveResult(square, "off board", bool(success), error, disturbed,
                          self._step_count - start_steps, reason, waypoints_precise=bool(executed))

    # -- observation ---------------------------------------------------------

    def observe(self, images: bool = True) -> Observation:
        imgs = {cam: self.render(cam) for cam in self.cameras} if images else {}
        return Observation(time=float(self.data.time),
                           joint_pos=self.arm_joint_positions().copy(),
                           joint_vel=self.data.qvel[self._joint_dof].copy(),
                           images=imgs, fen=self.board.fen())

    def render(self, camera: str = "top") -> np.ndarray:
        if self._renderer is None:
            w, h = self._image_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def arm_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._joint_qpos]

    def _current_action(self) -> np.ndarray:
        return np.array([self.data.ctrl[a] for a in self._actuators])

    def jaw_clearance(self, xy, direction=None, exclude: PieceSlot | None = None):
        """Clearance margin (meters) of a jaw-span `direction` at `xy`, or of
        the best direction as (direction, margin) when none is given.

        The moving jaw opens toward +direction and needs ~MOVING_JAW_ROOM of
        free space past the piece; the fixed prong sits on -direction and needs
        ~FIXED_PRONG_ROOM. The margin is the smaller of the two surpluses.
        """
        xy = np.asarray(xy, dtype=float)
        others = [(self.piece_position(sl)[:2] - xy, sl.geometry.collider_radius)
                  for sl in self.active_slots() if sl is not exclude]
        lane = 0.75 * self.board_spec.square

        def room(d):
            """Free distance along `d` to the nearest piece surface."""
            best = 0.30
            for rel, radius in others:
                along, across = np.dot(rel, d), float(np.cross(d, rel))
                if along > 0 and abs(across) < lane:
                    best = min(best, along - radius)
            return best

        def margin(d):
            return float(min(room(d) - MOVING_JAW_ROOM, room(-d) - FIXED_PRONG_ROOM))

        if direction is not None:
            return margin(np.asarray(direction, dtype=float))
        return max(((d, margin(d)) for d in SPAN_DIRECTIONS), key=lambda o: o[1])

    def span_options(self, xy, exclude: PieceSlot | None = None):
        """All four jaw-span directions at `xy` with their clearance margins,
        best first."""
        return sorted(((d, self.jaw_clearance(xy, d, exclude)) for d in SPAN_DIRECTIONS),
                      key=lambda o: -o[1])

    def grasp_span(self, point, offset, exclude: PieceSlot | None = None) -> np.ndarray | None:
        """Jaw-span direction to work at world `point` (tool point `offset`):
        the direction with the most clearance that the arm can realize with the
        tool near vertical and without folding into itself. None when no
        direction has both clearance and a solution."""
        for direction, margin in self.span_options(point[:2], exclude):
            if margin < 0:
                break
            res = self.ik.solve(self.data, point, offset=offset, span=direction)
            if res.position_error <= MAX_IK_ERROR and res.tilt <= MAX_TILT:
                return direction
        return None

    def executable_moves(self) -> list[chess.Move]:
        """Quiet legal moves the expert can execute here: reachable endpoints
        where some jaw span has clearance from the neighbors and a
        collision-free arm solution."""
        moves = []
        for mv in self.reach.executable_moves(self.board):
            slot = self._square_slot[mv.from_square]
            plan = grasp_plan(slot.geometry, self.board_spec)
            src = np.array([*self.board_spec.square_center(mv.from_square), plan.z_grasp])
            dst = np.array([*self.board_spec.square_center(mv.to_square), plan.z_grasp])
            if (self.grasp_span(src, plan.off_open, exclude=slot) is not None
                    and self.grasp_span(dst, plan.off_hold, exclude=slot) is not None):
                moves.append(mv)
        return moves

    # -- pieces --------------------------------------------------------------

    def slot_at(self, square: str) -> PieceSlot | None:
        return self._square_slot.get(parse_square(square))

    def piece_position(self, slot: PieceSlot) -> np.ndarray:
        return self.data.xpos[self._slot_body[slot.body]].copy()

    def piece_upright(self, slot: PieceSlot) -> float:
        return float(self.data.xmat[self._slot_body[slot.body]].reshape(3, 3)[2, 2])

    def set_piece_pose(self, slot: PieceSlot, position, quat, zero_velocity=False) -> None:
        adr = self._slot_qpos[slot.body]
        self.data.qpos[adr:adr + 3] = position
        self.data.qpos[adr + 3:adr + 7] = quat
        if zero_velocity:
            dof = self._slot_dof[slot.body]
            self.data.qvel[dof:dof + 6] = 0

    def _place(self, slot: PieceSlot, x: float, y: float, surface_z: float) -> None:
        z = surface_z - slot.geometry.bottom_offset + 0.0015
        self.set_piece_pose(slot, [x - slot.geometry.center[0], y - slot.geometry.center[1], z],
                            [1, 0, 0, 0], zero_velocity=True)

    # -- export --------------------------------------------------------------

    def export_xml(self, path: str) -> str:
        """Standalone MuJoCo XML of the scene in its current position: open it
        with `python -m mujoco.viewer --mjcf <path>`."""
        return export_xml(self.spec, path, self.model, self.data)
