"""ChessSimEnv: the user-facing simulation of an SO-101 playing chess.

    env = ChessSimEnv()
    env.reset("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    env.move("g1", "f3")            # scripted expert; env.board mirrors the result
    obs = env.step(action)          # or drive the joints yourself (6 targets, rad)
    frame = env.render("external")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import chess
import mujoco
import numpy as np

from .board import START_FEN, BoardSpec, parse_square
from .controller import PickPlaceController
from .ik import So101Ik
from .reach import ReachMap
from .scene import ARM_PREFIX, CAMERA_NAMES, PieceSlot, arm_rest_pose, build_scene, export_xml, piece_slots

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
PLACEMENT_TOLERANCE = 0.011   # piece axis vs square center
DISTURB_TOLERANCE = 0.010     # any other piece moving more than this fails a move
UPRIGHT_MIN = 0.85            # z-axis alignment of a standing piece
# free space the open jaw needs beyond the grasped piece's surface
MOVING_JAW_ROOM = 0.020       # moving prong swings out this far past the piece
FIXED_PRONG_ROOM = 0.009      # fixed prong: ~3 mm slack + prong thickness


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


class ChessSimEnv:
    def __init__(self, board_spec: BoardSpec = BoardSpec(),
                 cameras: tuple[str, ...] = CAMERA_NAMES,
                 image_size: tuple[int, int] = (640, 480),
                 control_hz: int = 30, grasp: str = "physical"):
        """`grasp`: "physical" (friction between fingertip pads) or "kinematic"
        (piece attached to the gripper at pinch time)."""
        self.board_spec = board_spec
        self.spec = build_scene(board_spec)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.control_hz = control_hz
        self.substeps = max(1, int(round(1.0 / (control_hz * self.model.opt.timestep))))
        self.board = chess.Board(None)

        self.slots = piece_slots(board_spec)
        self._slot_body = {s.body: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, s.body)
                           for s in self.slots}
        self._slot_qpos = {s.body: self.model.jnt_qposadr[self.model.body_jntadr[self._slot_body[s.body]]]
                           for s in self.slots}
        self._slot_dof = {s.body: self.model.jnt_dofadr[self.model.body_jntadr[self._slot_body[s.body]]]
                          for s in self.slots}
        self._square_slot: dict[int, PieceSlot] = {}

        self.ik = So101Ik(self.model, prefix=ARM_PREFIX)
        self._joint_qpos = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_PREFIX + n)] for n in JOINTS])
        self._joint_dof = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_PREFIX + n)] for n in JOINTS])
        self._actuators = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_PREFIX + n)
                           for n in JOINTS]

        self.cameras = tuple(cameras)
        self._image_size = image_size
        self._renderer: mujoco.Renderer | None = None
        self.controller = PickPlaceController(self, mode=grasp)
        self._step_count = 0
        self.reach = ReachMap.compute(board_spec, self._tool_query, grasp_height=0.016)

    def _tool_query(self, target: np.ndarray):
        res = self.ik.solve(self.data, target)
        _, mat = self.ik.forward(res.q, self.data, np.zeros(3))
        return res, mat

    # -- lifecycle -----------------------------------------------------------

    def reset(self, fen: str = START_FEN, settle_seconds: float = 0.8) -> Observation:
        """Lay out a position (FEN board field or full FEN) and park the arm."""
        position = chess.Board(fen if " " in fen else f"{fen} w - - 0 1")
        self.board = position
        self.controller.grasp = None
        self._square_slot = {}
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
            self.controller.update_grasp()
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

    # -- observation ---------------------------------------------------------

    def observe(self, images: bool = True) -> Observation:
        imgs = {cam: self.render(cam) for cam in self.cameras} if images else {}
        return Observation(time=float(self.data.time),
                           joint_pos=self.arm_joint_positions().copy(),
                           joint_vel=self.data.qvel[self._joint_dof].copy(),
                           images=imgs, fen=self.board.fen())

    def render(self, camera: str = "external") -> np.ndarray:
        if self._renderer is None:
            w, h = self._image_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def arm_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._joint_qpos]

    def _current_action(self) -> np.ndarray:
        return np.array([self.data.ctrl[a] for a in self._actuators])

    def jaw_clearance(self, xy, exclude: PieceSlot | None = None):
        """Best jaw-span direction at `xy` and its clearance margin (meters).

        The moving jaw opens toward +direction and needs ~MOVING_JAW_ROOM of
        free space past the piece; the fixed prong sits on -direction and needs
        ~FIXED_PRONG_ROOM. The margin is the smaller of the two surpluses.
        """
        xy = np.asarray(xy, dtype=float)
        others = [(self.piece_position(sl)[:2] - xy, sl.geometry.collider_radius)
                  for sl in self._square_slot.values() if sl is not exclude]
        lane = 0.75 * self.board_spec.square

        def room(direction):
            """Free distance along `direction` to the nearest piece surface."""
            best = 0.30
            for rel, radius in others:
                along, across = np.dot(rel, direction), float(np.cross(direction, rel))
                if along > 0 and abs(across) < lane:
                    best = min(best, along - radius)
            return best

        best_dir, best_margin = None, -1.0
        for d in (np.array([0.0, 1.0]), np.array([0.0, -1.0]),
                  np.array([1.0, 0.0]), np.array([-1.0, 0.0])):
            margin = min(room(d) - MOVING_JAW_ROOM, room(-d) - FIXED_PRONG_ROOM)
            if margin > best_margin:
                best_dir, best_margin = d, margin
        return best_dir, float(best_margin)

    def free_span_direction(self, xy, exclude: PieceSlot | None = None) -> np.ndarray:
        return self.jaw_clearance(xy, exclude)[0]

    def executable_moves(self) -> list[chess.Move]:
        """Quiet legal moves the expert can execute here without touching
        neighbors: reachable endpoints with adequate jaw clearance at both."""
        moves = []
        for mv in self.reach.executable_moves(self.board):
            slot = self._square_slot[mv.from_square]
            src_xy = self.board_spec.square_center(mv.from_square)
            dst_xy = self.board_spec.square_center(mv.to_square)
            if (self.jaw_clearance(src_xy, exclude=slot)[1] >= 0
                    and self.jaw_clearance(dst_xy, exclude=slot)[1] >= 0):
                moves.append(mv)
        return moves

    # -- pieces --------------------------------------------------------------

    def slot_at(self, square: str) -> PieceSlot | None:
        return self._square_slot.get(parse_square(square))

    def piece_position(self, slot: PieceSlot) -> np.ndarray:
        return self.data.xpos[self._slot_body[slot.body]].copy()

    def piece_upright(self, slot: PieceSlot) -> float:
        return float(self.data.xmat[self._slot_body[slot.body]].reshape(3, 3)[2, 2])

    def piece_yaw(self, slot: PieceSlot) -> float:
        m = self.data.xmat[self._slot_body[slot.body]].reshape(3, 3)
        return float(np.arctan2(m[1, 0], m[0, 0]))

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
        """Standalone MuJoCo XML of the scene (pieces at their graveyard slots)."""
        return export_xml(self.spec, path)
