"""ChessSimEnv: the SO-101 playing chess in MuJoCo.

    env = ChessSimEnv(EnvConfig())
    env.reset(START_FEN)
    result = env.execute(MoveTask(from_square="g1", to_square="f3"))  # scripted expert
    obs = env.step(action)                                            # or drive the joints yourself
    frame = env.render(Camera.EXTERNAL)

`execute` and a policy rollout are scored by the same rule (`evaluate`), so an
expert demonstration and a learned attempt are directly comparable.
"""
from __future__ import annotations

from typing import Callable

import chess
import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from .assets import piece_asset_name
from .config import (JOINTS, START_FEN, AppearanceConfig, Camera, EnvConfig, FailureReason,
                     JointPose, Square, square_at, square_index)
from .controller import PickPlaceController, grasp_plan
from .ik import So101Ik
from .reach import MAX_IK_ERROR, MAX_TILT, ReachMap
from .scene import (ARM_PREFIX, KEY_LIGHT_DIFFUSE, PieceSlot, arm_rest_pose, build_arm, build_scene,
                    export_xml, piece_slots)
from .tasks import AnyTask, CaptureTask, MoveTask, Task, TaskFamily

# candidate jaw-span directions: the moving jaw opens along one board axis
SPAN_DIRECTIONS = (np.array([0.0, 1.0]), np.array([0.0, -1.0]),
                   np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
GRASP_HEIGHT = 0.016      # tool height above the board used to map out reachable squares

OnStep = Callable[[np.ndarray], None]


class Record(BaseModel):
    """Frozen output value; holds numpy arrays as they are."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class CameraImages(Record):
    """What each camera saw, HxWx3 uint8. Indexed by camera: `images[Camera.TOP]`.

    The shape is checked on the way in, which is worth the microsecond once
    frames come from a real camera rather than the renderer.
    """

    frames: dict[Camera, np.ndarray] = {}

    @field_validator("frames")
    @classmethod
    def _check_frames(cls, frames: dict) -> dict:
        for camera, frame in frames.items():
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"{camera} frame must be HxWx3 uint8, got "
                                 f"{frame.shape} {frame.dtype}")
        return frames

    def __getitem__(self, camera: Camera) -> np.ndarray:
        return self.frames[camera]

    def __contains__(self, camera: Camera) -> bool:
        return camera in self.frames

    def __iter__(self):
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def items(self):
        return self.frames.items()


class Observation(Record):
    time: float
    joint_pos: np.ndarray                      # the six joints, radians
    joint_vel: np.ndarray
    images: CameraImages = CameraImages()      # one frame per configured camera
    fen: str = ""


class Layout(Record):
    """Where the board and the arm were placed for this episode."""

    board_origin: tuple[float, float]
    arm_start: JointPose


class PieceSnapshot(Record):
    """Where every piece stood at a moment - the baseline `evaluate` scores against."""

    pieces: dict[Square, tuple[float, float]]

    def __getitem__(self, square: Square) -> tuple[float, float]:
        return self.pieces[square]

    def __contains__(self, square: Square) -> bool:
        return square in self.pieces

    def items(self):
        return self.pieces.items()


class TaskResult(Record):
    """The verdict on one attempt, scripted or learned."""

    task: AnyTask
    success: bool
    placement_error: float                     # meters from the target
    upright: bool
    disturbed: list[Square]                    # squares whose pieces moved
    picked: Square | None                      # square of the piece that actually moved
    right_piece: bool
    steps: int
    reason: FailureReason | None = None
    waypoints_precise: bool = True             # diagnostic: the expert's waypoints hit tolerance

    @property
    def instruction(self) -> str:
        return self.task.instruction


class ChessSimEnv:
    def __init__(self, config: EnvConfig = EnvConfig()):
        self.config = config
        self.board = config.board                     # live geometry: reset() may shift its origin
        self.position = chess.Board(None)             # mirror of what stands on the board
        self.spec = build_scene(config.board, config.appearance)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self._board_mocap = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "board")]
        self.layout = Layout(board_origin=config.board.origin, arm_start=arm_rest_pose())
        self.control_hz = config.control.hz
        self.substeps = max(1, int(round(1.0 / (self.control_hz * self.model.opt.timestep))))

        self.slots = piece_slots(config.board, config.appearance.piece_scale)
        self._slot_body = {s.body: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, s.body)
                           for s in self.slots}
        self._slot_qpos = {s.body: self.model.jnt_qposadr[self.model.body_jntadr[self._slot_body[s.body]]]
                           for s in self.slots}
        self._slot_dof = {s.body: self.model.jnt_dofadr[self.model.body_jntadr[self._slot_body[s.body]]]
                          for s in self.slots}
        self._square_slot: dict[int, PieceSlot] = {}
        self._captured: list[PieceSlot] = []

        self.ik = So101Ik(self.model, build_arm(config.board).compile(), prefix=ARM_PREFIX)
        self._joint_qpos = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_PREFIX + n)] for n in JOINTS])
        self._joint_dof = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_PREFIX + n)] for n in JOINTS])
        self._actuators = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_PREFIX + n)
                           for n in JOINTS]

        self.cameras = tuple(config.control.cameras)
        self._renderer: mujoco.Renderer | None = None
        self.controller = PickPlaceController(self)
        self._step_count = 0
        self.reach = self._compute_reach()

    # -- lifecycle -----------------------------------------------------------

    def reset(self, position: str | chess.Board = START_FEN, rng=None) -> Observation:
        """Lay out a position (FEN board field, full FEN or board) and park the arm.

        With `rng` (anything with `uniform`, e.g. random.Random) the board is
        shifted and the arm's start pose jittered within `RandomizationConfig`;
        without it both are nominal. The draw is kept in `layout`.
        """
        nominal = self.config.board.origin
        shift = self.config.randomization.board_shift
        origin = nominal if rng is None else (nominal[0] + rng.uniform(-shift, shift),
                                              nominal[1] + rng.uniform(-shift, shift))
        self._move_board(origin)
        if isinstance(position, chess.Board):
            position = position.board_fen()
        self.position = chess.Board(position if " " in position else f"{position} w - - 0 1")
        self._square_slot = {}
        self._captured = []
        pool = {s.body: s for s in self.slots}
        for square, piece in self.position.piece_map().items():
            slot = next((s for s in pool.values() if s.piece == piece), None)
            if slot is None:
                raise ValueError(f"position needs more {piece} than the set contains")
            del pool[slot.body]
            self._square_slot[square] = slot
            x, y = self.board.square_center(square)
            self._place(slot, x, y, self.board.top)
        for i, slot in enumerate(pool.values()):
            x, y = self.board.graveyard_slot(i)
            self._place(slot, x, y, self.board.table_top)

        start = arm_rest_pose().to_array()
        if rng is not None:
            jitter = self.config.randomization.arm_joint_jitter
            ranges = self.model.actuator_ctrlrange[self._actuators]
            start = np.array([float(np.clip(v + rng.uniform(-jitter, jitter), *limits))
                              for v, limits in zip(start, ranges)])
        self.layout = Layout(board_origin=self.board.origin, arm_start=JointPose.from_array(start))
        self.data.qpos[self._joint_qpos] = start
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        action = start
        for _ in range(int(self.config.control.settle_seconds * self.control_hz)):
            self.apply_action(action)
        self._step_count = 0
        return self.observe()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def recolor(self, appearance: AppearanceConfig) -> None:
        """Apply colors and lighting to the compiled scene without recompiling.
        Piece size and the board texture stay as built."""
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
        self.config = self.config.model_copy(update={"appearance": self.config.appearance.model_copy(
            update={k: getattr(appearance, k) for k in
                    ("white_rgba", "black_rgba", "table_rgb", "light_intensity", "light_dir")})})

    def _move_board(self, origin) -> None:
        """Put the board's centre at `origin` (world x/y). Pieces are not moved;
        reset() lays them out afterwards."""
        origin = (float(origin[0]), float(origin[1]))
        self.data.mocap_pos[self._board_mocap][:2] = origin
        if origin != self.board.origin:
            self.board = self.board.model_copy(update={"origin": origin})
            # which squares the scripted grasp can work on depends on where they are
            self.reach = self._compute_reach()

    def _compute_reach(self) -> ReachMap:
        return ReachMap.compute(self.board, self._tool_query, grasp_height=GRASP_HEIGHT)

    def _tool_query(self, target: np.ndarray):
        res = self.ik.solve(self.data, target)
        _, mat = self.ik.forward(res.q, np.zeros(3))
        return res, mat

    # -- control -------------------------------------------------------------

    def apply_action(self, action: np.ndarray, on_step: OnStep | None = None) -> None:
        """Command joint targets (six joints, radians) for one control period."""
        action = np.asarray(action, dtype=float)
        if on_step is not None:
            on_step(action)
        for aid, target in zip(self._actuators, action):
            self.data.ctrl[aid] = target
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

    def step(self, action: np.ndarray, on_step: OnStep | None = None,
             images: bool = True) -> Observation:
        """One control period. Reward and termination are the caller's rule -
        see `chess_sim.rewards` and `env.evaluate`. Pass `images=False` when the
        caller will not look at them: rendering is most of the cost of a step."""
        self.apply_action(action, on_step)
        return self.observe(images=images)

    # -- scripted expert -----------------------------------------------------

    def execute(self, task: Task, on_step: OnStep | None = None) -> TaskResult:
        """Carry out a task with the scripted expert and score it.

        This plans against the simulator state, so it is a demonstrator, not an
        inference-time policy: use it to collect data and to set the bar.
        """
        before = self.piece_snapshot()
        started = self._step_count
        slot = self._slot_for(task)
        if slot is None:
            return self._rejected(task, FailureReason.NO_PIECE)
        if task.family is TaskFamily.MOVE:
            if square_index(task.to_square) in self._square_slot:
                return self._rejected(task, FailureReason.OCCUPIED)
            target, target_z = self.board.square_center(task.to_square), None
        else:
            tray = self.free_tray_slot()
            if tray is None:
                return self._rejected(task, FailureReason.TRAY_FULL)
            target, target_z = tray, self.board.table_top

        precise = self.controller.pick_place(slot, target, on_step, target_z=target_z)
        for _ in range(int(self.config.control.release_seconds * self.control_hz)):
            self.apply_action(self._current_action(), on_step)
        result = self.evaluate(task, before, steps=self._step_count - started,
                               waypoints_precise=bool(precise))
        self._commit(task, result)
        return result

    def move(self, from_square, to_square, on_step: OnStep | None = None) -> TaskResult:
        return self.execute(MoveTask(from_square=square_at(from_square),
                                     to_square=square_at(to_square)), on_step)

    def capture(self, square, on_step: OnStep | None = None) -> TaskResult:
        return self.execute(CaptureTask(square=square_at(square)), on_step)

    # -- scoring -------------------------------------------------------------

    def piece_snapshot(self) -> PieceSnapshot:
        """Where every piece on the board stands right now; pass it to `evaluate`."""
        return PieceSnapshot(pieces={square_at(sq): tuple(self.piece_position(slot)[:2])
                                     for sq, slot in self._square_slot.items()})

    def evaluate(self, task: Task, before: PieceSnapshot, steps: int = 0,
                 waypoints_precise: bool = True) -> TaskResult:
        """Score an attempt against the snapshot it started from.

        A move succeeds when the piece stands within `tolerances.placement` of
        the target square, upright, and nothing else was displaced. A capture is
        looser about where it lands - what matters is that the piece left the
        board, stayed upright and disturbed nothing still in play.
        """
        tol = self.config.tolerances
        slot = self._slot_for(task)
        if slot is None:
            return self._rejected(task, FailureReason.NO_PIECE, steps)
        position = self.piece_position(slot)
        target = np.asarray(task.target_xy(self))
        error = float(np.linalg.norm(position[:2] - target))
        upright = self.piece_upright(slot) > tol.upright_min
        moved = {square: float(np.linalg.norm(
            self.piece_position(self._square_slot[square_index(square)])[:2] - np.asarray(xy)))
            for square, xy in before.items()}
        disturbed = [square for square, distance in moved.items()
                     if square != task.source and distance > tol.disturbance]
        # Which piece actually moved. With several pieces in play, going to the
        # wrong square is a failure of grounding and needs telling apart from a
        # clumsy grasp of the right one.
        picked = max(moved, key=moved.get, default=None)
        if picked is None or moved[picked] <= tol.disturbance:
            picked = None

        if task.family is TaskFamily.CAPTURE:
            off_board = not self.board.on_field(position[:2])
            success = off_board and error < tol.tray and upright and not disturbed
            reason = None if success else (
                FailureReason.STILL_ON_BOARD if not off_board else
                FailureReason.MISSED_TRAY if error >= tol.tray else
                FailureReason.FELL if not upright else FailureReason.DISTURBED)
        else:
            success = error < tol.placement and upright and not disturbed
            reason = None if success else (
                FailureReason.PLACEMENT if error >= tol.placement else
                FailureReason.FELL if not upright else FailureReason.DISTURBED)
        return TaskResult(task=task, success=bool(success), placement_error=error, upright=upright,
                          disturbed=disturbed, picked=picked, right_piece=picked == task.source,
                          steps=steps, reason=reason, waypoints_precise=waypoints_precise)

    def _commit(self, task: Task, result: TaskResult) -> None:
        """Keep the mirror honest: a piece that physically reached its target has
        moved, even when the attempt is flagged for disturbing a neighbor."""
        tol = self.config.tolerances
        source = square_index(task.source)
        if not result.upright or source not in self._square_slot:
            return
        if task.family is TaskFamily.CAPTURE:
            if result.placement_error < tol.tray and not self.board.on_field(
                    self.piece_position(self._square_slot[source])[:2]):
                self._captured.append(self._square_slot.pop(source))
                self.position.remove_piece_at(source)
            return
        if result.placement_error >= tol.placement:
            return
        destination = square_index(task.to_square)
        self._square_slot[destination] = self._square_slot.pop(source)
        move = chess.Move(source, destination)
        if move in self.position.legal_moves:
            self.position.push(move)
        else:   # keep the mirror consistent even for moves no chess rule allows
            self.position.set_piece_at(destination, self.position.remove_piece_at(source))

    def _rejected(self, task: Task, reason: FailureReason, steps: int = 0) -> TaskResult:
        return TaskResult(task=task, success=False, placement_error=0.0, upright=True, disturbed=[],
                          picked=None, right_piece=False, steps=steps, reason=reason)

    def _slot_for(self, task: Task) -> PieceSlot | None:
        return self._square_slot.get(square_index(task.source))

    # -- what the arm can attempt here ---------------------------------------

    def executable_moves(self) -> list[chess.Move]:
        """Quiet legal moves the expert can execute here: reachable endpoints
        where some jaw span has clearance from the neighbors and a
        collision-free arm solution."""
        moves = []
        for move in self.reach.executable_moves(self.position):
            slot = self._square_slot[move.from_square]
            plan = grasp_plan(slot.geometry, self.board)
            source = np.array([*self.board.square_center(move.from_square), plan.z_grasp])
            target = np.array([*self.board.square_center(move.to_square), plan.z_grasp])
            if (self.grasp_span(source, plan.off_open, exclude=slot) is not None
                    and self.grasp_span(target, plan.off_hold, exclude=slot) is not None):
                moves.append(move)
        return moves

    def executable_captures(self) -> list[int]:
        """Squares whose piece the arm can lift and set down in the tray."""
        slot_xy = self.free_tray_slot()
        if slot_xy is None:
            return []
        tray = np.array(slot_xy)
        out = []
        for square, slot in self._square_slot.items():
            if square not in self.reach.squares:
                continue
            plan = grasp_plan(slot.geometry, self.board)
            source = np.array([*self.board.square_center(square), plan.z_grasp])
            target = np.array([*tray, plan.z_grasp])
            if (self.grasp_span(source, plan.off_open, exclude=slot) is not None
                    and self.grasp_span(target, plan.off_hold, exclude=slot) is not None):
                out.append(square)
        return out

    def free_tray_slot(self) -> tuple[float, float] | None:
        """The next unused discard position, or None once the tray is full."""
        if len(self._captured) >= self.board.tray_slots:
            return None
        return self.board.tray_slot(len(self._captured))

    def active_slots(self) -> list[PieceSlot]:
        """Every piece the arm has to work around: those on squares and those
        already in the discard tray."""
        return list(self._square_slot.values()) + self._captured

    # -- observation ---------------------------------------------------------

    def observe(self, images: bool = True) -> Observation:
        frames = {cam: self.render(cam) for cam in self.cameras} if images else {}
        return Observation(time=float(self.data.time),
                           joint_pos=self.arm_joint_positions().copy(),
                           joint_vel=self.data.qvel[self._joint_dof].copy(),
                           images=CameraImages(frames=frames), fen=self.position.fen())

    def render(self, camera: Camera | str = Camera.TOP) -> np.ndarray:
        if self._renderer is None:
            width, height = self.config.control.image_size
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=str(camera))
        return self._renderer.render()

    def arm_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._joint_qpos]

    def _current_action(self) -> np.ndarray:
        return np.array([self.data.ctrl[a] for a in self._actuators])

    # -- pieces --------------------------------------------------------------

    def slot_at(self, square: int | str | Square) -> PieceSlot | None:
        return self._square_slot.get(square_index(square))

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

    # -- grasp geometry ------------------------------------------------------

    def jaw_clearance(self, xy, direction=None, exclude: PieceSlot | None = None):
        """Clearance margin (meters) of a jaw-span `direction` at `xy`, or of the
        best direction as (direction, margin) when none is given.

        The moving jaw opens toward +direction and needs `moving_jaw_room` of
        free space past the piece; the fixed prong sits on -direction and needs
        `fixed_prong_room`. The margin is the smaller of the two surpluses.
        """
        clearance = self.config.clearance
        xy = np.asarray(xy, dtype=float)
        others = [(self.piece_position(sl)[:2] - xy, sl.geometry.collider_radius)
                  for sl in self.active_slots() if sl is not exclude]
        lane = clearance.lane_fraction * self.board.square

        def room(d):
            """Free distance along `d` to the nearest piece surface."""
            best = clearance.probe_distance
            for rel, radius in others:
                along, across = np.dot(rel, d), float(np.cross(d, rel))
                if along > 0 and abs(across) < lane:
                    best = min(best, along - radius)
            return best

        def margin(d):
            return float(min(room(d) - clearance.moving_jaw_room,
                             room(-d) - clearance.fixed_prong_room))

        if direction is not None:
            return margin(np.asarray(direction, dtype=float))
        return max(((d, margin(d)) for d in SPAN_DIRECTIONS), key=lambda o: o[1])

    def span_options(self, xy, exclude: PieceSlot | None = None):
        """All four jaw-span directions at `xy` with their clearance margins, best first."""
        return sorted(((d, self.jaw_clearance(xy, d, exclude)) for d in SPAN_DIRECTIONS),
                      key=lambda o: -o[1])

    def grasp_span(self, point, offset, exclude: PieceSlot | None = None) -> np.ndarray | None:
        """Jaw-span direction to work at world `point` (tool point `offset`): the
        direction with the most clearance that the arm can realize with the tool
        near vertical and without folding into itself. None when no direction has
        both clearance and a solution."""
        for direction, margin in self.span_options(point[:2], exclude):
            if margin < 0:
                break
            res = self.ik.solve(self.data, point, offset=offset, span=direction)
            if res.position_error <= MAX_IK_ERROR and res.tilt <= MAX_TILT:
                return direction
        return None

    # -- export --------------------------------------------------------------

    def export_xml(self, path: str) -> str:
        """Standalone MuJoCo XML of the scene in its current position: open it
        with `python -m mujoco.viewer --mjcf <path>`."""
        return export_xml(self.spec, path, self.model, self.data)
