"""Typed configuration for the simulator.

Every number that shapes the scene, the success rule or the control loop lives
here, in a frozen pydantic model that can be printed, diffed and stored next to
a run:

    config = EnvConfig(board=BoardConfig(square=0.030),
                       control=ControlConfig(hz=30, cameras=ROBOT_CAMERAS))
    env = ChessSimEnv(config)

The defaults are the measured ones - the values every result in docs/EXPERIMENTS.md
was produced with - so `EnvConfig()` reproduces the published setup.
"""
from __future__ import annotations

from collections.abc import Sequence

from enum import StrEnum

import chess
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

START_FEN = chess.STARTING_BOARD_FEN
FILES = "abcdefgh"

# The 64 squares as an enum, in python-chess index order: Square.E4 is "e4", and
# a plain "e4" still validates into it.
Square = StrEnum("Square", {name.upper(): name for name in chess.SQUARE_NAMES})
Square.__doc__ = "A board square, a1 ... h8."
SQUARES: tuple["Square", ...] = tuple(Square)


class Config(BaseModel):
    """Frozen and strict: a config is a value, and an unknown field is a typo."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Camera(StrEnum):
    TOP = "top"          # overhead, sees the whole board
    WRIST = "wrist"      # on the gripper, sees what it is about to grasp
    EXTERNAL = "external"  # viewing only: never recorded, never given to a policy


ROBOT_CAMERAS: tuple[Camera, ...] = (Camera.TOP, Camera.WRIST)   # what the real arm has


class FailureReason(StrEnum):
    """Why an attempt did not count. The wording is what the logs have always said."""

    NO_PIECE = "no piece on source square"
    OCCUPIED = "target square occupied"
    TRAY_FULL = "discard tray is full"
    PLACEMENT = "placement error"
    STILL_ON_BOARD = "still on the board"
    MISSED_TRAY = "missed the tray"
    FELL = "piece fell"
    DISTURBED = "disturbed other pieces"


class RecoveryTrigger(StrEnum):
    """What made a scripted expert take over from a policy mid-episode."""

    WRONG_PIECE = "engaged the wrong piece"
    DISTURBED = "disturbed a neighbour"
    NUDGED = "pushed the named piece without lifting it"
    TOPPLED = "knocked the named piece over"    # nothing to correct: the expert has no grasp for it
    STALLED = "named piece had not moved"


class JointName(StrEnum):
    """The six SO-101 joints, in the order every action and state vector uses."""

    SHOULDER_PAN = "shoulder_pan"
    SHOULDER_LIFT = "shoulder_lift"
    ELBOW_FLEX = "elbow_flex"
    WRIST_FLEX = "wrist_flex"
    WRIST_ROLL = "wrist_roll"
    GRIPPER = "gripper"


JOINTS: tuple[JointName, ...] = tuple(JointName)
DOF = len(JOINTS)


class JointPose(Config):
    """One angle per joint, in radians - an arm pose you can store and diff."""

    shoulder_pan: float = 0.0
    shoulder_lift: float = 0.0
    elbow_flex: float = 0.0
    wrist_flex: float = 0.0
    wrist_roll: float = 0.0
    gripper: float = 0.0

    @classmethod
    def from_array(cls, values) -> "JointPose":
        return cls(**{str(joint): float(v) for joint, v in zip(JOINTS, values)})

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, str(joint)) for joint in JOINTS])

    def __getitem__(self, joint: JointName) -> float:
        return getattr(self, str(joint))


class BoardConfig(Config):
    """Physical layout of the board and the arm mount, in meters.

    The board is procedural - a bordered slab with an 8x8 checker texture - so
    every square center is an exact analytic coordinate in the world frame. The
    board center sits at `origin` in x/y and the arm sits on the -y side.
    """

    square: float = 0.028          # square edge (mini set: SO-101 reach is ~33 cm)
    border: float = 0.015          # wood border around the 8x8 field
    thickness: float = 0.012       # board slab thickness
    table_top: float = 0.43        # table height above the floor
    arm_gap: float = 0.08          # distance from board edge to arm base center
    arm_riser: float = 0.06        # pedestal height under the arm base
    # where the board's centre sits on the table; the arm, camera mast, tray and
    # parked pieces do not move with it
    origin: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0                # radians: the board turned about its centre, as a person sets it down

    # Discard tray: the camera mast stands at x = +0.16 and unused pieces park
    # beyond x = +0.187, so the tray goes to the arm's left where nothing else is.
    tray_x: float = -0.16
    tray_y0: float = -0.10
    tray_pitch: float = 0.04
    tray_slots: int = 4

    # Parking grid for pieces absent from the position, off the board's right edge.
    graveyard_gap: float = 0.06
    graveyard_pitch: float = 0.035
    graveyard_columns: int = 4

    @property
    def field(self) -> float:
        return 8 * self.square

    @property
    def width(self) -> float:
        return self.field + 2 * self.border

    @property
    def top(self) -> float:
        """World z of the board's playing surface."""
        return self.table_top + self.thickness

    @property
    def arm_base(self) -> tuple[float, float, float]:
        """World position of the SO-101 base frame. Independent of `origin`:
        shifting the board moves it relative to the arm."""
        return (0.0, -(self.width / 2 + self.arm_gap), self.table_top + self.arm_riser)

    def square_center(self, square: int | str | Square) -> tuple[float, float]:
        """World x/y of a square, as a `Square`, a name or an index (0 = a1 ... 63 = h8)."""
        square = square_index(square)
        f, r = chess.square_file(square), chess.square_rank(square)
        dx, dy = (f - 3.5) * self.square, (r - 3.5) * self.square
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return self.origin[0] + c * dx - s * dy, self.origin[1] + s * dx + c * dy

    @property
    def half_extent(self) -> float:
        """Half the board's footprint along x once turned - how far it reaches sideways."""
        return self.width / 2 * (abs(np.cos(self.yaw)) + abs(np.sin(self.yaw)))

    def on_field(self, xy) -> bool:
        """Whether a world x/y lies over the 8x8 playing field."""
        half = self.field / 2
        dx, dy = xy[0] - self.origin[0], xy[1] - self.origin[1]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return abs(c * dx + s * dy) <= half and abs(-s * dx + c * dy) <= half

    @classmethod
    def sample(cls, rng: np.random.Generator, per_episode_placement: bool = False) -> "BoardConfig":
        """A board as people own them: 25-31 mm squares, a thin to wide border, a
        vinyl roll-up to a thick wooden slab, set down a little nearer or further
        from the arm. The tray follows the board's edge. A quarter are the
        published board; which squares the arm can work on is recomputed for each
        board (`ReachMap`), and tasks only come from those."""
        if rng.random() < PUBLISHED_BOARD_CHANCE:
            return cls()
        u = rng.uniform
        square, border = float(u(*BOARD_SQUARE_RANGE)), float(u(*BOARD_BORDER_RANGE))
        # the turn and the sideways offset are drawn here once per scene, or left for
        # `RandomizationConfig` to draw every episode (`ChessSimEnv.reset`)
        yaw = 0.0 if per_episode_placement else float(np.radians(u(-BOARD_YAW_DEG, BOARD_YAW_DEG)))
        side = 0.0 if per_episode_placement else float(u(-BOARD_SIDEWAYS, BOARD_SIDEWAYS))
        riser = 0.0 if rng.random() < ARM_ON_TABLE_CHANCE else float(u(*ARM_RISER_RANGE))
        thickness = float(u(*BOARD_THICKNESS_RANGE))
        # a board top more than ~12 mm above the arm's base puts the board edge in the
        # forearm's path on the near ranks: a thick board needs the arm raised
        riser = max(riser, thickness - MAX_BOARD_ABOVE_ARM)
        board = cls(square=square, border=border, thickness=thickness,
                    arm_gap=float(u(*BOARD_ARM_GAP_RANGE)), arm_riser=riser,
                    origin=(side, 0.0), yaw=yaw)
        return board.model_copy(update={
            "tray_x": side - (board.half_extent + float(u(*TRAY_MARGIN_RANGE)))})

    @property
    def calibrated(self) -> bool:
        """Is this the board the per-square servo calibration was measured on?"""
        return self.model_copy(update={"origin": (0.0, 0.0)}) == BoardConfig()

    def tray_slot(self, index: int) -> tuple[float, float]:
        """Where a piece taken off the board is set down.

        On the arm's left, clear of the camera mast and of the parked pieces
        that both sit on the right, and inside the overhead camera's frame so a
        policy can see where it is putting the piece. Measured at 67/72 across
        four slots, six source squares and three piece types."""
        return self.tray_x, self.tray_y0 + index * self.tray_pitch

    def graveyard_slot(self, index: int) -> tuple[float, float]:
        """Off-board parking spot for pieces absent from the position."""
        row, col = divmod(index, self.graveyard_columns)
        return (self.origin[0] + self.half_extent + self.graveyard_gap + col * self.graveyard_pitch,
                -self.field / 2 + row * self.graveyard_pitch)


# Boards as people own them (BoardConfig.sample)
PUBLISHED_BOARD_CHANCE = 0.25
BOARD_SQUARE_RANGE = (0.025, 0.031)
BOARD_BORDER_RANGE = (0.006, 0.025)
BOARD_THICKNESS_RANGE = (0.003, 0.020)       # a vinyl roll-up to a thick wooden board
BOARD_ARM_GAP_RANGE = (0.050, 0.110)
BOARD_YAW_DEG = 12.0                         # set down a little turned
BOARD_SIDEWAYS = 0.04                        # the arm clamped off the board's centre line
PER_EPISODE_PLACEMENT = dict(board_yaw=float(np.radians(BOARD_YAW_DEG)), board_side=BOARD_SIDEWAYS, camera=True)
ARM_ON_TABLE_CHANCE = 0.4                    # clamped straight to the table, no riser
ARM_RISER_RANGE = (0.02, 0.06)
MAX_BOARD_ABOVE_ARM = 0.012                  # measured: 17 mm misses by a square, 12 mm is clean
TRAY_MARGIN_RANGE = (0.025, 0.045)           # tray beside the board's left edge

# Plausible looks for domain randomization: wood or painted boards, ivory-to-cream
# white sets, black-to-dark-colored black sets, varied table and lighting.
PIECE_SCALE_LIMITS = (0.7, 1.1)              # below/above this pieces stop fitting their squares
PIECE_SCALE_SAMPLED = (0.85, 1.1)
LIGHT_SQUARE_RANGE = ((160, 130, 90), (240, 225, 200))
DARK_SQUARE_RANGE = ((30, 20, 15), (130, 95, 70))
BORDER_RANGE = ((40, 25, 15), (150, 110, 80))
PAINTED_BOARD_CHANCE = 0.3                   # green/blue/red dark squares instead of wood
PAINTED_BASE_RANGE = (20, 60)
PAINTED_HUE_BOOST = 70
WHITE_TINT_RANGE = ((0.8, 0.75, 0.6), (1.0, 1.0, 1.0))
BLACK_TINT_RANGE = (0.5, 1.0)
TABLE_RGB_RANGE = ((0.2, 0.15, 0.1), (0.75, 0.65, 0.55))
TABLE_SURFACES = ("plain", "wood", "cloth", "marble", "laminate")
BACKDROP_TINT_RANGE = (0.7, 1.0)
BACKDROP_BRIGHTNESS_RANGE = (0.3, 0.8)
LIGHT_INTENSITY_RANGE = (0.45, 1.6)
WARM_LIGHT = (1.0, 0.80, 0.60)               # a tungsten lamp
COOL_LIGHT = (0.86, 0.93, 1.06)              # overcast daylight from a window
FILL_INTENSITY_RANGE = (0.3, 1.6)
AMBIENT_RANGE = (0.0, 0.25)
SHADOW_SOFTNESS_RANGE = (0.02, 0.40)
LIGHT_ELEVATION_RANGE = (40, 75)             # degrees above the table
BOARD_LABELS_CHANCE = 0.5
# board finishes: (reflectance of the surface, sliding friction pieces meet on it).
# A magnetic board's pull is modelled as grip - pieces resist sliding off their square.
BOARD_FINISHES = {"wood": (0.04, 1.0), "plastic": (0.12, 0.6), "vinyl": (0.02, 1.2),
                  "magnetic": (0.20, 1.6), "cardboard": (0.02, 0.9)}
BOARD_FINISH_WEIGHTS = (0.35, 0.2, 0.2, 0.15, 0.1)
# overhead camera: a third of the episodes keep the published mast rig; the rest put
# a camera anywhere from the mast to a boom arm over the board, 35-72 cm up (a C920
# on a desk arm frames the board from ~37 cm), aimed within 12 mm of the centre
PUBLISHED_RIG_CHANCE = 0.33
CAMERA_HEIGHT_RANGE = (0.35, 0.75)
CAMERA_ANYWHERE_CHANCE = 0.6                 # else on the line from the mast to above the board
CAMERA_OFFSET_MAX = 0.25                     # lens this far off the board centre, any direction
SIDE_VIEW_CHANCE = 0.25                      # of the non-rig cameras: beside the board, not above it
SIDE_VIEW_ELEVATION_DEG = (20.0, 50.0)       # above the table, looking across the board
SIDE_VIEW_DISTANCE = (0.35, 0.60)            # lens to board centre
SIDE_VIEW_ARM_SECTOR = 35.0                  # degrees either side of straight behind the arm
ARM_HEADING = -np.pi / 2                     # the arm sits on the board's -y side
CAMERA_FRAMED_HALF_WIDTH = 0.150             # half the strip that should fill the frame
CAMERA_FOVY_SLACK = (0.93, 1.10)
CAMERA_AIM = 0.012
CAMERA_ROLL = 8.0


class AppearanceConfig(Config):
    """Visual variety of the scene: piece size and colors, board and table colors, lighting.

    Piece size and the board texture are baked into the compiled scene; every
    other attribute can also be changed at run time with `ChessSimEnv.recolor`.
    """

    piece_scale: float = Field(1.0, ge=PIECE_SCALE_LIMITS[0], le=PIECE_SCALE_LIMITS[1])
    piece_set: str = "default"            # see chess_sim.assets.available_piece_sets()
    board_labels: bool = False            # a-h and 1-8 printed on the border
    board_finish: str = "wood"            # one of BOARD_FINISHES: how the board looks and grips
    # The overhead camera. None keeps the published rig: the lens on the mast beside
    # the arm, 678 mm above the board, 24 degrees. Otherwise the lens sits `camera_height`
    # above the board, `camera_over_board` of the way from the mast to straight above
    # the board centre (a boom arm), looking at the centre plus `camera_aim`.
    camera_height: float | None = None
    camera_fovy: float = 24.0
    camera_over_board: float = 0.0
    camera_aim: tuple[float, float] = (0.0, 0.0)
    camera_roll: float = 0.0              # degrees about the view axis
    # a boom-arm camera anywhere above the table: offset of the lens from the board
    # centre (x, y); overrides camera_over_board when set
    camera_offset: tuple[float, float] | None = None
    white_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # tint over the texture
    black_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    light_square: tuple[int, int, int] = (214, 178, 132)   # board colors, 0-255
    dark_square: tuple[int, int, int] = (99, 64, 40)
    border: tuple[int, int, int] = (92, 58, 32)
    table_rgb: tuple[float, float, float] = (0.42, 0.28, 0.17)   # 0-1
    light_intensity: float = 1.0                                 # key light brightness multiplier
    light_dir: tuple[float, float, float] = (-0.3, 0.3, -1.0)    # key light direction (world)
    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)    # key light tint: warm lamp to cool daylight
    fill_intensity: float = 1.0                                  # the soft light from the room
    ambient: float = 0.0                                         # light from everywhere, flattens shadows
    shadow_softness: float = 0.02                                # key light's radius, metres: hard to soft shadows
    table_surface: str = "plain"                                 # one of TABLE_SURFACES; `table_rgb` tints it
    backdrop_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # the room behind the table
    backdrop_brightness: float = 0.55

    @staticmethod
    def sample_camera(rng: np.random.Generator) -> dict:
        """Where the workspace camera stands: the published mast a third of the time,
        otherwise a boom arm over the board or a tripod beside it. Every field is
        returned, so applying the draw to a scene on the mast moves it off, and back."""
        u = rng.uniform
        camera = dict(camera_height=None, camera_offset=None, camera_fovy=24.0,
                      camera_over_board=0.0, camera_aim=(0.0, 0.0), camera_roll=0.0)
        if rng.random() >= PUBLISHED_RIG_CHANCE:
            height = float(u(*CAMERA_HEIGHT_RANGE))
            offset = None
            if rng.random() < SIDE_VIEW_CHANCE:
                # a phone or webcam on a tripod beside the table, looking across the board;
                # from anywhere but straight behind the arm, which would fill the frame
                while True:
                    heading = float(u(0, 2 * np.pi))
                    if abs((heading - ARM_HEADING + np.pi) % (2 * np.pi) - np.pi) > np.radians(SIDE_VIEW_ARM_SECTOR):
                        break
                elevation, distance = np.radians(u(*SIDE_VIEW_ELEVATION_DEG)), float(u(*SIDE_VIEW_DISTANCE))
                ground = distance * float(np.cos(elevation))
                offset = (ground * float(np.cos(heading)), ground * float(np.sin(heading)))
                height = distance * float(np.sin(elevation))
            elif rng.random() < CAMERA_ANYWHERE_CHANCE:
                radius, heading = CAMERA_OFFSET_MAX * float(np.sqrt(u(0, 1))), float(u(0, 2 * np.pi))
                offset = (radius * float(np.cos(heading)), radius * float(np.sin(heading)))
            reach = float(np.hypot(height, np.hypot(*offset))) if offset else height
            # frame the board whatever the distance, the way a person sets a camera up
            framed = float(np.degrees(2 * np.arctan(CAMERA_FRAMED_HALF_WIDTH / reach)))
            camera = dict(camera_height=height, camera_offset=offset,
                          camera_fovy=framed * float(u(*CAMERA_FOVY_SLACK)),
                          camera_over_board=float(u(0.0, 1.0)),
                          camera_aim=tuple(float(v) for v in u(-CAMERA_AIM, CAMERA_AIM, 2)),
                          camera_roll=float(u(-CAMERA_ROLL, CAMERA_ROLL)))
        return camera

    @classmethod
    def sample(cls, rng: np.random.Generator,
               piece_scale_range: tuple[float, float] = PIECE_SCALE_SAMPLED,
               piece_sets: Sequence[str] = ("default",)) -> "AppearanceConfig":
        """A random look drawn from the ranges above, the piece set among `piece_sets`."""
        u = rng.uniform
        piece_set = str(piece_sets[int(rng.integers(len(piece_sets)))])
        light = tuple(int(v) for v in u(*LIGHT_SQUARE_RANGE))
        dark = tuple(int(v) for v in u(*DARK_SQUARE_RANGE))
        if rng.random() < PAINTED_BOARD_CHANCE:
            hue = rng.integers(3)
            dark = tuple(int(u(*PAINTED_BASE_RANGE) + (PAINTED_HUE_BOOST if hue == c else 0))
                         for c in range(3))
        azimuth, elevation = u(0, 2 * np.pi), np.radians(u(*LIGHT_ELEVATION_RANGE))
        camera = cls.sample_camera(rng)
        warmth = float(u(0.0, 1.0))
        light_color = tuple(float(w + (c - w) * warmth) for w, c in zip(WARM_LIGHT, COOL_LIGHT))
        return cls(piece_scale=float(u(*piece_scale_range)), piece_set=piece_set,
                   light_color=light_color, fill_intensity=float(u(*FILL_INTENSITY_RANGE)),
                   ambient=float(u(*AMBIENT_RANGE)), shadow_softness=float(u(*SHADOW_SOFTNESS_RANGE)),
                   table_surface=str(TABLE_SURFACES[int(rng.integers(len(TABLE_SURFACES)))]),
                   backdrop_tint=tuple(float(v) for v in u(*BACKDROP_TINT_RANGE, 3)),
                   backdrop_brightness=float(u(*BACKDROP_BRIGHTNESS_RANGE)),
                   board_labels=bool(rng.random() < BOARD_LABELS_CHANCE), **camera,
                   board_finish=str(rng.choice(list(BOARD_FINISHES), p=list(BOARD_FINISH_WEIGHTS))),
                   white_rgba=(*(float(v) for v in u(*WHITE_TINT_RANGE)), 1.0),
                   black_rgba=(*(float(u(*BLACK_TINT_RANGE)) for _ in range(3)), 1.0),
                   light_square=light, dark_square=dark,
                   border=tuple(int(v) for v in u(*BORDER_RANGE)),
                   table_rgb=tuple(float(v) for v in u(*TABLE_RGB_RANGE)),
                   light_intensity=float(u(*LIGHT_INTENSITY_RANGE)),
                   light_dir=(float(np.cos(azimuth) * np.cos(elevation)),
                              float(np.sin(azimuth) * np.cos(elevation)),
                              -float(np.sin(elevation))))

    @property
    def board_key(self) -> str:
        """Identifies the board texture (its colors and labels) for caching."""
        key = "-".join(f"{c:02x}" for rgb in (self.light_square, self.dark_square, self.border)
                       for c in rgb)
        return (key + ("-labels" if self.board_labels else "")
                + ("" if self.board_finish == "wood" else f"-{self.board_finish}"))


class ControlConfig(Config):
    """How the arm is driven and what the env renders."""

    hz: int = 30                                   # control rate; one action per period
    cameras: tuple[Camera, ...] = ROBOT_CAMERAS    # rendered into every observation
    image_size: tuple[int, int] = (640, 480)
    settle_seconds: float = 0.8                    # holding the start pose after a reset
    release_seconds: float = 0.3                   # letting a released piece come to rest


class RandomizationConfig(Config):
    """How far the layout is drawn from nominal on a randomized reset.

    A real board is never set down in exactly the same spot and a real arm is
    never parked in exactly the same pose; a policy that only ever saw one of
    each can key on it instead of looking.
    """

    board_shift: float = 0.010        # m on each axis - about a third of a square
    arm_joint_jitter: float = 0.10    # rad on each joint
    # Per-episode placement, off by default so the published results reproduce. The
    # first wide collection drew these once per 25-episode chunk - some three hundred
    # placements in seven thousand episodes - and the policy learned none of them.
    board_yaw: float = 0.0            # rad either way: the board set down turned
    board_side: float = 0.0           # m either way: the board off the arm's centre line
    camera: bool = False              # the workspace camera redrawn every episode (AppearanceConfig.sample_camera)


class ActionNoiseConfig(Config):
    """Perturb what the arm executes while recording what the expert commanded.

    A clean demonstration never shows how to get back on course. Executing the
    expert's commands with a slow random offset puts the arm where a drifting
    policy ends up, and the expert's corrections from there are what gets
    recorded - the label is always the command, never the perturbed one.
    """

    joint_std: float = 0.0        # rad per joint; 0 turns it off
    hold: int = 10                # control steps each draw lasts: a drift, not a tremor
    perturb_gripper: bool = False # a perturbed jaw drops the piece, which nothing recovers

    @property
    def enabled(self) -> bool:
        return self.joint_std > 0


class ToleranceConfig(Config):
    """The success rule, shared by the scripted expert and by policy evaluation."""

    placement: float = 0.011      # piece axis vs the target square center
    tray: float = 0.030           # a discarded piece only has to land in its tray slot
    disturbance: float = 0.010    # any other piece moving more than this fails the episode
    upright_min: float = 0.85     # z-axis alignment of a standing piece


class ClearanceConfig(Config):
    """Free space the open jaw needs around the piece it is about to grasp."""

    moving_jaw_room: float = 0.020   # the moving prong swings out this far past the piece
    fixed_prong_room: float = 0.009  # fixed prong: ~3 mm slack plus prong thickness
    lane_fraction: float = 0.75      # of a square: how far off-axis a piece still blocks the jaw
    probe_distance: float = 0.30     # nothing further away counts as an obstacle


class EnvConfig(Config):
    """Everything `ChessSimEnv` needs. Compose one and keep it with the run."""

    board: BoardConfig = BoardConfig()
    appearance: AppearanceConfig = AppearanceConfig()
    control: ControlConfig = ControlConfig()
    randomization: RandomizationConfig = RandomizationConfig()
    tolerances: ToleranceConfig = ToleranceConfig()
    clearance: ClearanceConfig = ClearanceConfig()
    action_noise: ActionNoiseConfig = ActionNoiseConfig()


def square_index(square: int | str | Square) -> int:
    """A `Square`, a name or an index -> python-chess square index."""
    return square if isinstance(square, int) else chess.parse_square(str(square))


def square_at(square: int | str | Square) -> Square:
    """Anything that names a square -> the `Square` member."""
    return Square(square if isinstance(square, str) else chess.square_name(square))
