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
        return self.origin[0] + (f - 3.5) * self.square, self.origin[1] + (r - 3.5) * self.square

    def on_field(self, xy) -> bool:
        """Whether a world x/y lies over the 8x8 playing field."""
        half = self.field / 2
        return abs(xy[0] - self.origin[0]) <= half and abs(xy[1] - self.origin[1]) <= half

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
        return (self.width / 2 + self.graveyard_gap + col * self.graveyard_pitch,
                -self.field / 2 + row * self.graveyard_pitch)


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
LIGHT_INTENSITY_RANGE = (0.6, 1.4)
LIGHT_ELEVATION_RANGE = (40, 75)             # degrees above the table


class AppearanceConfig(Config):
    """Visual variety of the scene: piece size and colors, board and table colors, lighting.

    Piece size and the board texture are baked into the compiled scene; every
    other attribute can also be changed at run time with `ChessSimEnv.recolor`.
    """

    piece_scale: float = Field(1.0, ge=PIECE_SCALE_LIMITS[0], le=PIECE_SCALE_LIMITS[1])
    white_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # tint over the texture
    black_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    light_square: tuple[int, int, int] = (214, 178, 132)   # board colors, 0-255
    dark_square: tuple[int, int, int] = (99, 64, 40)
    border: tuple[int, int, int] = (92, 58, 32)
    table_rgb: tuple[float, float, float] = (0.42, 0.28, 0.17)   # 0-1
    light_intensity: float = 1.0                                 # key light brightness multiplier
    light_dir: tuple[float, float, float] = (-0.3, 0.3, -1.0)    # key light direction (world)

    @classmethod
    def sample(cls, rng: np.random.Generator,
               piece_scale_range: tuple[float, float] = PIECE_SCALE_SAMPLED) -> "AppearanceConfig":
        """A random look drawn from the ranges above."""
        u = rng.uniform
        light = tuple(int(v) for v in u(*LIGHT_SQUARE_RANGE))
        dark = tuple(int(v) for v in u(*DARK_SQUARE_RANGE))
        if rng.random() < PAINTED_BOARD_CHANCE:
            hue = rng.integers(3)
            dark = tuple(int(u(*PAINTED_BASE_RANGE) + (PAINTED_HUE_BOOST if hue == c else 0))
                         for c in range(3))
        azimuth, elevation = u(0, 2 * np.pi), np.radians(u(*LIGHT_ELEVATION_RANGE))
        return cls(piece_scale=float(u(*piece_scale_range)),
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
        """Identifies the board texture (its colors) for caching."""
        return "-".join(f"{c:02x}" for rgb in (self.light_square, self.dark_square, self.border)
                        for c in rgb)


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


def square_index(square: int | str | Square) -> int:
    """A `Square`, a name or an index -> python-chess square index."""
    return square if isinstance(square, int) else chess.parse_square(str(square))


def square_at(square: int | str | Square) -> Square:
    """Anything that names a square -> the `Square` member."""
    return Square(square if isinstance(square, str) else chess.square_name(square))
