"""Scripted pick-and-place expert for the SO-101.

Executes one chess move as a sequence of tool-space waypoints solved by IK:
approach along the tool axis, close the jaws onto the piece, carry it
world-upright above the other pieces, and set it down on the target square.
The piece is held purely by friction between the fingertip pads: the jaws
close SQUEEZE past its surface and the resulting clamp carries it, so the
carry is a slow straight tool path and the piece-to-pocket offset is measured
live before placement.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import gripper
from .scene import PieceSlot

StepCallback = Callable[[np.ndarray], None]  # receives the 6-D action applied

TRANSIT = 0.105
APPROACH_LENGTH = 0.06
PRECISION = 0.004            # executed pocket error accepted at grasp/place
SERVO_ROUNDS_PRECISE = 11
SERVO_ROUNDS_TRANSIT = 2
SERVO_ROUND_STEPS = 8
SETTLE_STEPS = 12            # control steps holding the closed jaws before lifting


class PickPlaceController:
    def __init__(self, env):
        self.env = env
        self._span = None            # preferred jaw-span direction for the current phase
        self._span_grasp = None
        self._span_place = None

    # -- public --------------------------------------------------------------

    def pick_place(self, slot: PieceSlot, target_xy: tuple[float, float],
                   on_step: StepCallback | None = None) -> bool:
        """Move `slot`'s piece so its axis stands on `target_xy`. True if all
        waypoints executed within tolerance (final outcome is verified by env)."""
        env, board, g = self.env, self.env.board_spec, slot.geometry
        px, py, _ = env.piece_position(slot)
        self._span_grasp = env.free_span_direction((px, py), exclude=slot)
        self._span_place = env.free_span_direction(target_xy, exclude=slot)
        # Pocket height: the piece's grasp height, unless the fingertips would
        # touch the board or the pads would reach down into a wider segment
        # below the one they clamp (a rook's base flange).
        z_grasp = board.top + max(g.waist,
                                  gripper.TIP_CLEARANCE + gripper.TIP_X - gripper.POCKET_X,
                                  g.flange_top + gripper.PAD_REACH)
        # Open prongs must clear everything the pads span vertically on the way
        # in and out (a rook's base flange as much as a king's crown), so they
        # are positioned around the widest radius in that band; the hold gap
        # closes SQUEEZE past the grasp radius so the pads clamp the piece.
        pad_lo = z_grasp - board.top - gripper.PAD_REACH
        pad_hi = z_grasp - board.top + gripper.PAD_REACH
        clear = g.radius_between(pad_lo, pad_hi) + 0.003
        gap_open = 2 * clear + 0.010
        gap_hold = 2 * g.grasp_radius - gripper.SQUEEZE
        gap_release = gap_open
        ok = True

        grasp_pt = np.array([px, py, z_grasp])
        place_pt = np.array([target_xy[0], target_xy[1], z_grasp])
        off_hold = gripper.pocket_offset(gap_hold)
        # while the jaws are open, keep the piece as close to the FIXED prong as
        # the widest radius in the pad band allows: the moving jaw opens toward
        # the free side, so the opening slack goes there rather than into an
        # occupied neighbor square
        off_open = gripper.pocket_offset(2 * clear)
        off_release = off_open

        # approach the piece along the tool axis, close, let the clamp build
        self._span = self._span_grasp
        self._go(np.array([px, py, board.top + TRANSIT]), gap_open, off_open, 22, on_step)
        approach = self._approach_axis(grasp_pt, off_open)
        self._go(grasp_pt - approach * APPROACH_LENGTH, gap_open, off_open, 14, on_step)
        ok &= self._descend(grasp_pt, gap_open, off_open, approach, on_step)
        self._go(grasp_pt, gap_hold, off_hold, 10, on_step)
        self._settle(SETTLE_STEPS, gap_hold, on_step)

        # lift and carry high enough to clear standing pieces; a friction-held
        # piece cannot take jerk, so the carry is a slow straight tool path
        lift_pt = grasp_pt - approach * (APPROACH_LENGTH + 0.01)
        self._line(grasp_pt, lift_pt, gap_hold, off_hold, on_step)
        carry_z = board.top + TRANSIT
        above_src = np.array([px, py, carry_z])
        above_dst = np.array([target_xy[0], target_xy[1], carry_z])
        self._line(lift_pt, above_src, gap_hold, off_hold, on_step, segments=2, steps=6)
        self._line(above_src, above_dst, gap_hold, off_hold, on_step,
                   segments=max(2, int(np.linalg.norm(above_dst - above_src) / 0.03)), steps=8)

        # the piece sits wherever the pads caught it: measure the offset between
        # its axis and the pocket so the PIECE lands centered on the square
        pocket, _ = env.ik.tool_pose(env.data, off_hold)
        axis = env.piece_position(slot)[:2] + g.center[:2]
        place_pt[:2] -= axis - pocket[:2]

        # set down along the tool axis, release, retreat
        self._span = self._span_place
        approach = self._approach_axis(place_pt, off_hold)
        self._go(place_pt - approach * APPROACH_LENGTH, gap_hold, off_hold, 14, on_step)
        ok &= self._descend(place_pt, gap_hold, off_hold, approach, on_step)
        # open the jaws, then shift the tool so the piece sits centered between
        # the open prongs (the wider gap's pocket lies further from the fixed
        # prong) and retreat straight up: backing out along a tilted tool axis
        # would sweep a prong through the piece
        self._go(place_pt, gap_release, off_hold, 8, on_step)
        self._go(place_pt, gap_release, off_release, 6, on_step)
        clear_pt = np.array([target_xy[0], target_xy[1], board.top + g.height + 0.03])
        self._line(place_pt, clear_pt, gap_release, off_release, on_step, segments=4)
        self._go(np.array([target_xy[0], target_xy[1], board.top + TRANSIT]), gap_open, off_open, 10, on_step)
        return bool(ok)

    # -- motion primitives ---------------------------------------------------

    def _approach_axis(self, target, offset) -> np.ndarray:
        res = self.env.ik.solve(self.env.data, target, offset=offset, span=self._span)
        _, mat = self.env.ik.forward(res.q, offset)
        return mat[:, 0]

    def _line(self, start, target, gap, offset, on_step, precise=False,
              segments: int = 3, steps: int = 5) -> bool:
        """Straight tool-space path: joint-space interpolation over a long move
        bows sideways by more than the clearance around a standing piece."""
        for k in range(1, segments):
            self._go(start + (target - start) * (k / segments), gap, offset, steps, on_step)
        return self._go(target, gap, offset, steps + 3, on_step, precise=precise)

    def _settle(self, steps: int, gap: float, on_step) -> None:
        """Hold the current arm command (bias included) for a few control steps."""
        action = self.env._current_action()
        action[5] = gripper.gap_to_angle(gap)
        for _ in range(steps):
            self.env.apply_action(action, on_step)

    def _descend(self, target, gap, offset, approach, on_step) -> bool:
        return self._line(target - approach * APPROACH_LENGTH, target, gap, offset,
                          on_step, precise=True)

    def _go(self, target, gap, offset, steps, on_step, precise=False) -> bool:
        """Interpolate to the IK solution, then hold until the servos converge."""
        env = self.env
        res = env.ik.solve(env.data, target, offset=offset, span=self._span)
        q_start = env.arm_joint_positions()[:5]
        grip = gripper.gap_to_angle(gap)
        for i in range(1, steps + 1):
            q = q_start + (res.q - q_start) * (i / steps)
            env.apply_action(np.append(q, grip), on_step)

        # Position servos have pose-dependent steady-state error (no gravity
        # compensation). Bias the command every few steps until the executed
        # tool point matches; per-step updates would limit-cycle.
        bias = np.zeros(5)
        rounds = SERVO_ROUNDS_PRECISE if precise else SERVO_ROUNDS_TRANSIT
        for _ in range(rounds):
            pos, _ = env.ik.tool_pose(env.data, offset)
            if np.linalg.norm(pos - target) < PRECISION:
                break
            bias = np.clip(bias + 0.6 * (res.q - env.arm_joint_positions()[:5]), -0.45, 0.45)
            for _ in range(SERVO_ROUND_STEPS):
                env.apply_action(np.append(res.q + bias, grip), on_step)
        pos, _ = env.ik.tool_pose(env.data, offset)
        return bool(np.linalg.norm(pos - target) < 2.5 * PRECISION) if precise else True
