"""Joint conventions: how simulator radians map to dataset units.

The simulator records MuJoCo joint targets in radians, whose zero is mid-range
like LeRobot's post-0.5.0 convention. The public SO-100/101 datasets use degrees
with an offset (shoulder_lift and elbow_flex mirrored about 90 degrees), so a
checkpoint pretrained on them expects that convention - export in it, and undo it
before feeding actions back to the simulator.
"""
from __future__ import annotations

from enum import StrEnum

import numpy as np

from .config import DOF, Config


class ConventionName(StrEnum):
    SO101 = "so101"        # degrees with an offset, as the public SO-100/101 datasets use
    RADIANS = "radians"


class JointConvention(Config):
    name: ConventionName
    degrees: bool = True
    signs: tuple[float, ...] = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
    offsets: tuple[float, ...] = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)

    def from_radians(self, values) -> np.ndarray:
        """Simulator radians -> dataset units."""
        values = np.asarray(values, dtype=float)
        if not self.degrees:
            return values
        return np.degrees(values) * np.asarray(self.signs) + np.asarray(self.offsets)

    def to_radians(self, values) -> np.ndarray:
        """Dataset units -> simulator radians."""
        values = np.asarray(values, dtype=float)
        if not self.degrees:
            return values
        return np.radians((values - np.asarray(self.offsets)) / np.asarray(self.signs))


SO101_DEGREES = JointConvention(name=ConventionName.SO101)           # what the public SO-101 datasets use
RADIANS = JointConvention(name=ConventionName.RADIANS, degrees=False, signs=(1.0,) * DOF, offsets=(0.0,) * DOF)
CONVENTIONS = {c.name: c for c in (SO101_DEGREES, RADIANS)}
