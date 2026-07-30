"""Movement calculations for tracked flies."""

from __future__ import annotations

import math


def euclidean_distance(
        previous_x: float,
        previous_y: float,
        current_x: float,
        current_y: float,
) -> float:
    """Return straight-line pixel displacement between two positions."""
    return math.hypot(current_x - previous_x, current_y - previous_y)


def apply_jitter_deadband(distance_px: float, jitter_threshold_px: float) -> float:
    """Ignore small motion and subtract the deadband from larger motion."""
    if distance_px < jitter_threshold_px:
        return 0.0
    return max(0.0, distance_px - jitter_threshold_px)
