"""Rolling five-minute movement accumulation and sleep classification."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class RollingSleepState:
    """Maintain one fly's rolling movement history."""

    window_samples: int
    awake_threshold_px: float
    distances: deque[float] = field(init=False)
    rolling_distance_px: float = 0.0

    def __post_init__(self) -> None:
        if self.window_samples < 1:
            raise ValueError("window_samples must be at least 1.")
        self.distances = deque()

    def reset(self) -> None:
        """Clear all accumulated movement for a newly assigned identity slot."""
        self.distances.clear()
        self.rolling_distance_px = 0.0

    def update(self, distance_px: float) -> tuple[float, str, int]:
        """Add one sample, remove the expired sample, and return state."""
        if len(self.distances) == self.window_samples:
            self.rolling_distance_px -= self.distances.popleft()

        self.distances.append(float(distance_px))
        self.rolling_distance_px += float(distance_px)

        # Clamp tiny floating-point residue to zero.
        if abs(self.rolling_distance_px) < 1e-12:
            self.rolling_distance_px = 0.0

        state = (
            "AWAKE"
            if self.rolling_distance_px > self.awake_threshold_px
            else "ASLEEP"
        )
        return self.rolling_distance_px, state, len(self.distances)
