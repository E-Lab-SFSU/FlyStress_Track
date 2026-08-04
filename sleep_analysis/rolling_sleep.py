"""Time-based rolling movement and continuous-immobility state."""
from __future__ import annotations
from collections import deque


class RollingMovement:
    def __init__(self, window_seconds: float) -> None:
        self.window_seconds = float(window_seconds)
        self._values: deque[tuple[float, float]] = deque()
        self.total = 0.0

    def advance(self, timestamp_s: float) -> float:
        cutoff = timestamp_s - self.window_seconds
        while self._values and self._values[0][0] < cutoff:
            _, value = self._values.popleft()
            self.total -= value
        return max(0.0, self.total)

    def add(self, timestamp_s: float, distance_px: float) -> float:
        self.advance(timestamp_s)
        value = max(0.0, float(distance_px))
        self._values.append((float(timestamp_s), value))
        self.total += value
        return self.total

    @property
    def samples(self) -> int:
        return len(self._values)
