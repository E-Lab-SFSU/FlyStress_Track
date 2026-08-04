"""One persistent fly track per well."""
from __future__ import annotations
from dataclasses import dataclass
from sleep_analysis.fly_detection import FlyDetection
from sleep_analysis.movement import apply_jitter_deadband, euclidean_distance
from sleep_analysis.rolling_sleep import RollingMovement


@dataclass(frozen=True)
class TrackResult:
    well: str
    fly_name: str
    state: str
    valid_tracking: bool
    reason: str
    detected: bool
    x_px: float | None
    y_px: float | None
    local_x_px: float | None
    local_y_px: float | None
    area_px: int | None
    threshold_value: int | None
    raw_distance_px: float | None
    distance_px: float | None
    rolling_distance_px: float
    rolling_samples: int
    immobile_duration_seconds: float


class SingleFlyTracker:
    def __init__(self, well_names: list[str], *, jitter_threshold_px: float,
                 rolling_window_seconds: float, sleep_duration_seconds: float,
                 max_position_jump_px: float, max_valid_sample_gap_seconds: float) -> None:
        self.jitter = float(jitter_threshold_px)
        self.sleep_seconds = float(sleep_duration_seconds)
        self.max_jump = float(max_position_jump_px)
        self.max_gap = float(max_valid_sample_gap_seconds)
        self.data = {
            well: dict(x=None, y=None, last_t=None, immobile=0.0,
                       rolling=RollingMovement(rolling_window_seconds))
            for well in well_names
        }

    def _choose(self, well: str, candidates: list[FlyDetection]) -> FlyDetection | None:
        if not candidates:
            return None
        state = self.data[well]
        if state["x"] is None:
            return max(candidates, key=lambda d: d.area_px)
        ranked = sorted(candidates, key=lambda d: euclidean_distance(state["x"], state["y"], d.x, d.y))
        best = ranked[0]
        return best if euclidean_distance(state["x"], state["y"], best.x, best.y) <= self.max_jump else None

    def update(self, well: str, candidates: list[FlyDetection], timestamp_s: float,
               registration_ok: bool) -> TrackResult:
        state = self.data[well]
        rolling = state["rolling"]
        rolling.advance(timestamp_s)
        fly_name = f"{well}_Fly"
        if not registration_ok:
            return TrackResult(well, fly_name, "UNKNOWN", False, "registration_failed", False,
                               None, None, None, None, None, None, None, None,
                               rolling.total, rolling.samples, state["immobile"])
        detection = self._choose(well, candidates)
        if detection is None:
            return TrackResult(well, fly_name, "UNKNOWN", False, "fly_not_detected", False,
                               None, None, None, None, None, None, None, None,
                               rolling.total, rolling.samples, state["immobile"])
        previous_x, previous_y, previous_t = state["x"], state["y"], state["last_t"]
        state["x"], state["y"], state["last_t"] = detection.x, detection.y, timestamp_s
        if previous_x is None or previous_t is None:
            state["immobile"] = 0.0
            rolling.add(timestamp_s, 0.0)
            reason = "first_valid_position"
            raw_distance = distance = 0.0
        else:
            dt = timestamp_s - previous_t
            if dt <= 0 or dt > self.max_gap:
                state["immobile"] = 0.0
                reason = "reacquired_after_gap"
                raw_distance = distance = 0.0
                rolling.add(timestamp_s, 0.0)
            else:
                raw_distance = euclidean_distance(previous_x, previous_y, detection.x, detection.y)
                distance = apply_jitter_deadband(raw_distance, self.jitter)
                rolling.add(timestamp_s, distance)
                if distance > 0:
                    state["immobile"] = 0.0
                else:
                    state["immobile"] += dt
                reason = "valid"
        state_name = "ASLEEP" if state["immobile"] >= self.sleep_seconds else "AWAKE"
        return TrackResult(well, fly_name, state_name, True, reason, True,
                           detection.x, detection.y, detection.local_x, detection.local_y,
                           detection.area_px, detection.threshold_value,
                           raw_distance, distance, rolling.total, rolling.samples,
                           state["immobile"])
