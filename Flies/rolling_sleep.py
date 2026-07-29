"""
rolling_sleep.py

Sleep classification using a rolling 5-minute sum of jitter-corrected
distance traveled, per the pipeline guide's Step 3/4 pseudocode:

    d = sqrt(dx*dx + dy*dy)
    if d < jitterThreshold: d = 0
    else: d -= jitterThreshold
    rollingDistance += d
    rollingDistance -= distanceFrom300SecondsAgo
    state = AWAKE if rollingDistance > awakeThreshold else ASLEEP

This differs from Flies.fly_sleep.FlySleepTracker, which uses a
continuous inactivity timer that resets to zero on any single
above-threshold displacement. This tracker instead keeps a rolling
sum: one small movement nudges the sum without wiping out the whole
inactivity window, matching the guide's stated behavior. Select which
one ImagePipeline uses via config.SLEEP_MODEL.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Optional

import config


AWAKE = "awake"
ASLEEP = "asleep"


@dataclass
class _FlyRollingRecord:
    object_id: int
    well_label: Optional[str]

    x: float
    y: float

    history: deque  # deque of jitter-corrected distances, maxlen = window frames
    rolling_distance: float = 0.0
    state: str = ASLEEP


@dataclass(frozen=True)
class RollingSleepResult:
    frame_idx: int
    time_s: float

    object_id: int
    well_label: Optional[str]

    x: float
    y: float
    dx: float
    dy: float
    raw_distance_px: float
    jitter_corrected_distance_px: float

    rolling_distance_px: float
    state: str

    @property
    def is_awake(self) -> bool:
        return self.state == AWAKE

    @property
    def is_asleep(self) -> bool:
        return self.state == ASLEEP


class RollingDistanceSleepTracker:
    """Rolling-window, jitter-deadbanded sleep classifier (one fly per object_id)."""

    def __init__(
            self,
            jitter_threshold_px: Optional[float] = None,
            awake_threshold_px: Optional[float] = None,
            window_seconds: Optional[float] = None,
            fps: Optional[float] = None,
    ) -> None:
        self.jitter_threshold_px = float(
            jitter_threshold_px
            if jitter_threshold_px is not None
            else config.JITTER_THRESHOLD_PX
        )
        self.awake_threshold_px = float(
            awake_threshold_px
            if awake_threshold_px is not None
            else config.AWAKE_THRESHOLD_PX
        )
        self.window_seconds = float(
            window_seconds
            if window_seconds is not None
            else config.ROLLING_WINDOW_SEC
        )
        self.fps = float(fps if fps is not None else config.FPS)

        if self.fps <= 0:
            raise ValueError("fps must be greater than zero")

        self.window_frames = max(1, int(round(self.window_seconds * self.fps)))

        self.records: dict[int, _FlyRollingRecord] = {}

    def update_fly(
            self,
            frame_idx: int,
            time_s: float,
            object_id: int,
            x: float,
            y: float,
            well_label: Optional[str] = None,
    ) -> RollingSleepResult:
        object_id = int(object_id)
        x = float(x)
        y = float(y)

        record = self.records.get(object_id)

        if record is None:
            record = _FlyRollingRecord(
                object_id=object_id,
                well_label=well_label,
                x=x,
                y=y,
                history=deque(maxlen=self.window_frames),
            )
            self.records[object_id] = record

            return self._make_result(
                frame_idx=frame_idx,
                time_s=time_s,
                record=record,
                dx=0.0,
                dy=0.0,
                raw_d=0.0,
                d=0.0,
            )

        px, py = record.x, record.y
        dx, dy = x - px, y - py
        raw_d = hypot(dx, dy)

        # Jitter deadband
        if raw_d < self.jitter_threshold_px:
            d = 0.0
        else:
            d = raw_d - self.jitter_threshold_px

        # Rolling window: add newest, evict the value that ages out
        # (deque with maxlen drops the oldest automatically on append,
        # so we must subtract it from the running sum first).
        if len(record.history) == record.history.maxlen:
            record.rolling_distance -= record.history[0]
        record.history.append(d)
        record.rolling_distance += d

        record.state = (
            AWAKE if record.rolling_distance > self.awake_threshold_px else ASLEEP
        )

        record.x = x
        record.y = y
        record.well_label = well_label

        return self._make_result(
            frame_idx=frame_idx,
            time_s=time_s,
            record=record,
            dx=dx,
            dy=dy,
            raw_d=raw_d,
            d=d,
        )

    def update_detection(
            self,
            frame_idx: int,
            time_s: float,
            detection: dict,
    ) -> Optional[RollingSleepResult]:
        object_id = detection.get("id")
        centroid = detection.get("centroid")

        if object_id is None or centroid is None or len(centroid) != 2:
            return None

        x, y = centroid
        return self.update_fly(
            frame_idx=frame_idx,
            time_s=time_s,
            object_id=object_id,
            x=x,
            y=y,
            well_label=detection.get("well_label"),
        )

    def update_frame(self, frame_idx: int, time_s: float, detections) -> list:
        results = []
        for detection in detections:
            result = self.update_detection(frame_idx, time_s, detection)
            if result is not None:
                results.append(result)
        return results

    def reset(self) -> None:
        self.records.clear()

    @staticmethod
    def _make_result(frame_idx, time_s, record, dx, dy, raw_d, d) -> RollingSleepResult:
        return RollingSleepResult(
            frame_idx=int(frame_idx),
            time_s=float(time_s),
            object_id=record.object_id,
            well_label=record.well_label,
            x=record.x,
            y=record.y,
            dx=dx,
            dy=dy,
            raw_distance_px=raw_d,
            jitter_corrected_distance_px=d,
            rolling_distance_px=record.rolling_distance,
            state=record.state,
        )