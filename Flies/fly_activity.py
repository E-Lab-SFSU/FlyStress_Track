"""
fly_activity.py

Determines whether fly is awake or inactive.
Saves data is csv file

"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional

import config


AWAKE = "awake"
INACTIVE = "inactive"
UNKNOWN = "unknown"


@dataclass
class FlyActivityRecord:
    """Current position and activity information for one tracked fly."""

    object_id: int
    well_label: Optional[str]

    x: float
    y: float

    previous_x: Optional[float] = None
    previous_y: Optional[float] = None

    displacement_px: float = 0.0
    activity_state: str = INACTIVE

    last_seen_frame: Optional[int] = None
    last_seen_time_s: Optional[float] = None


@dataclass(frozen=True)
class FlyActivityResult:
    """Activity result returned for one fly in one frame."""

    frame_idx: int
    time_s: float
    object_id: int
    well_label: Optional[str]

    x: float
    y: float

    previous_x: Optional[float]
    previous_y: Optional[float]

    displacement_px: float
    activity_state: str

    @property
    def is_awake(self) -> bool:
        return self.activity_state == AWAKE

    @property
    def is_inactive(self) -> bool:
        return self.activity_state == INACTIVE


class FlyActivityTracker:
    """Maintains per-fly positions and classifies current activity."""

    def __init__(
            self,
            inactive_range_px: Optional[float] = None,
    ) -> None:
        """
        Args:
            inactive_range_px:
                Maximum displacement still treated as inactivity.

                When omitted, config.INACTIVE_RNG is used.
        """

        if inactive_range_px is None:
            inactive_range_px = config.INACTIVE_RNG

        if inactive_range_px < 0:
            raise ValueError(
                "inactive_range_px cannot be negative"
            )

        self.inactive_range_px = float(
            inactive_range_px
        )

        self.records: dict[int, FlyActivityRecord] = {}

    def update_fly(
            self,
            frame_idx: int,
            time_s: float,
            object_id: int,
            x: float,
            y: float,
            well_label: Optional[str] = None,
    ) -> FlyActivityResult:
        """
        Update one fly using its position in the current frame.

        Returns:
            FlyActivityResult containing displacement and activity state.
        """

        if frame_idx < 0:
            raise ValueError("frame_idx cannot be negative")

        if time_s < 0:
            raise ValueError("time_s cannot be negative")

        object_id = int(object_id)
        x = float(x)
        y = float(y)

        record = self.records.get(object_id)

        if record is None:
            record = FlyActivityRecord(
                object_id=object_id,
                well_label=well_label,
                x=x,
                y=y,
                displacement_px=0.0,
                activity_state=INACTIVE,
                last_seen_frame=frame_idx,
                last_seen_time_s=float(time_s),
            )

            self.records[object_id] = record

            return self._make_result(
                frame_idx=frame_idx,
                time_s=time_s,
                record=record,
            )

        previous_x = record.x
        previous_y = record.y

        displacement_px = self.calculate_distance(
            old_x=previous_x,
            old_y=previous_y,
            new_x=x,
            new_y=y,
        )

        if displacement_px > self.inactive_range_px:
            activity_state = AWAKE
        else:
            activity_state = INACTIVE

        record.previous_x = previous_x
        record.previous_y = previous_y

        record.x = x
        record.y = y
        record.well_label = well_label

        record.displacement_px = displacement_px
        record.activity_state = activity_state

        record.last_seen_frame = frame_idx
        record.last_seen_time_s = float(time_s)

        return self._make_result(
            frame_idx=frame_idx,
            time_s=time_s,
            record=record,
        )

    def update_detection(
            self,
            frame_idx: int,
            time_s: float,
            detection: dict,
    ) -> Optional[FlyActivityResult]:
        """
        Update activity using one pipeline detection dictionary.

        Expected fields:
            id
            centroid

        Optional field:
            well_label

        Returns None when required data is missing.
        """

        object_id = detection.get("id")
        centroid = detection.get("centroid")

        if object_id is None:
            return None

        if centroid is None or len(centroid) != 2:
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

    def update_frame(
            self,
            frame_idx: int,
            time_s: float,
            detections: Iterable[dict],
    ) -> list[FlyActivityResult]:
        """Update every valid detection in one frame."""

        results: list[FlyActivityResult] = []

        for detection in detections:
            result = self.update_detection(
                frame_idx=frame_idx,
                time_s=time_s,
                detection=detection,
            )

            if result is not None:
                results.append(result)

        return results

    def get_state(
            self,
            object_id: int,
    ) -> str:
        """Return the current activity state of one fly."""

        record = self.records.get(int(object_id))

        if record is None:
            return UNKNOWN

        return record.activity_state

    def get_record(
            self,
            object_id: int,
    ) -> Optional[FlyActivityRecord]:
        """Return the current activity record for one fly."""

        return self.records.get(int(object_id))

    def reset_fly(
            self,
            object_id: int,
    ) -> None:
        """Remove one fly's stored activity history."""

        self.records.pop(int(object_id), None)

    def reset(self) -> None:
        """Remove all stored activity history."""

        self.records.clear()

    @staticmethod
    def calculate_distance(
            old_x: float,
            old_y: float,
            new_x: float,
            new_y: float,
    ) -> float:
        """Calculate Euclidean displacement between two positions."""

        return hypot(
            float(new_x) - float(old_x),
            float(new_y) - float(old_y),
            )

    @staticmethod
    def _make_result(
            frame_idx: int,
            time_s: float,
            record: FlyActivityRecord,
    ) -> FlyActivityResult:
        """Build an immutable result from a stored activity record."""

        return FlyActivityResult(
            frame_idx=int(frame_idx),
            time_s=float(time_s),
            object_id=record.object_id,
            well_label=record.well_label,
            x=record.x,
            y=record.y,
            previous_x=record.previous_x,
            previous_y=record.previous_y,
            displacement_px=record.displacement_px,
            activity_state=record.activity_state,
        )