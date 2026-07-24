"""
fly_sleep.py

determine if a fly is asleep using 'active' + 'inactive' states
Saves info in csv file

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import config

from Flies.fly_activity import (
    AWAKE,
    INACTIVE,
    UNKNOWN,
    FlyActivityResult,
)


SLEEP = "sleep"


@dataclass
class FlySleepRecord:
    """Stored sleep timing information for one tracked fly."""

    object_id: int
    well_label: Optional[str]

    state: str = INACTIVE

    inactive_start_time_s: Optional[float] = None
    inactive_duration_s: float = 0.0

    last_seen_frame: Optional[int] = None
    last_seen_time_s: Optional[float] = None


@dataclass(frozen=True)
class FlySleepResult:
    """Sleep classification for one fly in one frame."""

    frame_idx: int
    time_s: float

    object_id: int
    well_label: Optional[str]

    activity_state: str
    state: str

    inactive_start_time_s: Optional[float]
    inactive_duration_s: float
    sleep_threshold_s: float

    @property
    def is_awake(self) -> bool:
        return self.state == AWAKE

    @property
    def is_inactive(self) -> bool:
        return self.state == INACTIVE

    @property
    def is_asleep(self) -> bool:
        return self.state == SLEEP


class FlySleepTracker:
    """Maintain continuous inactivity timers for tracked flies."""

    def __init__(
            self,
            sleep_threshold_s: Optional[float] = None,
    ) -> None:
        """
        Args:
            sleep_threshold_s:
                Continuous inactivity required for sleep.

                When omitted, config.SLEEP_SEC is used.
        """

        if sleep_threshold_s is None:
            sleep_threshold_s = config.SLEEP_SEC

        if sleep_threshold_s <= 0:
            raise ValueError(
                "sleep_threshold_s must be greater than zero"
            )

        self.sleep_threshold_s = float(
            sleep_threshold_s
        )

        self.records: dict[int, FlySleepRecord] = {}

    def update(
            self,
            activity: FlyActivityResult,
    ) -> FlySleepResult:
        """
        Update the sleep timer for one fly.

        Awake movement resets the inactivity timer.
        """

        object_id = int(activity.object_id)
        current_time_s = float(activity.time_s)

        record = self.records.get(object_id)

        if record is None:
            record = FlySleepRecord(
                object_id=object_id,
                well_label=activity.well_label,
                state=INACTIVE,
                inactive_start_time_s=current_time_s,
                inactive_duration_s=0.0,
                last_seen_frame=activity.frame_idx,
                last_seen_time_s=current_time_s,
            )

            self.records[object_id] = record

        record.well_label = activity.well_label
        record.last_seen_frame = activity.frame_idx
        record.last_seen_time_s = current_time_s

        if activity.activity_state == AWAKE:
            record.state = AWAKE
            record.inactive_start_time_s = None
            record.inactive_duration_s = 0.0

        elif activity.activity_state == INACTIVE:
            if record.inactive_start_time_s is None:
                record.inactive_start_time_s = current_time_s

            record.inactive_duration_s = max(
                0.0,
                current_time_s
                - record.inactive_start_time_s,
                )

            if (
                    record.inactive_duration_s
                    >= self.sleep_threshold_s
            ):
                record.state = SLEEP
            else:
                record.state = INACTIVE

        else:
            record.state = UNKNOWN

        return self._make_result(
            activity=activity,
            record=record,
        )

    def update_frame(
            self,
            activity_results: Iterable[FlyActivityResult],
    ) -> list[FlySleepResult]:
        """Update sleep state for every activity result in a frame."""

        return [
            self.update(activity)
            for activity in activity_results
        ]

    def get_state(
            self,
            object_id: int,
    ) -> str:
        """Return the current state of one fly."""

        record = self.records.get(int(object_id))

        if record is None:
            return UNKNOWN

        return record.state

    def get_record(
            self,
            object_id: int,
    ) -> Optional[FlySleepRecord]:
        """Return the sleep record for one fly."""

        return self.records.get(int(object_id))

    def reset_fly(
            self,
            object_id: int,
    ) -> None:
        """Remove one fly's stored sleep history."""

        self.records.pop(int(object_id), None)

    def reset(self) -> None:
        """Remove all stored sleep history."""

        self.records.clear()

    def state_counts(self) -> dict[str, int]:
        """Return counts for all currently registered fly states."""

        counts = {
            AWAKE: 0,
            INACTIVE: 0,
            SLEEP: 0,
            UNKNOWN: 0,
        }

        for record in self.records.values():
            if record.state in counts:
                counts[record.state] += 1
            else:
                counts[UNKNOWN] += 1

        return counts

    def sleeping_object_ids(self) -> tuple[int, ...]:
        """Return tracker IDs currently classified as asleep."""

        return tuple(
            object_id
            for object_id, record in self.records.items()
            if record.state == SLEEP
        )

    def sleeping_wells(self) -> tuple[str, ...]:
        """Return well labels containing currently sleeping flies."""

        labels = {
            record.well_label
            for record in self.records.values()
            if (
                    record.state == SLEEP
                    and record.well_label is not None
            )
        }

        return tuple(sorted(labels))

    def _make_result(
            self,
            activity: FlyActivityResult,
            record: FlySleepRecord,
    ) -> FlySleepResult:
        """Build an immutable sleep result."""

        return FlySleepResult(
            frame_idx=activity.frame_idx,
            time_s=activity.time_s,
            object_id=activity.object_id,
            well_label=activity.well_label,
            activity_state=activity.activity_state,
            state=record.state,
            inactive_start_time_s=(
                record.inactive_start_time_s
            ),
            inactive_duration_s=(
                record.inactive_duration_s
            ),
            sleep_threshold_s=self.sleep_threshold_s,
        )