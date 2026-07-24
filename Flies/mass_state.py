"""

mass_state.py

Calculates amount of flies awake and sleep.
When x% of flies are asleep, default 50%, changed is saved.

to be used later when activating shaker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import config

from Flies.fly_activity import AWAKE, INACTIVE, UNKNOWN
from Flies.fly_sleep import SLEEP, FlySleepResult


@dataclass(frozen=True)
class MassStateResult:
    """Population-level behavior result for one frame."""

    frame_idx: Optional[int]
    time_s: Optional[float]

    expected_total: int
    observed_total: int

    awake_count: int
    inactive_count: int
    sleep_count: int
    unknown_count: int

    sleep_percent: float
    required_sleep_percent: float

    threshold_met: bool

    sleeping_object_ids: tuple[int, ...]
    sleeping_wells: tuple[str, ...]

    @property
    def threshold_text(self) -> str:
        """Return YES or NO for display and CSV output."""

        return "YES" if self.threshold_met else "NO"

    @property
    def message(self) -> str:
        """Return a readable threshold description."""

        if self.threshold_met:
            return (
                f"YES ({self.sleep_percent:.1f}% of flies "
                f"are asleep)"
            )

        return (
            f"NO ({self.sleep_percent:.1f}% of flies "
            f"are asleep; {self.required_sleep_percent:.1f}% "
            f"required)"
        )


class MassStateTracker:
    """Calculate experiment-wide sleep statistics."""

    def __init__(
            self,
            expected_total_flies: Optional[int] = None,
            required_sleep_percent: Optional[float] = None,
    ) -> None:
        """
        Args:
            expected_total_flies:
                Number of flies expected in the experiment.

                When omitted, config.E_TOTAL_FLIES is used.

            required_sleep_percent:
                Percentage required to meet the mass-sleep condition.

                When omitted, config.SLEEP_AMT is used.
        """

        if expected_total_flies is None:
            expected_total_flies = config.E_TOTAL_FLIES

        if required_sleep_percent is None:
            required_sleep_percent = config.SLEEP_AMT

        if expected_total_flies <= 0:
            raise ValueError(
                "expected_total_flies must be greater than zero"
            )

        if not 0.0 <= required_sleep_percent <= 100.0:
            raise ValueError(
                "required_sleep_percent must be between 0 and 100"
            )

        self.expected_total_flies = int(
            expected_total_flies
        )

        self.required_sleep_percent = float(
            required_sleep_percent
        )

        self.last_result: Optional[MassStateResult] = None

        # Easy-to-call variable for later functions.
        self.sleep_amount_reached = False

    def update(
            self,
            sleep_results: Iterable[FlySleepResult],
            frame_idx: Optional[int] = None,
            time_s: Optional[float] = None,
    ) -> MassStateResult:
        """
        Calculate the population state from current per-fly results.

        Only one result per object ID is counted.
        """

        latest_by_object: dict[int, FlySleepResult] = {}

        for result in sleep_results:
            latest_by_object[int(result.object_id)] = result

        awake_count = 0
        inactive_count = 0
        sleep_count = 0
        explicit_unknown_count = 0

        sleeping_object_ids: list[int] = []
        sleeping_wells: set[str] = set()

        for result in latest_by_object.values():
            if result.state == AWAKE:
                awake_count += 1

            elif result.state == INACTIVE:
                inactive_count += 1

            elif result.state == SLEEP:
                sleep_count += 1
                sleeping_object_ids.append(
                    result.object_id
                )

                if result.well_label is not None:
                    sleeping_wells.add(
                        result.well_label
                    )

            else:
                explicit_unknown_count += 1

        observed_total = len(latest_by_object)

        unobserved_count = max(
            0,
            self.expected_total_flies - observed_total,
            )

        unknown_count = (
                explicit_unknown_count
                + unobserved_count
        )

        sleep_percent = (
                sleep_count
                / self.expected_total_flies
                * 100.0
        )

        threshold_met = (
                sleep_percent
                >= self.required_sleep_percent
        )

        self.sleep_amount_reached = threshold_met

        result = MassStateResult(
            frame_idx=frame_idx,
            time_s=time_s,
            expected_total=self.expected_total_flies,
            observed_total=observed_total,
            awake_count=awake_count,
            inactive_count=inactive_count,
            sleep_count=sleep_count,
            unknown_count=unknown_count,
            sleep_percent=sleep_percent,
            required_sleep_percent=(
                self.required_sleep_percent
            ),
            threshold_met=threshold_met,
            sleeping_object_ids=tuple(
                sorted(sleeping_object_ids)
            ),
            sleeping_wells=tuple(
                sorted(sleeping_wells)
            ),
        )

        self.last_result = result
        return result

    def threshold_is_met(self) -> bool:
        """Return the current mass-sleep threshold variable."""

        return self.sleep_amount_reached

    def get_last_result(
            self,
    ) -> Optional[MassStateResult]:
        """Return the most recently calculated population result."""

        return self.last_result

    @staticmethod
    def summary_text(
            result: MassStateResult,
    ) -> str:
        """Create a readable population summary."""

        frame_text = (
            f"Frame {result.frame_idx}"
            if result.frame_idx is not None
            else "Mass state"
        )

        return (
            f"{frame_text}: "
            f"awake={result.awake_count}, "
            f"inactive={result.inactive_count}, "
            f"sleep={result.sleep_count}, "
            f"unknown={result.unknown_count}, "
            f"sleep_percent={result.sleep_percent:.1f}%, "
            f"threshold={result.threshold_text}"
        )