"""
occupancy.py

Counts tracked flies in every well and compares the observed counts
against the expected number of flies per well.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import config


@dataclass(frozen=True)
class WellOccupancy:
    """Occupancy result for one well."""

    well_label: str
    expected_count: int
    detected_count: int
    object_ids: tuple[int, ...] = field(
        default_factory=tuple
    )

    @property
    def difference(self) -> int:
        """
        Return detected minus expected.

        Negative:
            Too few flies.

        Zero:
            Expected occupancy.

        Positive:
            Too many flies.
        """

        return self.detected_count - self.expected_count

    @property
    def is_valid(self) -> bool:
        """Return True when observed occupancy matches expectation."""

        return self.detected_count == self.expected_count

    @property
    def status(self) -> str:
        """Return a readable occupancy status."""

        if self.detected_count < self.expected_count:
            return "MISSING"

        if self.detected_count > self.expected_count:
            return "EXCESS"

        return "OK"


@dataclass(frozen=True)
class OccupancyReport:
    """Complete occupancy result for one frame."""

    frame_idx: Optional[int]
    wells: dict[str, WellOccupancy]
    outside_object_ids: tuple[int, ...]
    duplicate_object_ids: tuple[int, ...]

    @property
    def expected_total(self) -> int:
        """Return the expected total number of flies."""

        return sum(
            result.expected_count
            for result in self.wells.values()
        )

    @property
    def detected_inside_total(self) -> int:
        """Return the number of flies assigned to wells."""

        return sum(
            result.detected_count
            for result in self.wells.values()
        )

    @property
    def outside_count(self) -> int:
        """Return the number of detections outside all wells."""

        return len(self.outside_object_ids)

    @property
    def detected_total(self) -> int:
        """Return all unique detections inside and outside wells."""

        return (
                self.detected_inside_total
                + self.outside_count
        )

    @property
    def valid_well_count(self) -> int:
        """Return how many wells match expected occupancy."""

        return sum(
            result.is_valid
            for result in self.wells.values()
        )

    @property
    def invalid_well_count(self) -> int:
        """Return how many wells do not match expectation."""

        return len(self.wells) - self.valid_well_count

    @property
    def empty_wells(self) -> tuple[str, ...]:
        """Return labels of wells containing zero detected flies."""

        return tuple(
            label
            for label, result in self.wells.items()
            if result.detected_count == 0
        )

    @property
    def missing_wells(self) -> tuple[str, ...]:
        """Return wells with fewer flies than expected."""

        return tuple(
            label
            for label, result in self.wells.items()
            if result.detected_count < result.expected_count
        )

    @property
    def excess_wells(self) -> tuple[str, ...]:
        """Return wells with more flies than expected."""

        return tuple(
            label
            for label, result in self.wells.items()
            if result.detected_count > result.expected_count
        )

    @property
    def is_valid(self) -> bool:
        """
        Return True when:
        - Every well has its expected number of flies.
        - No flies are outside the wells.
        - No duplicate tracker IDs were received.
        """

        return (
                self.invalid_well_count == 0
                and self.outside_count == 0
                and len(self.duplicate_object_ids) == 0
        )

    def count_for_well(
            self,
            well_label: str,
    ) -> int:
        """Return the detected count for one well."""

        result = self.wells.get(
            well_label.strip().upper()
        )

        if result is None:
            raise KeyError(
                f"Unknown well label: {well_label}"
            )

        return result.detected_count


class OccupancyValidator:
    """Count and validate tracked flies in each configured well."""

    def __init__(
            self,
            well_labels: Iterable[str],
            expected_per_well: int,
    ) -> None:
        """
        Args:
            well_labels:
                Every valid well label, preferably from plate.wells.

            expected_per_well:
                Expected number of flies in each well.
        """

        if expected_per_well < 0:
            raise ValueError(
                "expected_per_well cannot be negative"
            )

        normalized_labels = []

        for label in well_labels:
            normalized = str(label).strip().upper()

            if not normalized:
                raise ValueError(
                    "well labels cannot be empty"
                )

            if normalized in normalized_labels:
                raise ValueError(
                    f"Duplicate well label: {normalized}"
                )

            normalized_labels.append(normalized)

        if not normalized_labels:
            raise ValueError(
                "At least one well label is required"
            )

        self.well_labels = tuple(normalized_labels)
        self.expected_per_well = int(
            expected_per_well
        )

        self.last_report: Optional[OccupancyReport] = None

    @classmethod
    def from_plate(
            cls,
            plate,
            expected_per_well: Optional[int] = None,
    ) -> "OccupancyValidator":
        """
        Create a validator from a WellPlate instance.

        When expected_per_well is omitted, config.E_FLY_PER_WELL
        is used.
        """

        if expected_per_well is None:
            expected_per_well = config.E_FLY_PER_WELL

        return cls(
            well_labels=[
                well.label
                for well in plate.wells
            ],
            expected_per_well=expected_per_well,
        )

    def analyze(
            self,
            detections: Iterable[dict],
            frame_idx: Optional[int] = None,
    ) -> OccupancyReport:
        """
        Count flies in every well for one frame.

        Each detection should contain:
            id
            well_label

        Detections with well_label=None are counted as outside.
        """

        objects_by_well: dict[str, list[int]] = {
            label: []
            for label in self.well_labels
        }

        outside_object_ids: list[int] = []
        duplicate_object_ids: list[int] = []
        seen_object_ids: set[int] = set()

        for detection in detections:
            object_id = detection.get("id")

            if object_id is None:
                continue

            object_id = int(object_id)

            if object_id in seen_object_ids:
                duplicate_object_ids.append(object_id)
                continue

            seen_object_ids.add(object_id)

            well_label = detection.get("well_label")

            if well_label is None:
                outside_object_ids.append(object_id)
                continue

            normalized_label = str(
                well_label
            ).strip().upper()

            if normalized_label not in objects_by_well:
                outside_object_ids.append(object_id)
                continue

            objects_by_well[normalized_label].append(
                object_id
            )

        well_results = {
            label: WellOccupancy(
                well_label=label,
                expected_count=self.expected_per_well,
                detected_count=len(object_ids),
                object_ids=tuple(object_ids),
            )
            for label, object_ids in objects_by_well.items()
        }

        report = OccupancyReport(
            frame_idx=frame_idx,
            wells=well_results,
            outside_object_ids=tuple(
                outside_object_ids
            ),
            duplicate_object_ids=tuple(
                duplicate_object_ids
            ),
        )

        self.last_report = report
        return report

    @staticmethod
    def summary_text(
            report: OccupancyReport,
    ) -> str:
        """Return a concise readable report."""

        frame_text = (
            f"Frame {report.frame_idx}"
            if report.frame_idx is not None
            else "Occupancy"
        )

        lines = [
            (
                f"{frame_text}: "
                f"{report.detected_inside_total}/"
                f"{report.expected_total} flies inside wells"
            ),
            (
                f"Valid wells: "
                f"{report.valid_well_count}/"
                f"{len(report.wells)}"
            ),
            f"Outside detections: {report.outside_count}",
            (
                "Overall status: "
                f"{'OK' if report.is_valid else 'CHECK'}"
            ),
        ]

        if report.missing_wells:
            lines.append(
                "Missing flies: "
                + ", ".join(report.missing_wells)
            )

        if report.excess_wells:
            lines.append(
                "Excess flies: "
                + ", ".join(report.excess_wells)
            )

        if report.duplicate_object_ids:
            duplicate_text = ", ".join(
                str(object_id)
                for object_id in report.duplicate_object_ids
            )

            lines.append(
                f"Duplicate tracker IDs: {duplicate_text}"
            )

        return "\n".join(lines)