"""CSV output helpers for FlyStress analysis."""

from __future__ import annotations

import csv
from pathlib import Path


RESULT_FIELDS = [
    "timestamp_iso",
    "elapsed_seconds",
    "image",
    "frame_number",
    "well",
    "fly_name",
    "fly_slot",
    "fly_state",
    "detected",
    "x_px",
    "y_px",
    "well_relative_x_px",
    "well_relative_y_px",
    "area_px",
    "threshold_value",
    "raw_distance_px",
    "distance_px",
    "rolling_distance_px",
    "rolling_samples",
    "missed_frames",
    "registration_succeeded",
    "registration_score",
]


class ResultsLogger:
    """Write rows incrementally so partial results survive an interrupted run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=RESULT_FIELDS)
        self.writer.writeheader()
        self.file.flush()

    def write(self, row: dict[str, object]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()

    def __enter__(self) -> "ResultsLogger":
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self.close()
