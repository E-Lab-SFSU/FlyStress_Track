"""Incremental CSV logging."""
from __future__ import annotations
import csv
from pathlib import Path

FIELDS = [
    "timestamp_iso", "elapsed_seconds", "image", "frame_number", "well", "fly_name",
    "fly_state", "valid_tracking", "tracking_reason", "detected", "registration_succeeded",
    "registration_score", "x_px", "y_px", "well_relative_x_px", "well_relative_y_px",
    "area_px", "threshold_value", "raw_distance_px", "distance_px", "rolling_distance_px",
    "rolling_samples", "immobile_duration_seconds"
]


class ResultsLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8", buffering=1)
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()

    def write(self, row: dict[str, object]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
