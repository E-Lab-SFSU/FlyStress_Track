"""Incremental CSV logging."""
from __future__ import annotations
import csv
from pathlib import Path

FIELDS = [
    "timestamp_iso", "elapsed_seconds", "image", "frame_number", "well", "fly_name",
    "fly_state", "valid_tracking", "tracking_reason", "detected", "registration_succeeded",
    "registration_score", "x_px", "y_px", "well_relative_x_px", "well_relative_y_px",
    "area_px", "threshold_value", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "raw_distance_px", "distance_px", "rolling_distance_px",
    "rolling_samples", "immobile_duration_seconds"
]

POSITION_FIELDS = [
    "timestamp_iso", "elapsed_seconds", "image", "frame_number", "well", "fly_name",
    "detected", "valid_tracking", "tracking_reason", "tracking_confidence", "detection_stage",
    "wall_mode", "radial_fraction",
    "arrival_change", "arrival_fraction", "departure_change",
    "x_px", "y_px", "well_relative_x_px", "well_relative_y_px",
    "center_gray", "component_median_gray", "area_px",
    "raw_step_distance_px", "step_distance_px",
    "cumulative_raw_distance_px", "cumulative_distance_px",
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


class PositionLogger:
    """Write one position/movement record per fly per analyzed frame.

    ``step_distance_px`` uses the same jitter deadband as the sleep analysis,
    while ``raw_step_distance_px`` is the direct centroid-to-centroid distance.
    Cumulative distances only grow across consecutive valid samples; a
    reacquisition after a long/missing gap does not invent movement through the
    period in which the fly was not visible.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8", buffering=1)
        self.writer = csv.DictWriter(self.file, fieldnames=POSITION_FIELDS)
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
