"""
fly_position.py

Saves the tracked position of each detected fly for every frame.

One CSV row is written for every tracked fly detected in a frame.

Expected detection fields:
    id
    centroid
    well_label
    well_number
    well_row
    well_column

This module does not:
- Detect flies.
- Track flies.
- Assign flies to wells.
- Determine activity or sleep state.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional


class FlyPositionLogger:
    # Write tracked fly positions to a dedicated CSV file.

    FIELDNAMES = [
        "frame_idx",
        "time_s",
        "object_id",
        "fly_label",
        "well_label",
        "well_number",
        "well_row",
        "well_column",
        "centroid_x",
        "centroid_y",
    ]

    def __init__(
            self,
            csv_path: str | Path,
            fps: float,
    ) -> None:

        # Create a fly-position CSV logger.

        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        self.csv_path = Path(csv_path)
        self.fps = float(fps)

        self.csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._csv_file = self.csv_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        )

        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=self.FIELDNAMES,
        )

        self._writer.writeheader()

        self.rows_written = 0
        self.closed = False

    def log_frame(
            self,
            frame_idx: int,
            detections: Iterable[dict],
            time_s: Optional[float] = None,
    ) -> int:
        """
        Save all valid tracked detections from one frame.

        Args:
            frame_idx: index 0-n

            detections: tracked detections after well assignment.

            time_s:
                Optional explicit frame time. When omitted, it is
                calculated as frame_idx / fps.

        Returns:
            Number of rows written for this frame.
        """

        self._ensure_open()

        if frame_idx < 0:
            raise ValueError("frame_idx cannot be negative")

        if time_s is None:
            time_s = frame_idx / self.fps

        rows_this_frame = 0

        for detection in detections:
            if self.log_detection(
                    frame_idx=frame_idx,
                    time_s=time_s,
                    detection=detection,
            ):
                rows_this_frame += 1

        return rows_this_frame

    def log_detection(
            self,
            frame_idx: int,
            time_s: float,
            detection: dict,
    ) -> bool:
        """
        Save one tracked detection.

        Returns:
            True if a row was written.
            False if the detection lacked required position data.
        """

        self._ensure_open()

        centroid = detection.get("centroid")
        object_id = detection.get("id")

        if object_id is None:
            return False

        if centroid is None or len(centroid) != 2:
            return False

        x, y = centroid

        if x is None or y is None:
            return False

        well_label = detection.get("well_label")

        fly_label = self.make_fly_label(
            well_label=well_label,
            object_id=object_id,
        )

        self._writer.writerow(
            {
                "frame_idx": int(frame_idx),
                "time_s": f"{float(time_s):.6f}",
                "object_id": int(object_id),
                "fly_label": fly_label,
                "well_label": (
                    well_label
                    if well_label is not None
                    else ""
                ),
                "well_number": self._optional_value(
                    detection.get("well_number")
                ),
                "well_row": self._optional_value(
                    detection.get("well_row")
                ),
                "well_column": self._optional_value(
                    detection.get("well_column")
                ),
                "centroid_x": f"{float(x):.3f}",
                "centroid_y": f"{float(y):.3f}",
            }
        )

        self.rows_written += 1
        return True

    @staticmethod
    def make_fly_label(
            well_label: Optional[str],
            object_id: int,
    ) -> str:
        """
        Create a readable fly label.
            A1-Fly0
            C4-Fly17
            Outside-Fly8
        Outside is typically not a fly
        The tracker ID remains part of the name so two flies detected
        in the same well do not receive the same identity.
        """

        if well_label is None:
            return f"Outside-Fly{object_id}"

        return f"{well_label}-Fly{object_id}"

    def flush(self) -> None:
        self._ensure_open()
        self._csv_file.flush()

    def close(self) -> None:
        if self.closed:
            return

        self._csv_file.flush()
        self._csv_file.close()
        self.closed = True

    def _ensure_open(self) -> None:
        if self.closed or self._csv_file.closed:
            raise RuntimeError(
                "FlyPositionLogger is already closed"
            )

    @staticmethod
    def _optional_value(value) -> object:
        """Convert None to an empty CSV field."""

        if value is None:
            return ""

        return value

    def __enter__(self) -> "FlyPositionLogger":
        return self

    def __exit__(
            self,
            exception_type,
            exception_value,
            traceback,
    ) -> None:
        self.close()
