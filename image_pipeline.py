# Copyright (c) 2025 Thomas Zimmerman — MIT License
"""
image_pipeline.py

Sleep-detection pipeline for a directory of sequential plate images
(one image per second, imageNNNNNN.png), per the simplified guide:

    align each image to a reference -> diff -> per-well fly position
    -> jitter-gated displacement -> rolling 5-min distance -> sleep state

Mirrors pipeline.FlyPipeline's structure (same tracker, well-assignment,
activity, sleep, mass-state, and CSV-writing components) but sources
frames from a directory of images instead of cv2.VideoCapture, and uses
image_detector's phase-correlation alignment instead of median
background subtraction.
"""

from __future__ import annotations

import csv
import glob
import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np

import config
from image_detector import to_gray_blur, detect_objects_in_sequence
from tracker import CentroidTracker

from Flies.fly_activity import FlyActivityTracker
from Flies.fly_sleep import FlySleepTracker
from Flies.rolling_sleep import RollingDistanceSleepTracker
from Flies.mass_state import MassStateTracker
from Flies.occupancy import OccupancyValidator
from WellPlate.detect_wells import detect_plate_wells_adjusted
from WellPlate.well_overlay import add_well_overlay


class ImagePipeline:
    """Coordinate alignment, detection, tracking, sleep state, and CSV output."""

    def __init__(self, image_dir: str, show: bool = False) -> None:
        self.image_dir = os.path.abspath(image_dir)
        self.show = bool(show)

        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "image*.png")))
        if len(self.image_paths) < 2:
            raise ValueError(
                f"Need at least 2 sequential images in {self.image_dir}"
            )

        self.fps = float(config.FPS)

        reference_frame = cv2.imread(self.image_paths[0])
        if reference_frame is None:
            raise IOError(f"Cannot read reference image: {self.image_paths[0]}")

        self.ref_blur = to_gray_blur(reference_frame)

        # Detect the 48 wells once, from the reference frame. Every
        # later frame is aligned back to this same reference (see
        # image_detector.align_to_reference), so wells only need to be
        # translated by that frame's shift, not re-detected.
        self.base_plate = detect_plate_wells_adjusted(reference_frame)
        print(f"Detected {self.base_plate.total_wells} wells from reference frame.")

        self.tracker = CentroidTracker()

        self.occupancy_validator = OccupancyValidator.from_plate(
            plate=self.base_plate,
            expected_per_well=config.E_FLY_PER_WELL,
        )
        self.fly_activity_tracker = FlyActivityTracker(
            inactive_range_px=config.INACTIVE_RNG,
        )

        self.sleep_model = config.SLEEP_MODEL
        if self.sleep_model == "rolling_window":
            self.sleep_tracker = RollingDistanceSleepTracker(fps=self.fps)
        else:
            self.sleep_tracker = FlySleepTracker(
                sleep_threshold_s=config.SLEEP_SEC,
            )

        self.mass_state_tracker = MassStateTracker(
            expected_total_flies=config.E_TOTAL_FLIES,
            required_sleep_percent=config.SLEEP_AMT,
        )

        self.frame_idx = 0
        self._closed = False

        self._init_output_files()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            for frame_idx, path in enumerate(self.image_paths):
                self.frame_idx = frame_idx
                self.process_image(path)

                if self.show:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        finally:
            self.cleanup()

    def process_image(self, path: str) -> None:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Skipping unreadable image: {path}")
            return

        time_s = self.frame_idx / self.fps

        detections, foreground_mask, shift_x, shift_y, response, reliable = (
            detect_objects_in_sequence(frame, self.ref_blur)
        )

        if not reliable:
            print(
                f"Frame {self.frame_idx} ({os.path.basename(path)}): "
                f"low alignment confidence (response={response:.3f})."
            )

        plate = detect_plate_wells_adjusted(
            reference_frame=frame,
            shift_x=shift_x,
            shift_y=shift_y,
            base_plate=self.base_plate,
        )

        detections = self.tracker.update(detections)
        detections = plate.assign_detections(detections)

        occupancy_report = self.occupancy_validator.analyze(
            detections=detections,
            frame_idx=self.frame_idx,
        )

        sleep_results = self._add_activity_and_sleep(detections, time_s)
        mass_state = self.mass_state_tracker.update(
            sleep_results=sleep_results,
            frame_idx=self.frame_idx,
            time_s=time_s,
        )

        for detection in detections:
            self._write_tracking_row(detection, time_s, shift_x, shift_y, response)

        self._write_mass_state_row(mass_state)

        flush_interval = max(1, int(config.CSV_FLUSH_INTERVAL_FRAMES))
        if self.frame_idx % flush_interval == 0:
            self._flush_outputs()

        if self.show:
            display_frame = add_well_overlay(
                frame=frame,
                plate=plate,
                draw_assignment_boundary=config.SHOW_ASSIGNMENT_BOUNDARY,
            )
            cv2.imshow("Foreground Mask", foreground_mask)
            cv2.imshow("Fly Tracking", display_frame)

    # ------------------------------------------------------------------
    # Per-frame enrichment
    # ------------------------------------------------------------------

    def _add_activity_and_sleep(self, detections: list[dict[str, Any]], time_s: float):
        """Attach activity/sleep fields; returns per-fly sleep results for this frame."""

        activity_results = self.fly_activity_tracker.update_frame(
            frame_idx=self.frame_idx,
            time_s=time_s,
            detections=detections,
        )
        activity_by_id = {r.object_id: r for r in activity_results}

        if self.sleep_model == "rolling_window":
            sleep_results = self.sleep_tracker.update_frame(
                frame_idx=self.frame_idx,
                time_s=time_s,
                detections=detections,
            )
        else:
            sleep_results = self.sleep_tracker.update_frame(activity_results)

        sleep_by_id = {r.object_id: r for r in sleep_results}

        for detection in detections:
            object_id = int(detection["id"])
            activity = activity_by_id.get(object_id)
            sleep = sleep_by_id.get(object_id)

            detection["displacement_px"] = (
                activity.displacement_px if activity is not None else None
            )
            detection["activity_state"] = (
                activity.activity_state if activity is not None else "unknown"
            )

            if sleep is not None:
                detection["fly_state"] = sleep.state
                if self.sleep_model == "rolling_window":
                    detection["rolling_distance_px"] = sleep.rolling_distance_px
                else:
                    detection["inactive_duration_s"] = sleep.inactive_duration_s
            else:
                detection["fly_state"] = "unknown"

        return sleep_results

    # ------------------------------------------------------------------
    # Output initialization
    # ------------------------------------------------------------------

    def _init_output_files(self) -> None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        dir_name = os.path.basename(self.image_dir.rstrip(os.sep)) or "images"
        prefix = f"{dir_name}_{timestamp}"

        self.csv_dir = os.path.join(self.image_dir, "csv")
        os.makedirs(self.csv_dir, exist_ok=True)

        extra_field = (
            "rolling_distance_px"
            if self.sleep_model == "rolling_window"
            else "inactive_duration_s"
        )

        self.tracking_csv_path = os.path.join(self.csv_dir, f"{prefix}_tracking.csv")
        self.tracking_csv_file = open(
            self.tracking_csv_path, "w", newline="", encoding="utf-8"
        )
        self.tracking_csv_writer = csv.DictWriter(
            self.tracking_csv_file,
            fieldnames=[
                "frame_idx", "time_s", "object_id", "fly_label",
                "well_label", "well_number", "well_row", "well_column",
                "centroid_x", "centroid_y",
                "displacement_px", "activity_state", extra_field, "fly_state",
                "align_shift_x", "align_shift_y", "align_response",
            ],
        )
        self.tracking_csv_writer.writeheader()
        self._extra_field = extra_field

        self.mass_csv_path = os.path.join(self.csv_dir, f"{prefix}_mass_state.csv")
        self.mass_csv_file = open(
            self.mass_csv_path, "w", newline="", encoding="utf-8"
        )
        self.mass_csv_writer = csv.DictWriter(
            self.mass_csv_file,
            fieldnames=[
                "frame_idx", "time_s", "expected_total", "observed_total",
                "awake_count", "inactive_count", "sleep_count", "unknown_count",
                "sleep_percent", "required_sleep_percent",
                "threshold_met", "threshold_message",
            ],
        )
        self.mass_csv_writer.writeheader()

    # ------------------------------------------------------------------
    # CSV writers
    # ------------------------------------------------------------------

    def _write_tracking_row(self, detection, time_s, shift_x, shift_y, response) -> None:
        object_id = int(detection["id"])
        x, y = detection["centroid"]
        well_label = detection.get("well_label")
        fly_label = (
            f"{well_label}-Fly{object_id}"
            if well_label is not None
            else f"Outside-Fly{object_id}"
        )

        extra_value = detection.get(self._extra_field)

        self.tracking_csv_writer.writerow({
            "frame_idx": self.frame_idx,
            "time_s": f"{time_s:.6f}",
            "object_id": object_id,
            "fly_label": fly_label,
            "well_label": well_label or "",
            "well_number": self._optional(detection.get("well_number")),
            "well_row": self._optional(detection.get("well_row")),
            "well_column": self._optional(detection.get("well_column")),
            "centroid_x": f"{float(x):.3f}",
            "centroid_y": f"{float(y):.3f}",
            "displacement_px": self._number(detection.get("displacement_px"), 3),
            "activity_state": detection.get("activity_state", "unknown"),
            self._extra_field: self._number(extra_value, 3),
            "fly_state": detection.get("fly_state", "unknown"),
            "align_shift_x": f"{shift_x:.3f}",
            "align_shift_y": f"{shift_y:.3f}",
            "align_response": f"{response:.4f}",
        })

    def _write_mass_state_row(self, mass_state) -> None:
        self.mass_csv_writer.writerow({
            "frame_idx": mass_state.frame_idx,
            "time_s": self._number(mass_state.time_s, 6),
            "expected_total": mass_state.expected_total,
            "observed_total": mass_state.observed_total,
            "awake_count": mass_state.awake_count,
            "inactive_count": mass_state.inactive_count,
            "sleep_count": mass_state.sleep_count,
            "unknown_count": mass_state.unknown_count,
            "sleep_percent": f"{mass_state.sleep_percent:.3f}",
            "required_sleep_percent": f"{mass_state.required_sleep_percent:.3f}",
            "threshold_met": mass_state.threshold_text,
            "threshold_message": mass_state.message,
        })

    # ------------------------------------------------------------------
    # Cleanup and helpers
    # ------------------------------------------------------------------

    def _flush_outputs(self) -> None:
        if not self.tracking_csv_file.closed:
            self.tracking_csv_file.flush()
        if not self.mass_csv_file.closed:
            self.mass_csv_file.flush()

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True

        if not self.tracking_csv_file.closed:
            self.tracking_csv_file.flush()
            self.tracking_csv_file.close()
        if not self.mass_csv_file.closed:
            self.mass_csv_file.flush()
            self.mass_csv_file.close()

        cv2.destroyAllWindows()
        print("\nAnalysis complete.")
        print(f"CSV directory: {self.csv_dir}")
        print(f"Tracking CSV: {self.tracking_csv_path}")

    @staticmethod
    def _number(value: Any, decimal_places: int) -> str:
        if value is None:
            return ""
        return f"{float(value):.{decimal_places}f}"

    @staticmethod
    def _optional(value: Any) -> Any:
        return "" if value is None else value