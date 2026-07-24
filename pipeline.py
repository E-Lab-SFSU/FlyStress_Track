# Copyright (c) 2025 Thomas Zimmerman — MIT License
"""
pipeline.py

Main fly-tracking pipeline.

Processing order:
    detection -> tracking -> well assignment -> occupancy
    -> activity -> sleep -> mass state -> CSV -> display
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np

import config
from detector import detect_objects
from features_motion import binary_motion_detect, windowed_velocity
from kinematics import head_tail_from_bbox
from movement_state import MovementStateTracker
from tracker import CentroidTracker

from Flies.fly_activity import FlyActivityTracker
from Flies.fly_position import FlyPositionLogger
from Flies.fly_sleep import FlySleepTracker
from Flies.mass_state import MassStateTracker
from Flies.occupancy import OccupancyValidator
from WellPlate.well_overlay import add_well_overlay
from WellPlate.well_plate import create_plate_from_config


class FlyPipeline:
    """Coordinate detection, tracking, state analysis, output, and display."""

    def __init__(self, video_path: str, show: bool = True) -> None:
        self.video_path = os.path.abspath(video_path)
        self.show = bool(show)

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        video_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = video_fps if video_fps > 0 else float(config.FPS)

        self.tracker = CentroidTracker()
        self.movement_state_tracker = MovementStateTracker()
        self.plate = create_plate_from_config()

        self.occupancy_validator = OccupancyValidator.from_plate(
            plate=self.plate,
            expected_per_well=config.E_FLY_PER_WELL,
        )
        self.fly_activity_tracker = FlyActivityTracker(
            inactive_range_px=config.INACTIVE_RNG,
        )
        self.fly_sleep_tracker = FlySleepTracker(
            sleep_threshold_s=config.SLEEP_SEC,
        )
        self.mass_state_tracker = MassStateTracker(
            expected_total_flies=config.E_TOTAL_FLIES,
            required_sleep_percent=config.SLEEP_AMT,
        )

        self.frame_idx = 0
        self.last_velocity_angle: dict[int, float] = {}
        self._closed = False

        self._init_output_files()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Process the video until completion or until q/ESC is pressed."""

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break

                self.process_frame(frame)
                self.frame_idx += 1

                if self.show:
                    key = cv2.waitKey(config.PLAYBACK_DELAY) & 0xFF
                    if key in (27, ord("q")):
                        break
        finally:
            self.cleanup()

    def process_frame(self, frame: np.ndarray) -> None:
        """Process one frame and enrich each detection dictionary in place."""

        time_s = self.frame_idx / self.fps

        detections, foreground_mask = detect_objects(frame, self.cap)

        if self.show:
            cv2.imshow("Foreground Mask", foreground_mask)

        detections = self.tracker.update(detections)
        detections = self.plate.assign_detections(detections)

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

        display_frame = None
        if self.show:
            display_frame = add_well_overlay(
                frame=frame,
                plate=self.plate,
                draw_assignment_boundary=config.SHOW_ASSIGNMENT_BOUNDARY,
            )

        for detection in detections:
            head, tail = self._add_motion_and_pose(
                detection=detection,
                foreground_mask=foreground_mask,
            )
            self._write_tracking_row(detection, time_s)

            if display_frame is not None:
                self.draw_object(display_frame, detection, head, tail)

        if self.position_logger is not None:
            self.position_logger.log_frame(
                frame_idx=self.frame_idx,
                detections=detections,
                time_s=time_s,
            )

        self._write_mass_state_row(mass_state)

        flush_interval = max(1, int(config.CSV_FLUSH_INTERVAL_FRAMES))
        if self.frame_idx % flush_interval == 0:
            self._flush_outputs()

        if display_frame is not None:
            self._draw_mass_summary(
                frame=display_frame,
                mass_state=mass_state,
                occupancy_report=occupancy_report,
            )
            cv2.imshow("Fly Tracking", display_frame)

    # ------------------------------------------------------------------
    # Per-frame enrichment
    # ------------------------------------------------------------------

    def _add_activity_and_sleep(
            self,
            detections: list[dict[str, Any]],
            time_s: float,
    ) -> list[Any]:
        """Attach activity and sleep fields and return current sleep results."""

        activity_results = self.fly_activity_tracker.update_frame(
            frame_idx=self.frame_idx,
            time_s=time_s,
            detections=detections,
        )
        activity_by_id = {
            result.object_id: result for result in activity_results
        }

        sleep_results = self.fly_sleep_tracker.update_frame(activity_results)
        sleep_by_id = {result.object_id: result for result in sleep_results}

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
            detection["fly_state"] = (
                sleep.state if sleep is not None else "unknown"
            )
            detection["inactive_duration_s"] = (
                sleep.inactive_duration_s if sleep is not None else 0.0
            )

        return sleep_results

    def _add_motion_and_pose(
            self,
            detection: dict[str, Any],
            foreground_mask: np.ndarray,
    ) -> tuple[Any, Any]:
        """Attach legacy motion, pose, and movement-state fields."""

        object_id = int(detection["id"])
        track = self.tracker.objects.get(object_id)

        speed = 0.0
        velocity_angle = self.last_velocity_angle.get(object_id)

        if track is not None:
            history = track["history"]
            recent_deltas = track["recent_deltas"]
            is_moving = binary_motion_detect(recent_deltas)

            if is_moving:
                speed, velocity_angle = windowed_velocity(history)
                if velocity_angle is not None:
                    self.last_velocity_angle[object_id] = velocity_angle
                else:
                    velocity_angle = self.last_velocity_angle.get(object_id)

        detection["speed_avg_px_s"] = speed
        detection["velocity_angle_deg"] = velocity_angle

        pose, head, tail, axis_vector = head_tail_from_bbox(
            foreground_mask,
            detection["contour"],
        )
        detection["pose"] = pose

        head_aligned = True
        if pose == "ELONGATED" and velocity_angle is not None:
            motion_vector = np.array(
                [
                    np.cos(np.deg2rad(velocity_angle)),
                    np.sin(np.deg2rad(velocity_angle)),
                ]
            )
            head_aligned = bool(np.dot(axis_vector, motion_vector) >= 0)

        detection["movement_state"] = self.movement_state_tracker.update(
            object_id,
            speed,
            pose,
            head_aligned,
        )
        return head, tail

    # ------------------------------------------------------------------
    # Output initialization
    # ------------------------------------------------------------------

    def _init_output_files(self) -> None:
        video_directory = os.path.dirname(self.video_path)
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        prefix = f"{video_name}_{timestamp}"

        self.csv_dir = os.path.join(video_directory, "csv")
        os.makedirs(self.csv_dir, exist_ok=True)

        self.tracking_csv_path = os.path.join(
            self.csv_dir,
            f"{prefix}_tracking.csv",
        )
        self.tracking_csv_file = open(
            self.tracking_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )
        self.tracking_csv_writer = csv.DictWriter(
            self.tracking_csv_file,
            fieldnames=[
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
                "displacement_px",
                "activity_state",
                "inactive_duration_s",
                "fly_state",
                "pose",
                "speed_avg_px_s",
                "velocity_angle_deg",
                "movement_state",
            ],
        )
        self.tracking_csv_writer.writeheader()

        self.position_logger = None
        if config.SAVE_POSITION_CSV:
            self.position_logger = FlyPositionLogger(
                csv_path=os.path.join(self.csv_dir, f"{prefix}_positions.csv"),
                fps=self.fps,
            )

        self.mass_csv_file = None
        self.mass_csv_writer = None
        if config.SAVE_MASS_STATE_CSV:
            self.mass_csv_file = open(
                os.path.join(self.csv_dir, f"{prefix}_mass_state.csv"),
                "w",
                newline="",
                encoding="utf-8",
            )
            self.mass_csv_writer = csv.DictWriter(
                self.mass_csv_file,
                fieldnames=[
                    "frame_idx",
                    "time_s",
                    "expected_total",
                    "observed_total",
                    "awake_count",
                    "inactive_count",
                    "sleep_count",
                    "unknown_count",
                    "sleep_percent",
                    "required_sleep_percent",
                    "threshold_met",
                    "threshold_message",
                ],
            )
            self.mass_csv_writer.writeheader()

        if config.SAVE_WELL_GEOMETRY_CSV:
            self._save_well_geometry_csv(
                os.path.join(self.csv_dir, f"{prefix}_wells.csv")
            )

    def _save_well_geometry_csv(self, csv_path: str) -> None:
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "well_number",
                    "well_label",
                    "row",
                    "column",
                    "center_x",
                    "center_y",
                    "diameter_px",
                    "radius_px",
                    "assignment_radius_px",
                ],
            )
            writer.writeheader()
            for well in self.plate.wells:
                writer.writerow(
                    {
                        "well_number": well.number,
                        "well_label": well.label,
                        "row": well.row,
                        "column": well.column,
                        "center_x": f"{well.center_x:.3f}",
                        "center_y": f"{well.center_y:.3f}",
                        "diameter_px": f"{well.diameter_px:.3f}",
                        "radius_px": f"{well.radius_px:.3f}",
                        "assignment_radius_px": (
                            f"{self.plate.assignment_radius_px:.3f}"
                        ),
                    }
                )

    # ------------------------------------------------------------------
    # CSV writers
    # ------------------------------------------------------------------

    def _write_tracking_row(
            self,
            detection: dict[str, Any],
            time_s: float,
    ) -> None:
        object_id = int(detection["id"])
        x, y = detection["centroid"]
        well_label = detection.get("well_label")
        fly_label = (
            f"{well_label}-Fly{object_id}"
            if well_label is not None
            else f"Outside-Fly{object_id}"
        )

        self.tracking_csv_writer.writerow(
            {
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
                "displacement_px": self._number(
                    detection.get("displacement_px"), 3
                ),
                "activity_state": detection.get("activity_state", "unknown"),
                "inactive_duration_s": self._number(
                    detection.get("inactive_duration_s"), 3
                ),
                "fly_state": detection.get("fly_state", "unknown"),
                "pose": detection.get("pose", ""),
                "speed_avg_px_s": self._number(
                    detection.get("speed_avg_px_s"), 3
                ),
                "velocity_angle_deg": self._number(
                    detection.get("velocity_angle_deg"), 2
                ),
                "movement_state": detection.get("movement_state", ""),
            }
        )

    def _write_mass_state_row(self, mass_state: Any) -> None:
        if self.mass_csv_writer is None:
            return

        self.mass_csv_writer.writerow(
            {
                "frame_idx": mass_state.frame_idx,
                "time_s": self._number(mass_state.time_s, 6),
                "expected_total": mass_state.expected_total,
                "observed_total": mass_state.observed_total,
                "awake_count": mass_state.awake_count,
                "inactive_count": mass_state.inactive_count,
                "sleep_count": mass_state.sleep_count,
                "unknown_count": mass_state.unknown_count,
                "sleep_percent": f"{mass_state.sleep_percent:.3f}",
                "required_sleep_percent": (
                    f"{mass_state.required_sleep_percent:.3f}"
                ),
                "threshold_met": mass_state.threshold_text,
                "threshold_message": mass_state.message,
            }
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def draw_object(
            self,
            frame: np.ndarray,
            detection: dict[str, Any],
            head: Any,
            tail: Any,
    ) -> None:
        x, y = detection["centroid"]
        x, y = int(x), int(y)
        object_id = int(detection["id"])
        pose = detection.get("pose", "BALL")
        speed = detection.get("speed_avg_px_s", 0.0)

        rect = cv2.minAreaRect(detection["contour"])
        box = np.int32(cv2.boxPoints(rect))
        edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
        lengths = [np.linalg.norm(p1 - p0) for p0, p1 in edges]
        short_indices = np.argsort(lengths)[:2]
        long_indices = np.argsort(lengths)[2:]

        for index in short_indices:
            p0, p1 = edges[index]
            midpoint = ((p0 + p1) / 2).astype(int)
            if pose == "BALL":
                color = (255, 0, 0)
            elif head is not None and np.linalg.norm(midpoint - head) < 15:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.line(frame, tuple(p0), tuple(p1), color, 2)

        for index in long_indices:
            p0, p1 = edges[index]
            color = (
                (0, 255, 255)
                if speed is not None and speed >= config.MIN_MOVEMENT_SPEED_PX_S
                else (255, 255, 0)
            )
            cv2.line(frame, tuple(p0), tuple(p1), color, 2)

        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

        well_label = detection.get("well_label") or "Outside"
        fly_state = detection.get("fly_state", "unknown")
        inactive_s = float(detection.get("inactive_duration_s", 0.0))

        cv2.putText(
            frame,
            f"{well_label} | ID {object_id}",
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{fly_state} {inactive_s:.1f}s",
            (x + 5, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            self._fly_state_color(fly_state),
            1,
            cv2.LINE_AA,
        )
        if speed is not None:
            cv2.putText(
                frame,
                f"{float(speed):.2f}px/s",
                (x + 5, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

    def _draw_mass_summary(
            self,
            frame: np.ndarray,
            mass_state: Any,
            occupancy_report: Any,
    ) -> None:
        lines = [
            (
                f"Awake: {mass_state.awake_count}  "
                f"Inactive: {mass_state.inactive_count}  "
                f"Sleep: {mass_state.sleep_count}"
            ),
            (
                f"Sleep: {mass_state.sleep_percent:.1f}%  "
                f"Threshold: {mass_state.threshold_text}"
            ),
            (
                f"Observed: {mass_state.observed_total}/"
                f"{mass_state.expected_total}  "
                f"Outside: {occupancy_report.outside_count}"
            ),
        ]

        y = 22
        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 20

    # ------------------------------------------------------------------
    # Cleanup and helpers
    # ------------------------------------------------------------------

    def _flush_outputs(self) -> None:
        if not self.tracking_csv_file.closed:
            self.tracking_csv_file.flush()
        if self.position_logger is not None:
            self.position_logger.flush()
        if self.mass_csv_file is not None and not self.mass_csv_file.closed:
            self.mass_csv_file.flush()

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self.cap is not None:
            self.cap.release()
        if not self.tracking_csv_file.closed:
            self.tracking_csv_file.flush()
            self.tracking_csv_file.close()
        if self.position_logger is not None:
            self.position_logger.close()
        if self.mass_csv_file is not None and not self.mass_csv_file.closed:
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

    @staticmethod
    def _fly_state_color(state: str) -> tuple[int, int, int]:
        if state == "awake":
            return 0, 255, 0
        if state == "inactive":
            return 0, 255, 255
        if state == "sleep":
            return 255, 0, 255
        return 180, 180, 180
