# Copyright (c) 2025 Thomas Zimmerman — MIT License
"""
pipeline.py

Main processing pipeline for plankton detection, tracking,
motion analysis, and behavior classification.

Key design:
- Detection → Tracking → Binary motion detect → Windowed velocity
- Velocity is gated to suppress jitter
- Movement state derived downstream
"""

import os
import csv
import cv2
import numpy as np
from datetime import datetime

import config
from detector import detect_objects
from tracker import CentroidTracker
from kinematics import head_tail_from_bbox
from features_motion import windowed_velocity, binary_motion_detect
from movement_state import MovementStateTracker

from Flies.fly_position import FlyPositionLogger
from Flies.fly_activity import FlyActivityTracker
from Flies.fly_sleep import FlySleepTracker
from Flies.mass_state import MassStateTracker
from Flies.occupancy import OccupancyValidator

from WellPlate.well_plate import create_plate_from_config
from WellPlate.well_overlay import add_well_overlay


class FlyPipeline:
    def __init__(self, video_path, show=True):
        self.video_path = video_path
        self.show = show

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        video_fps = self.cap.get(cv2.CAP_PROP_FPS)

        if video_fps > 0:
            self.fps = float(video_fps)
        else:
            self.fps = float(config.FPS)

        self.tracker = CentroidTracker()
        self.state_tracker = MovementStateTracker()

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
        self.last_velocity_angle = {}

        self._init_output_files(video_path)

    # --------------------------------------------------
    # CSV initialization
    # --------------------------------------------------
    def _init_output_files(self, video_path):
        """
        Create all CSV outputs for this run using one shared timestamp.
        """

        video_dir = os.path.dirname(video_path)
        video_base = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        self.run_timestamp = datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S"
        )

        self.csv_dir = os.path.join(video_dir, "csv")
        os.makedirs(self.csv_dir, exist_ok=True)

        run_prefix = (
            f"{video_base}_{self.run_timestamp}"
        )

        # --------------------------------------------------------
        # Full per-fly analysis CSV
        # --------------------------------------------------------

        self.tracking_csv_path = os.path.join(
            self.csv_dir,
            f"{run_prefix}_tracking.csv",
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

        # --------------------------------------------------------
        # Simple position-only CSV
        # --------------------------------------------------------

        self.position_logger = None

        if config.SAVE_POSITION_CSV:
            position_path = os.path.join(
                self.csv_dir,
                f"{run_prefix}_positions.csv",
            )

            self.position_logger = FlyPositionLogger(
                csv_path=position_path,
                fps=self.fps,
            )

        # --------------------------------------------------------
        # One population summary per frame
        # --------------------------------------------------------

        self.mass_csv_file = None
        self.mass_csv_writer = None

        if config.SAVE_MASS_STATE_CSV:
            mass_path = os.path.join(
                self.csv_dir,
                f"{run_prefix}_mass_state.csv",
            )

            self.mass_csv_file = open(
                mass_path,
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

        # --------------------------------------------------------
        # Save the well geometry once for this run
        # --------------------------------------------------------

        if config.SAVE_WELL_GEOMETRY_CSV:
            well_path = os.path.join(
                self.csv_dir,
                f"{run_prefix}_wells.csv",
            )

            self._save_well_geometry_csv(well_path)

    # -----------------------
    # Well Dimensions
    # -----------------------
    def _save_well_geometry_csv(self, csv_path):
        """
        Save every calculated well center and radius once per run.
        """

        with open(
                csv_path,
                "w",
                newline="",
                encoding="utf-8",
        ) as csv_file:
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
                        "diameter_px": (
                            f"{well.diameter_px:.3f}"
                        ),
                        "radius_px": (
                            f"{well.radius_px:.3f}"
                        ),
                        "assignment_radius_px": (
                            f"{self.plate.assignment_radius_px:.3f}"
                        ),
                    }
                )

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    break

                self.process_frame(frame)
                self.frame_idx += 1

                if self.show:
                    key = cv2.waitKey(
                        config.PLAYBACK_DELAY
                    ) & 0xFF

                    if key == 27 or key == ord("q"):
                        break

        finally:
            self.cleanup()

    # --------------------------------------------------
    # Per-frame processing
    # --------------------------------------------------
    def process_frame(self, frame):
        """Process one video frame."""

        time_s = self.frame_idx / self.fps

        # --------------------------------------------------
        # 1. Fly detection
        # --------------------------------------------------

        detections, fg_mask = detect_objects(
            frame,
            self.cap,
        )

        if self.show:
            cv2.imshow(
                "Foreground Mask",
                fg_mask,
            )

        # --------------------------------------------------
        # 2. Persistent tracker IDs
        # --------------------------------------------------

        detections = self.tracker.update(detections)

        # --------------------------------------------------
        # 3. Assign each tracked fly to a well
        # --------------------------------------------------

        detections = self.plate.assign_detections(
            detections
        )

        # --------------------------------------------------
        # 4. Count flies in each well
        # --------------------------------------------------

        occupancy_report = self.occupancy_validator.analyze(
            detections=detections,
            frame_idx=self.frame_idx,
        )

        # --------------------------------------------------
        # 5. Awake versus inactive
        # --------------------------------------------------

        activity_results = (
            self.fly_activity_tracker.update_frame(
                frame_idx=self.frame_idx,
                time_s=time_s,
                detections=detections,
            )
        )

        # --------------------------------------------------
        # 6. Awake, inactive, or sleep
        # --------------------------------------------------

        sleep_results = (
            self.fly_sleep_tracker.update_frame(
                activity_results
            )
        )

        activity_by_id = {
            result.object_id: result
            for result in activity_results
        }

        sleep_by_id = {
            result.object_id: result
            for result in sleep_results
        }

        # Copy activity/sleep results into detection dictionaries.
        for det in detections:
            object_id = det["id"]

            activity = activity_by_id.get(object_id)
            sleep = sleep_by_id.get(object_id)

            if activity is None:
                det["displacement_px"] = None
                det["activity_state"] = "unknown"
            else:
                det["displacement_px"] = (
                    activity.displacement_px
                )
                det["activity_state"] = (
                    activity.activity_state
                )

            if sleep is None:
                det["fly_state"] = "unknown"
                det["inactive_duration_s"] = 0.0
            else:
                det["fly_state"] = sleep.state
                det["inactive_duration_s"] = (
                    sleep.inactive_duration_s
                )

        # --------------------------------------------------
        # 7. Whole-experiment sleep state
        # --------------------------------------------------

        mass_state = self.mass_state_tracker.update(
            sleep_results=sleep_results,
            frame_idx=self.frame_idx,
            time_s=time_s,
        )

        # --------------------------------------------------
        # 8. Create display-only frame with well overlay
        # --------------------------------------------------

        display_frame = None

        if self.show:
            display_frame = add_well_overlay(
                frame=frame,
                plate=self.plate,
                draw_assignment_boundary=(
                    config.SHOW_ASSIGNMENT_BOUNDARY
                ),
            )

        # --------------------------------------------------
        # 9. Existing pose and movement-state processing
        # --------------------------------------------------

        for det in detections:
            oid = det["id"]
            cx, cy = det["centroid"]

            track = self.tracker.objects.get(oid)

            if track is None:
                continue

            history = track["history"]
            recent_deltas = track["recent_deltas"]

            is_moving = binary_motion_detect(
                recent_deltas
            )

            if not is_moving:
                speed = 0.0
                vel_angle = self.last_velocity_angle.get(
                    oid
                )
            else:
                speed, vel_angle = windowed_velocity(
                    history
                )

                if vel_angle is not None:
                    self.last_velocity_angle[oid] = (
                        vel_angle
                    )
                else:
                    vel_angle = (
                        self.last_velocity_angle.get(oid)
                    )

            det["speed_avg_px_s"] = speed
            det["velocity_angle_deg"] = vel_angle

            pose, head, tail, axis_vec = (
                head_tail_from_bbox(
                    fg_mask,
                    det["contour"],
                )
            )

            det["pose"] = pose

            head_aligned = True

            if (
                    pose == "ELONGATED"
                    and vel_angle is not None
            ):
                motion_vec = np.array(
                    [
                        np.cos(
                            np.deg2rad(vel_angle)
                        ),
                        np.sin(
                            np.deg2rad(vel_angle)
                        ),
                    ]
                )

                head_aligned = (
                        np.dot(axis_vec, motion_vec)
                        >= 0
                )

            movement_state = (
                self.state_tracker.update(
                    oid,
                    speed,
                    pose,
                    head_aligned,
                )
            )

            det["movement_state"] = movement_state

            # Complete tracking/state CSV.
            self._write_tracking_row(
                detection=det,
                time_s=time_s,
            )

            if self.show:
                self.draw_object(
                    display_frame,
                    det,
                    head,
                    tail,
                )

        # --------------------------------------------------
        # 10. Simple position CSV
        # --------------------------------------------------

        if self.position_logger is not None:
            self.position_logger.log_frame(
                frame_idx=self.frame_idx,
                detections=detections,
                time_s=time_s,
            )

        # --------------------------------------------------
        # 11. Population state CSV
        # --------------------------------------------------

        self._write_mass_state_row(
            mass_state
        )

        # Periodically force buffered data onto disk.
        flush_interval = max(
            1,
            int(config.CSV_FLUSH_INTERVAL_FRAMES),
        )

        if self.frame_idx % flush_interval == 0:
            self._flush_outputs()

        # --------------------------------------------------
        # 12. Main tracking display
        # --------------------------------------------------

        if self.show:
            self._draw_mass_summary(
                frame=display_frame,
                mass_state=mass_state,
                occupancy_report=occupancy_report,
            )

            cv2.imshow(
                "Fly Tracking",
                display_frame,
            )

    # writing lines in csv files
    def _write_tracking_row(
            self,
            detection,
            time_s,
    ):
        """Write one complete tracked-fly CSV row."""

        object_id = detection["id"]
        cx, cy = detection["centroid"]

        well_label = detection.get(
            "well_label"
        )

        if well_label is None:
            fly_label = f"Outside-Fly{object_id}"
        else:
            fly_label = (
                f"{well_label}-Fly{object_id}"
            )

        self.tracking_csv_writer.writerow(
            {
                "frame_idx": self.frame_idx,
                "time_s": f"{time_s:.6f}",
                "object_id": object_id,
                "fly_label": fly_label,
                "well_label": well_label or "",
                "well_number": self._optional_value(
                    detection.get("well_number")
                ),
                "centroid_x": f"{float(cx):.3f}",
                "centroid_y": f"{float(cy):.3f}",
                "displacement_px": self._format_number(
                    detection.get(
                        "displacement_px"
                    ),
                    decimal_places=3,
                ),
                "activity_state": detection.get(
                    "activity_state",
                    "unknown",
                ),
                "inactive_duration_s": (
                    self._format_number(
                        detection.get(
                            "inactive_duration_s"
                        ),
                        decimal_places=3,
                    )
                ),
                "fly_state": detection.get(
                    "fly_state",
                    "unknown",
                ),
                "pose": detection.get(
                    "pose",
                    "",
                ),
                "speed_avg_px_s": self._format_number(
                    detection.get(
                        "speed_avg_px_s"
                    ),
                    decimal_places=3,
                ),
                "velocity_angle_deg": (
                    self._format_number(
                        detection.get(
                            "velocity_angle_deg"
                        ),
                        decimal_places=2,
                    )
                ),
                "movement_state": detection.get(
                    "movement_state",
                    "",
                ),
            }
        )

    def _write_mass_state_row(
            self,
            mass_state,
    ):
        """Write one population summary row per frame."""

        if self.mass_csv_writer is None:
            return

        self.mass_csv_writer.writerow(
            {
                "frame_idx": (
                    mass_state.frame_idx
                ),
                "time_s": self._format_number(
                    mass_state.time_s,
                    decimal_places=6,
                ),
                "expected_total": (
                    mass_state.expected_total
                ),
                "observed_total": (
                    mass_state.observed_total
                ),
                "awake_count": (
                    mass_state.awake_count
                ),
                "inactive_count": (
                    mass_state.inactive_count
                ),
                "sleep_count": (
                    mass_state.sleep_count
                ),
                "unknown_count": (
                    mass_state.unknown_count
                ),
                "sleep_percent": (
                    f"{mass_state.sleep_percent:.3f}"
                ),
                "required_sleep_percent": (
                    f"{mass_state.required_sleep_percent:.3f}"
                ),
                "threshold_met": (
                    mass_state.threshold_text
                ),
                "threshold_message": (
                    mass_state.message
                ),
            }
        )

    def _draw_mass_summary(
            self,
            frame,
            mass_state,
            occupancy_report,
    ):
        """Draw experiment-wide status in the tracking window."""

        lines = [
            (
                f"Awake: {mass_state.awake_count}   "
                f"Inactive: {mass_state.inactive_count}   "
                f"Sleep: {mass_state.sleep_count}"
            ),
            (
                f"Sleep percent: "
                f"{mass_state.sleep_percent:.1f}%   "
                f"Threshold: {mass_state.threshold_text}"
            ),
            (
                f"Observed: "
                f"{mass_state.observed_total}/"
                f"{mass_state.expected_total}   "
                f"Outside wells: "
                f"{occupancy_report.outside_count}"
            ),
        ]

        y_position = 22

        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            y_position += 20
    def _flush_outputs(self):
        # Force buffered CSV data onto disk to prevent crashing

        if (
                self.tracking_csv_file is not None
                and not self.tracking_csv_file.closed
        ):
            self.tracking_csv_file.flush()

        if self.position_logger is not None:
            self.position_logger.flush()

        if (
                self.mass_csv_file is not None
                and not self.mass_csv_file.closed
        ):
            self.mass_csv_file.flush()
    # --------------------------------------------------
    # Drawing
    # --------------------------------------------------
    def draw_object(self, frame, det, head, tail):
        cx, cy = det["centroid"]
        oid = det["id"]
        speed = det.get("speed_avg_px_s", 0.0)
        pose = det.get("pose", "BALL")
    
        # --------------------------------------------------
        # Rotated bounding box from contour
        # --------------------------------------------------
        rect = cv2.minAreaRect(det["contour"])
        box = cv2.boxPoints(rect)
        box = np.int32(box)
    
        # Box edges
        edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
        lengths = [np.linalg.norm(p1 - p0) for p0, p1 in edges]
    
        short_idxs = np.argsort(lengths)[:2]
        long_idxs = np.argsort(lengths)[2:]
    
        # --------------------------------------------------
        # Draw short edges (polarity / ball)
        # --------------------------------------------------
        for idx in short_idxs:
            p0, p1 = edges[idx]
            mid = ((p0 + p1) / 2).astype(int)
    
            if pose == "BALL":
                color = (255, 0, 0)   # BLUE
            else:
                if head is not None and np.linalg.norm(mid - head) < 15:
                    color = (0, 255, 0)   # GREEN = head
                else:
                    color = (0, 0, 255)   # RED = tail
    
            cv2.line(frame, tuple(p0), tuple(p1), color, 2)
    
        # --------------------------------------------------
        # Draw long edges (movement)
        # --------------------------------------------------
        for idx in long_idxs:
            p0, p1 = edges[idx]
    
            if speed is not None and speed >= config.MIN_MOVEMENT_SPEED_PX_S:
                color = (0, 255, 255)   # YELLOW = moving
            else:
                color = (255, 255, 0)   # CYAN = attached
    
            cv2.line(frame, tuple(p0), tuple(p1), color, 2)

        # --------------------------------------------------
        # Centroid, well, tracker ID, and fly state
        # --------------------------------------------------

        cv2.circle(
            frame,
            (int(cx), int(cy)),
            3,
            (255, 255, 255),
            -1,
        )

        well_label = det.get("well_label")
        fly_state = det.get(
            "fly_state",
            "unknown",
        )

        inactive_duration_s = float(
            det.get(
                "inactive_duration_s",
                0.0,
            )
        )

        if well_label is None:
            identity_text = (
                f"Outside | ID {oid}"
            )
        else:
            identity_text = (
                f"{well_label} | ID {oid}"
            )

        state_text = (
            f"{fly_state} "
            f"{inactive_duration_s:.1f}s"
        )

        cv2.putText(
            frame,
            identity_text,
            (int(cx) + 5, int(cy) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            state_text,
            (int(cx) + 5, int(cy) + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            self._fly_state_color(
                fly_state
            ),
            1,
            cv2.LINE_AA,
        )
        # --------------------------------------------------
        # Optional speed overlay (useful for tuning)
        # --------------------------------------------------
        if speed is not None:
            cv2.putText(
                frame,
                f"{speed:.2f}px/s",
                (cx + 5, cy + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (200, 200, 200),
                1
            )

    # def draw_object(self, frame, det, head, tail):
    #     cx, cy = det["centroid"]
    #     oid = det["id"]

    #     cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
    #     cv2.putText(
    #         frame,
    #         f"ID {oid}",
    #         (cx + 5, cy - 5),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.4,
    #         (255, 255, 255),
    #         1
    #     )

    #     if head is not None and tail is not None:
    #         cv2.circle(frame, head, 4, (0, 255, 0), -1)
    #         cv2.circle(frame, tail, 4, (0, 0, 255), -1)
    # ---------------------
    # CSV writing + visualization helpers
    # ----------------------
    @staticmethod
    def _format_number(
            value,
            decimal_places,
    ):
        """Format an optional numeric value for CSV."""

        if value is None:
            return ""

        return (
            f"{float(value):.}"
            f"{decimal_places}f"
        )


    @staticmethod
    def _optional_value(value):
        """Convert None into an empty CSV field."""

        if value is None:
            return ""

        return value

    @staticmethod
    def _fly_state_color(state):
        """Return an OpenCV BGR color for a fly state."""

        if state == "awake":
            return 0, 255, 0

        if state == "inactive":
            return 0, 255, 255

        if state == "sleep":
            return 255, 0, 255

        return 180, 180, 180
    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    def cleanup(self):
        """Release video and close every output safely."""

        if self.cap is not None:
            self.cap.release()

        if (
                hasattr(self, "tracking_csv_file")
                and self.tracking_csv_file is not None
                and not self.tracking_csv_file.closed
        ):
            self.tracking_csv_file.flush()
            self.tracking_csv_file.close()

        if (
                hasattr(self, "position_logger")
                and self.position_logger is not None
        ):
            self.position_logger.close()

        if (
                hasattr(self, "mass_csv_file")
                and self.mass_csv_file is not None
                and not self.mass_csv_file.closed
        ):
            self.mass_csv_file.flush()
            self.mass_csv_file.close()

        cv2.destroyAllWindows()

        print("\nAnalysis complete.")
        print(f"CSV directory: {self.csv_dir}")
        print(
            f"Tracking CSV: "
            f"{self.tracking_csv_path}"
        )
