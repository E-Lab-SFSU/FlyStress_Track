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
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.process_frame(frame)
            self.frame_idx += 1

            if self.show:
                key = cv2.waitKey(config.PLAYBACK_DELAY) & 0xFF
                if key == 27 or key == ord('q'):
                    break

        self.cleanup()

    # --------------------------------------------------
    # Per-frame processing
    # --------------------------------------------------
    def process_frame(self, frame):
        time_s = self.frame_idx / config.FPS

        detections, fg_mask = detect_objects(frame, self.cap)
        
        cv2.imshow("Foreground Mask", fg_mask)

        detections = self.tracker.update(detections)

        for det in detections:
            oid = det["id"]
            cx, cy = det["centroid"]        # center position of well

            track = self.tracker.objects[oid]
            history = track["history"]
            recent_deltas = track["recent_deltas"]

            # ------------------------------
            # Binary motion detection
            # ------------------------------
            is_moving = binary_motion_detect(recent_deltas)

            # ------------------------------
            # Windowed velocity (gated)
            # ------------------------------
            if not is_moving:
                speed = 0.0
                vel_angle = self.last_velocity_angle.get(oid, None)
            else:
                speed, vel_angle = windowed_velocity(history)

                if vel_angle is not None:
                    self.last_velocity_angle[oid] = vel_angle
                else:
                    vel_angle = self.last_velocity_angle.get(oid, None)

            det["speed_avg_px_s"] = speed
            det["velocity_angle_deg"] = vel_angle

            # ------------------------------
            # Morphology / pose
            # ------------------------------
            pose, head, tail, axis_vec = head_tail_from_bbox(
                fg_mask, det["contour"]
            )
            det["pose"] = pose

            # ------------------------------
            # Head alignment
            # ------------------------------
            head_aligned = True
            if pose == "ELONGATED" and vel_angle is not None:
                motion_vec = np.array([
                    np.cos(np.deg2rad(vel_angle)),
                    np.sin(np.deg2rad(vel_angle))
                ])
                head_aligned = np.dot(axis_vec, motion_vec) >= 0

            # ------------------------------
            # Movement state
            # ------------------------------
            movement_state = self.state_tracker.update(
                oid,
                speed,
                pose,
                head_aligned
            )

            det["movement_state"] = movement_state

            # ------------------------------
            # CSV output
            # ------------------------------
            self.csv_writer.writerow([
                self.frame_idx,
                f"{time_s:.3f}",
                oid,
                int(cx),
                int(cy),
                pose,
                f"{speed:.3f}" if speed is not None else "",
                f"{vel_angle:.2f}" if vel_angle is not None else "",
                movement_state,
                det.get("area", ""),
                det.get("perimeter", ""),
                det.get("aspect_ratio", ""),
                det.get("solidity", ""),
            ])

            # ------------------------------
            # Visualization
            # ------------------------------
            if self.show:
                self.draw_object(frame, det, head, tail)

        if self.show:
            cv2.imshow("Fly Tracking", frame)

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
        # Centroid + ID
        # --------------------------------------------------
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
        cv2.putText(
            frame,
            f"ID {oid}",
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
    
        # --------------------------------------------------
        # Optional speed overlay (useful for tuning)
        # --------------------------------------------------
        if speed is not None:
            cv2.putText(
                frame,
                f"{speed:.2f}px/s",
                (cx + 5, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
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

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    def cleanup(self):
        self.cap.release()
        self.csv_file.close()
        cv2.destroyAllWindows()
