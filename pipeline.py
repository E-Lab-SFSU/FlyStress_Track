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


class PlanktonPipeline:
    def __init__(self, video_path, show=True):
        self.video_path = video_path
        self.show = show

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.tracker = CentroidTracker()
        self.state_tracker = MovementStateTracker()

        self.frame_idx = 0
        self.last_velocity_angle = {}

        self._init_csv(video_path)

    # --------------------------------------------------
    # CSV initialization
    # --------------------------------------------------
    def _init_csv(self, video_path):
        video_dir = os.path.dirname(video_path)
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        csv_dir = os.path.join(video_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        csv_name = f"{video_base}_{timestamp}.csv"
        self.csv_path = os.path.join(csv_dir, csv_name)

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "frame_idx",
            "time_s",
            "object_id",
            "centroid_x",
            "centroid_y",
            "pose",
            "speed_avg_px_s",
            "velocity_angle_deg",
            "movement_state",
            "area_px",
            "perimeter_px",
            "aspect_ratio",
            "solidity",
        ])

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
            cx, cy = det["centroid"]

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
