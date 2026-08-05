"""Synthetic integration test for reference subtraction, three slots and overlap."""
from pathlib import Path
import tempfile
import cv2
import numpy as np
import config
from sleep_analysis.fly_detection import detect_flies
from sleep_analysis.multi_fly_tracker import PerWellMultiFlyTracker

def circle(image, x, y, radius=3, value=30):
    cv2.circle(image, (x, y), radius, (value, value, value), -1)

def main():
    background = np.full((160, 160, 3), 180, np.uint8)
    current = background.copy()
    circle(current, 65, 75)
    circle(current, 80, 80)
    circle(current, 95, 85)
    wells = [dict(well="A1", x=80, y=80, radius=45)]
    tracker = PerWellMultiFlyTracker(["A1"], flies_per_well=3,
                                     max_match_distance_px=30, jitter_threshold_px=1,
                                     rolling_window_seconds=300, sleep_duration_seconds=300,
                                     max_valid_sample_gap_seconds=2.5, low_confidence_frames_after_split=3)
    settings = dict(max_components=4, inner_mask_scale=.95, edge_exclusion_px=1,
                    difference_blur_kernel=3, fixed_threshold=16, use_otsu_floor=True,
                    otsu_min_threshold=10, otsu_max_threshold=40, min_area_px=5,
                    max_single_area_px=180, max_component_area_px=520,
                    min_fill_ratio=.10, max_aspect_ratio=5, morph_kernel=3,
                    open_iterations=1, close_iterations=1, enable_overlap=True,
                    min_samples_for_overlap=8, overlap_two_multiplier=1.65,
                    overlap_three_multiplier=2.65)
    detections, mask, _ = detect_flies(current, background, wells,
                                       area_hints=tracker.area_hints(), area_sample_counts=tracker.area_sample_counts(), **settings)
    assert len(detections["A1"]) == 3, detections["A1"]
    results = tracker.update_all(detections, 0.0, True)
    assert sum(r.observation_status == "DETECTED" for r in results) == 3
    results = tracker.update_all(detections, 1.0, True)
    assert all(r.observation_status == "" for r in results)
    assert mask.any()
    print("FlyStress Track v2 self-test passed.")

if __name__ == "__main__":
    main()
