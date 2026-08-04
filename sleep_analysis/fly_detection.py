"""Detect up to three dark flies independently inside manually selected wells."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class FlyDetection:
    well: str
    x: float
    y: float
    local_x: float
    local_y: float
    area_px: int
    threshold_value: int
    estimated_fly_count: int = 1


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _normalize_roi(roi: np.ndarray, circle: np.ndarray, use_clahe: bool,
                   clahe_clip_limit: float, clahe_tile_size: int) -> np.ndarray:
    # Fill outside the circle with the median so CLAHE does not amplify the black mask edge.
    pixels = roi[circle > 0]
    median = int(np.median(pixels)) if pixels.size else 127
    work = roi.copy()
    work[circle == 0] = median
    if use_clahe:
        tile = max(2, int(clahe_tile_size))
        work = cv2.createCLAHE(float(clahe_clip_limit), (tile, tile)).apply(work)
    return work


def detect_candidates_in_well(gray: np.ndarray, well: dict[str, object], *,
                              max_components: int, mask_margin_px: int, dark_percentile: float,
                              threshold_offset: int, min_area_px: int, max_single_area_px: int,
                              max_overlap_area_px: int, morph_kernel: int, open_iterations: int,
                              close_iterations: int, use_clahe: bool, clahe_clip_limit: float,
                              clahe_tile_size: int, overlap_two_multiplier: float,
                              overlap_three_multiplier: float, expected_single_area: float | None = None):
    cx, cy, radius = int(float(well["x"])), int(float(well["y"])), int(float(well["radius"]))
    usable = max(3, radius - int(mask_margin_px))
    h, w = gray.shape
    left, right = max(0, cx-usable), min(w, cx+usable+1)
    top, bottom = max(0, cy-usable), min(h, cy+usable+1)
    roi = gray[top:bottom, left:right]
    full_mask = np.zeros_like(gray)
    if roi.size == 0:
        return [], full_mask
    circle = np.zeros_like(roi)
    cv2.circle(circle, (cx-left, cy-top), usable, 255, -1)
    normalized = _normalize_roi(roi, circle, use_clahe, clahe_clip_limit, clahe_tile_size)
    pixels = normalized[circle > 0]
    if pixels.size == 0:
        return [], full_mask
    # Local threshold per well. The low percentile resists dark well-to-well illumination differences.
    percentile = float(np.percentile(pixels, dark_percentile))
    median = float(np.median(pixels))
    threshold = int(np.clip(min(percentile + threshold_offset, median - 8.0), 0, 255))
    binary = cv2.inRange(normalized, 0, threshold)
    binary = cv2.bitwise_and(binary, circle)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(morph_kernel), _odd(morph_kernel)))
    if open_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=int(open_iterations))
    if close_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=int(close_iterations))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    baseline = float(expected_single_area or max_single_area_px * 0.45)
    baseline = max(float(min_area_px), baseline)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not (min_area_px <= area <= max_overlap_area_px):
            continue
        estimated = 1
        if area >= baseline * overlap_three_multiplier:
            estimated = 3
        elif area >= baseline * overlap_two_multiplier:
            estimated = 2
        lx, ly = centroids[label]
        components.append((label, left+float(lx), top+float(ly), area, estimated))
    components.sort(key=lambda item: item[3], reverse=True)
    components = components[:max_components]
    detections = []
    for label, x, y, area, estimated in components:
        component = np.zeros_like(binary)
        component[labels == label] = 255
        full_mask[top:bottom, left:right] = cv2.bitwise_or(full_mask[top:bottom, left:right], component)
        detections.append(FlyDetection(str(well["well"]), x, y, x-cx, y-cy, area, threshold, estimated))
    return detections, full_mask


def detect_flies(image_bgr: np.ndarray, wells: list[dict[str, object]], *, area_hints=None, **settings):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    all_detections = {}
    combined = np.zeros_like(gray)
    for well in wells:
        name = str(well["well"])
        detections, mask = detect_candidates_in_well(
            gray, well, expected_single_area=(area_hints or {}).get(name), **settings)
        all_detections[name] = detections
        combined = cv2.bitwise_or(combined, mask)
    return all_detections, combined
