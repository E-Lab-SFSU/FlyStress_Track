"""Detect dark fly candidates independently inside manually defined wells."""
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


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def detect_candidates_in_well(gray: np.ndarray, well: dict[str, object], *,
                              max_candidates: int, mask_margin_px: int,
                              dark_percentile: float, threshold_offset: int,
                              min_area_px: int, max_area_px: int,
                              morph_kernel: int, open_iterations: int,
                              close_iterations: int) -> tuple[list[FlyDetection], np.ndarray]:
    cx, cy, radius = int(float(well["x"])), int(float(well["y"])), int(float(well["radius"]))
    usable = max(3, radius - int(mask_margin_px))
    height, width = gray.shape
    left, right = max(0, cx-usable), min(width, cx+usable+1)
    top, bottom = max(0, cy-usable), min(height, cy+usable+1)
    roi = gray[top:bottom, left:right]
    full_mask = np.zeros_like(gray)
    if roi.size == 0:
        return [], full_mask
    circle = np.zeros_like(roi)
    cv2.circle(circle, (cx-left, cy-top), usable, 255, -1)
    pixels = roi[circle > 0]
    if pixels.size == 0:
        return [], full_mask
    # Otsu generally separates a dark fly from the brighter well interior.
    # The percentile setting remains as a conservative upper bound for uneven lighting.
    otsu_threshold, _ = cv2.threshold(
        pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    percentile_limit = float(np.percentile(pixels, dark_percentile))
    median_limit = float(np.median(pixels)) - 5.0
    threshold = int(np.clip(min(otsu_threshold + threshold_offset, percentile_limit, median_limit), 0, 255))
    binary = cv2.inRange(roi, 0, threshold)
    binary = cv2.bitwise_and(binary, circle)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(morph_kernel), _odd(morph_kernel)))
    if open_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=int(open_iterations))
    if close_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=int(close_iterations))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    candidates: list[tuple[int, float, float, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area_px <= area <= max_area_px:
            lx, ly = centroids[label]
            candidates.append((label, left+float(lx), top+float(ly), area))
    candidates.sort(key=lambda item: item[3], reverse=True)
    candidates = candidates[:max_candidates]
    detections: list[FlyDetection] = []
    for label, x, y, area in candidates:
        component = np.zeros_like(binary)
        component[labels == label] = 255
        full_mask[top:bottom, left:right] = cv2.bitwise_or(full_mask[top:bottom, left:right], component)
        detections.append(FlyDetection(str(well["well"]), x, y, x-cx, y-cy, area, threshold))
    return detections, full_mask


def detect_flies(image_bgr: np.ndarray, wells: list[dict[str, object]], **settings):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    all_detections: dict[str, list[FlyDetection]] = {}
    combined = np.zeros_like(gray)
    for well in wells:
        detections, mask = detect_candidates_in_well(gray, well, **settings)
        all_detections[str(well["well"])] = detections
        combined = cv2.bitwise_or(combined, mask)
    return all_detections, combined
