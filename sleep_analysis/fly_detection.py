"""Conservative darker-than-median detector for manually selected wells."""
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
    width_px: int = 0
    height_px: int = 0
    fill_ratio: float = 0.0


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def detect_candidates_in_well(
        current_gray: np.ndarray,
        background_gray: np.ndarray,
        well: dict[str, object],
        *,
        expected_single_area: float | None,
        single_area_sample_count: int,
        max_components: int,
        inner_mask_scale: float,
        edge_exclusion_px: int,
        difference_blur_kernel: int,
        fixed_threshold: int,
        use_otsu_floor: bool,
        otsu_min_threshold: int,
        otsu_max_threshold: int,
        min_area_px: int,
        max_single_area_px: int,
        max_component_area_px: int,
        min_fill_ratio: float,
        max_aspect_ratio: float,
        morph_kernel: int,
        open_iterations: int,
        close_iterations: int,
        enable_overlap: bool,
        min_samples_for_overlap: int,
        overlap_two_multiplier: float,
        overlap_three_multiplier: float,
        overlap_max_total_flies_per_well: int = 3,
):
    cx = int(round(float(well["x"])))
    cy = int(round(float(well["y"])))
    radius = int(round(float(well["radius"])))
    usable = max(3, int(round(radius * float(inner_mask_scale))))
    height_full, width_full = current_gray.shape
    left, right = max(0, cx - usable), min(width_full, cx + usable + 1)
    top, bottom = max(0, cy - usable), min(height_full, cy + usable + 1)
    current = current_gray[top:bottom, left:right]
    background = background_gray[top:bottom, left:right]
    full_mask = np.zeros_like(current_gray)
    if current.size == 0 or current.shape != background.shape:
        return [], full_mask, int(fixed_threshold)

    local_center = (cx - left, cy - top)
    circle = np.zeros_like(current)
    inner_radius = max(1, usable - int(edge_exclusion_px))
    cv2.circle(circle, local_center, inner_radius, 255, -1)

    blur = _odd(difference_blur_kernel)
    current_blur = cv2.GaussianBlur(current, (blur, blur), 0)
    background_blur = cv2.GaussianBlur(background, (blur, blur), 0)

    # Crucial change: only pixels that became DARKER than the temporal median
    # are foreground. absdiff incorrectly accepted highlights and plate changes.
    signed_difference = (
            background_blur.astype(np.int16) - current_blur.astype(np.int16)
    )
    dark_difference = np.clip(signed_difference, 0, 255).astype(np.uint8)
    dark_difference[circle == 0] = 0

    threshold = int(fixed_threshold)
    if use_otsu_floor:
        pixels = dark_difference[circle > 0]
        if pixels.size:
            otsu, _ = cv2.threshold(pixels.reshape(-1, 1), 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold = max(threshold, int(np.clip(
                otsu, otsu_min_threshold, otsu_max_threshold)))

    _, binary = cv2.threshold(dark_difference, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, circle)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(morph_kernel), _odd(morph_kernel)))
    if open_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,
                                  iterations=int(open_iterations))
    if close_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                  iterations=int(close_iterations))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    overlap_ready = (
            enable_overlap
            and expected_single_area is not None
            and single_area_sample_count >= min_samples_for_overlap
    )
    baseline = max(float(min_area_px), float(expected_single_area or 0.0))

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area_px or area > max_component_area_px:
            continue
        fill = area / max(1, width * height)
        aspect = max(width, height) / max(1, min(width, height))
        if fill < min_fill_ratio or aspect > max_aspect_ratio:
            continue
        local_x, local_y = centroids[label]
        radial = float(np.hypot(
            local_x - local_center[0], local_y - local_center[1]))
        if radial > inner_radius:
            continue

        estimated = 1
        if overlap_ready:
            if area >= baseline * overlap_three_multiplier:
                estimated = 3
            elif area >= baseline * overlap_two_multiplier:
                estimated = 2
        elif area > max_single_area_px:
            # Do not manufacture overlap before a trustworthy per-well area
            # baseline exists. The tracker will output UNKNOWN instead.
            continue

        candidates.append((label, left + float(local_x), top + float(local_y),
                           area, estimated, width, height, fill))

    # Keep the most plausible components. Normal single-fly components are
    # preferred, followed by components closest to the learned area baseline.
    target_area = expected_single_area or min(max_single_area_px * 0.45, 70.0)
    candidates.sort(key=lambda item: (
        item[4] != 1,
        abs(item[3] / max(1, item[4]) - target_area),
        -item[6],
    ))
    candidates = candidates[:max_components]

    # Never claim more than the configured number of flies in a well.
    accepted = []
    running_count = 0
    for candidate in candidates:
        estimated = candidate[4]
        if running_count + estimated > overlap_max_total_flies_per_well:
            continue
        accepted.append(candidate)
        running_count += estimated

    detections = []
    for label, x, y, area, estimated, width, height, fill in accepted:
        component = np.zeros_like(binary)
        component[labels == label] = 255
        roi_mask = full_mask[top:bottom, left:right]
        full_mask[top:bottom, left:right] = cv2.bitwise_or(roi_mask, component)
        detections.append(FlyDetection(
            str(well["well"]), x, y, x - cx, y - cy, area, threshold,
            estimated, width, height, fill))
    return detections, full_mask, threshold


def detect_flies(image_bgr: np.ndarray, background_bgr: np.ndarray,
                 wells: list[dict[str, object]], *, area_hints=None,
                 area_sample_counts=None, **settings):
    current_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    if current_gray.shape != background_gray.shape:
        raise ValueError("Current image and background must have identical dimensions.")
    all_detections: dict[str, list[FlyDetection]] = {}
    combined = np.zeros_like(current_gray)
    thresholds: dict[str, int] = {}
    for well in wells:
        name = str(well["well"])
        detections, mask, threshold = detect_candidates_in_well(
            current_gray, background_gray, well,
            expected_single_area=(area_hints or {}).get(name),
            single_area_sample_count=(area_sample_counts or {}).get(name, 0),
            **settings)
        all_detections[name] = detections
        thresholds[name] = threshold
        combined = cv2.bitwise_or(combined, mask)
    return all_detections, combined, thresholds
