"""Detect multiple dark fly-shaped blobs independently inside each well."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlyDetection:
    """One fly candidate detected inside one well."""

    well: str
    detection_index: int
    x: float
    y: float
    local_x: float
    local_y: float
    area_px: int
    threshold_value: int


def _odd_kernel(size: int) -> int:
    if size < 1:
        return 1
    return size if size % 2 == 1 else size + 1


def detect_flies_in_well(
        gray: np.ndarray,
        well: dict[str, object],
        *,
        max_flies: int = 4,
        mask_margin_px: int = 8,
        dark_percentile: float = 16.0,
        threshold_offset: int = 4,
        min_area_px: int = 12,
        max_area_px: int = 450,
        morph_kernel: int = 3,
        open_iterations: int = 1,
        close_iterations: int = 1,
) -> tuple[list[FlyDetection], np.ndarray]:
    """Detect up to ``max_flies`` dark objects inside one circular well.

    The returned binary mask has the same size as ``gray`` and contains only
    accepted fly components for this well.
    """
    center_x = int(float(well["x"]))
    center_y = int(float(well["y"]))
    radius = int(float(well["radius"]))
    usable_radius = max(3, radius - mask_margin_px)

    height, width = gray.shape
    left = max(0, center_x - usable_radius)
    right = min(width, center_x + usable_radius + 1)
    top = max(0, center_y - usable_radius)
    bottom = min(height, center_y + usable_radius + 1)

    roi = gray[top:bottom, left:right]
    if roi.size == 0:
        return [], np.zeros_like(gray)

    local_center = (center_x - left, center_y - top)
    circular_mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.circle(circular_mask, local_center, usable_radius, 255, thickness=-1)

    pixels = roi[circular_mask > 0]
    if pixels.size == 0:
        return [], np.zeros_like(gray)

    percentile_value = float(np.percentile(pixels, dark_percentile))
    threshold_value = int(np.clip(percentile_value + threshold_offset, 0, 255))

    # Dark flies become white foreground pixels.
    binary = cv2.inRange(roi, 0, threshold_value)
    binary = cv2.bitwise_and(binary, circular_mask)

    kernel_size = _odd_kernel(morph_kernel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    if open_iterations > 0:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=open_iterations,
        )
    if close_iterations > 0:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations,
        )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    candidates: list[tuple[int, float, float, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area_px:
            continue

        local_x, local_y = centroids[label]
        global_x = float(left + local_x)
        global_y = float(top + local_y)
        candidates.append((label, global_x, global_y, area))

    # Larger components are more likely to be flies than isolated camera noise.
    candidates.sort(key=lambda item: item[3], reverse=True)
    candidates = candidates[:max_flies]
    # Give deterministic per-frame numbering from left to right, then top to bottom.
    candidates.sort(key=lambda item: (item[1], item[2]))

    accepted_global_mask = np.zeros_like(gray)
    detections: list[FlyDetection] = []
    well_name = str(well["well"])

    for detection_index, (label, global_x, global_y, area) in enumerate(
            candidates,
            start=1,
    ):
        accepted_component = np.zeros_like(binary)
        accepted_component[labels == label] = 255
        accepted_global_mask[top:bottom, left:right] = cv2.bitwise_or(
            accepted_global_mask[top:bottom, left:right],
            accepted_component,
        )
        detections.append(
            FlyDetection(
                well=well_name,
                detection_index=detection_index,
                x=global_x,
                y=global_y,
                local_x=global_x - center_x,
                local_y=global_y - center_y,
                area_px=area,
                threshold_value=threshold_value,
            )
        )

    return detections, accepted_global_mask


def detect_flies(
        image_bgr: np.ndarray,
        wells: list[dict[str, object]],
        **settings,
) -> tuple[list[FlyDetection], np.ndarray]:
    """Detect fly candidates in all wells of one plate image."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot detect flies in an empty image.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    all_detections: list[FlyDetection] = []
    combined_mask = np.zeros(gray.shape, dtype=np.uint8)

    for well in wells:
        detections, mask = detect_flies_in_well(gray, well, **settings)
        all_detections.extend(detections)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    return all_detections, combined_mask


def annotate_detections(
        image_bgr: np.ndarray,
        wells: list[dict[str, object]],
        detections: list[FlyDetection],
) -> np.ndarray:
    """Draw well boundaries and per-frame fly detection labels."""
    output = image_bgr.copy()

    for well in wells:
        center = (int(float(well["x"])), int(float(well["y"])))
        radius = int(float(well["radius"]))
        cv2.circle(output, center, radius, (160, 160, 160), 1)
        cv2.putText(
            output,
            str(well["well"]),
            (center[0] - radius, center[1] - radius - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for detection in detections:
        center = (int(round(detection.x)), int(round(detection.y)))
        label = f"{detection.well}-{detection.detection_index}"
        cv2.circle(output, center, 5, (0, 0, 255), 2)
        cv2.drawMarker(
            output,
            center,
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )
        cv2.putText(
            output,
            label,
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output
