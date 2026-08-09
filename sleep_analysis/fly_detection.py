"""Detect fly candidates independently inside manually defined wells.

The manual well circles remain the authoritative ROIs. Candidate ranking then
suppresses well rims and static material (food/debris) using geometry plus an
optional empty-plate background reference and previous-frame motion.
"""
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
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    candidate_score: float = 0.0
    background_change: float = 0.0
    motion_change: float = 0.0
    fill_ratio: float = 0.0
    aspect_ratio: float = 1.0
    center_gray: float = 0.0
    component_median_gray: float = 0.0
    edge_clearance_px: float = 0.0
    radial_fraction: float = 0.0
    arrival_change: float = 0.0
    arrival_fraction: float = 0.0
    departure_change: float = 0.0


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _mean_component_value(image: np.ndarray | None, component: np.ndarray) -> float:
    if image is None:
        return 0.0
    values = image[component > 0]
    return float(np.mean(values)) if values.size else 0.0


def detect_candidates_in_well(
        gray: np.ndarray,
        well: dict[str, object],
        *,
        max_candidates: int,
        mask_margin_px: int,
        dark_percentile: float,
        threshold_offset: int,
        min_area_px: int,
        max_area_px: int,
        morph_kernel: int,
        open_iterations: int,
        close_iterations: int,
        background_gray: np.ndarray | None = None,
        previous_gray: np.ndarray | None = None,
        background_difference_threshold: int = 8,
        motion_difference_threshold: int = 6,
        max_aspect_ratio: float = 4.0,
        min_fill_ratio: float = 0.15,
        edge_exclusion_px: int = 3,
        relax_shape_filter: bool = False,
        use_darkness_gate: bool = False,
        max_gray_value: int = 92,
        preferred_center: tuple[float, float] | None = None,
        tracking_search_radius_px: float | None = None,
        local_background_percentile: float = 70.0,
        min_contrast_from_background: float = 12.0,
        wall_mode_radial_fraction: float = 0.72,
        wall_edge_exclusion_px: int = 1,
        normalize_per_well_illumination: bool = True,
        per_well_arrival_threshold: float = 4.0,
        per_well_arrival_min_contrast: float = 6.0,
        per_well_motion_dilate_iterations: int = 1,
) -> tuple[list[FlyDetection], np.ndarray]:
    cx, cy, radius = int(float(well["x"])), int(float(well["y"])), int(float(well["radius"]))
    usable = max(3, radius - int(mask_margin_px))
    height, width = gray.shape
    left, right = max(0, cx - usable), min(width, cx + usable + 1)
    top, bottom = max(0, cy - usable), min(height, cy + usable + 1)
    roi = gray[top:bottom, left:right]
    full_mask = np.zeros_like(gray)
    if roi.size == 0:
        return [], full_mask

    circle = np.zeros_like(roi)
    cv2.circle(circle, (cx - left, cy - top), usable, 255, -1)
    pixels = roi[circle > 0]
    if pixels.size == 0:
        return [], full_mask

    # Dark-component segmentation is deliberately restricted to the manual ROI.
    otsu_threshold, _ = cv2.threshold(
        pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    percentile_limit = float(np.percentile(pixels, dark_percentile))
    median_limit = float(np.median(pixels)) - 3.0

    # Estimate illumination separately inside every well. This is intentionally
    # local: flies in brighter/dimmer wells should not share one global cutoff.
    # The contrast limit lets a somewhat lighter fly survive in a bright well.
    local_background = float(np.percentile(pixels, local_background_percentile))
    contrast_limit = local_background - float(min_contrast_from_background)
    threshold = int(np.clip(min(otsu_threshold + threshold_offset, percentile_limit, median_limit, contrast_limit), 0, 255))

    # This is only a safety ceiling now, not the main detector. A higher default
    # prevents lighter flies from being discarded while still excluding the
    # brightest floor/rim pixels.
    if use_darkness_gate:
        threshold = min(threshold, int(np.clip(max_gray_value, 0, 255)))

    binary = cv2.inRange(roi, 0, threshold)
    binary = cv2.bitwise_and(binary, circle)

    # If the previous accepted track is already near the wall, preserve tiny dot-like
    # components by skipping the opening pass and allowing candidates closer to the rim.
    preferred_wall_mode = False
    if preferred_center is not None:
        pref_radial = float(np.hypot(preferred_center[0] - cx, preferred_center[1] - cy)) / max(1.0, float(usable))
        preferred_wall_mode = pref_radial >= float(wall_mode_radial_fraction)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(morph_kernel), _odd(morph_kernel)))
    effective_open_iterations = 0 if preferred_wall_mode else int(open_iterations)
    if effective_open_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=effective_open_iterations)
    if close_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=int(close_iterations))

    # Difference images are only used to rank candidates. A stationary fly is
    # therefore still allowed; it is not thrown away just for not moving.
    background_roi = None
    if background_gray is not None and background_gray.shape == gray.shape:
        background_roi = cv2.absdiff(roi, background_gray[top:bottom, left:right])
    motion_roi = None
    signed_motion = None
    departure_change = 0.0
    if previous_gray is not None and previous_gray.shape == gray.shape:
        previous_roi = previous_gray[top:bottom, left:right]
        raw_signed_motion = previous_roi.astype(np.int16) - roi.astype(np.int16)

        # IMPORTANT: motion is normalized independently inside each well. A small
        # camera/lighting fluctuation can change thousands of pixels across the
        # plate, but it should not create a fly candidate in every well. We remove
        # the median signed change within this well before looking for arrivals.
        signed_motion = raw_signed_motion
        if normalize_per_well_illumination:
            valid_motion = raw_signed_motion[circle > 0]
            if valid_motion.size:
                signed_motion = raw_signed_motion - int(round(float(np.median(valid_motion))))

        motion_roi = np.abs(signed_motion).astype(np.uint8)

        # Add a small, strictly per-well arrival cue to the dark-object mask. This
        # helps when a wall-climbing fly becomes a tiny dot that ordinary morphology
        # would otherwise erase. The pixel still has to be darker than this well's
        # own local background, so bright reflections do not become candidates.
        arrival_mask = np.zeros_like(binary)
        arrival_pixels = (signed_motion >= float(per_well_arrival_threshold)) & \
                         (roi <= (local_background - float(per_well_arrival_min_contrast))) & \
                         (circle > 0)
        arrival_mask[arrival_pixels] = 255
        if int(per_well_motion_dilate_iterations) > 0:
            tiny_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            arrival_mask = cv2.dilate(arrival_mask, tiny_kernel, iterations=int(per_well_motion_dilate_iterations))
        binary = cv2.bitwise_or(binary, arrival_mask)
        binary = cv2.bitwise_and(binary, circle)

        if preferred_center is not None:
            pcx = int(round(preferred_center[0] - left))
            pcy = int(round(preferred_center[1] - top))
            r = 5
            x1, x2 = max(0, pcx-r), min(roi.shape[1], pcx+r+1)
            y1, y2 = max(0, pcy-r), min(roi.shape[0], pcy+r+1)
            patch = signed_motion[y1:y2, x1:x2]
            if patch.size:
                departure_change = float(np.mean(np.maximum(-patch, 0)))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    ranked: list[tuple[float, int, float, float, int, int, int, int, int, float, float, float, float, float, float]] = []

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not (min_area_px <= area <= max_area_px):
            continue

        lx, ly = centroids[label]
        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue

        # Reject components whose centroid is still too close to the usable ROI
        # edge. This is the strongest protection against the circular well wall.
        radial_distance = float(np.hypot(float(lx) - (cx - left), float(ly) - (cy - top)))
        edge_clearance = float(usable) - radial_distance
        effective_edge_exclusion = float(wall_edge_exclusion_px) if preferred_wall_mode else float(edge_exclusion_px)
        if edge_clearance < effective_edge_exclusion:
            continue

        aspect_ratio = max(float(bw) / float(bh), float(bh) / float(bw))
        fill_ratio = float(area) / float(bw * bh)
        if not relax_shape_filter and (aspect_ratio > float(max_aspect_ratio) or fill_ratio < float(min_fill_ratio)):
            continue

        component = np.zeros_like(binary)
        component[labels == label] = 255
        bg_change = _mean_component_value(background_roi, component)
        motion_change = _mean_component_value(motion_roi, component)
        arrival_change = 0.0
        arrival_fraction = 0.0
        if signed_motion is not None:
            vals = signed_motion[labels == label]
            if vals.size:
                positive = np.maximum(vals, 0)
                arrival_change = float(np.mean(positive))
                arrival_fraction = float(np.mean(vals >= float(motion_difference_threshold)))

        # Grayscale identity features.  The fly is usually substantially darker
        # than the well background, and its grayscale signature changes more
        # slowly than a fixed shadow/rim artifact.  Use both a small patch around
        # the centroid and the median of the segmented component.  The latter is
        # especially robust to a single bright wing/reflection pixel.
        center_x = int(round(float(lx)))
        center_y = int(round(float(ly)))
        patch_radius = 3
        px1 = max(0, center_x - patch_radius)
        py1 = max(0, center_y - patch_radius)
        px2 = min(roi.shape[1], center_x + patch_radius + 1)
        py2 = min(roi.shape[0], center_y + patch_radius + 1)
        center_patch = roi[py1:py2, px1:px2]
        center_gray = float(np.median(center_patch)) if center_patch.size else float(roi[center_y, center_x])
        component_values = roi[labels == label]
        component_median_gray = float(np.median(component_values)) if component_values.size else center_gray

        # Score rather than hard-threshold the differences. Static food tends to
        # match an empty-background image; moving flies tend to differ strongly.
        bg_score = min(bg_change / max(1.0, float(background_difference_threshold)), 3.0) if background_roi is not None else 0.0
        motion_score = min(motion_change / max(1.0, float(motion_difference_threshold)), 3.0) if motion_roi is not None else 0.0
        arrival_score = min(arrival_change / max(1.0, float(motion_difference_threshold)), 3.0) if signed_motion is not None else 0.0
        edge_score = min(max(edge_clearance, 0.0) / max(1.0, usable * 0.35), 1.5)
        shape_score = min(fill_ratio / 0.45, 1.5) + min(2.0 / max(aspect_ratio, 1.0), 1.0)

        # Dynamic tracking prior: the well itself remains fixed, but candidate
        # ranking follows the fly's last accepted centroid.  This is a soft
        # preference rather than a hard crop, so reacquisition remains possible.
        tracking_score = 0.0
        if preferred_center is not None and tracking_search_radius_px is not None:
            global_x = left + float(lx)
            global_y = top + float(ly)
            distance_to_track = float(np.hypot(global_x - preferred_center[0],
                                               global_y - preferred_center[1]))
            radius_for_score = max(1.0, float(tracking_search_radius_px))
            tracking_score = 3.0 * max(0.0, 1.0 - distance_to_track / radius_for_score)

        # Background agreement is most useful for excluding food/walls; motion
        # is secondary so a resting fly can keep being tracked.
        # Signed arrival evidence gets more weight than absolute motion. Absolute
        # difference alone can score both the new fly location and the spot it
        # just left; arrival evidence only favors pixels that became darker now.
        score = ((2.0 * bg_score) + (0.55 * motion_score) + (2.4 * arrival_score) +
                 (1.0 * arrival_fraction) + (0.7 * edge_score) +
                 (0.5 * shape_score) + tracking_score)

        ranked.append((
            score, label, left + float(lx), top + float(ly), area,
            left + bx, top + by, left + bx + bw - 1, top + by + bh - 1,
            bg_change, motion_change, fill_ratio, aspect_ratio,
            center_gray, component_median_gray, edge_clearance, radial_distance / max(1.0, float(usable)), arrival_change, arrival_fraction, departure_change,
        ))

    ranked.sort(key=lambda item: item[0], reverse=True)
    ranked = ranked[:max_candidates]

    detections: list[FlyDetection] = []
    for score, label, x, y, area, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bg_change, motion_change, fill_ratio, aspect_ratio, center_gray, component_median_gray, edge_clearance, radial_fraction, arrival_change, arrival_fraction, departure_change in ranked:
        component = np.zeros_like(binary)
        component[labels == label] = 255
        full_mask[top:bottom, left:right] = cv2.bitwise_or(full_mask[top:bottom, left:right], component)
        detections.append(FlyDetection(
            str(well["well"]), x, y, x - cx, y - cy, area, threshold,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            candidate_score=float(score), background_change=float(bg_change),
            motion_change=float(motion_change), fill_ratio=float(fill_ratio),
            aspect_ratio=float(aspect_ratio),
            center_gray=float(center_gray),
            component_median_gray=float(component_median_gray),
            edge_clearance_px=float(edge_clearance), radial_fraction=float(radial_fraction),
            arrival_change=float(arrival_change), arrival_fraction=float(arrival_fraction),
            departure_change=float(departure_change),
                                     ))
    return detections, full_mask


def detect_flies(
        image_bgr: np.ndarray,
        wells: list[dict[str, object]],
        *,
        background_bgr: np.ndarray | None = None,
        previous_bgr: np.ndarray | None = None,
        tracking_hints: dict[str, dict[str, float]] | None = None,
        **settings,
):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background_gray = None
    if background_bgr is not None and background_bgr.shape == image_bgr.shape:
        background_gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    previous_gray = None
    if previous_bgr is not None and previous_bgr.shape == image_bgr.shape:
        previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)

    all_detections: dict[str, list[FlyDetection]] = {}
    combined = np.zeros_like(gray)
    for well in wells:
        well_name = str(well["well"])
        hint = (tracking_hints or {}).get(well_name)
        preferred_center = None
        search_radius = None
        if hint:
            preferred_center = (float(hint["x"]), float(hint["y"]))
            search_radius = float(hint["search_radius_px"])
        detections, mask = detect_candidates_in_well(
            gray, well, background_gray=background_gray, previous_gray=previous_gray,
            preferred_center=preferred_center, tracking_search_radius_px=search_radius,
            **settings
        )
        all_detections[well_name] = detections
        combined = cv2.bitwise_or(combined, mask)
    return all_detections, combined
