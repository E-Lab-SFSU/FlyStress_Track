# Copyright (c) 2025 Thomas Zimmerman — MIT License
"""
image_detector.py

Foreground detection for the one-image-per-second workflow.

Fixes vs. the original alignment prototype script:
- Phase correlation runs on blurred grayscale, not binary Canny edges
  (binary edge maps are sparse/high-frequency and destabilize the
  sub-pixel FFT shift estimate).
- A Hanning window is passed into phaseCorrelate to suppress the
  spurious energy created by the plate's straight image borders.
- Every frame is aligned back to a single reference frame (the first
  image), not chained frame-to-frame, so small per-frame alignment
  errors don't accumulate over a long sequence and drift the well
  grid out from under the wells.

Public API mirrors detector.detect_objects() so it can be dropped into
the same downstream pipeline stages (tracker, well assignment, etc.):
    detections, mask = detect_objects_in_sequence(...)
"""

import cv2
import numpy as np
import config


def to_gray_blur(img, ksize=(3, 3)):
    """Grayscale + light blur to reduce camera noise."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.GaussianBlur(gray, ksize, 0)


_hann_cache = {}


def _get_hann(shape):
    if shape not in _hann_cache:
        h, w = shape
        _hann_cache[shape] = cv2.createHanningWindow((w, h), cv2.CV_32F)
    return _hann_cache[shape]


def align_to_reference(ref_blur, cur_blur, min_response=None):
    """
    Estimate the sub-pixel shift that aligns cur_blur onto ref_blur
    using windowed phase correlation on blurred grayscale images.

    Returns (aligned_cur, shift_x, shift_y, response, reliable).
    """
    if min_response is None:
        min_response = config.MIN_ALIGN_RESPONSE

    ref_f = ref_blur.astype(np.float32)
    cur_f = cur_blur.astype(np.float32)
    win = _get_hann(ref_blur.shape)

    (shift_x, shift_y), response = cv2.phaseCorrelate(ref_f, cur_f, win)

    h, w = cur_blur.shape[:2]
    M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
    aligned_cur = cv2.warpAffine(
        cur_blur, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    reliable = response >= min_response
    return aligned_cur, shift_x, shift_y, response, reliable


def motion_mask(ref_blur, aligned_cur, diff_thresh=None, open_ksize=3):
    """Difference two aligned grayscale images and threshold to isolate motion."""
    if diff_thresh is None:
        diff_thresh = config.IMAGE_DIFF_THRESH

    diff = cv2.absdiff(ref_blur, aligned_cur)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((open_ksize, open_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return diff, mask


def _contours_to_detections(mask):
    """Apply the same size/shape filters detector.py uses, on this mask's contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    h_img, w_img = mask.shape

    for c in contours:
        area = cv2.contourArea(c)
        if area < config.MIN_A or area > config.MAX_A:
            continue

        rect = cv2.minAreaRect(c)
        (x, y), (rw, rh), _ = rect
        w = int(min(rw, rh))
        h = int(max(rw, rh))

        if x <= 2 or y <= 2 or x >= w_img - 2 or y >= h_img - 2:
            continue
        if w < config.MIN_WH or w > config.MAX_WH:
            continue
        if h < config.MIN_WH or h > config.MAX_WH:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        detections.append({"contour": c, "centroid": (cx, cy)})

    return detections


def detect_objects_in_sequence(frame, ref_blur):
    """
    Detect fly contours in one frame of an image sequence.

    Parameters
    ----------
    frame : np.ndarray (BGR)
        Current raw image.
    ref_blur : np.ndarray
        Blurred grayscale reference frame (typically the sequence's
        first image), already prepared with to_gray_blur().

    Returns
    -------
    detections : list of dict ({"contour", "centroid"})
    foreground_mask : np.ndarray (uint8)
    shift_x, shift_y : float
        Estimated plate shift for this frame relative to ref_blur.
        Feed into WellPlate.shifted() / detect_plate_wells_adjusted()
        to keep well ROIs aligned with the plate.
    response : float
        phaseCorrelate confidence for this frame.
    reliable : bool
        False if response is below config.MIN_ALIGN_RESPONSE; the
        shift and detections for this frame should be treated with
        caution (e.g. held over from the previous frame) upstream.
    """
    cur_blur = to_gray_blur(frame)

    aligned_cur, shift_x, shift_y, response, reliable = align_to_reference(
        ref_blur, cur_blur
    )

    _, mask = motion_mask(ref_blur, aligned_cur)
    detections = _contours_to_detections(mask)

    return detections, mask, shift_x, shift_y, response, reliable