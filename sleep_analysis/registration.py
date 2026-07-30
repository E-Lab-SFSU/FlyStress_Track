"""Align consecutive plate images and create motion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    """Outputs from registering one current image to one reference image."""

    aligned_bgr: np.ndarray
    reference_gray: np.ndarray
    aligned_gray: np.ndarray
    difference: np.ndarray
    thresholded_difference: np.ndarray
    overlay: np.ndarray
    warp_matrix: np.ndarray
    correlation: float
    succeeded: bool


def preprocess_for_registration(image_bgr: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
    """Convert a BGR image to blurred grayscale for registration."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot preprocess an empty image.")
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be a positive odd integer.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)


def _motion_model_code(name: str) -> int:
    models = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    try:
        return models[name.lower()]
    except KeyError as exc:
        valid = ", ".join(models)
        raise ValueError(f"Unknown motion model {name!r}. Choose: {valid}.") from exc


def _identity_warp(motion_model: int) -> np.ndarray:
    if motion_model == cv2.MOTION_HOMOGRAPHY:
        return np.eye(3, 3, dtype=np.float32)
    return np.eye(2, 3, dtype=np.float32)


def _make_overlay(reference_gray: np.ndarray, aligned_gray: np.ndarray) -> np.ndarray:
    """Create a green/magenta overlay; aligned pixels appear gray."""
    return cv2.merge((aligned_gray, reference_gray, aligned_gray))


def register_pair(
        reference_bgr: np.ndarray,
        current_bgr: np.ndarray,
        *,
        motion_model: str = "euclidean",
        blur_kernel: int = 5,
        max_iterations: int = 100,
        epsilon: float = 1e-6,
        difference_threshold: int = 18,
) -> RegistrationResult:
    """Register ``current_bgr`` to ``reference_bgr`` using ECC.

    If ECC cannot converge, the unregistered current image is returned and
    ``succeeded`` is False. This keeps a long experiment from terminating due
    to one difficult image pair.
    """
    if reference_bgr.shape != current_bgr.shape:
        raise ValueError(
            "Consecutive images must have identical dimensions. "
            f"Reference={reference_bgr.shape}, current={current_bgr.shape}."
        )

    reference_gray = preprocess_for_registration(reference_bgr, blur_kernel)
    current_gray = preprocess_for_registration(current_bgr, blur_kernel)

    model_code = _motion_model_code(motion_model)
    warp_matrix = _identity_warp(model_code)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iterations,
        epsilon,
    )

    correlation = float("nan")
    succeeded = True

    try:
        correlation, warp_matrix = cv2.findTransformECC(
            reference_gray,
            current_gray,
            warp_matrix,
            model_code,
            criteria,
            None,
            1,
        )

        height, width = reference_gray.shape
        aligned_bgr = cv2.warpAffine(
            current_bgr,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error:
        # Registration can fail if a frame is badly exposed or too different.
        # Preserve the frame and mark the row as failed for later inspection.
        aligned_bgr = current_bgr.copy()
        succeeded = False

    aligned_gray = preprocess_for_registration(aligned_bgr, blur_kernel)
    difference = cv2.absdiff(reference_gray, aligned_gray)
    _, thresholded = cv2.threshold(
        difference,
        difference_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    overlay = _make_overlay(reference_gray, aligned_gray)

    return RegistrationResult(
        aligned_bgr=aligned_bgr,
        reference_gray=reference_gray,
        aligned_gray=aligned_gray,
        difference=difference,
        thresholded_difference=thresholded,
        overlay=overlay,
        warp_matrix=warp_matrix,
        correlation=float(correlation),
        succeeded=succeeded,
    )
