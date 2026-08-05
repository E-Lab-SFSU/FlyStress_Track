#!/usr/bin/env python3
"""
Standalone temporal-median background and frame-difference program.

Main reusable functions
-----------------------
1. create_median_frame(image_directory, output_path=None)
       Reads all images in a directory and returns the per-pixel temporal
       median image.

2. difference_from_median(median_frame, new_frame, threshold=8,
                          minimum_object_area=3)
       Compares one frame with the median background and returns a binary
       image:
           white = object darker than the median background
           black = no significant dark change

The functions accept either NumPy arrays or image-file paths where noted.

Dependencies
------------
    pip install opencv-python numpy

Standalone example
------------------
Edit IMAGE_DIRECTORY and NEW_FRAME near the bottom, then run:

    python median_frame_difference.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# ---------------------------------------------------------------------
# User-adjustable defaults
# ---------------------------------------------------------------------

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)

DEFAULT_THRESHOLD = 8
DEFAULT_MINIMUM_OBJECT_AREA = 3


# ---------------------------------------------------------------------
# File-loading helpers
# ---------------------------------------------------------------------

def list_image_files(
        image_directory: str | Path,
        extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> list[Path]:
    """
    Return image files in filename order.

    Parameters
    ----------
    image_directory:
        Directory containing the image sequence.
    extensions:
        Allowed filename extensions.
    """
    directory = Path(image_directory)

    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    allowed = {extension.lower() for extension in extensions}

    image_files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    )

    if not image_files:
        raise FileNotFoundError(
            f"No supported image files were found in: {directory}"
        )

    return image_files


def load_grayscale_image(image: str | Path | np.ndarray) -> np.ndarray:
    """
    Load or validate a grayscale image.

    Accepts:
        - a file path
        - a two-dimensional grayscale NumPy array
        - a three-channel BGR NumPy array
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)

        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        raise ValueError(
            f"Unsupported NumPy image shape: {image.shape}"
        )

    path = Path(image)
    loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if loaded is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return loaded


# ---------------------------------------------------------------------
# Function 1: build the temporal median frame
# ---------------------------------------------------------------------

def create_median_frame(
        image_directory: str | Path,
        output_path: str | Path | None = None,
) -> np.ndarray:
    """
    Create a per-pixel temporal median frame from an image directory.

    For each pixel location, the function collects that pixel's brightness
    across every frame and selects the median brightness.

    Parameters
    ----------
    image_directory:
        Directory containing the image sequence.
    output_path:
        Optional path for saving the median image.

    Returns
    -------
    median_frame:
        Two-dimensional uint8 grayscale NumPy array.
    """
    image_files = list_image_files(image_directory)

    frames: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None

    for image_path in image_files:
        frame = load_grayscale_image(image_path)

        if expected_shape is None:
            expected_shape = frame.shape
        elif frame.shape != expected_shape:
            raise ValueError(
                f"Image dimensions do not match. "
                f"{image_path.name} has shape {frame.shape}; "
                f"expected {expected_shape}."
            )

        frames.append(frame)

    # Stack shape:
    #     number_of_frames x image_height x image_width
    frame_stack = np.stack(frames, axis=0)

    # Median brightness at every pixel location.
    median_frame = np.median(frame_stack, axis=0).astype(np.uint8)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), median_frame):
            raise IOError(f"Could not save median frame: {output_path}")

    return median_frame


# ---------------------------------------------------------------------
# Function 2: compare one frame with the median
# ---------------------------------------------------------------------

def difference_from_median(
        median_frame: str | Path | np.ndarray,
        new_frame: str | Path | np.ndarray,
        threshold: int = DEFAULT_THRESHOLD,
        minimum_object_area: int = DEFAULT_MINIMUM_OBJECT_AREA,
) -> np.ndarray:
    """
    Compare a new frame against a temporal median background.

    This function detects objects that are darker than the median background.

    Calculation:
        dark_difference = median_background - new_frame

    Output:
        white pixels = new frame is darker than the median by more than threshold
        black pixels = no significant dark change

    Parameters
    ----------
    median_frame:
        Median-background image path or grayscale NumPy array.
    new_frame:
        New image path or grayscale NumPy array.
    threshold:
        Required grayscale decrease. Example: 8 means a pixel must be at
        least 9 grayscale levels darker than the median to become white.
    minimum_object_area:
        Connected white regions smaller than this many pixels are removed.
        Set to 1 to disable practical area filtering.

    Returns
    -------
    binary_difference:
        Two-dimensional uint8 image containing only 0 and 255.
    """
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")

    if minimum_object_area < 1:
        raise ValueError("minimum_object_area must be at least 1")

    background = load_grayscale_image(median_frame)
    current = load_grayscale_image(new_frame)

    if background.shape != current.shape:
        raise ValueError(
            f"Median and new frame dimensions differ: "
            f"{background.shape} versus {current.shape}"
        )

    # Signed arithmetic is required so negative differences are retained
    # correctly during subtraction.
    dark_difference = (
            background.astype(np.int16)
            - current.astype(np.int16)
    )

    # Newly darker pixels become white.
    binary = np.zeros(background.shape, dtype=np.uint8)
    binary[dark_difference > threshold] = 255

    # Remove isolated noise and tiny regions.
    if minimum_object_area > 1:
        component_count, labels, statistics, _ = (
            cv2.connectedComponentsWithStats(binary, connectivity=8)
        )

        cleaned = np.zeros_like(binary)

        for component_index in range(1, component_count):
            area = int(
                statistics[
                    component_index,
                    cv2.CC_STAT_AREA,
                ]
            )

            if area >= minimum_object_area:
                cleaned[labels == component_index] = 255

        binary = cleaned

    return binary


def save_difference_image(
        median_frame: str | Path | np.ndarray,
        new_frame: str | Path | np.ndarray,
        output_path: str | Path,
        threshold: int = DEFAULT_THRESHOLD,
        minimum_object_area: int = DEFAULT_MINIMUM_OBJECT_AREA,
) -> np.ndarray:
    """
    Convenience wrapper that calculates and saves the binary difference.
    """
    binary_difference = difference_from_median(
        median_frame=median_frame,
        new_frame=new_frame,
        threshold=threshold,
        minimum_object_area=minimum_object_area,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), binary_difference):
        raise IOError(f"Could not save difference image: {output_path}")

    return binary_difference


# ---------------------------------------------------------------------
# Standalone example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Edit these paths for your computer.
    IMAGE_DIRECTORY = Path(
        r"C:\Users\teaze\OneDrive\Documents\Project\FlyShaker"
        r"\FlyStress_Track-main\frames"
    )

    MEDIAN_OUTPUT = IMAGE_DIRECTORY / "median_frame.png"

    # Example: compare the final numbered frame with the median.
    # Replace this with any frame you want to test.
    NEW_FRAME = IMAGE_DIRECTORY / "frame_000278.png"

    DIFFERENCE_OUTPUT = IMAGE_DIRECTORY / "frame_000278_difference.png"

    THRESHOLD = 8
    MINIMUM_OBJECT_AREA = 3

    print(f"Reading images from: {IMAGE_DIRECTORY}")

    median = create_median_frame(
        image_directory=IMAGE_DIRECTORY,
        output_path=MEDIAN_OUTPUT,
    )

    difference = save_difference_image(
        median_frame=median,
        new_frame=NEW_FRAME,
        output_path=DIFFERENCE_OUTPUT,
        threshold=THRESHOLD,
        minimum_object_area=MINIMUM_OBJECT_AREA,
    )

    white_pixels = int(cv2.countNonZero(difference))

    print(f"Median frame saved to: {MEDIAN_OUTPUT}")
    print(f"Difference image saved to: {DIFFERENCE_OUTPUT}")
    print(f"Threshold: {THRESHOLD}")
    print(f"White difference pixels: {white_pixels}")
