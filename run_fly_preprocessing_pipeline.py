#!/usr/bin/env python3
"""
Master pipeline script for fly-well preprocessing.

This script IMPORTS the two earlier standalone modules:

    1. auto_plate_wells_from_images.py
    2. median_frame_difference.py

and runs them in sequence:

    Step 1: detect wells automatically and write plate_wells.csv
    Step 2: build a temporal median frame from the image directory
    Step 3: generate one binary difference image for every frame

Outputs
-------
- plate_wells.csv
- median_frame.png
- one binary difference image per frame:
      white = darker than temporal median background
      black = no significant dark change

This script is intended to be easy for the programmer to inspect and integrate.

Required files in the same directory as this script
---------------------------------------------------
- auto_plate_wells_from_images.py
- median_frame_difference.py

Dependencies
------------
    pip install opencv-python numpy

Typical standalone use
----------------------
Edit IMAGE_DIRECTORY and OUTPUT_ROOT near the bottom, then run:

    python run_fly_preprocessing_pipeline.py
"""

from __future__ import annotations

from pathlib import Path
import csv

import cv2

from auto_plate_wells_from_images import auto_generate_plate_wells_csv
from median_frame_difference import (
    create_median_frame,
    difference_from_median,
    list_image_files,
)


# ---------------------------------------------------------------------
# Helper function: process all frames against the median background
# ---------------------------------------------------------------------

def generate_difference_sequence(
        image_directory: str | Path,
        median_frame_path: str | Path,
        output_directory: str | Path,
        threshold: int = 8,
        minimum_object_area: int = 3,
) -> list[Path]:
    """
    Generate one binary median-difference image for each input frame.

    Parameters
    ----------
    image_directory:
        Directory containing input frames.
    median_frame_path:
        Path to the saved median background image.
    output_directory:
        Directory where binary difference images are written.
    threshold:
        Pixel must be darker than the median by more than this many
        grayscale levels to become white.
    minimum_object_area:
        Tiny white blobs smaller than this many pixels are removed.

    Returns
    -------
    output_paths:
        List of written image paths.
    """
    image_files = list_image_files(image_directory)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    for index, image_path in enumerate(image_files, start=1):
        binary_difference = difference_from_median(
            median_frame=median_frame_path,
            new_frame=image_path,
            threshold=threshold,
            minimum_object_area=minimum_object_area,
        )

        output_name = f"diff_{index:06d}.png"
        output_path = output_directory / output_name

        if not cv2.imwrite(str(output_path), binary_difference):
            raise IOError(f"Could not save difference image: {output_path}")

        output_paths.append(output_path)

    return output_paths


def write_processing_summary(
        image_files: list[Path],
        difference_files: list[Path],
        summary_csv_path: str | Path,
) -> Path:
    """
    Write a simple CSV linking each source frame to its difference image.
    """
    if len(image_files) != len(difference_files):
        raise ValueError("image_files and difference_files must have equal length")

    summary_csv_path = Path(summary_csv_path)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "source_frame",
            "difference_image",
        ])

        for index, (source_path, diff_path) in enumerate(
                zip(image_files, difference_files),
                start=1,
        ):
            writer.writerow([
                index,
                source_path.name,
                diff_path.name,
            ])

    return summary_csv_path


# ---------------------------------------------------------------------
# Main reusable pipeline function
# ---------------------------------------------------------------------

def run_fly_preprocessing_pipeline(
        image_directory: str | Path,
        output_root: str | Path,
        rows: int = 4,
        cols: int = 8,
        threshold: int = 8,
        minimum_object_area: int = 3,
) -> dict[str, Path]:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    image_directory:
        Directory containing the input image frames.
    output_root:
        Root directory for all outputs.
    rows, cols:
        Plate layout for well detection.
    threshold:
        Threshold used in median-frame differencing.
    minimum_object_area:
        Area filter for removing tiny white blobs.

    Returns
    -------
    outputs:
        Dictionary of important output paths.
    """
    image_directory = Path(image_directory)
    output_root = Path(output_root)

    plate_dir = output_root / "plate"
    plate_debug_dir = output_root / "plate_debug"
    difference_dir = output_root / "difference_frames"

    plate_dir.mkdir(parents=True, exist_ok=True)
    difference_dir.mkdir(parents=True, exist_ok=True)

    plate_wells_csv = plate_dir / "plate_wells.csv"
    median_frame_path = output_root / "median_frame.png"
    summary_csv_path = output_root / "difference_frame_index.csv"

    # Step 1: automatic well detection
    auto_generate_plate_wells_csv(
        image_dir=image_directory,
        output_csv=plate_wells_csv,
        rows=rows,
        cols=cols,
        debug_dir=plate_debug_dir,
        reference_mode="sum",
        apply_clahe=True,
    )

    # Step 2: create temporal median background
    create_median_frame(
        image_directory=image_directory,
        output_path=median_frame_path,
    )

    # Step 3: generate all binary difference frames
    image_files = list_image_files(image_directory)

    difference_files = generate_difference_sequence(
        image_directory=image_directory,
        median_frame_path=median_frame_path,
        output_directory=difference_dir,
        threshold=threshold,
        minimum_object_area=minimum_object_area,
    )

    write_processing_summary(
        image_files=image_files,
        difference_files=difference_files,
        summary_csv_path=summary_csv_path,
    )

    return {
        "plate_wells_csv": plate_wells_csv,
        "median_frame": median_frame_path,
        "difference_directory": difference_dir,
        "difference_summary_csv": summary_csv_path,
        "plate_debug_directory": plate_debug_dir,
    }


# ---------------------------------------------------------------------
# Standalone example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    IMAGE_DIRECTORY = Path(
        r"C:\Users\teaze\OneDrive\Documents\Project\FlyShaker"
        r"\FlyStress_Track-main\frames"
    )

    OUTPUT_ROOT = Path(
        r"C:\Users\teaze\OneDrive\Documents\Project\FlyShaker"
        r"\FlyStress_Track-main\preprocessing_output"
    )

    ROWS = 4
    COLS = 8

    # IMPORTANT:
    # The difference images are BINARY, not grayscale.
    # White means: the frame is darker than the temporal median by more than
    # THRESHOLD grayscale levels.
    # Black means: no significant dark change.
    THRESHOLD = 8
    MINIMUM_OBJECT_AREA = 3

    outputs = run_fly_preprocessing_pipeline(
        image_directory=IMAGE_DIRECTORY,
        output_root=OUTPUT_ROOT,
        rows=ROWS,
        cols=COLS,
        threshold=THRESHOLD,
        minimum_object_area=MINIMUM_OBJECT_AREA,
    )

    print("Pipeline completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")
