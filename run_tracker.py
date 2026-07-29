# -*- coding: utf-8 -*-
"""
run_tracker.py

Entry point for FlyStress_Track.

Processing order:
    original video
    -> crop around well plate
    -> save cropped video
    -> adjust well coordinates
    -> run FlyPipeline on cropped video
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import config
import detector
from crop_well_plate_video import crop_video
from pipeline import FlyPipeline


# -------------------------------------------------
# USER CONFIG
# -------------------------------------------------

RUN_SINGLE_FILE = True

# Single-video mode
VIDEO_PATH = r"C:\Users\chana\Videos\flies.mp4"

# Batch mode
VIDEO_DIR = r"C:\Users\chana\Videos\Screen Recordings"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

SHOW = True

# Cropping
ENABLE_CROPPING = True
CROP_MARGIN_PX = 30

# True:
#   reuse an existing cropped video and JSON file.
#
# False:
#   overwrite the old crop and ask for a new selection.
REUSE_EXISTING_CROP = True


# -------------------------------------------------
# CROP HELPERS
# -------------------------------------------------

def cropped_video_path(source_path: Path) -> Path:
    """Return the output location for a cropped video."""

    crop_directory = source_path.parent / "cropped"
    return crop_directory / f"{source_path.stem}_cropped.mp4"


def crop_metadata_path(cropped_path: Path) -> Path:
    """Return the JSON sidecar location for a cropped video."""

    return cropped_path.with_suffix(".crop.json")


def prepare_cropped_video(source_path: Path) -> tuple[Path, dict[str, Any]]:
    """
    Crop a source video or reuse its existing cropped version.

    Returns
    -------
    cropped_path
        Path to the video that should be analyzed.

    metadata
        Crop information containing crop_x, crop_y, width, and height.
    """

    output_path = cropped_video_path(source_path)
    metadata_path = crop_metadata_path(output_path)

    crop_already_exists = (
            output_path.is_file()
            and metadata_path.is_file()
    )

    if REUSE_EXISTING_CROP and crop_already_exists:
        print(f"Reusing cropped video: {output_path}")

    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print()
        print(f"Select the well plate for: {source_path.name}")
        print(
            f"A {CROP_MARGIN_PX}-pixel margin will be added automatically."
        )

        crop_video(
            input_path=source_path,
            output_path=output_path,
            margin=CROP_MARGIN_PX,
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    return output_path, metadata


# -------------------------------------------------
# CONFIG COORDINATE HELPERS
# -------------------------------------------------

def shifted_point(
        point: tuple[float, float],
        crop_x: int,
        crop_y: int,
) -> tuple[float, float]:
    """Convert an original-frame point to cropped-frame coordinates."""

    return (
        float(point[0]) - crop_x,
        float(point[1]) - crop_y,
    )


def save_coordinate_config() -> dict[str, Any]:
    """Save coordinate-dependent config values before shifting them."""

    saved = {
        "WELL_TL": config.WELL_TL,
        "WELL_TR": config.WELL_TR,
        "WELL_BL": config.WELL_BL,
        "WELL_BR": config.WELL_BR,
    }

    if hasattr(config, "MASK_XC"):
        saved["MASK_XC"] = config.MASK_XC

    if hasattr(config, "MASK_YC"):
        saved["MASK_YC"] = config.MASK_YC

    return saved


def restore_coordinate_config(saved: dict[str, Any]) -> None:
    """Restore original full-frame coordinate settings."""

    for name, value in saved.items():
        setattr(config, name, value)


def apply_crop_offset(metadata: dict[str, Any]) -> None:
    """
    Shift full-frame well coordinates into cropped-frame coordinates.
    """

    crop_x = int(metadata["crop_x"])
    crop_y = int(metadata["crop_y"])

    config.WELL_TL = shifted_point(
        config.WELL_TL,
        crop_x,
        crop_y,
    )
    config.WELL_TR = shifted_point(
        config.WELL_TR,
        crop_x,
        crop_y,
    )
    config.WELL_BL = shifted_point(
        config.WELL_BL,
        crop_x,
        crop_y,
    )
    config.WELL_BR = shifted_point(
        config.WELL_BR,
        crop_x,
        crop_y,
    )

    # Shift the optional circular mask too.
    if hasattr(config, "MASK_XC"):
        config.MASK_XC = float(config.MASK_XC) - crop_x

    if hasattr(config, "MASK_YC"):
        config.MASK_YC = float(config.MASK_YC) - crop_y

    print(f"Crop offset: x={crop_x}, y={crop_y}")
    print(f"Cropped WELL_TL: {config.WELL_TL}")
    print(f"Cropped WELL_TR: {config.WELL_TR}")
    print(f"Cropped WELL_BL: {config.WELL_BL}")
    print(f"Cropped WELL_BR: {config.WELL_BR}")


# -------------------------------------------------
# PIPELINE RUNNER
# -------------------------------------------------

def run_single_video(video_path: str | Path) -> None:
    """
    Crop one video and run the existing pipeline on the cropped result.
    """

    source_path = Path(video_path).expanduser().resolve()

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    print()
    print("=" * 70)
    print(f"Source video: {source_path}")
    print("=" * 70)

    saved_coordinates = save_coordinate_config()

    try:
        if ENABLE_CROPPING:
            analysis_path, metadata = prepare_cropped_video(source_path)
            apply_crop_offset(metadata)
        else:
            analysis_path = source_path

        # detector.py keeps the median background in a global variable.
        # A new video must build a new background.
        detector._BG_MEDIAN_V = None

        print(f"Analyzing: {analysis_path}")

        pipeline = FlyPipeline(
            str(analysis_path),
            show=SHOW,
        )
        pipeline.run()

    finally:
        # Essential for batch mode. Otherwise, crop offsets accumulate.
        restore_coordinate_config(saved_coordinates)

        # Prevent the final background from being reused elsewhere.
        detector._BG_MEDIAN_V = None


def run_video_directory(video_dir: str | Path) -> None:
    """Process every original video in a directory."""

    directory = Path(video_dir).expanduser().resolve()

    if not directory.is_dir():
        raise NotADirectoryError(directory)

    video_files = [
        path
        for path in sorted(directory.iterdir())
        if (
                path.is_file()
                and path.suffix.lower() in VIDEO_EXTS
                and not path.stem.endswith("_cropped")
        )
    ]

    if not video_files:
        print("No source video files found.")
        return

    print(f"Found {len(video_files)} source videos.")

    for index, video_path in enumerate(video_files, start=1):
        print()
        print(f"[{index}/{len(video_files)}] {video_path.name}")

        try:
            run_single_video(video_path)

        except KeyboardInterrupt:
            print("Batch processing interrupted by user.")
            break

        except Exception as error:
            print(f"Could not process {video_path.name}: {error}")


def main() -> None:
    if RUN_SINGLE_FILE:
        if not VIDEO_PATH:
            raise ValueError(
                "RUN_SINGLE_FILE is True, but VIDEO_PATH is empty."
            )

        run_single_video(VIDEO_PATH)

    else:
        if not VIDEO_DIR:
            raise ValueError(
                "RUN_SINGLE_FILE is False, but VIDEO_DIR is empty."
            )

        run_video_directory(VIDEO_DIR)


if __name__ == "__main__":
    main()