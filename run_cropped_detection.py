"""Run the existing FlyStress_Track pipeline on a cropped video.

This script reads the .crop.json sidecar produced by crop_well_plate_video.py
and shifts the WellPlate coordinates (and optional circular mask center) into
the cropped coordinate system before FlyPipeline is constructed.

Run from the FlyStress_Track repository root:
    python run_cropped_detection.py path/to/video_cropped.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
import detector
from pipeline import FlyPipeline


def shifted(point, x_offset: int, y_offset: int):
    return float(point[0]) - x_offset, float(point[1]) - y_offset


def apply_crop_offset(metadata: dict) -> None:
    x_offset = int(metadata["crop_x"])
    y_offset = int(metadata["crop_y"])

    config.WELL_TL = shifted(config.WELL_TL, x_offset, y_offset)
    config.WELL_TR = shifted(config.WELL_TR, x_offset, y_offset)
    config.WELL_BL = shifted(config.WELL_BL, x_offset, y_offset)
    config.WELL_BR = shifted(config.WELL_BR, x_offset, y_offset)

    # Preserve mask behavior if it is enabled in config.py.
    if hasattr(config, "MASK_XC"):
        config.MASK_XC = float(config.MASK_XC) - x_offset
    if hasattr(config, "MASK_YC"):
        config.MASK_YC = float(config.MASK_YC) - y_offset

    print(f"Applied crop offset x={x_offset}, y={y_offset}")
    print(f"Shifted WELL_TL to {config.WELL_TL}")
    print(f"Shifted WELL_BR to {config.WELL_BR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Cropped video path")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Crop JSON path (default: VIDEO.crop.json)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without OpenCV display windows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata is not None
        else video_path.with_suffix(".crop.json")
    )

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing crop metadata: {metadata_path}. "
            "Use crop_well_plate_video.py to create the cropped video."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    apply_crop_offset(metadata)

    # detector.py caches one median background globally. Reset it for this run.
    detector._BG_MEDIAN_V = None

    FlyPipeline(str(video_path), show=not args.no_display).run()


if __name__ == "__main__":
    main()