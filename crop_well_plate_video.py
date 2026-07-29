"""Crop a video around the well plate and save only the cropped frames.

Run from the FlyStress_Track repository root:
    python crop_well_plate_video.py path/to/input.mp4

On the first frame, drag a rectangle tightly around the OUTER edge of the
well plate and press Enter/Space. The program adds a 30-pixel margin on all
sides, writes a cropped MP4, and saves crop metadata beside it as JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def even(value: int) -> int:
    """Return an even dimension for video codecs that require one."""
    return value if value % 2 == 0 else value - 1


def choose_crop(first_frame, margin: int) -> tuple[int, int, int, int]:
    """Interactively select the plate and expand the selection by margin."""
    x, y, width, height = cv2.selectROI(
        "Select OUTER well-plate edge, then press Enter",
        first_frame,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow("Select OUTER well-plate edge, then press Enter")

    if width <= 0 or height <= 0:
        raise RuntimeError("No valid well-plate region was selected.")

    frame_h, frame_w = first_frame.shape[:2]
    x1 = max(0, int(x) - margin)
    y1 = max(0, int(y) - margin)
    x2 = min(frame_w, int(x + width) + margin)
    y2 = min(frame_h, int(y + height) + margin)

    # Keep encoded dimensions even without moving the top-left origin.
    x2 = x1 + even(x2 - x1)
    y2 = y1 + even(y2 - y1)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("The expanded crop has invalid dimensions.")

    return x1, y1, x2, y2


def crop_video(input_path: Path, output_path: Path, margin: int) -> Path:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {input_path}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read the first frame.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_h, source_w = first_frame.shape[:2]
    x1, y1, x2, y2 = choose_crop(first_frame, margin)
    crop_w = x2 - x1
    crop_h = y2 - y1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (crop_w, crop_h),
    )
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Cannot create output video: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame[y1:y2, x1:x2])
            written += 1
            if written % 300 == 0:
                total = frame_count if frame_count > 0 else "?"
                print(f"Cropped {written}/{total} frames")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    metadata = {
        "source_video": str(input_path.resolve()),
        "cropped_video": str(output_path.resolve()),
        "margin_px": margin,
        "crop_x": x1,
        "crop_y": y1,
        "crop_width": crop_w,
        "crop_height": crop_h,
        "source_width": source_w,
        "source_height": source_h,
        "fps": fps,
        "frames_written": written,
    }
    metadata_path = output_path.with_suffix(".crop.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved cropped video: {output_path}")
    print(f"Saved crop metadata: {metadata_path}")
    print(f"Crop offset: x={x1}, y={y1}; size={crop_w}x{crop_h}")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path (default: INPUT_cropped.mp4)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=30,
        help="Margin around selected plate in pixels (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.video.expanduser().resolve()
    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_cropped.mp4")
    else:
        output_path = output_path.expanduser().resolve()
    crop_video(input_path, output_path, max(0, args.margin))


if __name__ == "__main__":
    main()