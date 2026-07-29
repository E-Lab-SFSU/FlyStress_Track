"""Create a four-quadrant debugging video.

Quadrants:
    top-left:     raw cropped frame
    top-right:    background-subtracted image
    bottom-left:  binary threshold image
    bottom-right: detected image with colored bounding boxes

Run from the FlyStress_Track repository root:
    python make_detection_quadrants.py path/to/video_cropped.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import config


def build_median_background(cap: cv2.VideoCapture) -> np.ndarray:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise RuntimeError("Invalid frame count while building background.")

    sample_count = min(int(config.BG_SAMPLES), frame_count)
    indices = np.linspace(0, frame_count - 1, sample_count).astype(int)
    values = []

    print(f"Building median background from {sample_count} frames")
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            values.append(hsv[:, :, 2])

    if not values:
        raise RuntimeError("No frames were collected for the background.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(np.stack(values, axis=0), axis=0).astype(np.uint8)


def detect_stages(frame: np.ndarray, background_v: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    foreground = cv2.subtract(background_v, value)

    if str(config.THRESH_METHOD).lower() == "otsu":
        _, binary = cv2.threshold(
            foreground, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            foreground, int(config.THRESH), 255, cv2.THRESH_BINARY
        )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    image_h, image_w = binary.shape
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.MIN_A or area > config.MAX_A:
            continue

        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (rect_w, rect_h), _ = rect
        short_side = int(min(rect_w, rect_h))
        long_side = int(max(rect_w, rect_h))

        if (
                center_x <= 2
                or center_y <= 2
                or center_x >= image_w - 2
                or center_y >= image_h - 2
        ):
            continue
        if short_side < config.MIN_WH or short_side > config.MAX_WH:
            continue
        if long_side < config.MIN_WH or long_side > config.MAX_WH:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centroid = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        detections.append((contour, rect, centroid, area))

    return foreground, binary, detections


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (260, 30), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def create_debug_video(input_path: Path, output_path: Path, show: bool) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame.")
    frame_h, frame_w = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    background_v = build_median_background(cap)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w * 2, frame_h * 2),
    )
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Cannot create output video: {output_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            foreground, binary, detections = detect_stages(frame, background_v)
            detected = frame.copy()
            for contour, rect, centroid, area in detections:
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(detected, [box], 0, (0, 255, 0), 2)
                cv2.circle(detected, centroid, 3, (0, 0, 255), -1)
                cv2.putText(
                    detected,
                    f"A={area:.0f}",
                    (centroid[0] + 4, centroid[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            foreground_bgr = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            top = np.hstack(
                [label_panel(frame, "Raw image"),
                 label_panel(foreground_bgr, "Background subtracted")]
            )
            bottom = np.hstack(
                [label_panel(binary_bgr, "Binary threshold"),
                 label_panel(detected, f"Detected: {len(detections)}")]
            )
            dashboard = np.vstack([top, bottom])
            writer.write(dashboard)

            if show:
                cv2.imshow("Detection stages", dashboard)
                key = cv2.waitKey(max(1, int(config.PLAYBACK_DELAY))) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1
            if frame_index % 300 == 0:
                print(f"Processed {frame_index} frames")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved four-quadrant video: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Cropped input video")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path (default: INPUT_detection_quadrants.mp4)",
    )
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.video.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_detection_quadrants.mp4")
    )
    create_debug_video(input_path, output_path, show=not args.no_display)


if __name__ == "__main__":
    main()