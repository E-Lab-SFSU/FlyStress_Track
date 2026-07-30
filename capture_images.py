from __future__ import annotations

import csv
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2

import config
from detect_plate_wells import annotate, detect_plate_wells, save_csv


def create_experiment_folder(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    experiment_number = 1
    while True:
        folder = root / f"exp{experiment_number:05d}"
        if not folder.exists():
            folder.mkdir()
            return folder
        experiment_number += 1


def open_camera() -> cv2.VideoCapture:
    system = platform.system()

    if system == "Windows":
        return cv2.VideoCapture(
            config.CAMERA_INDEX,
            cv2.CAP_DSHOW,
        )

    if system == "Linux":
        camera_device = Path("/dev/video0")

        if not camera_device.exists():
            raise RuntimeError(
                f"Camera device does not exist: {camera_device}\n"
                "Run: v4l2-ctl --list-devices"
            )

        return cv2.VideoCapture(
            str(camera_device),
            cv2.CAP_V4L2,
        )

    return cv2.VideoCapture(config.CAMERA_INDEX)


def configure_camera(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)


def read_frame(cap: cv2.VideoCapture):
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("The camera opened, but no image could be read.")
    return frame


def crop_bounds_from_wells(wells, image_shape):
    image_height, image_width = image_shape[:2]
    left = min(int(w["x"]) - int(w["radius"]) for w in wells)
    right = max(int(w["x"]) + int(w["radius"]) for w in wells)
    top = min(int(w["y"]) - int(w["radius"]) for w in wells)
    bottom = max(int(w["y"]) + int(w["radius"]) for w in wells)

    left = max(0, left - config.CROP_PADDING_PX)
    top = max(0, top - config.CROP_PADDING_PX)
    right = min(image_width, right + config.CROP_PADDING_PX)
    bottom = min(image_height, bottom + config.CROP_PADDING_PX)

    if right <= left or bottom <= top:
        raise RuntimeError("Calculated plate crop is invalid.")

    return left, top, right, bottom


def adjust_wells_to_crop(wells, left: int, top: int):
    adjusted = []
    for well in wells:
        item = dict(well)
        item["x"] = int(item["x"]) - left
        item["y"] = int(item["y"]) - top
        adjusted.append(item)
    return adjusted


def write_capture_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "image",
        "image_number",
        "timestamp_iso",
        "elapsed_seconds",
        "width_px",
        "height_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    experiment_folder = create_experiment_folder(config.OUTPUT_ROOT)
    images_folder = experiment_folder / "images"
    plate_folder = experiment_folder / "plate"
    images_folder.mkdir()
    plate_folder.mkdir()

    print(f"Experiment folder: {experiment_folder.resolve()}")

    cap = open_camera()
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {config.CAMERA_INDEX}. "
            "Try CAMERA_INDEX 0, 1, or 2 in config.py."
        )

    try:
        configure_camera(cap)
        print(
            "Actual camera mode: "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} at "
            f"{cap.get(cv2.CAP_PROP_FPS):.1f} FPS"
        )

        warmup_end = time.monotonic() + config.CAMERA_WARMUP_SECONDS
        while time.monotonic() < warmup_end:
            read_frame(cap)

        reference_frame = read_frame(cap)
        bolts, wells = detect_plate_wells(
            reference_frame,
            rows=config.PLATE_ROWS,
            columns=config.PLATE_COLUMNS,
        )

        if len(wells) != config.EXPECTED_WELLS:
            raise RuntimeError(
                f"Detected {len(wells)} wells; expected {config.EXPECTED_WELLS}."
            )

        left, top, right, bottom = crop_bounds_from_wells(
            wells,
            reference_frame.shape,
        )
        cropped_reference = reference_frame[top:bottom, left:right]
        cropped_wells = adjust_wells_to_crop(wells, left, top)

        annotated_full = annotate(reference_frame, bolts, wells)
        annotated_crop = annotate(cropped_reference, [], cropped_wells)

        cv2.imwrite(str(plate_folder / "plate_reference_full.png"), reference_frame)
        cv2.imwrite(str(plate_folder / "plate_wells_detected_full.png"), annotated_full)
        cv2.imwrite(str(plate_folder / "plate_wells_detected_cropped.png"), annotated_crop)
        save_csv(plate_folder / "plate_wells.csv", cropped_wells)

        with (plate_folder / "crop_bounds.csv").open(
                "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["left", "top", "right", "bottom"])
            writer.writerow([left, top, right, bottom])

        print(f"Detected {len(cropped_wells)} wells.")
        print(f"Plate crop: left={left}, top={top}, right={right}, bottom={bottom}")
        print("Saving one cropped image per second. Press q to stop.")

        metadata_rows: list[dict[str, object]] = []
        image_number = 1
        start_time = time.monotonic()
        next_capture_time = start_time

        while True:
            frame = read_frame(cap)
            cropped = frame[top:bottom, left:right]

            cv2.imshow(config.PREVIEW_WINDOW_NAME, cropped)
            now = time.monotonic()

            if now >= next_capture_time:
                filename = f"image{image_number:06d}{config.IMAGE_EXTENSION}"
                image_path = images_folder / filename

                if not cv2.imwrite(str(image_path), cropped):
                    raise RuntimeError(f"Could not save image: {image_path}")

                elapsed = now - start_time
                metadata_rows.append(
                    {
                        "image": filename,
                        "image_number": image_number,
                        "timestamp_iso": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        "elapsed_seconds": f"{elapsed:.3f}",
                        "width_px": cropped.shape[1],
                        "height_px": cropped.shape[0],
                    }
                )
                write_capture_metadata(
                    experiment_folder / "capture_metadata.csv",
                    metadata_rows,
                    )

                print(f"Saved {filename}")
                image_number += 1
                next_capture_time += config.CAPTURE_INTERVAL_SECONDS

                if next_capture_time < now:
                    next_capture_time = now + config.CAPTURE_INTERVAL_SECONDS

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Capture stopped.")


if __name__ == "__main__":
    main()
