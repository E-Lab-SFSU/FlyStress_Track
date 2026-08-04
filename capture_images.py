from __future__ import annotations

import csv
import platform
import time
from datetime import datetime
from pathlib import Path
from live_views import LiveViewManager

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
        # --------------------------------------------------
        # Configure and warm up the camera
        # --------------------------------------------------

        configure_camera(cap)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print(
            "Actual camera mode: "
            f"{actual_width} x {actual_height} at "
            f"{actual_fps:.1f} FPS"
        )

        warmup_end = (
                time.monotonic()
                + config.CAMERA_WARMUP_SECONDS
        )

        while time.monotonic() < warmup_end:
            read_frame(cap)

        # --------------------------------------------------
        # Capture the first/background frame
        # --------------------------------------------------

        reference_frame = read_frame(cap)

        print(
            "Reference frame shape:",
            reference_frame.shape,
        )

        debug_path = (
                plate_folder
                / "reference_frame_debug.png"
        )

        if not cv2.imwrite(
                str(debug_path),
                reference_frame,
        ):
            raise RuntimeError(
                f"Could not save debug image: {debug_path}"
            )

        print(
            "Saved reference frame:",
            debug_path.resolve(),
        )

        # --------------------------------------------------
        # Attempt to detect wells
        #
        # Detection failure is no longer fatal.
        # --------------------------------------------------

        bolts = []
        wells = []
        cropped_wells = []

        well_detection_error = None

        try:
            bolts, wells = detect_plate_wells(
                reference_frame,
                rows=config.PLATE_ROWS,
                columns=config.PLATE_COLUMNS,
            )

            if len(wells) != config.EXPECTED_WELLS:
                raise RuntimeError(
                    f"Detected {len(wells)} wells; "
                    f"expected {config.EXPECTED_WELLS}."
                )

            print(
                f"Successfully detected "
                f"{len(wells)} wells."
            )

        except Exception as error:
            well_detection_error = str(error)

            print()
            print("WARNING: Well detection failed.")
            print(well_detection_error)
            print(
                "Capture and diagnostic windows "
                "will continue using the full frame."
            )
            print()

            error_path = (
                    plate_folder
                    / "well_detection_error.txt"
            )

            error_path.write_text(
                well_detection_error,
                encoding="utf-8",
            )

        # --------------------------------------------------
        # Set the processing area
        #
        # If wells were detected:
        #     crop to the plate.
        #
        # If wells were not detected:
        #     use the full camera frame.
        # --------------------------------------------------

        if wells:
            left, top, right, bottom = (
                crop_bounds_from_wells(
                    wells,
                    reference_frame.shape,
                )
            )

            processing_reference = (
                reference_frame[
                    top:bottom,
                    left:right,
                ]
            )

            cropped_wells = adjust_wells_to_crop(
                wells,
                left,
                top,
            )

            annotated_full = annotate(
                reference_frame,
                bolts,
                wells,
            )

            annotated_crop = annotate(
                processing_reference,
                [],
                cropped_wells,
            )

            cv2.imwrite(
                str(
                    plate_folder
                    / "plate_reference_full.png"
                ),
                reference_frame,
            )

            cv2.imwrite(
                str(
                    plate_folder
                    / "plate_wells_detected_full.png"
                ),
                annotated_full,
            )

            cv2.imwrite(
                str(
                    plate_folder
                    / "plate_wells_detected_cropped.png"
                ),
                annotated_crop,
            )

            save_csv(
                plate_folder / "plate_wells.csv",
                cropped_wells,
                )

            with (
                    plate_folder / "crop_bounds.csv"
            ).open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)

                writer.writerow(
                    [
                        "left",
                        "top",
                        "right",
                        "bottom",
                    ]
                )

                writer.writerow(
                    [
                        left,
                        top,
                        right,
                        bottom,
                    ]
                )

            print(
                f"Plate crop: "
                f"left={left}, "
                f"top={top}, "
                f"right={right}, "
                f"bottom={bottom}"
            )

        else:
            image_height, image_width = (
                reference_frame.shape[:2]
            )

            left = 0
            top = 0
            right = image_width
            bottom = image_height

            processing_reference = (
                reference_frame.copy()
            )

            cv2.imwrite(
                str(
                    plate_folder
                    / "plate_reference_full.png"
                ),
                reference_frame,
            )

            with (
                    plate_folder / "crop_bounds.csv"
            ).open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)

                writer.writerow(
                    [
                        "left",
                        "top",
                        "right",
                        "bottom",
                    ]
                )

                writer.writerow(
                    [
                        left,
                        top,
                        right,
                        bottom,
                    ]
                )

            print(
                "Using the full camera frame "
                "because wells were not detected."
            )

        # --------------------------------------------------
        # Create the live diagnostic windows
        #
        # The background image is the first frame.
        # --------------------------------------------------

        live_views = LiveViewManager(
            processing_reference,
            display_width=640,
            binary_threshold=25,
            minimum_motion_area=20,
        )

        print()
        print(
            "Displaying:"
        )
        print("  Background Image")
        print("  Grayscale Image")
        print("  Binary Image")
        print("  Detect Image")
        print("  Tracking Window")
        print()
        print(
            "Saving one image per second."
        )
        print(
            "Press q in an OpenCV window to stop."
        )

        # --------------------------------------------------
        # Prepare image capture
        # --------------------------------------------------

        metadata_rows: list[
            dict[str, object]
        ] = []

        image_number = 1

        start_time = time.monotonic()
        next_capture_time = start_time

        # --------------------------------------------------
        # Main capture loop
        # --------------------------------------------------

        while True:
            frame = read_frame(cap)

            processing_frame = frame[
                top:bottom,
                left:right,
            ]

            display_wells = (
                cropped_wells
                if cropped_wells
                else None
            )

            quit_requested = live_views.show(
                processing_frame,
                wells=display_wells,
                tracks=None,
                well_detection_error=(
                    well_detection_error
                ),
            )

            now = time.monotonic()

            # ----------------------------------------------
            # Save one image per interval
            # ----------------------------------------------

            if now >= next_capture_time:
                filename = (
                    f"image{image_number:06d}"
                    f"{config.IMAGE_EXTENSION}"
                )

                image_path = (
                        images_folder / filename
                )

                saved = cv2.imwrite(
                    str(image_path),
                    processing_frame,
                )

                if not saved:
                    raise RuntimeError(
                        "Could not save image: "
                        f"{image_path}"
                    )

                elapsed = now - start_time

                metadata_rows.append(
                    {
                        "image": filename,
                        "image_number": (
                            image_number
                        ),
                        "timestamp_iso": (
                            datetime.now()
                            .astimezone()
                            .isoformat(
                                timespec="milliseconds"
                            )
                        ),
                        "elapsed_seconds": (
                            f"{elapsed:.3f}"
                        ),
                        "width_px": (
                            processing_frame.shape[1]
                        ),
                        "height_px": (
                            processing_frame.shape[0]
                        ),
                    }
                )

                write_capture_metadata(
                    experiment_folder
                    / "capture_metadata.csv",
                    metadata_rows,
                    )

                print(
                    f"Saved {filename}"
                )

                image_number += 1

                next_capture_time += (
                    config
                    .CAPTURE_INTERVAL_SECONDS
                )

                # Prevent the program from rapidly saving
                # multiple images if it falls behind.
                if next_capture_time < now:
                    next_capture_time = (
                            now
                            + config
                            .CAPTURE_INTERVAL_SECONDS
                    )

            # ----------------------------------------------
            # Stop when q is pressed
            # ----------------------------------------------

            if quit_requested:
                print(
                    "Stop requested."
                )
                break

    except KeyboardInterrupt:
        print()
        print(
            "Capture stopped with Ctrl+C."
        )

    finally:
        cap.release()
        cv2.destroyAllWindows()

        print(
            "Camera released."
        )
        print(
            f"Experiment saved in: "
            f"{experiment_folder.resolve()}"
        )

if __name__ == "__main__":
    main()
