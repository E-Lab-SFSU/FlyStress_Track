"""Run the complete FlyStress still-image sleep-analysis pipeline.

Examples:
    python run_analysis.py "C:/Users/chana/Downloads/FS_IMG/exp00001"
    python run_analysis.py ~/FS_IMG/exp00001

If no path is supplied, the newest experiment inside config.OUTPUT_ROOT is used.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from sleep_analysis.fly_detection import detect_flies
from sleep_analysis.multi_fly_tracker import (
    PerWellMultiFlyTracker,
    TrackResult,
    TrackerDetection,
)
from sleep_analysis.registration import RegistrationResult, register_pair
from sleep_analysis.results_logger import ResultsLogger


def find_latest_experiment(root: Path) -> Path:
    experiments = sorted(
        folder
        for folder in root.glob("exp*")
        if folder.is_dir() and (folder / "images").is_dir()
    )
    if not experiments:
        raise FileNotFoundError(
            f"No experiment folders containing images were found in {root.resolve()}."
        )
    return experiments[-1]


def image_number(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else -1


def collect_images(images_folder: Path) -> list[Path]:
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(
        (path for path in images_folder.iterdir() if path.suffix.lower() in supported),
        key=image_number,
    )
    if len(images) < 2:
        raise RuntimeError(
            f"At least two images are required. Found {len(images)} in "
            f"{images_folder.resolve()}."
        )
    return images


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_wells(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing well coordinates: {path}. Run capture_images.py first."
        )

    numeric_fields = {
        "row", "column", "x", "y", "radius", "diameter",
        "detected_radius", "detected_diameter",
    }
    wells: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            converted: dict[str, object] = dict(row)
            for field in numeric_fields:
                value = row.get(field, "")
                if value != "":
                    converted[field] = int(float(value))
            wells.append(converted)

    if len(wells) != config.EXPECTED_WELLS:
        raise RuntimeError(
            f"Expected {config.EXPECTED_WELLS} wells in {path}, found {len(wells)}."
        )
    return wells


def load_capture_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["image"]: row for row in csv.DictReader(file)}


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save image: {path}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def detection_settings() -> dict[str, object]:
    return {
        "max_flies": config.FLIES_PER_WELL_MAX,
        "mask_margin_px": config.WELL_MASK_MARGIN_PX,
        "dark_percentile": config.FLY_DARK_PERCENTILE,
        "threshold_offset": config.FLY_THRESHOLD_OFFSET,
        "min_area_px": config.FLY_MIN_AREA_PX,
        "max_area_px": config.FLY_MAX_AREA_PX,
        "morph_kernel": config.FLY_MORPH_KERNEL,
        "open_iterations": config.FLY_MORPH_OPEN_ITERATIONS,
        "close_iterations": config.FLY_MORPH_CLOSE_ITERATIONS,
    }


def warp_values(result: RegistrationResult) -> tuple[float, float, float]:
    matrix = result.warp_matrix
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    rotation = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    return dx, dy, rotation


def aligned_to_raw_point(
        x: float,
        y: float,
        warp_matrix: np.ndarray,
) -> tuple[float, float]:
    """Convert a point from the aligned image back to the current raw frame."""
    # register_pair uses cv2.WARP_INVERSE_MAP, so the ECC matrix maps an
    # aligned destination coordinate directly into the current raw source.
    matrix = warp_matrix.astype(np.float32)
    raw_x = float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
    raw_y = float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
    return raw_x, raw_y


def group_tracker_detections(detections, warp_matrix=None):
    grouped: dict[str, list[TrackerDetection]] = defaultdict(list)
    for detection in detections:
        if warp_matrix is None:
            raw_x, raw_y = detection.x, detection.y
        else:
            raw_x, raw_y = aligned_to_raw_point(
                detection.x,
                detection.y,
                warp_matrix,
            )
        grouped[detection.well].append(
            TrackerDetection(detection=detection, raw_x=raw_x, raw_y=raw_y)
        )
    return grouped


def track_color(slot: int) -> tuple[int, int, int]:
    colors = {
        1: (0, 0, 255),
        2: (255, 0, 0),
        3: (0, 255, 0),
        4: (0, 255, 255),
    }
    return colors.get(slot, (255, 255, 255))


def annotate_tracks(
        image: np.ndarray,
        wells: list[dict[str, object]],
        results: list[TrackResult],
) -> np.ndarray:
    output = image.copy()
    for well in wells:
        center = (int(float(well["x"])), int(float(well["y"])))
        radius = int(float(well["radius"]))
        cv2.circle(output, center, radius, (160, 160, 160), 1)
        cv2.putText(
            output,
            str(well["well"]),
            (center[0] - radius, center[1] - radius - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for result in results:
        if not result.detected or result.x_px is None or result.y_px is None:
            continue
        center = (int(round(result.x_px)), int(round(result.y_px)))
        color = track_color(result.fly_slot)
        cv2.circle(output, center, 6, color, 2)
        cv2.putText(
            output,
            f"{result.fly_name} {result.fly_state}",
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def result_row(
        result: TrackResult,
        image_path: Path,
        frame_number: int,
        metadata: dict[str, dict[str, str]],
        registration_succeeded: bool,
        registration_score: float | None,
) -> dict[str, object]:
    image_metadata = metadata.get(image_path.name, {})

    def number_or_blank(value: float | int | None, digits: int = 3):
        if value is None:
            return ""
        if isinstance(value, int):
            return value
        return f"{value:.{digits}f}"

    return {
        "timestamp_iso": image_metadata.get("timestamp_iso", ""),
        "elapsed_seconds": image_metadata.get("elapsed_seconds", ""),
        "image": image_path.name,
        "frame_number": frame_number,
        "well": result.well,
        "fly_name": result.fly_name,
        "fly_slot": result.fly_slot,
        "fly_state": result.fly_state,
        "detected": result.detected,
        "x_px": number_or_blank(result.x_px),
        "y_px": number_or_blank(result.y_px),
        "well_relative_x_px": number_or_blank(result.well_relative_x_px),
        "well_relative_y_px": number_or_blank(result.well_relative_y_px),
        "area_px": number_or_blank(result.area_px),
        "threshold_value": number_or_blank(result.threshold_value),
        "raw_distance_px": f"{result.raw_distance_px:.3f}",
        "distance_px": f"{result.distance_px:.3f}",
        "rolling_distance_px": f"{result.rolling_distance_px:.3f}",
        "rolling_samples": result.rolling_samples,
        "missed_frames": result.missed_frames,
        "registration_succeeded": registration_succeeded,
        "registration_score": (
            f"{registration_score:.8f}"
            if registration_score is not None and math.isfinite(registration_score)
            else ""
        ),
    }


def analyze_experiment(experiment_folder: Path) -> None:
    experiment_folder = experiment_folder.expanduser().resolve()
    images_folder = experiment_folder / "images"
    if not images_folder.is_dir():
        raise FileNotFoundError(
            f"The experiment must contain an images folder: {images_folder}"
        )

    images = collect_images(images_folder)
    wells = load_wells(experiment_folder / "plate" / "plate_wells.csv")
    metadata = load_capture_metadata(experiment_folder / "capture_metadata.csv")

    analysis_folder = experiment_folder / "analysis"
    folders = {
        "registered": analysis_folder / "registered",
        "difference": analysis_folder / "difference",
        "threshold": analysis_folder / "difference_thresholded",
        "overlay": analysis_folder / "overlay",
        "detection_masks": analysis_folder / "fly_detection_masks",
        "tracked": analysis_folder / "tracked_fly_overlays",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    tracker = PerWellMultiFlyTracker(
        [str(well["well"]) for well in wells],
        max_flies_per_well=config.FLIES_PER_WELL_MAX,
        max_match_distance_px=config.TRACK_MAX_MATCH_DISTANCE_PX,
        max_missed_frames=config.TRACK_MAX_MISSED_FRAMES,
        jitter_threshold_px=config.JITTER_THRESHOLD_PX,
        rolling_window_samples=config.ROLLING_WINDOW_SECONDS,
        awake_threshold_px=config.AWAKE_THRESHOLD_PX,
    )

    registration_rows: list[dict[str, object]] = []
    registration_fields = [
        "reference_image", "current_image", "registration_succeeded",
        "ecc_correlation", "translation_x_px", "translation_y_px",
        "rotation_degrees", "mean_absolute_difference", "changed_pixels",
    ]

    print(f"Experiment: {experiment_folder}")
    print(f"Images found: {len(images)}")
    print(f"Wells loaded: {len(wells)}")

    reference_path = images[0]
    reference_image = load_image(reference_path)

    with ResultsLogger(analysis_folder / "sleep_results.csv") as logger:
        first_detections, first_mask = detect_flies(
            reference_image,
            wells,
            **detection_settings(),
        )
        grouped = group_tracker_detections(first_detections)
        first_results: list[TrackResult] = []
        for well in wells:
            first_results.extend(
                tracker.update_well(str(well["well"]), grouped.get(str(well["well"]), []))
            )
        save_image(folders["detection_masks"] / f"mask_{reference_path.stem}.png", first_mask)
        save_image(
            folders["tracked"] / f"tracked_{reference_path.stem}.png",
            annotate_tracks(reference_image, wells, first_results),
            )
        for item in first_results:
            logger.write(result_row(item, reference_path, 1, metadata, True, None))
        print(f"[1/{len(images)}] {reference_path.name}: {len(first_detections)} detections")

        for frame_number, current_path in enumerate(images[1:], start=2):
            current_image = load_image(current_path)
            registration = register_pair(
                reference_image,
                current_image,
                motion_model=config.REGISTRATION_MOTION_MODEL,
                blur_kernel=config.REGISTRATION_BLUR_KERNEL,
                max_iterations=config.REGISTRATION_MAX_ITERATIONS,
                epsilon=config.REGISTRATION_EPSILON,
                difference_threshold=config.DIFFERENCE_THRESHOLD,
            )

            stem = current_path.stem
            save_image(folders["registered"] / f"registered_{stem}.png", registration.aligned_bgr)
            save_image(folders["difference"] / f"difference_{stem}.png", registration.difference)
            save_image(
                folders["threshold"] / f"difference_thresholded_{stem}.png",
                registration.thresholded_difference,
                )
            save_image(folders["overlay"] / f"overlay_{stem}.png", registration.overlay)

            detections, detection_mask = detect_flies(
                registration.aligned_bgr,
                wells,
                **detection_settings(),
            )
            grouped = group_tracker_detections(detections, registration.warp_matrix)
            frame_results: list[TrackResult] = []
            for well in wells:
                well_name = str(well["well"])
                frame_results.extend(
                    tracker.update_well(well_name, grouped.get(well_name, []))
                )

            save_image(folders["detection_masks"] / f"mask_{stem}.png", detection_mask)
            save_image(
                folders["tracked"] / f"tracked_{stem}.png",
                annotate_tracks(registration.aligned_bgr, wells, frame_results),
                )
            for item in frame_results:
                logger.write(
                    result_row(
                        item,
                        current_path,
                        frame_number,
                        metadata,
                        registration.succeeded,
                        registration.correlation,
                    )
                )

            dx, dy, rotation = warp_values(registration)
            correlation = (
                f"{registration.correlation:.8f}"
                if math.isfinite(registration.correlation)
                else ""
            )
            registration_rows.append(
                {
                    "reference_image": reference_path.name,
                    "current_image": current_path.name,
                    "registration_succeeded": registration.succeeded,
                    "ecc_correlation": correlation,
                    "translation_x_px": f"{dx:.6f}",
                    "translation_y_px": f"{dy:.6f}",
                    "rotation_degrees": f"{rotation:.6f}",
                    "mean_absolute_difference": f"{float(registration.difference.mean()):.6f}",
                    "changed_pixels": int(cv2.countNonZero(registration.thresholded_difference)),
                }
            )
            write_csv(
                analysis_folder / "registration_summary.csv",
                registration_fields,
                registration_rows,
                )

            status = "OK" if registration.succeeded else "FAILED - unaligned frame used"
            print(
                f"[{frame_number}/{len(images)}] {current_path.name}: {status}, "
                f"{len(detections)} detections, {len(frame_results)} tracked rows"
            )

            # The guide specifies consecutive-image registration. Track positions
            # are converted back to the current raw frame before the next update.
            reference_path = current_path
            reference_image = current_image

    print("Phase 4 analysis complete.")
    print(f"Sleep and position log: {analysis_folder / 'sleep_results.csv'}")
    print(f"Tracked overlays: {folders['tracked']}")
    print(
        "Tune JITTER_THRESHOLD_PX and AWAKE_THRESHOLD_PX in config.py using "
        "your diagnostic images and experimental validation."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        type=Path,
        help="Experiment folder, for example FS_IMG/exp00001. Defaults to newest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = args.experiment or find_latest_experiment(config.OUTPUT_ROOT)
    analyze_experiment(experiment)


if __name__ == "__main__":
    main()
