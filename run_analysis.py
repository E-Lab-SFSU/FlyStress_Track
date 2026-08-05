"""FlyStress Track v2: reference-background tracking for three flies per well.

Offline:
    python run_analysis.py "C:/path/to/experiment"

Folder layouts supported:
    experiment/images/*.png
    experiment/empty_reference/*.png   (recommended)

or images directly in the supplied folder. In that case outputs are written to
FlyStress_analysis inside that folder.
"""
from __future__ import annotations
import argparse
import csv
import math
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import config
from live_views import LiveViewManager
from manual_well_calibration import calibrate as manual_calibrate
from sleep_analysis.background_reference import resolve_reference
from sleep_analysis.fly_detection import detect_flies
from sleep_analysis.registration import register_pair
from sleep_analysis.multi_fly_tracker import PerWellMultiFlyTracker
from sleep_analysis.results_logger import ExperimentLoggers

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def create_experiment_folder(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while (root / f"{config.EXPERIMENT_PREFIX}{number:05d}").exists():
        number += 1
    path = root / f"{config.EXPERIMENT_PREFIX}{number:05d}"
    path.mkdir()
    return path

def collect_images(folder: Path) -> list[Path]:
    def number(path: Path):
        digits = "".join(c for c in path.stem if c.isdigit())
        return int(digits) if digits else -1
    images = sorted((p for p in folder.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=number)
    if len(images) < 2:
        raise RuntimeError(f"At least two images are required; found {len(images)} in {folder}")
    return images

def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image

def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save image: {path}")

def load_wells(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    wells = [dict(well=row["well"], row=int(float(row.get("row", 0) or 0)),
                  column=int(float(row.get("column", 0) or 0)),
                  x=int(float(row["x"])), y=int(float(row["y"])),
                  radius=int(float(row["radius"]))) for row in rows]
    if len(wells) != config.EXPECTED_WELLS:
        raise RuntimeError(f"Expected {config.EXPECTED_WELLS} wells; found {len(wells)}")
    return wells

def resolve_wells(reference: np.ndarray, plate_folder: Path) -> list[dict[str, object]]:
    plate_folder.mkdir(parents=True, exist_ok=True)
    csv_path = plate_folder / "plate_wells.csv"
    if config.REUSE_EXISTING_WELL_CALIBRATION and csv_path.is_file():
        return load_wells(csv_path)
    if config.MANUAL_WELL_CSV and Path(config.MANUAL_WELL_CSV).is_file():
        shutil.copy2(config.MANUAL_WELL_CSV, csv_path)
        return load_wells(csv_path)
    image_path = plate_folder / "manual_calibration_reference.png"
    save_image(image_path, reference)
    if not config.SHOW_WINDOWS:
        raise RuntimeError("Manual calibration is required, but SHOW_WINDOWS=False.")
    if not manual_calibrate(image_path, csv_path, load_existing=False):
        raise RuntimeError("Calibration was canceled.")
    return load_wells(csv_path)

def registration_settings() -> dict[str, object]:
    return dict(motion_model=config.REGISTRATION_MOTION_MODEL,
                blur_kernel=config.REGISTRATION_BLUR_KERNEL,
                max_iterations=config.REGISTRATION_MAX_ITERATIONS,
                epsilon=config.REGISTRATION_EPSILON,
                difference_threshold=config.DIFFERENCE_THRESHOLD)

def detection_settings() -> dict[str, object]:
    return dict(max_components=config.MAX_DETECTION_COMPONENTS_PER_WELL,
                inner_mask_scale=config.WELL_INNER_MASK_SCALE,
                edge_exclusion_px=config.WELL_EDGE_EXCLUSION_PX,
                difference_blur_kernel=config.DIFFERENCE_BLUR_KERNEL,
                fixed_threshold=config.BACKGROUND_DIFFERENCE_THRESHOLD,
                use_otsu_floor=config.DIFFERENCE_USE_OTSU_FLOOR,
                otsu_min_threshold=config.DIFFERENCE_OTSU_MIN_THRESHOLD,
                otsu_max_threshold=config.DIFFERENCE_OTSU_MAX_THRESHOLD,
                min_area_px=config.FLY_MIN_AREA_PX,
                max_single_area_px=config.FLY_MAX_SINGLE_AREA_PX,
                max_component_area_px=config.FLY_MAX_COMPONENT_AREA_PX,
                min_fill_ratio=config.FLY_MIN_FILL_RATIO,
                max_aspect_ratio=config.FLY_MAX_ASPECT_RATIO,
                morph_kernel=config.FLY_MORPH_KERNEL,
                open_iterations=config.FLY_MORPH_OPEN_ITERATIONS,
                close_iterations=config.FLY_MORPH_CLOSE_ITERATIONS,
                enable_overlap=config.ENABLE_OVERLAP_DETECTION,
                min_samples_for_overlap=config.MIN_SINGLE_AREA_SAMPLES_FOR_OVERLAP,
                overlap_two_multiplier=config.OVERLAP_TWO_FLY_MULTIPLIER,
                overlap_three_multiplier=config.OVERLAP_THREE_FLY_MULTIPLIER,
                overlap_max_total_flies_per_well=config.OVERLAP_MAX_TOTAL_FLIES_PER_WELL)

def create_tracker(wells):
    return PerWellMultiFlyTracker([str(w["well"]) for w in wells],
                                  flies_per_well=config.FLIES_PER_WELL,
                                  max_match_distance_px=config.MAX_POSITION_JUMP_PX,
                                  jitter_threshold_px=config.JITTER_THRESHOLD_PX,
                                  rolling_window_seconds=config.ROLLING_WINDOW_SECONDS,
                                  sleep_duration_seconds=config.SLEEP_DURATION_SECONDS,
                                  max_valid_sample_gap_seconds=config.MAX_VALID_SAMPLE_GAP_SECONDS,
                                  low_confidence_frames_after_split=config.IDENTITY_LOW_CONFIDENCE_FRAMES_AFTER_SPLIT)

def annotate(image, wells, results):
    output = image.copy()
    well_lookup = {w["well"]: w for w in wells}
    for well in wells:
        inner = int(round(well["radius"] * config.WELL_INNER_MASK_SCALE))
        cv2.circle(output, (well["x"], well["y"]), inner, (150, 150, 150), 1)
    for result in results:
        if result.x_px is not None:
            point = (int(round(result.x_px)), int(round(result.y_px)))
            color = ((0, 255, 0) if result.activity_state == "AWAKE" else
                     (0, 0, 255) if result.activity_state == "ASLEEP" else (0, 255, 255))
            cv2.circle(output, point, 6, color, 2)
            text = f"{result.fly_id} {result.activity_state} {result.identity_confidence}"
            cv2.putText(output, text, (point[0] + 7, point[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1, cv2.LINE_AA)
        elif result.observation_status in ("OVERLAP", "UNKNOWN"):
            well = well_lookup[result.well]
            text = f"{result.fly_id}: {result.observation_status}"
            cv2.putText(output, text,
                        (well["x"] - well["radius"],
                         well["y"] - well["radius"] + 14 * result.fly_slot),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 165, 255), 1, cv2.LINE_AA)
    return output

def fmt(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"

def log_frame(loggers, results, detections, thresholds, wells, *, timestamp,
              elapsed, image, frame, registration_ok, registration_score):
    for result in results:
        common = dict(timestamp_iso=timestamp, elapsed_seconds=f"{elapsed:.3f}",
                      image=image, frame_number=frame, well=result.well,
                      fly_id=result.fly_id, fly_slot=result.fly_slot)
        loggers.positions.write(common | dict(x_px=fmt(result.x_px), y_px=fmt(result.y_px),
                                              well_relative_x_px=fmt(result.local_x_px), well_relative_y_px=fmt(result.local_y_px),
                                              area_px="" if result.area_px is None else result.area_px,
                                              raw_distance_px=fmt(result.raw_distance_px), distance_px=fmt(result.distance_px)))
        loggers.states.write(common | dict(observation_status=result.observation_status,
                                           activity_state=result.activity_state, identity_confidence=result.identity_confidence,
                                           overlap_group=result.overlap_group, overlap_count=result.overlap_count,
                                           rolling_distance_px=f"{result.rolling_distance_px:.3f}",
                                           rolling_samples=result.rolling_samples,
                                           immobile_duration_seconds=f"{result.immobile_duration_seconds:.3f}"))
    by_well = {w["well"]: [r for r in results if r.well == w["well"]] for w in wells}
    for well in wells:
        name = well["well"]
        ds = detections.get(name, [])
        rr = by_well[name]
        loggers.wells.write(dict(timestamp_iso=timestamp, elapsed_seconds=f"{elapsed:.3f}",
                                 image=image, frame_number=frame, well=name, configured_flies=config.FLIES_PER_WELL,
                                 separate_detections=sum(d.estimated_fly_count == 1 for d in ds),
                                 overlap_blobs=sum(d.estimated_fly_count > 1 for d in ds),
                                 estimated_flies_visible=sum(d.estimated_fly_count for d in ds),
                                 unknown_slots=sum(r.observation_status == "UNKNOWN" for r in rr),
                                 threshold_value=thresholds.get(name, ""), registration_succeeded=registration_ok,
                                 registration_score="" if registration_score is None or
                                                          not math.isfinite(registration_score) else f"{registration_score:.8f}"))

def output_folders(base: Path):
    analysis = base / "analysis"
    folders = {"analysis": analysis, "registered": analysis / "registered",
               "difference": analysis / "difference",
               "binary": analysis / "difference_thresholded",
               "masks": analysis / "fly_detection_masks",
               "tracking": analysis / "tracked_fly_overlays"}
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
    return folders

def process_frame(frame, alignment_reference, background_reference, tracker, wells,
                  elapsed, first=False):
    if first:
        aligned = frame
        registration_ok = True
        registration_score = None
        difference = cv2.absdiff(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(background_reference, cv2.COLOR_BGR2GRAY))
    else:
        result = register_pair(alignment_reference, frame, **registration_settings())
        aligned, registration_ok, registration_score = (
            result.aligned_bgr, result.succeeded, result.correlation)
        difference = cv2.absdiff(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(background_reference, cv2.COLOR_BGR2GRAY))
    detections, mask, thresholds = detect_flies(
        aligned, background_reference, wells, area_hints=tracker.area_hints(),
        area_sample_counts=tracker.area_sample_counts(), **detection_settings())
    results = tracker.update_all(detections, elapsed, registration_ok)
    overlay = annotate(aligned, wells, results)
    return (aligned, registration_ok, registration_score, difference, mask,
            detections, thresholds, results, overlay)

def analyze_offline(path: Path):
    source = path.expanduser().resolve()
    images_folder = source / "images" if (source / "images").is_dir() else source
    output_base = source if (source / "images").is_dir() else source / "FlyStress_analysis"
    experiment_root = source if (source / "images").is_dir() else source
    plate_folder = output_base / "plate"
    images = collect_images(images_folder)
    first = load_image(images[0])
    background, background_source = resolve_reference(
        experiment_root=experiment_root, experiment_images=images,
        empty_folder_name=config.EMPTY_REFERENCE_FOLDER_NAME,
        cache_filename=config.BACKGROUND_CACHE_FILENAME,
        max_images=config.BACKGROUND_MEDIAN_MAX_IMAGES,
        min_images=config.BACKGROUND_MIN_IMAGES,
        register_samples=config.BACKGROUND_REGISTER_SAMPLES,
        registration_settings=registration_settings(),
        prefer_temporal_experiment_median=config.PREFER_TEMPORAL_EXPERIMENT_MEDIAN)
    if background.shape != first.shape:
        raise RuntimeError("Background reference dimensions do not match experiment images.")
    print(f"Background source: {background_source}")
    wells = resolve_wells(first, plate_folder)
    tracker = create_tracker(wells)
    folders = output_folders(output_base)
    views = LiveViewManager(background, config.DISPLAY_WIDTH) if config.SHOW_WINDOWS else None
    storage = config.storage_settings()
    with ExperimentLoggers(folders["analysis"]) as logs:
        for number, image_path in enumerate(images, 1):
            frame = load_image(image_path)
            elapsed = (number - 1) * config.CAPTURE_INTERVAL_SECONDS
            (aligned, ok, score, difference, mask, detections, thresholds,
             results, overlay) = process_frame(frame, first, background, tracker,
                                               wells, elapsed, number == 1)
            log_frame(logs, results, detections, thresholds, wells, timestamp="",
                      elapsed=elapsed, image=image_path.name, frame=number,
                      registration_ok=ok, registration_score=score)
            every = int(storage["every_n_frames"])
            if number == 1 or (every and number % every == 0):
                save_image(folders["masks"] / f"mask_{image_path.stem}.png", mask)
                save_image(folders["tracking"] / f"tracked_{image_path.stem}.png", overlay)
                save_image(folders["difference"] / f"difference_{image_path.stem}.png", difference)
                save_image(folders["binary"] / f"binary_{image_path.stem}.png", mask)
            tracked = sum(r.x_px is not None for r in results)
            overlaps = sum(r.observation_status == "OVERLAP" for r in results)
            print(f"[{number}/{len(images)}] {image_path.name} | tracked={tracked}/{len(results)} overlap_slots={overlaps}")
            if views and views.show(aligned, overlay, "Offline analysis - q to stop", mask):
                break
    cv2.destroyAllWindows()
    print(f"Saved: {folders['analysis']}")

def open_camera():
    attempts = ([(config.CAMERA_INDEX, cv2.CAP_DSHOW),
                 (config.CAMERA_INDEX, cv2.CAP_ANY)] if platform.system() == "Windows" else
                [(config.CAMERA_DEVICE, cv2.CAP_V4L2),
                 (config.CAMERA_INDEX, cv2.CAP_ANY)])
    for source, backend in attempts:
        capture = cv2.VideoCapture(source, backend)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture()

def live_analysis():
    raise RuntimeError(
        "v2 live capture requires an empty-plate reference captured before flies are added. "
        "Use offline analysis today, or place experiment images and empty_reference images "
        "in an experiment folder and run: python run_analysis.py <folder>")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    live_analysis() if args.path is None else analyze_offline(args.path)

if __name__ == "__main__":
    main()
