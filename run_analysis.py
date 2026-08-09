"""FlyStress Track v1.0.

No path: live USB-camera capture and real-time analysis.
Path supplied: offline analysis of an experiment folder or image directory.
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
from manual_fly_food_setup import load_or_create as load_or_create_fly_food_setup
from sleep_analysis.fly_detection import detect_flies
from sleep_analysis.registration import register_pair
from sleep_analysis.results_logger import PositionLogger, ResultsLogger
from sleep_analysis.single_fly_tracker import SingleFlyTracker, TrackResult

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def create_experiment_folder(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        folder = root / f"{config.EXPERIMENT_PREFIX}{number:05d}"
        if not folder.exists():
            folder.mkdir()
            return folder
        number += 1


def open_camera() -> cv2.VideoCapture:
    system = platform.system()
    if system == "Linux":
        attempts = [(str(config.CAMERA_DEVICE), cv2.CAP_V4L2),
                    (str(config.CAMERA_DEVICE), cv2.CAP_ANY),
                    (int(config.CAMERA_INDEX), cv2.CAP_V4L2),
                    (int(config.CAMERA_INDEX), cv2.CAP_ANY)]
    elif system == "Windows":
        attempts = [(int(config.CAMERA_INDEX), cv2.CAP_MSMF),
                    (int(config.CAMERA_INDEX), cv2.CAP_DSHOW),
                    (int(config.CAMERA_INDEX), cv2.CAP_ANY)]
    else:
        attempts = [(int(config.CAMERA_INDEX), cv2.CAP_ANY)]
    for source, backend in attempts:
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            print(f"Opened camera {source!r} using backend {backend}.")
            return cap
        cap.release()
    return cv2.VideoCapture()


def configure_camera(cap: cv2.VideoCapture) -> None:
    if config.CAMERA_FOURCC:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)


def read_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Camera opened, but no frame could be read.")
    return frame


def image_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def collect_images(folder: Path) -> list[Path]:
    images = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=image_number)
    if len(images) < 2:
        raise RuntimeError(f"At least two images are required; found {len(images)} in {folder}.")
    return images


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save image: {path}")



def load_capture_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row.get("image", ""): row for row in csv.DictReader(file) if row.get("image")}


def load_wells(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    wells: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"well", "x", "y", "radius"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Well CSV missing required columns: {path}")
        for row in reader:
            wells.append({
                "well": row["well"],
                "row": int(float(row.get("row", 0) or 0)),
                "column": int(float(row.get("column", 0) or 0)),
                "x": int(round(float(row["x"]))),
                "y": int(round(float(row["y"]))),
                "radius": int(round(float(row["radius"]))),
                "diameter": int(round(float(row.get("diameter", 2 * float(row["radius"]))))),
            })
    if len(wells) != config.EXPECTED_WELLS:
        raise RuntimeError(f"Expected {config.EXPECTED_WELLS} wells; found {len(wells)} in {path}.")
    return wells


def resolve_wells(reference: np.ndarray, plate_folder: Path) -> list[dict[str, object]]:
    """Always use manual calibration; reuse a saved manual CSV when allowed."""
    plate_folder.mkdir(parents=True, exist_ok=True)
    experiment_csv = plate_folder / "plate_wells.csv"
    if config.REUSE_EXISTING_WELL_CALIBRATION and experiment_csv.is_file():
        print(f"Reusing manual calibration: {experiment_csv}")
        return load_wells(experiment_csv)
    if config.MANUAL_WELL_CSV:
        shared = Path(config.MANUAL_WELL_CSV).expanduser()
        if shared.is_file():
            shutil.copy2(shared, experiment_csv)
            print(f"Copied shared manual calibration: {shared}")
            return load_wells(experiment_csv)
    if not config.SHOW_WINDOWS:
        raise RuntimeError("Manual calibration is required. Set SHOW_WINDOWS=True or provide plate_wells.csv.")
    reference_path = plate_folder / "manual_calibration_reference.png"
    save_image(reference_path, reference)
    print("Manual calibration required. Draw A1-A8, B1-B8, C1-C8, D1-D8, then press S.")
    saved = manual_calibrate(reference_path, experiment_csv, load_existing=False)
    if not saved:
        raise RuntimeError("Manual well calibration was canceled.")
    return load_wells(experiment_csv)


def detection_settings() -> dict[str, object]:
    return dict(max_candidates=(getattr(config, "MANUAL_TRACKING_MAX_CANDIDATES_PER_WELL", config.MAX_DETECTION_CANDIDATES_PER_WELL) if getattr(config, "USE_MANUAL_FLY_FOOD_SETUP", False) else config.MAX_DETECTION_CANDIDATES_PER_WELL),
                mask_margin_px=config.WELL_MASK_MARGIN_PX,
                dark_percentile=config.FLY_DARK_PERCENTILE,
                threshold_offset=config.FLY_THRESHOLD_OFFSET,
                min_area_px=(getattr(config, "MANUAL_TRACKING_MIN_AREA_PX", config.FLY_MIN_AREA_PX) if getattr(config, "USE_MANUAL_FLY_FOOD_SETUP", False) else config.FLY_MIN_AREA_PX),
                max_area_px=(getattr(config, "MANUAL_TRACKING_MAX_AREA_PX", config.FLY_MAX_AREA_PX) if getattr(config, "USE_MANUAL_FLY_FOOD_SETUP", False) else config.FLY_MAX_AREA_PX),
                morph_kernel=config.FLY_MORPH_KERNEL,
                open_iterations=config.FLY_MORPH_OPEN_ITERATIONS,
                close_iterations=config.FLY_MORPH_CLOSE_ITERATIONS,
                background_difference_threshold=config.FLY_BACKGROUND_DIFFERENCE_THRESHOLD,
                motion_difference_threshold=config.FLY_MOTION_DIFFERENCE_THRESHOLD,
                max_aspect_ratio=config.FLY_MAX_ASPECT_RATIO,
                min_fill_ratio=config.FLY_MIN_FILL_RATIO,
                edge_exclusion_px=config.FLY_EDGE_EXCLUSION_PX,
                relax_shape_filter=getattr(config, "MANUAL_TRACKING_RELAX_SHAPE_FILTER", False),
                use_darkness_gate=getattr(config, "USE_FLY_DARKNESS_GATE", False),
                max_gray_value=getattr(config, "FLY_MAX_GRAY_VALUE", 135),
                local_background_percentile=getattr(config, "FLY_LOCAL_BACKGROUND_PERCENTILE", 70.0),
                min_contrast_from_background=getattr(config, "FLY_MIN_CONTRAST_FROM_BACKGROUND", 12.0),
                wall_mode_radial_fraction=getattr(config, "WALL_MODE_RADIAL_FRACTION", 0.72),
                wall_edge_exclusion_px=getattr(config, "WALL_EDGE_EXCLUSION_PX", 1),
                normalize_per_well_illumination=getattr(config, "NORMALIZE_PER_WELL_ILLUMINATION", True),
                per_well_arrival_threshold=getattr(config, "PER_WELL_ARRIVAL_THRESHOLD", 4.0),
                per_well_arrival_min_contrast=getattr(config, "PER_WELL_ARRIVAL_MIN_CONTRAST", 6.0),
                per_well_motion_dilate_iterations=getattr(config, "PER_WELL_MOTION_DILATE_ITERATIONS", 1))


def prepare_background_reference(reference: np.ndarray) -> np.ndarray | None:
    """Load and align the optional empty-plate reference to calibrated coordinates."""
    if not getattr(config, "USE_BACKGROUND_REFERENCE", False):
        return None
    configured = getattr(config, "BACKGROUND_REFERENCE_IMAGE", None)
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_file():
        print(f"Background reference not found: {path}. Continuing without it.")
        return None
    background = load_image(path)
    if background.shape != reference.shape:
        print(
            "Background reference dimensions do not match the tracking frames "
            f"({background.shape[1]}x{background.shape[0]} vs "
            f"{reference.shape[1]}x{reference.shape[0]}). Continuing without it."
        )
        return None
    registration = process_pair(reference, background)
    if registration.succeeded:
        print(f"Using aligned empty-background reference: {path}")
        return registration.aligned_bgr
    print(f"Could not align background reference {path}; using it without registration.")
    return background


def create_tracker(wells: list[dict[str, object]], manual_setup=None) -> SingleFlyTracker:
    tracker = SingleFlyTracker([str(w["well"]) for w in wells],
                               jitter_threshold_px=config.JITTER_THRESHOLD_PX,
                               rolling_window_seconds=config.ROLLING_WINDOW_SECONDS,
                               sleep_duration_seconds=config.SLEEP_DURATION_SECONDS,
                               max_position_jump_px=config.MAX_POSITION_JUMP_PX,
                               max_valid_sample_gap_seconds=config.MAX_VALID_SAMPLE_GAP_SECONDS,
                               food_overlap_penalty=getattr(config, "FOOD_OVERLAP_PENALTY", 35.0),
                               food_contact_max_jump_multiplier=getattr(config, "FOOD_CONTACT_MAX_JUMP_MULTIPLIER", 1.35),
                               food_contact_search_radius_px=getattr(config, "FOOD_CONTACT_SEARCH_RADIUS_PX", 35),
                               grayscale_identity_weight=getattr(config, "GRAYSCALE_IDENTITY_WEIGHT", 0.30),
                               reacquire_jump_growth=getattr(config, "REACQUIRE_JUMP_GROWTH", 0.75),
                               reacquire_max_jump_px=getattr(config, "REACQUIRE_MAX_JUMP_PX", 220.0),
                               full_well_reacquire_after_misses=getattr(config, "FULL_WELL_REACQUIRE_AFTER_MISSES", 2),
                               confidence_min_for_model_update=getattr(config, "TRACKING_CONFIDENCE_MIN_FOR_MODEL_UPDATE", 0.62),
                               confidence_min_accept=getattr(config, "TRACKING_CONFIDENCE_MIN_ACCEPT", 0.30),
                               wall_mode_radial_fraction=getattr(config, "WALL_MODE_RADIAL_FRACTION", 0.72),
                               wall_mode_hold_frames=getattr(config, "WALL_MODE_HOLD_FRAMES", 6),
                               wall_area_penalty_scale=getattr(config, "WALL_AREA_PENALTY_SCALE", 0.25),
                               motion_prediction_weight=getattr(config, "MOTION_PREDICTION_WEIGHT", 0.65),
                               appearance_update_alpha=getattr(config, "APPEARANCE_UPDATE_ALPHA", 0.06),
                               area_update_alpha=getattr(config, "AREA_UPDATE_ALPHA", 0.10),
                               wall_area_update_alpha=getattr(config, "WALL_AREA_UPDATE_ALPHA", 0.02),
                               frame_difference_weight=getattr(config, "FRAME_DIFFERENCE_WEIGHT", 1.25),
                               departure_motion_threshold=getattr(config, "DEPARTURE_MOTION_THRESHOLD", 4.0),
                               arrival_motion_threshold=getattr(config, "ARRIVAL_MOTION_THRESHOLD", 3.0),
                               moving_old_position_penalty=getattr(config, "MOVING_OLD_POSITION_PENALTY", 30.0))
    if manual_setup:
        entries = manual_setup.get("wells", {})
        for well in wells:
            name = str(well["well"])
            item = entries.get(name, {})
            tracker.initialize_manual(name, item.get("fly_bbox"), item.get("food_bboxes", []))
    return tracker


def annotate_tracking(image: np.ndarray, wells: list[dict[str, object]], results: list[TrackResult]) -> np.ndarray:
    """Draw exactly what the movement/sleep analysis is using.

    The moving box and crosshair are centered on result.x_px/result.y_px, which
    are the same accepted centroid coordinates written to fly_positions.csv and
    used to calculate frame-to-frame distance. Fixed food ROIs are hidden by
    default because they can otherwise look like stuck tracking boxes.
    """
    output = image.copy()
    results_by_well = {r.well: r for r in results}

    for well in wells:
        well_name = str(well["well"])
        well_center = (int(well["x"]), int(well["y"]))
        cv2.circle(output, well_center, int(well["radius"]), (150, 150, 150), 1)
        cv2.putText(output, well_name,
                    (well_center[0]-15, well_center[1]-int(well["radius"])-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        if bool(getattr(config, "SHOW_FOOD_BOXES_IN_TRACKING_OVERLAY", False)):
            for bx, by, bw, bh in well.get("food_bboxes", []):
                cv2.rectangle(output, (int(bx), int(by)), (int(bx+bw), int(by+bh)),
                              (0,165,255), 1)

        result = results_by_well.get(well_name)
        if result is not None and (result.x_px is None or result.y_px is None):
            # No position is logged or used for distance on a lost frame.
            # Mark the well clearly instead of leaving a stale fly box behind.
            x, y = well_center
            cv2.line(output, (x-7, y-7), (x+7, y+7), (0,0,255), 2)
            cv2.line(output, (x-7, y+7), (x+7, y-7), (0,0,255), 2)
            cv2.putText(output, f"{well_name} LOST", (x+10, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,255), 1, cv2.LINE_AA)

    for result in results:
        if result.x_px is None or result.y_px is None:
            continue

        # IMPORTANT: this center is the exact accepted tracking coordinate used
        # for movement calculations and written to fly_positions.csv.
        center = (int(round(result.x_px)), int(round(result.y_px)))
        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(result.well))
        color = (50 + (seed * 53) % 206, 50 + (seed * 97) % 206, 50 + (seed * 151) % 206)

        half = int(getattr(config, "TRACKING_BOX_HALF_SIZE_PX", 10))
        thickness = 2 if result.confidence >= 0.62 else 1
        cv2.rectangle(output,
                      (center[0] - half, center[1] - half),
                      (center[0] + half, center[1] + half),
                      color, thickness)

        # Crosshair = exact x/y used for distance. A contrasting center dot
        # remains visible even on a very dark fly or dark well wall.
        if bool(getattr(config, "DRAW_TRACKING_CROSSHAIR", True)):
            cross = int(getattr(config, "TRACKING_CROSSHAIR_HALF_SIZE_PX", 7))
            cv2.line(output, (center[0]-cross, center[1]), (center[0]+cross, center[1]), color, 2)
            cv2.line(output, (center[0], center[1]-cross), (center[0], center[1]+cross), color, 2)
            cv2.circle(output, center, 2, (255,255,255), -1)
            cv2.circle(output, center, 3, color, 1)
        else:
            cv2.circle(output, center, 4, color, -1)

        label = f"{result.well} {result.state} C={result.confidence:.2f}"
        if bool(getattr(config, "SHOW_STEP_DISTANCE_IN_OVERLAY", True)) and result.distance_px is not None:
            label += f" d={result.distance_px:.1f}px"
        label += f" ({center[0]},{center[1]})"
        cv2.putText(output, label, (center[0]+8, center[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
    return output


def fmt(value: float | int | None, digits: int = 3):
    if value is None:
        return ""
    return value if isinstance(value, int) else f"{value:.{digits}f}"


def result_row(result: TrackResult, *, timestamp_iso: str, elapsed: float, image_name: str,
               frame_number: int, registration_ok: bool, registration_score: float | None) -> dict[str, object]:
    score = "" if registration_score is None or not math.isfinite(registration_score) else f"{registration_score:.8f}"
    return {
        "timestamp_iso": timestamp_iso, "elapsed_seconds": f"{elapsed:.3f}", "image": image_name,
        "frame_number": frame_number, "well": result.well, "fly_name": result.fly_name,
        "fly_state": result.state, "valid_tracking": result.valid_tracking,
        "tracking_reason": result.reason, "detected": result.detected,
        "registration_succeeded": registration_ok, "registration_score": score,
        "x_px": fmt(result.x_px), "y_px": fmt(result.y_px),
        "well_relative_x_px": fmt(result.local_x_px), "well_relative_y_px": fmt(result.local_y_px),
        "area_px": fmt(result.area_px), "threshold_value": fmt(result.threshold_value),
        "bbox_x1": fmt(result.bbox_x1), "bbox_y1": fmt(result.bbox_y1),
        "bbox_x2": fmt(result.bbox_x2), "bbox_y2": fmt(result.bbox_y2),
        "raw_distance_px": fmt(result.raw_distance_px), "distance_px": fmt(result.distance_px),
        "rolling_distance_px": f"{result.rolling_distance_px:.3f}",
        "rolling_samples": result.rolling_samples,
        "immobile_duration_seconds": f"{result.immobile_duration_seconds:.3f}",
    }


def position_row(result: TrackResult, *, timestamp_iso: str, elapsed: float,
                 image_name: str, frame_number: int) -> dict[str, object]:
    return {
        "timestamp_iso": timestamp_iso,
        "elapsed_seconds": f"{elapsed:.3f}",
        "image": image_name,
        "frame_number": frame_number,
        "well": result.well,
        "fly_name": result.fly_name,
        "detected": result.detected,
        "valid_tracking": result.valid_tracking,
        "tracking_reason": result.reason,
        "tracking_confidence": f"{result.confidence:.3f}",
        "detection_stage": result.detection_stage,
        "wall_mode": result.wall_mode,
        "radial_fraction": fmt(result.radial_fraction),
        "arrival_change": fmt(result.arrival_change),
        "arrival_fraction": fmt(result.arrival_fraction),
        "departure_change": fmt(result.departure_change),
        "x_px": fmt(result.x_px),
        "y_px": fmt(result.y_px),
        "well_relative_x_px": fmt(result.local_x_px),
        "well_relative_y_px": fmt(result.local_y_px),
        "center_gray": fmt(result.center_gray),
        "component_median_gray": fmt(result.component_median_gray),
        "area_px": fmt(result.area_px),
        "raw_step_distance_px": fmt(result.raw_distance_px),
        "step_distance_px": fmt(result.distance_px),
        "cumulative_raw_distance_px": f"{result.cumulative_raw_distance_px:.3f}",
        "cumulative_distance_px": f"{result.cumulative_distance_px:.3f}",
    }


def update_tracks(tracker: SingleFlyTracker, wells, detections_by_well, timestamp_s: float,
                  registration_ok: bool) -> list[TrackResult]:
    return [tracker.update(str(w["well"]), detections_by_well.get(str(w["well"]), []),
                           timestamp_s, registration_ok) for w in wells]


def make_output_folders(base: Path) -> dict[str, Path]:
    analysis = base / "analysis"
    folders = {"analysis": analysis, "registered": analysis/"registered",
               "difference": analysis/"difference", "binary": analysis/"difference_thresholded",
               "masks": analysis/"fly_detection_masks", "tracking": analysis/"tracked_fly_overlays"}
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
    return folders


def should_save_diagnostic(frame_number: int) -> bool:
    every = int(config.storage_settings()["every_n_frames"])
    return every > 0 and frame_number % every == 0


def process_pair(reference: np.ndarray, current: np.ndarray):
    """Align every frame to the first calibrated frame so well coordinates stay fixed."""
    return register_pair(reference, current, motion_model=config.REGISTRATION_MOTION_MODEL,
                         blur_kernel=config.REGISTRATION_BLUR_KERNEL,
                         max_iterations=config.REGISTRATION_MAX_ITERATIONS,
                         epsilon=config.REGISTRATION_EPSILON,
                         difference_threshold=config.DIFFERENCE_THRESHOLD)


def analyze_offline(path: Path) -> None:
    source = path.expanduser().resolve()
    if (source / "images").is_dir():
        experiment = source
        images_folder = source / "images"
        plate_folder = source / "plate"
        output_base = source
    elif source.is_dir():
        images_folder = source
        output_base = source / "FlyStress_analysis"
        plate_folder = output_base / "plate"
    else:
        raise FileNotFoundError(source)
    images = collect_images(images_folder)
    metadata = load_capture_metadata(experiment / "capture_metadata.csv") if (source / "images").is_dir() else {}
    first = load_image(images[0])
    wells = resolve_wells(first, plate_folder)
    background_reference = prepare_background_reference(first)
    manual_setup = None
    if getattr(config, "USE_MANUAL_FLY_FOOD_SETUP", False):
        manual_setup = load_or_create_fly_food_setup(first, wells, plate_folder)
    tracker = create_tracker(wells, manual_setup)
    if manual_setup:
        for well in wells:
            item = manual_setup.get("wells", {}).get(str(well["well"]), {})
            well["food_bboxes"] = item.get("food_bboxes", [])
    folders = make_output_folders(output_base)
    views = LiveViewManager(first, config.DISPLAY_WIDTH, config.BACKGROUND_BINARY_THRESHOLD,
                            config.MINIMUM_MOTION_AREA_PX) if config.SHOW_WINDOWS else None
    start = 0.0
    with ResultsLogger(folders["analysis"] / "sleep_results.csv") as logger, \
            PositionLogger(folders["analysis"] / "fly_positions.csv") as position_logger:
        first_meta = metadata.get(images[0].name, {})
        first_elapsed = float(first_meta.get("elapsed_seconds", 0.0) or 0.0)
        detections, mask = detect_flies(first, wells, background_bgr=background_reference,
                                        tracking_hints=tracker.tracking_hints(),
                                        **detection_settings())
        results = update_tracks(tracker, wells, detections, first_elapsed, True)
        overlay = annotate_tracking(first, wells, results)
        for r in results:
            logger.write(result_row(r, timestamp_iso=first_meta.get("timestamp_iso", ""), elapsed=first_elapsed, image_name=images[0].name,
                                    frame_number=1, registration_ok=True, registration_score=None))
            position_logger.write(position_row(r, timestamp_iso=first_meta.get("timestamp_iso", ""),
                                               elapsed=first_elapsed, image_name=images[0].name,
                                               frame_number=1))
        reference = first
        previous_aligned = first
        for frame_number, image_path in enumerate(images[1:], start=2):
            current = load_image(image_path)
            item_meta = metadata.get(image_path.name, {})
            elapsed = float(item_meta.get("elapsed_seconds", (frame_number - 1) * config.CAPTURE_INTERVAL_SECONDS) or 0.0)
            registration = process_pair(reference, current)
            consecutive_difference = cv2.absdiff(
                cv2.cvtColor(previous_aligned, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(registration.aligned_bgr, cv2.COLOR_BGR2GRAY),
            )
            _, consecutive_binary = cv2.threshold(
                consecutive_difference, config.DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY
            )
            detections, mask = detect_flies(registration.aligned_bgr, wells,
                                            background_bgr=background_reference,
                                            previous_bgr=previous_aligned,
                                            tracking_hints=tracker.tracking_hints(),
                                            **detection_settings())
            results = update_tracks(tracker, wells, detections, elapsed, registration.succeeded)
            overlay = annotate_tracking(registration.aligned_bgr, wells, results)
            for r in results:
                logger.write(result_row(r, timestamp_iso=item_meta.get("timestamp_iso", ""), elapsed=elapsed, image_name=image_path.name,
                                        frame_number=frame_number, registration_ok=registration.succeeded,
                                        registration_score=registration.correlation))
                position_logger.write(position_row(r, timestamp_iso=item_meta.get("timestamp_iso", ""),
                                                   elapsed=elapsed, image_name=image_path.name,
                                                   frame_number=frame_number))
            if should_save_diagnostic(frame_number):
                settings = config.storage_settings()
                stem = image_path.stem
                if settings["save_registered"]: save_image(folders["registered"]/f"registered_{stem}.png", registration.aligned_bgr)
                if settings["save_difference"]: save_image(folders["difference"]/f"difference_{stem}.png", consecutive_difference)
                if settings["save_binary"]: save_image(folders["binary"]/f"binary_{stem}.png", consecutive_binary)
                if settings["save_detection_mask"]: save_image(folders["masks"]/f"mask_{stem}.png", mask)
                if settings["save_tracking_overlay"]: save_image(folders["tracking"]/f"tracked_{stem}.png", overlay)
            print(f"[{frame_number}/{len(images)}] {image_path.name} | registration={registration.succeeded}")
            if views and views.show(current, overlay, "Offline analysis - press q to stop",
                                    detection_mask=mask, wells=wells):
                break
            previous_aligned = registration.aligned_bgr
    if views: cv2.destroyAllWindows()
    print(f"Offline analysis saved in: {folders['analysis']}")


def check_disk_space(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / (1024 ** 3)
    if free < config.MIN_FREE_SPACE_GB:
        message = f"Low disk space: {free:.2f} GB remaining."
        if config.STOP_WHEN_DISK_LOW: raise RuntimeError(message)
        print("WARNING:", message)


def live_analysis() -> None:
    experiment = create_experiment_folder(Path(config.OUTPUT_ROOT).expanduser())
    images_folder, plate_folder = experiment/"images", experiment/"plate"
    images_folder.mkdir(); plate_folder.mkdir()
    folders = make_output_folders(experiment)
    cap = open_camera()
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera. Linux device={config.CAMERA_DEVICE}; Windows index={config.CAMERA_INDEX}.")
    metadata_path = experiment / "capture_metadata.csv"
    try:
        configure_camera(cap)
        print(f"Actual camera mode: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} at {cap.get(cv2.CAP_PROP_FPS):.1f} FPS")
        warmup_end = time.monotonic() + config.CAMERA_WARMUP_SECONDS
        while time.monotonic() < warmup_end: read_frame(cap)
        first = read_frame(cap)
        save_image(plate_folder/"background_image.png", first)
        wells = resolve_wells(first, plate_folder)
        background_reference = prepare_background_reference(first)
        tracker = create_tracker(wells)
        views = LiveViewManager(first, config.DISPLAY_WIDTH, config.BACKGROUND_BINARY_THRESHOLD,
                                config.MINIMUM_MOTION_AREA_PX) if config.SHOW_WINDOWS else None
        with metadata_path.open("w", newline="", encoding="utf-8", buffering=1) as meta_file, \
                ResultsLogger(folders["analysis"]/"sleep_results.csv") as logger, \
                PositionLogger(folders["analysis"]/"fly_positions.csv") as position_logger:
            meta = csv.DictWriter(meta_file, fieldnames=["image","image_number","timestamp_iso","elapsed_seconds","width_px","height_px"])
            meta.writeheader()
            start = time.monotonic(); next_sample = start; frame_number = 0
            reference = first.copy(); previous_aligned = None
            while True:
                frame = read_frame(cap)
                if views and previous_aligned is not None:
                    # Keep windows responsive between one-second analysis samples.
                    if views.show(frame, None, "Waiting for next one-second sample"):
                        break
                now = time.monotonic()
                if now < next_sample:
                    continue
                check_disk_space(Path(config.OUTPUT_ROOT).expanduser())
                frame_number += 1
                elapsed = now - start
                timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                filename = f"image{frame_number:06d}{config.IMAGE_EXTENSION}"
                if config.SAVE_CAPTURED_IMAGES: save_image(images_folder/filename, frame)
                meta.writerow(dict(image=filename, image_number=frame_number, timestamp_iso=timestamp,
                                   elapsed_seconds=f"{elapsed:.3f}", width_px=frame.shape[1], height_px=frame.shape[0]))
                meta_file.flush()
                if previous_aligned is None:
                    aligned = frame; registration_ok = True; score = None
                    detections, mask = detect_flies(aligned, wells,
                                                    background_bgr=background_reference,
                                                    tracking_hints=tracker.tracking_hints(),
                                                    **detection_settings())
                    results = update_tracks(tracker, wells, detections, elapsed, True)
                    overlay = annotate_tracking(aligned, wells, results)
                else:
                    registration = process_pair(reference, frame)
                    consecutive_difference = cv2.absdiff(
                        cv2.cvtColor(previous_aligned, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(registration.aligned_bgr, cv2.COLOR_BGR2GRAY),
                    )
                    _, consecutive_binary = cv2.threshold(
                        consecutive_difference, config.DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY
                    )
                    aligned = registration.aligned_bgr; registration_ok = registration.succeeded; score = registration.correlation
                    detections, mask = detect_flies(aligned, wells,
                                                    background_bgr=background_reference,
                                                    previous_bgr=previous_aligned,
                                                    tracking_hints=tracker.tracking_hints(),
                                                    **detection_settings())
                    results = update_tracks(tracker, wells, detections, elapsed, registration_ok)
                    overlay = annotate_tracking(aligned, wells, results)
                    if should_save_diagnostic(frame_number):
                        settings = config.storage_settings(); stem = Path(filename).stem
                        if settings["save_registered"]: save_image(folders["registered"]/f"registered_{stem}.png", aligned)
                        if settings["save_difference"]: save_image(folders["difference"]/f"difference_{stem}.png", consecutive_difference)
                        if settings["save_binary"]: save_image(folders["binary"]/f"binary_{stem}.png", consecutive_binary)
                        if settings["save_detection_mask"]: save_image(folders["masks"]/f"mask_{stem}.png", mask)
                        if settings["save_tracking_overlay"]: save_image(folders["tracking"]/f"tracked_{stem}.png", overlay)
                for r in results:
                    logger.write(result_row(r, timestamp_iso=timestamp, elapsed=elapsed, image_name=filename,
                                            frame_number=frame_number, registration_ok=registration_ok,
                                            registration_score=score))
                    position_logger.write(position_row(r, timestamp_iso=timestamp, elapsed=elapsed,
                                                       image_name=filename, frame_number=frame_number))
                print(f"Saved/analyzed {filename} | valid tracks={sum(r.valid_tracking for r in results)}/{len(results)}")
                if views and views.show(frame, overlay, "Live analysis - press q to stop",
                                        detection_mask=mask, wells=wells):
                    break
                previous_aligned = aligned.copy()
                next_sample = now + config.CAPTURE_INTERVAL_SECONDS
    except KeyboardInterrupt:
        print("Stopped with Ctrl+C.")
    finally:
        cap.release(); cv2.destroyAllWindows()
        print(f"Experiment saved in: {experiment}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="Existing experiment folder or directory of images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.path is None:
        live_analysis()
    else:
        analyze_offline(args.path)


if __name__ == "__main__":
    main()
