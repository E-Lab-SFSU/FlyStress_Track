"""Run 32-well, 3-4 flies-per-well sleep analysis on one experiment folder.

Example:
    python run_analysis.py ~/FS_IMG/exp00001
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import config
from sleep_analysis.fly_detection import detect_all_flies
from sleep_analysis.movement import calculate_distance
from sleep_analysis.multi_fly_tracker import PerWellMultiFlyTracker, draw_tracks
from sleep_analysis.registration import align_to_previous, difference_image, preprocess
from sleep_analysis.results_logger import ResultsLogger
from sleep_analysis.rolling_sleep import RollingSleepTracker



def load_wells(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        wells = list(csv.DictReader(file))
    for well in wells:
        for key in ("row", "column", "x", "y", "radius", "diameter"):
            well[key] = int(well[key])
    wells.sort(key=lambda item: (int(item["row"]), int(item["column"])))
    if len(wells) != config.NUMBER_OF_WELLS:
        raise RuntimeError(f"Found {len(wells)} wells; expected {config.NUMBER_OF_WELLS}.")
    return wells



def image_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob(f"image*{config.IMAGE_EXTENSION}"))
    if len(files) < 2:
        raise RuntimeError("The experiment folder must contain at least two images.")
    return files



def load_metadata(folder: Path) -> dict[str, dict[str, str]]:
    path = folder / "capture_metadata.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {row["image"]: row for row in csv.DictReader(file)}



def make_diagnostic_folders(folder: Path) -> dict[str, Path]:
    folders: dict[str, Path] = {}
    for enabled, name in (
            (config.SAVE_ALIGNED_IMAGES, "aligned"),
            (config.SAVE_DIFFERENCE_IMAGES, "differences"),
            (config.SAVE_DETECTION_OVERLAYS, "detections"),
    ):
        if enabled:
            path = folder / "diagnostics" / name
            path.mkdir(parents=True, exist_ok=True)
            folders[name] = path
    return folders



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_folder", type=Path)
    args = parser.parse_args()

    folder = args.experiment_folder.expanduser().resolve()
    wells = load_wells(folder / "plate_wells.csv")
    images = image_files(folder)
    metadata = load_metadata(folder)
    diagnostics = make_diagnostic_folders(folder)
    well_names = [str(well["well"]) for well in wells]

    first_color = cv2.imread(str(images[0]))
    if first_color is None:
        raise FileNotFoundError(images[0])
    previous_gray = preprocess(first_color)

    tracker = PerWellMultiFlyTracker(well_names)
    zero_difference = np.zeros_like(previous_gray)
    initial_detections = detect_all_flies(previous_gray, zero_difference, wells)
    for well_name in well_names:
        tracker.update(well_name, initial_detections[well_name])

    fly_ids = [
        f"{well}_Fly{index}"
        for well in well_names
        for index in range(1, config.MAX_FLIES_PER_WELL + 1)
    ]
    sleep_tracker = RollingSleepTracker(
        fly_ids,
        config.ROLLING_WINDOW_SECONDS,
        config.AWAKE_THRESHOLD_PX,
    )

    results_path = folder / "sleep_results.csv"
    with ResultsLogger(results_path) as logger:
        for frame_number, path in enumerate(images[1:], start=2):
            color = cv2.imread(str(path))
            if color is None:
                print(f"Skipping unreadable image: {path.name}")
                continue

            current_gray = preprocess(color)
            aligned_gray, _warp, registration_score = align_to_previous(previous_gray, current_gray)
            difference = difference_image(previous_gray, aligned_gray)
            detections_by_well = detect_all_flies(aligned_gray, difference, wells)

            for well_name in well_names:
                old_positions = {
                    track.fly_id: track.position
                    for track in tracker.tracks[well_name]
                }
                tracks = tracker.update(well_name, detections_by_well[well_name])

                for track in tracks:
                    current_position = track.position if track.detected else None
                    distance = calculate_distance(
                        old_positions[track.fly_id],
                        current_position,
                        config.JITTER_THRESHOLD_PX,
                    )
                    rolling_distance, state = sleep_tracker.update(track.fly_id, distance)
                    meta = metadata.get(path.name, {})
                    elapsed = meta.get(
                        "elapsed_seconds",
                        f"{(frame_number - 1) * config.CAPTURE_INTERVAL_SECONDS:.3f}",
                    )

                    logger.write({
                        "timestamp": meta.get("timestamp_iso", ""),
                        "elapsed_seconds": elapsed,
                        "image": path.name,
                        "frame_number": frame_number,
                        "well": well_name,
                        "fly_name": track.fly_id,
                        "fly_state": state,
                        "x": "" if track.position is None else f"{track.x:.3f}",
                        "y": "" if track.position is None else f"{track.y:.3f}",
                        "detected": track.detected,
                        "missed_frames": track.missed_frames,
                        "detection_area_px": f"{track.area:.3f}",
                        "moving_area_px": f"{track.moving_area:.3f}",
                        "distance_px": f"{distance:.3f}",
                        "rolling_distance_px": f"{rolling_distance:.3f}",
                        "registration_score": f"{registration_score:.6f}",
                    })

            if frame_number % max(1, config.DIAGNOSTIC_EVERY_N_FRAMES) == 0:
                stem = path.stem
                if "aligned" in diagnostics:
                    cv2.imwrite(str(diagnostics["aligned"] / f"{stem}.png"), aligned_gray)
                if "differences" in diagnostics:
                    cv2.imwrite(str(diagnostics["differences"] / f"{stem}.png"), difference)
                if "detections" in diagnostics:
                    overlay = draw_tracks(aligned_gray, wells, tracker.tracks)
                    cv2.imwrite(str(diagnostics["detections"] / f"{stem}.png"), overlay)

            previous_gray = aligned_gray
            print(f"Processed {path.name}")

    print(f"Finished. Results: {results_path}")
    print("Important: inspect diagnostics/detections before trusting individual fly IDs.")


if __name__ == "__main__":
    main()
