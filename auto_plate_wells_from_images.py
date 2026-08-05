#!/usr/bin/env python3
"""
Automatic well detection from a directory of image frames.

Purpose
-------
Build a reference image from a directory of grayscale or color frames,
detect circular wells automatically, fit them to a regular rows x cols grid,
and write a CSV in the same format as the existing manual plate_wells.csv:

    well,row,column,x,y,radius,diameter

Designed to be easy to integrate into a larger program. The main function is:

    auto_generate_plate_wells_csv(image_dir, output_csv, rows=4, cols=8, debug_dir=None)

Dependencies
------------
- Python 3
- OpenCV (cv2)
- NumPy

Typical use inside another program
----------------------------------
    auto_generate_plate_wells_csv(
        image_dir="test_frames",
        output_csv="test_frames/FlyStress_analysis/plate/plate_wells.csv",
        rows=4,
        cols=8,
        debug_dir="test_frames/FlyStress_analysis/plate_debug",
    )

Notes
-----
1. The reference image is built from all frames in the directory.
2. By default, the code uses a normalized SUM of frames plus CLAHE, because
   that made the wells stand out clearly in testing.
3. The circle detector may find too many or too few circles. The grid-fitting
   step forces a complete rows x cols plate layout and infers any missing wells.
"""


from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------
# User-adjustable defaults
# ---------------------------------------------------------------------

DEFAULT_ROWS = 4
DEFAULT_COLS = 8
DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Reference-image options
DEFAULT_REFERENCE_MODE = "sum"      # "sum", "mean", or "median"
DEFAULT_APPLY_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Hough circle-detection options
HOUGH_DP = 1.2
HOUGH_PARAM1 = 100
HOUGH_PARAM2_CANDIDATES = [18, 20, 22, 24, 26, 28, 30, 32]

# Radius search.
# If None, they are estimated from the image width and expected grid layout.
DEFAULT_MIN_RADIUS = None
DEFAULT_MAX_RADIUS = None

# Debug output
SAVE_DEBUG_IMAGES = True


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def list_image_files(image_dir: str | Path,
                     extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS) -> List[Path]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    files = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in tuple(ext.lower() for ext in extensions)
    )
    if not files:
        raise FileNotFoundError(f"No image files found in: {image_dir}")
    return files


def load_grayscale_frames(image_paths: List[Path]) -> np.ndarray:
    frames = []
    shape = None

    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            raise ValueError(
                f"Image size mismatch. "
                f"Expected {shape}, found {img.shape} in {path.name}"
            )

        frames.append(img.astype(np.float32))

    if not frames:
        raise RuntimeError("No readable images were loaded.")

    return np.stack(frames, axis=0)


def build_reference_image(stack: np.ndarray,
                          mode: str = DEFAULT_REFERENCE_MODE,
                          apply_clahe: bool = DEFAULT_APPLY_CLAHE) -> np.ndarray:
    """
    Build a single reference image from a stack of frames.

    mode:
        "sum"    -> sum all frames, then normalize to 0..255
        "mean"   -> mean of frames
        "median" -> median of frames
    """
    mode = mode.lower().strip()

    if mode == "sum":
        ref = stack.sum(axis=0)
        ref = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX)
        ref = ref.astype(np.uint8)
    elif mode == "mean":
        ref = np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)
    elif mode == "median":
        ref = np.median(stack, axis=0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown reference mode: {mode}")

    if apply_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )
        ref = clahe.apply(ref)

    return ref


def estimate_radius_bounds(image_shape: Tuple[int, int],
                           rows: int,
                           cols: int,
                           min_radius: int | None,
                           max_radius: int | None) -> Tuple[int, int]:
    """
    Estimate reasonable radius bounds if they are not supplied.

    This uses the image width and expected column count to estimate spacing.
    """
    h, w = image_shape
    if min_radius is not None and max_radius is not None:
        return int(min_radius), int(max_radius)

    estimated_spacing = w / (cols + 1)
    estimated_radius = estimated_spacing * 0.35

    rmin = int(max(8, round(estimated_radius * 0.70))) if min_radius is None else int(min_radius)
    rmax = int(max(rmin + 4, round(estimated_radius * 1.35))) if max_radius is None else int(max_radius)
    return rmin, rmax


def detect_circles_hough(reference_image: np.ndarray,
                         rows: int,
                         cols: int,
                         min_radius: int | None = DEFAULT_MIN_RADIUS,
                         max_radius: int | None = DEFAULT_MAX_RADIUS) -> np.ndarray:
    """
    Detect circles with HoughCircles.
    Returns an array of shape (N, 3) with columns [x, y, radius].
    """
    target_count = rows * cols
    rmin, rmax = estimate_radius_bounds(reference_image.shape, rows, cols, min_radius, max_radius)

    blur = cv2.medianBlur(reference_image, 5)
    min_dist = max(10, int(round(1.3 * ((rmin + rmax) / 2.0) * 2.0)))

    best_circles = None
    best_score = None

    for param2 in HOUGH_PARAM2_CANDIDATES:
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=min_dist,
            param1=HOUGH_PARAM1,
            param2=param2,
            minRadius=rmin,
            maxRadius=rmax,
        )

        count = 0 if circles is None else circles.shape[1]
        score = abs(count - target_count)

        if best_score is None or score < best_score:
            best_score = score
            best_circles = circles

    if best_circles is None:
        return np.empty((0, 3), dtype=np.float32)

    circles = np.round(best_circles[0]).astype(np.float32)
    return circles


def kmeans_1d(values: np.ndarray,
              k: int,
              max_iterations: int = 100) -> np.ndarray:
    """
    Simple 1D k-means for row and column center estimation.
    No external dependencies required.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        raise ValueError("Cannot cluster an empty set of values.")
    if k <= 0:
        raise ValueError("k must be positive.")

    if len(values) < k:
        # If fewer points than clusters, spread centers across the range.
        vmin = float(values.min())
        vmax = float(values.max())
        return np.linspace(vmin, vmax, k, dtype=np.float32)

    centers = np.linspace(values.min(), values.max(), k, dtype=np.float32)

    for _ in range(max_iterations):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = distances.argmin(axis=1)

        new_centers = centers.copy()
        for i in range(k):
            cluster_values = values[labels == i]
            if len(cluster_values) > 0:
                new_centers[i] = cluster_values.mean()

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return np.sort(centers)


def assign_to_grid(circles: np.ndarray,
                   rows: int,
                   cols: int,
                   image_shape: Tuple[int, int]) -> List[dict]:
    """
    Force detected circles into a regular rows x cols grid.

    Steps:
    1. Estimate row centers from circle y-values and column centers from x-values.
    2. Assign each detected circle to its nearest grid cell.
    3. If multiple circles land in one cell, keep the one nearest the cell center.
    4. If a cell has no detected circle, infer its center from the row/column
       centers and assign a median radius.

    Returns a list of dictionaries with:
        row_index, col_index, x, y, radius
    """
    h, w = image_shape

    if len(circles) == 0:
        # If nothing detected, synthesize a full grid across the image.
        row_centers = np.linspace(h * 0.2, h * 0.8, rows, dtype=np.float32)
        col_centers = np.linspace(w * 0.15, w * 0.85, cols, dtype=np.float32)
        median_radius = int(round(min(h / (rows * 3.0), w / (cols * 3.0))))
        detections = []
    else:
        row_centers = kmeans_1d(circles[:, 1], rows)
        col_centers = kmeans_1d(circles[:, 0], cols)
        median_radius = int(round(np.median(circles[:, 2])))
        detections = circles

    # Grid cells indexed by (row, col)
    cell_best = {}

    for x, y, r in detections:
        row_idx = int(np.argmin(np.abs(row_centers - y)))
        col_idx = int(np.argmin(np.abs(col_centers - x)))
        cell_center_x = float(col_centers[col_idx])
        cell_center_y = float(row_centers[row_idx])
        cell_distance = float(np.hypot(x - cell_center_x, y - cell_center_y))

        key = (row_idx, col_idx)
        candidate = {
            "row_index": row_idx,
            "col_index": col_idx,
            "x": int(round(float(x))),
            "y": int(round(float(y))),
            "radius": int(round(float(r))),
            "distance_to_cell_center": cell_distance,
            "inferred": False,
        }

        if key not in cell_best or cell_distance < cell_best[key]["distance_to_cell_center"]:
            cell_best[key] = candidate

    # Fill all required cells, inferring any missing wells.
    grid = []
    for r in range(rows):
        for c in range(cols):
            key = (r, c)
            if key in cell_best:
                grid.append(cell_best[key])
            else:
                grid.append({
                    "row_index": r,
                    "col_index": c,
                    "x": int(round(float(col_centers[c]))),
                    "y": int(round(float(row_centers[r]))),
                    "radius": median_radius,
                    "distance_to_cell_center": 0.0,
                    "inferred": True,
                })

    # Sort top-to-bottom, then left-to-right.
    grid.sort(key=lambda d: (d["row_index"], d["col_index"]))
    return grid


def label_for_well(row_index: int, col_index: int) -> str:
    """
    Convert zero-based row/column into A1, A2, ... D8.
    """
    return f"{chr(ord('A') + row_index)}{col_index + 1}"


def write_plate_wells_csv(wells: List[dict], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["well", "row", "column", "x", "y", "radius", "diameter"])

        for well in wells:
            row_num = well["row_index"] + 1
            col_num = well["col_index"] + 1
            radius = int(well["radius"])
            diameter = int(radius * 2)
            writer.writerow([
                label_for_well(well["row_index"], well["col_index"]),
                row_num,
                col_num,
                int(well["x"]),
                int(well["y"]),
                radius,
                diameter,
            ])

    return output_csv


def save_debug_outputs(reference_image: np.ndarray,
                       circles: np.ndarray,
                       wells: List[dict],
                       debug_dir: str | Path) -> None:
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(debug_dir / "reference_image.png"), reference_image)

    # Raw circles detected by Hough
    hough_overlay = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    for x, y, r in circles:
        cv2.circle(hough_overlay, (int(round(x)), int(round(y))), int(round(r)), (255, 0, 0), 2)
        cv2.circle(hough_overlay, (int(round(x)), int(round(y))), 2, (255, 0, 0), -1)
    cv2.imwrite(str(debug_dir / "hough_circles_overlay.png"), hough_overlay)

    # Final fitted grid
    final_overlay = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    for well in wells:
        x = int(well["x"])
        y = int(well["y"])
        r = int(well["radius"])
        color = (0, 255, 0) if not well["inferred"] else (0, 255, 255)
        cv2.circle(final_overlay, (x, y), r, color, 2)
        cv2.circle(final_overlay, (x, y), 2, color, -1)
        cv2.putText(
            final_overlay,
            label_for_well(well["row_index"], well["col_index"]),
            (x - 18, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(debug_dir / "final_plate_wells_overlay.png"), final_overlay)


# ---------------------------------------------------------------------
# Main function intended for integration
# ---------------------------------------------------------------------

def auto_generate_plate_wells_csv(image_dir: str | Path,
                                  output_csv: str | Path,
                                  rows: int = DEFAULT_ROWS,
                                  cols: int = DEFAULT_COLS,
                                  debug_dir: str | Path | None = None,
                                  reference_mode: str = DEFAULT_REFERENCE_MODE,
                                  apply_clahe: bool = DEFAULT_APPLY_CLAHE,
                                  min_radius: int | None = DEFAULT_MIN_RADIUS,
                                  max_radius: int | None = DEFAULT_MAX_RADIUS) -> List[dict]:
    """
    Build a reference image from a directory of frames, detect wells,
    fit them to a regular grid, and write a plate_wells.csv.

    Parameters
    ----------
    image_dir:
        Directory containing frame images.
    output_csv:
        Path to the CSV file to write.
    rows, cols:
        Plate layout, for example 4 x 8.
    debug_dir:
        Optional directory for saving debug images.
    reference_mode:
        "sum", "mean", or "median". Default is "sum".
    apply_clahe:
        Whether to apply CLAHE to the reference image.
    min_radius, max_radius:
        Optional circle-radius limits for Hough detection.

    Returns
    -------
    wells:
        A list of dictionaries. Each dictionary contains:
            row_index, col_index, x, y, radius, inferred, ...
    """
    image_paths = list_image_files(image_dir)
    stack = load_grayscale_frames(image_paths)
    reference_image = build_reference_image(
        stack,
        mode=reference_mode,
        apply_clahe=apply_clahe
    )

    circles = detect_circles_hough(
        reference_image,
        rows=rows,
        cols=cols,
        min_radius=min_radius,
        max_radius=max_radius
    )

    wells = assign_to_grid(
        circles,
        rows=rows,
        cols=cols,
        image_shape=reference_image.shape
    )

    write_plate_wells_csv(wells, output_csv)

    if debug_dir is not None and SAVE_DEBUG_IMAGES:
        save_debug_outputs(reference_image, circles, wells, debug_dir)

    return wells


# ---------------------------------------------------------------------
# Example standalone usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Edit these three paths when running as a standalone script.
    # -----------------------------------------------------------------
    IMAGE_DIR = r"test_frames"
    OUTPUT_CSV = r"test_frames/FlyStress_analysis/plate/plate_wells_auto.csv"
    DEBUG_DIR = r"test_frames/FlyStress_analysis/plate_debug"

    wells = auto_generate_plate_wells_csv(
        image_dir=IMAGE_DIR,
        output_csv=OUTPUT_CSV,
        rows=4,
        cols=8,
        debug_dir=DEBUG_DIR,
        reference_mode="sum",
        apply_clahe=True,
    )

    inferred_count = sum(1 for w in wells if w.get("inferred"))
    print(f"Generated {len(wells)} wells.")
    print(f"Inferred from missing detections: {inferred_count}")
    print(f"CSV written to: {OUTPUT_CSV}")
    print(f"Debug images written to: {DEBUG_DIR}")
