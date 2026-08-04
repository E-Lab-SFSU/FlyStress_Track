"""Detect a rectangular well grid while excluding the four mounting bolts.

Example:
    python detect_plate_wells.py frame.png --rows 4 --cols 8

Outputs:
    plate_wells_detected.png
    plate_wells.csv

Coordinates use OpenCV image coordinates:
    x increases left-to-right
    y increases top-to-bottom
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import config


@dataclass(frozen=True)
class Circle:
    x: int
    y: int
    radius: int

    @property
    def diameter(self) -> int:
        return 2 * self.radius


def _hough_circles(
        gray: np.ndarray,
        *,
        min_radius: int,
        max_radius: int,
        min_distance: int,
        param2: float,
) -> list[Circle]:
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_distance,
        param1=100,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    rounded = np.rint(circles[0]).astype(int)
    return [Circle(int(x), int(y), int(r)) for x, y, r in rounded]


def detect_mounting_bolts(gray: np.ndarray) -> list[Circle]:
    """Detect the four large mounting bolts near the image corners."""
    height, width = gray.shape
    candidates = _hough_circles(
        gray,
        min_radius=max(70, int(min(width, height) * 0.07)),
        max_radius=int(min(width, height) * 0.13),
        min_distance=int(min(width, height) * 0.35),
        param2=35,
    )

    # Assign at most one large circle to each image quadrant.
    quadrants: dict[str, list[Circle]] = {
        "top_left": [],
        "top_right": [],
        "bottom_left": [],
        "bottom_right": [],
    }
    for circle in candidates:
        vertical = "top" if circle.y < height / 2 else "bottom"
        horizontal = "left" if circle.x < width / 2 else "right"
        quadrants[f"{vertical}_{horizontal}"].append(circle)

    corners = {
        "top_left": (0, 0),
        "top_right": (width, 0),
        "bottom_left": (0, height),
        "bottom_right": (width, height),
    }

    bolts: list[Circle] = []
    for name, group in quadrants.items():
        if not group:
            continue
        corner_x, corner_y = corners[name]
        best = min(group, key=lambda c: (c.x - corner_x) ** 2 + (c.y - corner_y) ** 2)
        bolts.append(best)

    return bolts


def make_bolt_exclusion_mask(
        shape: tuple[int, int],
        bolts: list[Circle],
        padding: int = 35,
) -> np.ndarray:
    """Return a white mask with black exclusion regions around each bolt."""
    mask = np.full(shape, 255, dtype=np.uint8)
    for bolt in bolts:
        cv2.circle(mask, (bolt.x, bolt.y), bolt.radius + padding, 0, thickness=-1)
    return mask


def detect_well_candidates(gray: np.ndarray, exclusion_mask: np.ndarray) -> list[Circle]:
    """Detect wells using image-size-scaled parameters.

    Several Hough sensitivity levels are attempted so the same code can tolerate
    changes in camera resolution, distance, and zoom. The first result containing
    at least 32 candidates is preferred; otherwise the largest candidate set is
    returned for the caller's diagnostic error message.
    """
    height, width = gray.shape
    scale = min(width, height)

    # For the current plate geometry, a well radius is usually about 4-10% of
    # the shorter image dimension. These limits intentionally overlap to allow
    # moderate changes in zoom and camera placement.
    min_radius = max(12, int(scale * 0.035))
    max_radius = max(min_radius + 8, int(scale * 0.11))
    min_distance = max(30, int(scale * 0.075))

    best: list[Circle] = []
    for param2 in (30, 26, 22, 19, 16):
        candidates = _hough_circles(
            gray,
            min_radius=min_radius,
            max_radius=max_radius,
            min_distance=min_distance,
            param2=param2,
        )
        if len(candidates) > len(best):
            best = candidates
        if len(candidates) >= 32:
            return candidates

    return best


def _circle_is_excluded(circle: Circle, bolts: list[Circle], padding: int = 25) -> bool:
    for bolt in bolts:
        center_distance = np.hypot(circle.x - bolt.x, circle.y - bolt.y)
        if center_distance <= bolt.radius + padding:
            return True
    return False


def arrange_wells_in_grid(
        circles: list[Circle],
        rows: int,
        columns: int,
) -> list[dict[str, int | str]]:
    """Select and label wells as A1, A2, ... in row-major order."""
    expected = rows * columns
    if len(circles) < expected:
        raise RuntimeError(
            f"Only {len(circles)} well candidates were found; expected {expected}. "
            "Try reducing WELL_HOUGH_PARAM2 or adjusting the radius range."
        )

    # Cluster y coordinates into the requested number of rows.
    points = np.float32([[circle.y] for circle in circles])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _compactness, labels, centers = cv2.kmeans(
        points,
        rows,
        None,
        criteria,
        20,
        cv2.KMEANS_PP_CENTERS,
    )

    row_order = np.argsort(centers[:, 0])
    label_to_row = {int(label): row for row, label in enumerate(row_order)}

    grouped: list[list[Circle]] = [[] for _ in range(rows)]
    for circle, label in zip(circles, labels.ravel()):
        grouped[label_to_row[int(label)]].append(circle)

    selected: list[dict[str, int | str]] = []
    for row_index, row_circles in enumerate(grouped):
        row_circles.sort(key=lambda c: c.x)

        if len(row_circles) < columns:
            raise RuntimeError(
                f"Row {row_index + 1} has only {len(row_circles)} candidates; "
                f"expected {columns}."
            )

        # If there are extras, retain the subset whose spacing is most grid-like.
        if len(row_circles) > columns:
            best_subset = None
            best_score = float("inf")
            for start in range(len(row_circles) - columns + 1):
                subset = row_circles[start : start + columns]
                gaps = np.diff([c.x for c in subset])
                score = float(np.std(gaps))
                if score < best_score:
                    best_score = score
                    best_subset = subset
            row_circles = best_subset or row_circles[:columns]

        for column_index, circle in enumerate(row_circles):
            row_name = chr(ord("A") + row_index)
            selected.append(
                {
                    "well": f"{row_name}{column_index + 1}",
                    "row": row_index + 1,
                    "column": column_index + 1,
                    "x": circle.x,
                    "y": circle.y,
                    "radius": circle.radius,
                    "diameter": circle.diameter,
                }
            )

    if len(selected) != expected:
        raise RuntimeError(f"Selected {len(selected)} wells; expected {expected}.")

    return selected



def normalize_well_radii(
        wells: list[dict[str, int | str]],
        *,
        tolerance: int = 2,
) -> list[dict[str, int | str]]:
    """Replace inconsistent Hough radii with one robust plate-wide radius.

    HoughCircles can lock onto different visible rings caused by glare, liquid,
    or shadows. Because the physical wells are equal-size, estimate the common
    radius from the stronger detections and use it for every well.
    """
    if not wells:
        return wells

    radii = np.array([int(well["radius"]) for well in wells], dtype=np.int32)
    median_radius = float(np.median(radii))

    # Low radii are usually inner-ring detections. Favor detections at or above
    # the median, then take their median as the physical well radius.
    reliable = radii[radii >= median_radius]
    common_radius = int(round(float(np.median(reliable))))

    normalized: list[dict[str, int | str]] = []
    for well in wells:
        corrected = dict(well)
        detected_radius = int(well["radius"])
        corrected["detected_radius"] = detected_radius
        corrected["detected_diameter"] = detected_radius * 2

        if abs(detected_radius - common_radius) > tolerance:
            corrected["radius_corrected"] = "yes"
        else:
            corrected["radius_corrected"] = "no"

        corrected["radius"] = common_radius
        corrected["diameter"] = common_radius * 2
        normalized.append(corrected)

    return normalized

def annotate(
        image: np.ndarray,
        bolts: list[Circle],
        wells: list[dict[str, int | str]],
) -> np.ndarray:
    output = image.copy()

    for index, bolt in enumerate(sorted(bolts, key=lambda c: (c.y, c.x)), start=1):
        cv2.circle(output, (bolt.x, bolt.y), bolt.radius, (0, 0, 255), 5)
        cv2.circle(output, (bolt.x, bolt.y), 5, (0, 0, 255), -1)
        cv2.putText(
            output,
            f"BOLT {index} - EXCLUDED",
            (max(5, bolt.x - 115), max(30, bolt.y - bolt.radius - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for well in wells:
        x, y = int(well["x"]), int(well["y"])
        radius = int(well["radius"])
        label = str(well["well"])
        cv2.circle(output, (x, y), radius, (0, 255, 0), 3)
        cv2.circle(output, (x, y), 4, (255, 0, 0), -1)
        cv2.putText(
            output,
            label,
            (x - 22, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def save_csv(path: Path, wells: list[dict[str, int | str]]) -> None:
    fieldnames = ["well", "row", "column", "x", "y", "radius", "diameter", "detected_radius", "detected_diameter", "radius_corrected"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wells)


def detect_plate_wells(
        image: np.ndarray,
        rows: int = 4,
        columns: int = 8,
) -> tuple[list[Circle], list[dict[str, int | str]]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bolts = detect_mounting_bolts(gray)
    exclusion_mask = make_bolt_exclusion_mask(gray.shape, bolts)
    candidates = detect_well_candidates(gray, exclusion_mask)
    candidates = [c for c in candidates if not _circle_is_excluded(c, bolts, padding=5)]
    wells = arrange_wells_in_grid(candidates, rows, columns)
    wells = normalize_well_radii(wells)
    return bolts, wells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("plate_wells_detected.png"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("plate_wells.csv"),
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not open image: {args.image}")

    bolts, wells = detect_plate_wells(image, args.rows, args.cols)
    annotated = annotate(image, bolts, wells)

    if not cv2.imwrite(str(args.output_image), annotated):
        raise RuntimeError(f"Could not save image: {args.output_image}")
    save_csv(args.output_csv, wells)

    print(f"Detected {len(bolts)} mounting bolts (excluded).")
    print(f"Detected and labeled {len(wells)} wells.")
    for well in wells:
        print(
            f"{well['well']:>3}: center=({well['x']}, {well['y']}), "
            f"diameter={well['diameter']} px"
        )
    print(f"Annotated image: {args.output_image}")
    print(f"Coordinates CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
