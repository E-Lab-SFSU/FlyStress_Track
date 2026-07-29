"""
detect_wells.pt

Detect the wells center (x,y) + diameter manually using OpenCv and HOUGH

"""
from __future__ import annotations

from typing import Optional

import csv
from pathlib import Path

import cv2
import numpy as np

import config
from WellPlate.well_overlay import add_well_overlay
from WellPlate.well_plate import WellPlate, create_plate_from_config


def detect_well_grid(
        frame: np.ndarray,
        rows: int,
        columns: int,
        min_radius: int = 10,
        max_radius: int = 25,
) -> dict:
    # Attempt to detect well geometry using Hough circles (openCV function)

    if frame is None or frame.size == 0:
        raise ValueError("frame is empty")

    if min_radius <= 0:
        raise ValueError("min_radius must be greater than zero")

    if max_radius <= min_radius:
        raise ValueError(
            "max_radius must be greater than min_radius"
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    # uses Hough Gradient Method (cv2.HoughCircles) to detect wells in this case
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_radius * 1.5,
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    required_wells = rows * columns

    if circles is None:
        raise ValueError(
            f"No circles detected; expected {required_wells}"
        )

    detected = circles[0]

    if len(detected) < required_wells:
        raise ValueError(
            f"Only found {len(detected)} circles; "
            f"expected {required_wells}"
        )

    group_center_x = float(np.median(detected[:, 0]))
    group_center_y = float(np.median(detected[:, 1]))

    distances = np.hypot(
        detected[:, 0] - group_center_x,
        detected[:, 1] - group_center_y,
        )

    nearest_indices = np.argsort(distances)[:required_wells]
    detected = detected[nearest_indices]

    diameter_px = float(
        np.median(detected[:, 2]) * 2.0
    )

    sorted_by_y = detected[
        np.argsort(detected[:, 1])
    ]

    row_groups = np.array_split(
        sorted_by_y,
        rows,
    )

    grid: list[np.ndarray] = []

    for row_group in row_groups:
        row_sorted = row_group[
            np.argsort(row_group[:, 0])
        ]

        grid.append(row_sorted[:columns])

    return {
        "top_left": tuple(
            float(value)
            for value in grid[0][0][:2]
        ),
        "top_right": tuple(
            float(value)
            for value in grid[0][-1][:2]
        ),
        "bottom_left": tuple(
            float(value)
            for value in grid[-1][0][:2]
        ),
        "bottom_right": tuple(
            float(value)
            for value in grid[-1][-1][:2]
        ),
        "well_diameter_px": diameter_px,
    }


def create_plate_from_video_frame(
        frame: np.ndarray,
        rows: Optional[int] = None,
        columns: Optional[int] = None,
        well_margin_px: Optional[float] = None,
        min_radius: int = 10,
        max_radius: int = 25,
) -> WellPlate:
    #builds WellPlate from automatically detected circles (hough)"""

    actual_rows = (
        rows
        if rows is not None
        else config.WELL_ROWS
    )

    actual_columns = (
        columns
        if columns is not None
        else config.WELL_COLS
    )

    actual_margin = (
        well_margin_px
        if well_margin_px is not None
        else config.WELL_MARGIN
    )

    geometry = detect_well_grid(
        frame=frame,
        rows=actual_rows,
        columns=actual_columns,
        min_radius=min_radius,
        max_radius=max_radius,
    )

    save_calibration_csv(
        top_left=geometry["top_left"],
        top_right=geometry["top_right"],
        bottom_left=geometry["bottom_left"],
        bottom_right=geometry["bottom_right"],
        well_diameter_px=geometry["well_diameter_px"],
    )


    return WellPlate(
        rows=actual_rows,
        columns=actual_columns,
        top_left=geometry["top_left"],
        top_right=geometry["top_right"],
        bottom_left=geometry["bottom_left"],
        bottom_right=geometry["bottom_right"],
        well_diameter_px=geometry["well_diameter_px"],
        well_margin_px=actual_margin,
    )

def detect_plate_wells_adjusted(
        reference_frame: np.ndarray,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        base_plate: Optional[WellPlate] = None,
        rows: Optional[int] = None,
        columns: Optional[int] = None,
        well_margin_px: Optional[float] = None,
        min_radius: int = 10,
        max_radius: int = 25,
) -> WellPlate:
    """
    Return well positions adjusted for a small plate/camera movement.

    Two modes:

    1. base_plate is given (typical per-frame use):
       Skip Hough detection entirely and just translate the known
       well grid by (shift_x, shift_y). This is the cheap path meant
       to be called once per new image, after phase-correlation
       against the reference image has produced (shift_x, shift_y).

    2. base_plate is None (first frame / calibration):
       Run Hough-circle well detection on reference_frame to build
       the initial WellPlate, then translate it by (shift_x, shift_y)
       (normally 0, 0 for the very first frame).

    Parameters
    ----------
    reference_frame : np.ndarray
        Only used in calibration mode (base_plate=None).
    shift_x, shift_y : float
        The plate's estimated movement for this frame, in pixels,
        e.g. as returned by cv2.phaseCorrelate (see align_and_diff.py).
    base_plate : WellPlate, optional
        Previously detected/created plate to translate instead of
        re-detecting.

    Returns
    -------
    WellPlate
        Well grid positioned for this frame.
    """

    if base_plate is not None:
        return base_plate.shifted(shift_x, shift_y)

    plate = create_plate_from_video_frame(
        frame=reference_frame,
        rows=rows,
        columns=columns,
        well_margin_px=well_margin_px,
        min_radius=min_radius,
        max_radius=max_radius,
    )

    if shift_x or shift_y:
        plate = plate.shifted(shift_x, shift_y)

    return plate


def save_calibration_csv(
        top_left: tuple[float, float],
        top_right: tuple[float, float],
        bottom_left: tuple[float, float],
        bottom_right: tuple[float, float],
        well_diameter_px: float,
        csv_path: str = "well_calibration.csv",
) -> None:
    # Saves well dimensions to a CSV file.
    # The file stores the four corner-well centers and the average well diameter in pixels.

    output_path = Path(csv_path)

    with output_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "name",
                "x",
                "y",
                "diameter_px",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "name": "WELL_TL",
                "x": top_left[0],
                "y": top_left[1],
                "diameter_px": "",
            }
        )

        writer.writerow(
            {
                "name": "WELL_TR",
                "x": top_right[0],
                "y": top_right[1],
                "diameter_px": "",
            }
        )

        writer.writerow(
            {
                "name": "WELL_BL",
                "x": bottom_left[0],
                "y": bottom_left[1],
                "diameter_px": "",
            }
        )

        writer.writerow(
            {
                "name": "WELL_BR",
                "x": bottom_right[0],
                "y": bottom_right[1],
                "diameter_px": "",
            }
        )

        writer.writerow(
            {
                "name": "WELL_DIAMETER",
                "x": "",
                "y": "",
                "diameter_px": well_diameter_px,
            }
        )

    print(f"Saved well calibration to: {output_path.resolve()}")

def test_plate_on_video(
        video_path: str,
) -> None:
    # Shows the configured well overlay on the first video frame.

    plate = create_plate_from_config()
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    success, frame = capture.read()
    capture.release()

    if not success or frame is None:
        raise RuntimeError(
            "Could not read the first video frame"
        )

    display_frame = add_well_overlay(
        frame=frame,
        plate=plate,
        draw_assignment_boundary=True,
    )

    print(f"Generated {plate.total_wells} wells")
    print(
        f"Well diameter: "
        f"{plate.well_diameter_px:.2f} px"
    )
    print(
        f"Assignment radius: "
        f"{plate.assignment_radius_px:.2f} px"
    )

    cv2.imshow(
        "Well Plate Calibration Test",
        display_frame,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_plate_on_video("C:/Users/chana/Videos/Screen Recordings/Fly_Test_Vid.mp4")     # <-- Insert path to video file