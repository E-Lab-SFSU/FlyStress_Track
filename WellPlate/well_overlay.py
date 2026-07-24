"""
well_overlap.py

Draws visual of well plate as calculated. Adds labels and 'walls'

Can be used to determine if parameters are correct to actual well plate

"""
from __future__ import annotations

import cv2
import numpy as np

import config
from WellPlate.well_plate import WellPlate


def draw_wells(
        frame: np.ndarray,
        plate: WellPlate,
        draw_labels: bool = True,
        draw_assignment_boundary: bool = False,
) -> np.ndarray:

    if frame is None or frame.size == 0:
        raise ValueError("frame is empty")

    output = frame.copy()

    for well in plate.wells:
        center = (
            int(round(well.center_x)),
            int(round(well.center_y)),
        )

        physical_radius = int(round(well.radius_px))
        assignment_radius = int(
            round(plate.assignment_radius_px)
        )

        cv2.circle(
            output,
            center,
            physical_radius,
            (0, 255, 0),
            1,
        )

        cv2.circle(
            output,
            center,
            2,
            (0, 0, 255),
            -1,
        )

        if draw_assignment_boundary:
            cv2.circle(
                output,
                center,
                assignment_radius,
                (255, 255, 0),
                1,
            )

        if draw_labels:
            cv2.putText(
                output,
                well.label,
                (
                    center[0] - 10,
                    center[1] - physical_radius - 3,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return output


def add_well_overlay(
        frame: np.ndarray,
        plate: WellPlate,
        draw_assignment_boundary: bool = False,
) -> np.ndarray:
    """Apply the configured well overlay to a display frame."""

    if frame is None or frame.size == 0:
        raise ValueError("frame is empty")

    if not config.SHOW_OVERLAY:
        return frame.copy()

    return draw_wells(
        frame=frame,
        plate=plate,
        draw_labels=config.SHOW_LABELS,
        draw_assignment_boundary=draw_assignment_boundary,
    )