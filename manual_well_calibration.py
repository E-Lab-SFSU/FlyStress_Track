"""Interactive manual drag-circle calibration for plate wells.

Controls
--------
Left mouse: click at the well center, drag to the edge, and release.
Right mouse / U: undo the last accepted well.
R: reset all wells.
S: save after all expected wells are selected.
Q / Esc: cancel.

The circle is shown while dragging. Invalid radii are reported on screen instead
of being silently discarded.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import cv2

import config


_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_CYAN = (255, 255, 0)


def _well_name(index: int) -> str:
    row = index // config.PLATE_COLUMNS
    column = index % config.PLATE_COLUMNS
    return f"{chr(65 + row)}{column + 1}"


def _draw_text_with_outline(
        image,
        text: str,
        origin: tuple[int, int],
        *,
        scale: float = 0.55,
        color: tuple[int, int, int] = _WHITE,
        thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _BLACK,
        thickness + 3,
        cv2.LINE_AA,
        )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _load_existing_wells(output_csv: Path) -> list[tuple[int, int, int]]:
    if not output_csv.is_file():
        return []

    wells: list[tuple[int, int, int]] = []
    try:
        with output_csv.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                wells.append(
                    (
                        int(round(float(row["x"]))),
                        int(round(float(row["y"]))),
                        int(round(float(row["radius"]))),
                    )
                )
    except (OSError, KeyError, TypeError, ValueError):
        return []

    if len(wells) > config.EXPECTED_WELLS:
        return wells[: config.EXPECTED_WELLS]
    return wells


def _save_wells(output_csv: Path, wells: list[tuple[int, int, int]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["well", "row", "column", "x", "y", "radius", "diameter"],
        )
        writer.writeheader()
        for index, (x, y, radius) in enumerate(wells):
            writer.writerow(
                {
                    "well": _well_name(index),
                    "row": index // config.PLATE_COLUMNS + 1,
                    "column": index % config.PLATE_COLUMNS + 1,
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "diameter": 2 * radius,
                }
            )


def calibrate(
        image_path: Path,
        output_csv: Path,
        *,
        load_existing: bool = False,
        **_,
) -> bool:
    """Interactively select all plate wells and save them to ``output_csv``."""
    image_path = Path(image_path)
    output_csv = Path(output_csv)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read calibration image: {image_path}")

    scale = min(
        1.0,
        config.MANUAL_CALIBRATION_DISPLAY_MAX_WIDTH / image.shape[1],
        config.MANUAL_CALIBRATION_DISPLAY_MAX_HEIGHT / image.shape[0],
        )
    display_base = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image.copy()
    )

    wells = _load_existing_wells(output_csv) if load_existing else []

    drag_start_display: Optional[tuple[int, int]] = None
    drag_current_display: Optional[tuple[int, int]] = None
    status_message = "Click the center of A1, drag to its edge, then release."
    status_color = _CYAN

    min_radius = int(config.MANUAL_CALIBRATION_MIN_RADIUS)
    max_radius = int(config.MANUAL_CALIBRATION_MAX_RADIUS)

    def original_point(display_x: int, display_y: int) -> tuple[int, int]:
        return (
            int(round(display_x / scale)),
            int(round(display_y / scale)),
        )

    def callback(event, x, y, flags, param):
        nonlocal drag_start_display, drag_current_display, status_message, status_color

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(wells) >= config.EXPECTED_WELLS:
                status_message = "All wells are selected. Press S to save or U to undo."
                status_color = _YELLOW
                return
            drag_start_display = (x, y)
            drag_current_display = (x, y)
            status_message = f"Drawing {_well_name(len(wells))}..."
            status_color = _CYAN

        elif event == cv2.EVENT_MOUSEMOVE and drag_start_display is not None:
            drag_current_display = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and drag_start_display is not None:
            drag_current_display = (x, y)
            center_original = original_point(*drag_start_display)
            edge_original = original_point(x, y)
            radius = int(
                round(
                    math.hypot(
                        edge_original[0] - center_original[0],
                        edge_original[1] - center_original[1],
                        )
                )
            )

            if min_radius <= radius <= max_radius:
                wells.append((center_original[0], center_original[1], radius))
                status_message = (
                    f"Added {_well_name(len(wells) - 1)} with radius {radius}px. "
                    f"Next: {_well_name(len(wells))}."
                    if len(wells) < config.EXPECTED_WELLS
                    else "All wells selected. Press S to save."
                )
                status_color = _GREEN
            else:
                status_message = (
                    f"Circle rejected: radius {radius}px. "
                    f"Allowed range is {min_radius}-{max_radius}px. Try again."
                )
                status_color = _RED

            drag_start_display = None
            drag_current_display = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            if wells:
                removed_name = _well_name(len(wells) - 1)
                wells.pop()
                status_message = f"Removed {removed_name}. Draw it again."
                status_color = _YELLOW

    window_name = config.MANUAL_CALIBRATION_WINDOW_NAME
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_base.shape[1], display_base.shape[0])
    cv2.setMouseCallback(window_name, callback)

    saved = False
    final_display = display_base.copy()

    while True:
        display = display_base.copy()

        # Draw all accepted wells.
        for index, (x, y, radius) in enumerate(wells):
            center_display = (int(round(x * scale)), int(round(y * scale)))
            radius_display = max(1, int(round(radius * scale)))
            cv2.circle(display, center_display, radius_display, _GREEN, 3, cv2.LINE_AA)
            cv2.circle(display, center_display, 3, _YELLOW, -1, cv2.LINE_AA)
            _draw_text_with_outline(
                display,
                _well_name(index),
                (center_display[0] + 8, center_display[1] - 8),
                scale=0.6,
                color=_YELLOW,
                thickness=2,
            )

        # Draw a live preview while the mouse is being dragged.
        if drag_start_display is not None and drag_current_display is not None:
            preview_radius = int(
                round(
                    math.hypot(
                        drag_current_display[0] - drag_start_display[0],
                        drag_current_display[1] - drag_start_display[1],
                        )
                )
            )
            preview_radius_original = int(round(preview_radius / scale))
            preview_color = (
                _CYAN
                if min_radius <= preview_radius_original <= max_radius
                else _RED
            )
            cv2.circle(
                display,
                drag_start_display,
                max(1, preview_radius),
                preview_color,
                3,
                cv2.LINE_AA,
            )
            cv2.circle(display, drag_start_display, 4, _YELLOW, -1, cv2.LINE_AA)
            _draw_text_with_outline(
                display,
                f"{_well_name(len(wells))}: radius {preview_radius_original}px",
                (drag_start_display[0] + 10, drag_start_display[1] + 22),
                scale=0.55,
                color=preview_color,
                thickness=2,
            )

        # Add a dark banner so instructions remain readable over bright images.
        banner_height = 76
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], banner_height), _BLACK, -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

        _draw_text_with_outline(
            display,
            f"Wells: {len(wells)}/{config.EXPECTED_WELLS} | Left-drag: add | U/right-click: undo | R: reset | S: save | Q: cancel",
            (10, 26),
            scale=0.52,
            color=_WHITE,
            thickness=1,
        )
        _draw_text_with_outline(
            display,
            status_message,
            (10, 57),
            scale=0.52,
            color=status_color,
            thickness=1,
        )

        cv2.imshow(window_name, display)
        final_display = display.copy()
        key = cv2.waitKey(16) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("u"), ord("U")):
            if wells:
                removed_name = _well_name(len(wells) - 1)
                wells.pop()
                status_message = f"Removed {removed_name}. Draw it again."
                status_color = _YELLOW
            else:
                status_message = "There are no wells to undo."
                status_color = _YELLOW

        elif key in (ord("r"), ord("R")):
            wells.clear()
            drag_start_display = None
            drag_current_display = None
            status_message = "All wells cleared. Start again with A1."
            status_color = _YELLOW

        elif key in (ord("s"), ord("S")):
            if len(wells) != config.EXPECTED_WELLS:
                status_message = (
                    f"Cannot save yet: selected {len(wells)} of "
                    f"{config.EXPECTED_WELLS} wells."
                )
                status_color = _RED
            else:
                _save_wells(output_csv, wells)
                preview_name = getattr(
                    config,
                    "MANUAL_CALIBRATION_PREVIEW_FILENAME",
                    "plate_wells_manual_preview.png",
                )
                preview_path = output_csv.parent / preview_name
                cv2.imwrite(str(preview_path), final_display)
                saved = True
                break

    cv2.destroyWindow(window_name)
    return saved