"""Interactive manual well calibration.

Draw each well by pressing the left mouse button at its center, dragging to the
well edge, and releasing. Wells are labeled in row-major order (A1..A8,
B1..B8, C1..C8, D1..D8).

The script uses config.py when run directly, but run_analysis.py can call the
public calibrate() function with the current experiment's background image and
output path.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config


@dataclass
class ManualWell:
    x: int
    y: int
    radius: int

    @property
    def diameter(self) -> int:
        return self.radius * 2


def well_name(index: int, columns: int) -> str:
    row_index = index // columns
    column_index = index % columns
    return f"{chr(ord('A') + row_index)}{column_index + 1}"


def _fit_for_display(image: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return image.copy(), 1.0
    return cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    ), scale


def _to_original(x: int, y: int, scale: float, image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    ox = min(max(int(round(x / scale)), 0), width - 1)
    oy = min(max(int(round(y / scale)), 0), height - 1)
    return ox, oy


def _to_display(x: int, y: int, scale: float) -> tuple[int, int]:
    return int(round(x * scale)), int(round(y * scale))


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_wells_csv(path: Path) -> list[ManualWell]:
    wells: list[ManualWell] = []
    if not path.is_file():
        return wells
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            wells.append(
                ManualWell(
                    x=int(round(float(row["x"]))),
                    y=int(round(float(row["y"]))),
                    radius=int(round(float(row["radius"]))),
                )
            )
    return wells


def save_wells_csv(path: Path, wells: list[ManualWell], rows: int, columns: int) -> None:
    expected = rows * columns
    if len(wells) != expected:
        raise RuntimeError(f"Cannot save {len(wells)} wells; expected {expected}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["well", "row", "column", "x", "y", "radius", "diameter"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, well in enumerate(wells):
            writer.writerow(
                {
                    "well": well_name(index, columns),
                    "row": index // columns + 1,
                    "column": index % columns + 1,
                    "x": well.x,
                    "y": well.y,
                    "radius": well.radius,
                    "diameter": well.diameter,
                }
            )


def save_preview(path: Path, image: np.ndarray, wells: list[ManualWell], columns: int) -> None:
    preview = image.copy()
    for index, well in enumerate(wells):
        cv2.circle(preview, (well.x, well.y), well.radius, (0, 255, 0), 3)
        cv2.circle(preview, (well.x, well.y), 5, (0, 0, 255), -1)
        cv2.putText(
            preview,
            well_name(index, columns),
            (well.x + 8, well.y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), preview):
        raise RuntimeError(f"Could not save calibration preview: {path}")


def calibrate(
        image_path: Path,
        output_csv: Path,
        *,
        rows: int | None = None,
        columns: int | None = None,
        min_radius: int | None = None,
        max_radius: int | None = None,
        load_existing: bool | None = None,
) -> bool:
    """Run the calibration UI and return True when a calibration was saved."""
    rows = int(config.PLATE_ROWS if rows is None else rows)
    columns = int(config.PLATE_COLUMNS if columns is None else columns)
    expected = rows * columns
    min_radius = int(config.MANUAL_CALIBRATION_MIN_RADIUS if min_radius is None else min_radius)
    max_radius = int(config.MANUAL_CALIBRATION_MAX_RADIUS if max_radius is None else max_radius)
    load_existing = bool(
        config.MANUAL_CALIBRATION_LOAD_EXISTING if load_existing is None else load_existing
    )

    image_path = Path(image_path).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not open calibration image: {image_path}")

    display_base, scale = _fit_for_display(
        image,
        int(config.MANUAL_CALIBRATION_DISPLAY_MAX_WIDTH),
        int(config.MANUAL_CALIBRATION_DISPLAY_MAX_HEIGHT),
    )

    wells = load_wells_csv(output_csv) if load_existing else []
    if len(wells) > expected:
        wells = wells[:expected]

    drag_center: tuple[int, int] | None = None
    preview: ManualWell | None = None
    selected_index: int | None = None
    edit_mode: str | None = None  # "move" or "resize"
    window_name = str(config.MANUAL_CALIBRATION_WINDOW_NAME)

    def nearest_well(point: tuple[int, int]) -> tuple[int | None, str | None]:
        best_index: int | None = None
        best_mode: str | None = None
        best_distance = float("inf")
        for index, well in enumerate(wells):
            center_distance = _distance(point, (well.x, well.y))
            edge_distance = abs(center_distance - well.radius)
            if center_distance <= max(12, well.radius * 0.35) and center_distance < best_distance:
                best_index, best_mode, best_distance = index, "move", center_distance
            if edge_distance <= max(10, well.radius * 0.18) and edge_distance < best_distance:
                best_index, best_mode, best_distance = index, "resize", edge_distance
        return best_index, best_mode

    def mouse_callback(event: int, dx: int, dy: int, flags: int, userdata) -> None:
        nonlocal drag_center, preview, selected_index, edit_mode
        point = _to_original(dx, dy, scale, image)

        if event == cv2.EVENT_RBUTTONDOWN:
            index, _ = nearest_well(point)
            if index is not None:
                removed = wells.pop(index)
                print(f"Deleted {well_name(index, columns)} at ({removed.x}, {removed.y}).")
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            index, mode = nearest_well(point)
            if index is not None:
                selected_index = index
                edit_mode = mode
                drag_center = point
                return
            if len(wells) < expected:
                selected_index = None
                edit_mode = "new"
                drag_center = point
                preview = ManualWell(point[0], point[1], 0)

        elif event == cv2.EVENT_MOUSEMOVE and drag_center is not None:
            if edit_mode == "new":
                preview = ManualWell(drag_center[0], drag_center[1], int(round(_distance(drag_center, point))))
            elif selected_index is not None and edit_mode == "move":
                wells[selected_index].x = point[0]
                wells[selected_index].y = point[1]
            elif selected_index is not None and edit_mode == "resize":
                wells[selected_index].radius = int(round(_distance((wells[selected_index].x, wells[selected_index].y), point)))

        elif event == cv2.EVENT_LBUTTONUP and drag_center is not None:
            if edit_mode == "new" and preview is not None:
                if min_radius <= preview.radius <= max_radius:
                    wells.append(preview)
                    index = len(wells) - 1
                    print(
                        f"{well_name(index, columns)}: x={preview.x}, y={preview.y}, "
                        f"radius={preview.radius}"
                    )
                else:
                    print(
                        f"Circle rejected: radius {preview.radius}px is outside "
                        f"{min_radius}-{max_radius}px."
                    )
            elif selected_index is not None and edit_mode == "resize":
                wells[selected_index].radius = min(max(wells[selected_index].radius, min_radius), max_radius)
            drag_center = None
            preview = None
            selected_index = None
            edit_mode = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\nManual drag-circle calibration")
    print(f"Image: {image_path}")
    print(f"Output: {output_csv}")
    print("Left-drag empty area: draw next well")
    print("Left-drag center: move an existing well")
    print("Left-drag edge: resize an existing well")
    print("Right-click circle: delete it")
    print("U: undo last, R: reset, S: save, Q/Esc: cancel\n")

    saved = False
    while True:
        display = display_base.copy()
        for index, well in enumerate(wells):
            center = _to_display(well.x, well.y, scale)
            radius = max(1, int(round(well.radius * scale)))
            cv2.circle(display, center, radius, (0, 255, 0), 2)
            cv2.circle(display, center, 4, (0, 0, 255), -1)
            cv2.putText(
                display,
                well_name(index, columns),
                (center[0] + 7, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if preview is not None:
            center = _to_display(preview.x, preview.y, scale)
            radius = max(1, int(round(preview.radius * scale)))
            valid = min_radius <= preview.radius <= max_radius
            cv2.circle(display, center, radius, (255, 255, 0) if valid else (0, 0, 255), 2)

        cv2.rectangle(display, (0, 0), (display.shape[1], 76), (0, 0, 0), -1)
        next_text = (
            f"Draw {well_name(len(wells), columns)} ({len(wells)+1}/{expected})"
            if len(wells) < expected
            else f"All {expected} wells ready - press S to save"
        )
        cv2.putText(display, next_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            display,
            "Drag=new/move/resize | Right-click=delete | U=undo R=reset S=save Q=quit",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u") and wells:
            wells.pop()
        elif key == ord("r"):
            wells.clear()
        elif key == ord("s"):
            if len(wells) != expected:
                print(f"Cannot save: {len(wells)} of {expected} wells are defined.")
                continue
            save_wells_csv(output_csv, wells, rows, columns)
            preview_path = output_csv.parent / str(config.MANUAL_CALIBRATION_PREVIEW_FILENAME)
            save_preview(preview_path, image, wells, columns)
            print(f"Saved calibration: {output_csv}")
            print(f"Saved preview: {preview_path}")
            saved = True
            break

    cv2.destroyWindow(window_name)
    return saved


def main() -> None:
    calibrate(
        Path(config.MANUAL_CALIBRATION_IMAGE),
        Path(config.MANUAL_CALIBRATION_OUTPUT),
    )


if __name__ == "__main__":
    main()
