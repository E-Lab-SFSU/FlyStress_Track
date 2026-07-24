"""
well_plate.py

Purpose: to determine well positions (x,y) and detect flies in specific wells.

Features:
- Calculate each well center and average diameter
- Detect object (fly) within well plate
- Assign object (fly) a well

Notes:
WellPlate() - uses preset well positions from config.py
WellPlate.from_frame() - detect the well positions
px = pixel
coords = coordinates

"""
from __future__ import annotations

from math import hypot
from typing import Optional

import config
from WellPlate.well import Well

# full well plate

class WellPlate:
    # rectangular grid
    def __init__(self, rows: int, columns: int,
                 top_left: tuple[float, float], top_right: tuple[float, float],
                 bottom_left: tuple[float, float], bottom_right: tuple[float, float],
                 well_diameter_px: float, well_margin_px: float = 0.0,) -> None:

                # initializing values and placements
                self._validate_inputs(
                    rows=rows,
                    columns=columns,
                    well_diameter_px=well_diameter_px,
                    well_margin_px=well_margin_px,
                )

                self.rows = int(rows)
                self.columns = int(columns)

                self.top_left = self._float_point(top_left)
                self.top_right = self._float_point(top_right)
                self.bottom_left = self._float_point(bottom_left)
                self.bottom_right = self._float_point(bottom_right)

                self.well_diameter_px = float(well_diameter_px)
                self.well_margin_px = float(well_margin_px)

                self.wells = self._build_wells()

    # locate and build the wellplate without using preset coords from config
    # uses preset configs as fallbacks
    # used for first time recordings or if well plate was moved from previous video

    @staticmethod
    def _validate_inputs(
            rows: int,
            columns: int,
            well_diameter_px: float,
            well_margin_px: float,
    ) -> None:
        if rows <= 0:
            raise ValueError("rows must be greater than zero")

        if columns <= 0:
            raise ValueError("columns must be greater than zero")

        if well_diameter_px <= 0:
            raise ValueError(
                "well_diameter_px must be greater than zero"
            )

        if well_margin_px < 0:
            raise ValueError(
                "well_margin_px cannot be negative"
            )

        if well_margin_px >= well_diameter_px / 2.0:
            raise ValueError(
                "well_margin_px must be smaller than the well radius"
            )

    @staticmethod
    def _float_point(
            point: tuple[float, float],
    ) -> tuple[float, float]:
        if len(point) != 2:
            raise ValueError(
                "A point must contain exactly two values"
            )
        return float(point[0]), float(point[1])

    @staticmethod
    def _row_label(row: int) -> str:
        if row < 0:
            raise ValueError("row cannot be negative")

        label = ""
        value = row

        while True:
            value, remainder = divmod(value, 26)
            label = chr(ord("A") + remainder) + label

            if value == 0:
                break

            value -= 1

        return label

    @staticmethod
    def _interpolate_point(
            start: tuple[float, float],
            end: tuple[float, float],
            fraction: float,
    ) -> tuple[float, float]:
        start_x, start_y = start
        end_x, end_y = end

        x = start_x + fraction * (end_x - start_x)
        y = start_y + fraction * (end_y - start_y)

        return x, y

    def _calculate_well_center(
            self,
            row: int,
            column: int,
    ) -> tuple[float, float]:
        if self.rows == 1:
            row_fraction = 0.0
        else:
            row_fraction = row / (self.rows - 1)

        if self.columns == 1:
            column_fraction = 0.0
        else:
            column_fraction = column / (self.columns - 1)

        left_side = self._interpolate_point(
            start=self.top_left,
            end=self.bottom_left,
            fraction=row_fraction,
        )

        right_side = self._interpolate_point(
            start=self.top_right,
            end=self.bottom_right,
            fraction=row_fraction,
        )

        return self._interpolate_point(
            start=left_side,
            end=right_side,
            fraction=column_fraction,
        )

    def _build_wells(self) -> list[Well]:
        wells: list[Well] = []
        well_number = 1

        for row in range(self.rows):
            row_label = self._row_label(row)

            for column in range(self.columns):
                center_x, center_y = self._calculate_well_center(
                    row=row,
                    column=column,
                )

                wells.append(
                    Well(
                        number=well_number,
                        row=row,
                        column=column,
                        label=f"{row_label}{column + 1}",
                        center_x=center_x,
                        center_y=center_y,
                        diameter_px=self.well_diameter_px,
                    )
                )

                well_number += 1

        return wells

    @property
    def total_wells(self) -> int:
        return len(self.wells)

    @property
    def assignment_radius_px(self) -> float:
        return (
                self.well_diameter_px / 2.0
                - self.well_margin_px
        )

    @staticmethod
    def distance_to_well(
            x: float,
            y: float,
            well: Well,
    ) -> float:
        return hypot(
            float(x) - well.center_x,
            float(y) - well.center_y,
            )

    def point_is_inside_well(
            self,
            x: float,
            y: float,
            well: Well,
    ) -> bool:
        distance = self.distance_to_well(
            x=x,
            y=y,
            well=well,
        )

        return distance <= self.assignment_radius_px

    def well_from_point(
            self,
            x: float,
            y: float,
    ) -> Optional[Well]:
        nearest_well: Optional[Well] = None
        nearest_distance = float("inf")

        for well in self.wells:
            distance = self.distance_to_well(
                x=x,
                y=y,
                well=well,
            )

            if (
                    distance <= self.assignment_radius_px
                    and distance < nearest_distance
            ):
                nearest_well = well
                nearest_distance = distance

        return nearest_well

    def label_from_point(
            self,
            x: float,
            y: float,
    ) -> Optional[str]:
        well = self.well_from_point(x, y)

        if well is None:
            return None

        return well.label

    def get_well_by_label(
            self,
            label: str,
    ) -> Optional[Well]:
        normalized_label = label.strip().upper()

        for well in self.wells:
            if well.label == normalized_label:
                return well

        return None

    def get_well_by_number(
            self,
            number: int,
    ) -> Optional[Well]:
        for well in self.wells:
            if well.number == number:
                return well

        return None

    def count_points_by_well(
            self,
            positions: list[tuple[float, float]],
    ) -> dict[str, int]:
        counts = {
            well.label: 0
            for well in self.wells
        }

        for x, y in positions:
            well = self.well_from_point(x, y)

            if well is not None:
                counts[well.label] += 1

        return counts

    def assign_detections(
            self,
            detections: list[dict],
    ) -> list[dict]:
        assigned_detections: list[dict] = []

        for detection in detections:
            assigned = detection.copy()
            centroid = detection.get("centroid")

            if centroid is None or len(centroid) != 2:
                assigned["well_label"] = None
                assigned["well_number"] = None
                assigned["well_row"] = None
                assigned["well_column"] = None
                assigned_detections.append(assigned)
                continue

            x, y = centroid
            well = self.well_from_point(x, y)

            if well is None:
                assigned["well_label"] = None
                assigned["well_number"] = None
                assigned["well_row"] = None
                assigned["well_column"] = None
            else:
                assigned["well_label"] = well.label
                assigned["well_number"] = well.number
                assigned["well_row"] = well.row
                assigned["well_column"] = well.column

            assigned_detections.append(assigned)

        return assigned_detections


def create_plate_from_config() -> WellPlate:        # used config values, best when fixed position is established
    return WellPlate(
        rows=config.WELL_ROWS, columns=config.WELL_COLS,
        top_left=config.WELL_TL, top_right=config.WELL_TR,
        bottom_left=config.WELL_BL, bottom_right=config.WELL_BR,
        well_diameter_px=config.WELL_DIAMETER, well_margin_px=config.WELL_MARGIN,
    )