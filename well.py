"""
well.py

defines centery (x,y) & diameter of a well
px = pixels
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Well:                     # position of well and ID info

    number: int
    row: int
    column: int
    label: str
    center_x: float
    center_y: float
    diameter_px: float # pixels

    @property
    def radius_px(self) -> float:
        return self.diameter_px / 2.0      # returns radius in pixels

    @property
    def center(self) -> tuple[float, float]:
        return self.center_x, self.center_y         # returns center coords (x,y)
