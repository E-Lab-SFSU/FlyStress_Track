import math

def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)

def apply_jitter_deadband(distance_px: float, jitter_threshold_px: float) -> float:
    return 0.0 if distance_px < jitter_threshold_px else max(0.0, distance_px - jitter_threshold_px)
