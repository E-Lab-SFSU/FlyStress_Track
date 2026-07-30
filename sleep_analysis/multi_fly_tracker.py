"""Persistent multi-fly identity tracking, restricted to individual wells."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from sleep_analysis.fly_detection import FlyDetection
from sleep_analysis.movement import apply_jitter_deadband, euclidean_distance
from sleep_analysis.rolling_sleep import RollingSleepState


@dataclass(frozen=True)
class TrackerDetection:
    """A detection represented in both aligned and current raw coordinates."""

    detection: FlyDetection
    raw_x: float
    raw_y: float


@dataclass
class FlyTrack:
    well: str
    slot: int
    raw_x: float | None = None
    raw_y: float | None = None
    missed_frames: int = 0
    active: bool = False
    sleep: RollingSleepState | None = None

    @property
    def fly_name(self) -> str:
        return f"{self.well}_Fly{self.slot}"


@dataclass(frozen=True)
class TrackResult:
    well: str
    fly_name: str
    fly_slot: int
    detected: bool
    x_px: float | None
    y_px: float | None
    well_relative_x_px: float | None
    well_relative_y_px: float | None
    area_px: int | None
    threshold_value: int | None
    raw_distance_px: float
    distance_px: float
    rolling_distance_px: float
    rolling_samples: int
    fly_state: str
    missed_frames: int


class PerWellMultiFlyTracker:
    """Track up to a fixed number of flies independently in each well.

    Matching is based on nearest distance after registration. Because each well
    contains at most four flies, an exhaustive assignment search is small and
    avoids adding a SciPy dependency.
    """

    def __init__(
            self,
            well_names: Iterable[str],
            *,
            max_flies_per_well: int,
            max_match_distance_px: float,
            max_missed_frames: int,
            jitter_threshold_px: float,
            rolling_window_samples: int,
            awake_threshold_px: float,
    ) -> None:
        self.max_match_distance_px = float(max_match_distance_px)
        self.max_missed_frames = int(max_missed_frames)
        self.jitter_threshold_px = float(jitter_threshold_px)

        self.tracks: dict[str, list[FlyTrack]] = {}
        for well in well_names:
            self.tracks[well] = [
                FlyTrack(
                    well=well,
                    slot=slot,
                    sleep=RollingSleepState(
                        window_samples=rolling_window_samples,
                        awake_threshold_px=awake_threshold_px,
                    ),
                )
                for slot in range(1, max_flies_per_well + 1)
            ]

    def _best_assignment(
            self,
            tracks: list[FlyTrack],
            detections: list[TrackerDetection],
    ) -> dict[int, int]:
        """Return mapping from track-list index to detection-list index."""
        active_indices = [index for index, track in enumerate(tracks) if track.active]
        if not active_indices or not detections:
            return {}

        # Each active track may use any detection or remain unmatched (-1).
        choices = list(range(len(detections))) + [-1]
        unmatched_penalty = self.max_match_distance_px * 0.75
        best_cost = float("inf")
        best_mapping: dict[int, int] = {}

        for assignment in product(choices, repeat=len(active_indices)):
            used = [item for item in assignment if item >= 0]
            if len(used) != len(set(used)):
                continue

            cost = 0.0
            mapping: dict[int, int] = {}
            valid = True
            for track_index, detection_index in zip(active_indices, assignment):
                if detection_index < 0:
                    cost += unmatched_penalty
                    continue

                track = tracks[track_index]
                detection = detections[detection_index].detection
                assert track.raw_x is not None and track.raw_y is not None
                distance = euclidean_distance(
                    track.raw_x,
                    track.raw_y,
                    detection.x,
                    detection.y,
                )
                if distance > self.max_match_distance_px:
                    valid = False
                    break
                cost += distance
                mapping[track_index] = detection_index

            if valid and cost < best_cost:
                best_cost = cost
                best_mapping = mapping

        return best_mapping

    def update_well(
            self,
            well: str,
            detections: list[TrackerDetection],
    ) -> list[TrackResult]:
        tracks = self.tracks[well]
        initially_inactive = [track for track in tracks if not track.active]
        detections = sorted(
            detections,
            key=lambda item: (item.detection.x, item.detection.y),
        )
        assignment = self._best_assignment(tracks, detections)
        used_detections = set(assignment.values())
        results: list[TrackResult] = []

        # Update existing tracks.
        for track_index, track in enumerate(tracks):
            if not track.active:
                continue

            detection_index = assignment.get(track_index)
            if detection_index is None:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    track.active = False
                    track.raw_x = None
                    track.raw_y = None
                    track.missed_frames = 0
                rolling, state, samples = track.sleep.update(0.0)  # type: ignore[union-attr]
                results.append(
                    TrackResult(
                        well=well,
                        fly_name=track.fly_name,
                        fly_slot=track.slot,
                        detected=False,
                        x_px=None,
                        y_px=None,
                        well_relative_x_px=None,
                        well_relative_y_px=None,
                        area_px=None,
                        threshold_value=None,
                        raw_distance_px=0.0,
                        distance_px=0.0,
                        rolling_distance_px=rolling,
                        rolling_samples=samples,
                        fly_state=state,
                        missed_frames=track.missed_frames,
                    )
                )
                continue

            item = detections[detection_index]
            detection = item.detection
            assert track.raw_x is not None and track.raw_y is not None
            raw_distance = euclidean_distance(
                track.raw_x,
                track.raw_y,
                detection.x,
                detection.y,
            )
            filtered_distance = apply_jitter_deadband(
                raw_distance,
                self.jitter_threshold_px,
            )
            rolling, state, samples = track.sleep.update(filtered_distance)  # type: ignore[union-attr]

            # Preserve the current raw-frame position for matching to the next
            # frame. The aligned position is used for the current movement.
            track.raw_x = item.raw_x
            track.raw_y = item.raw_y
            track.missed_frames = 0

            results.append(
                TrackResult(
                    well=well,
                    fly_name=track.fly_name,
                    fly_slot=track.slot,
                    detected=True,
                    x_px=detection.x,
                    y_px=detection.y,
                    well_relative_x_px=detection.local_x,
                    well_relative_y_px=detection.local_y,
                    area_px=detection.area_px,
                    threshold_value=detection.threshold_value,
                    raw_distance_px=raw_distance,
                    distance_px=filtered_distance,
                    rolling_distance_px=rolling,
                    rolling_samples=samples,
                    fly_state=state,
                    missed_frames=0,
                )
            )

        # Assign unmatched detections to inactive slots.
        unused = [
            detection
            for index, detection in enumerate(detections)
            if index not in used_detections
        ]
        for track, item in zip(initially_inactive, unused):
            detection = item.detection
            track.sleep.reset()  # type: ignore[union-attr]
            track.active = True
            track.raw_x = item.raw_x
            track.raw_y = item.raw_y
            track.missed_frames = 0
            rolling, state, samples = track.sleep.update(0.0)  # type: ignore[union-attr]
            results.append(
                TrackResult(
                    well=well,
                    fly_name=track.fly_name,
                    fly_slot=track.slot,
                    detected=True,
                    x_px=detection.x,
                    y_px=detection.y,
                    well_relative_x_px=detection.local_x,
                    well_relative_y_px=detection.local_y,
                    area_px=detection.area_px,
                    threshold_value=detection.threshold_value,
                    raw_distance_px=0.0,
                    distance_px=0.0,
                    rolling_distance_px=rolling,
                    rolling_samples=samples,
                    fly_state=state,
                    missed_frames=0,
                )
            )

        return sorted(results, key=lambda result: result.fly_slot)
