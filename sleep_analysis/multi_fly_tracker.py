"""Three persistent fly identities per manually selected well."""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import permutations
from collections import deque
from statistics import median
from sleep_analysis.fly_detection import FlyDetection
from sleep_analysis.movement import euclidean_distance, apply_jitter_deadband

@dataclass
class FlyTrack:
    well: str
    slot: int
    x: float | None = None
    y: float | None = None
    initialized: bool = False
    initialization_reported: bool = False
    immobile_since: float | None = None
    rolling: deque = field(default_factory=deque)
    rolling_total: float = 0.0
    low_confidence_frames: int = 0

    @property
    def fly_id(self): return f"{self.well}_Fly{self.slot}"

@dataclass(frozen=True)
class TrackResult:
    well: str; fly_id: str; fly_slot: int
    x_px: float | None; y_px: float | None
    local_x_px: float | None; local_y_px: float | None
    area_px: int | None; threshold_value: int | None
    observation_status: str; activity_state: str; identity_confidence: str
    overlap_group: str; overlap_count: int
    raw_distance_px: float | None; distance_px: float | None
    rolling_distance_px: float; rolling_samples: int
    immobile_duration_seconds: float

class PerWellMultiFlyTracker:
    def __init__(self, well_names, *, flies_per_well=3, max_match_distance_px=55.0,
                 jitter_threshold_px=3.0, rolling_window_seconds=300.0,
                 sleep_duration_seconds=300.0, max_valid_sample_gap_seconds=2.5,
                 low_confidence_frames_after_split=5):
        self.flies_per_well = int(flies_per_well)
        self.max_match_distance = float(max_match_distance_px)
        self.jitter = float(jitter_threshold_px)
        self.window = float(rolling_window_seconds)
        self.sleep_duration = float(sleep_duration_seconds)
        self.max_gap = float(max_valid_sample_gap_seconds)
        self.low_after_split = int(low_confidence_frames_after_split)
        self.tracks = {w:[FlyTrack(w,i) for i in range(1,self.flies_per_well+1)] for w in well_names}
        self.last_time = {w:None for w in well_names}
        self.area_samples = {w:deque(maxlen=120) for w in well_names}
        self.overlap_serial = {w:0 for w in well_names}
        self.was_overlapping = {w:False for w in well_names}

    def area_hints(self):
        return {w: float(median(v)) for w,v in self.area_samples.items() if v}

    def _roll(self, track, timestamp, distance):
        cutoff = timestamp - self.window
        while track.rolling and track.rolling[0][0] < cutoff:
            track.rolling_total -= track.rolling.popleft()[1]
        if distance is not None:
            track.rolling.append((timestamp, distance)); track.rolling_total += distance
        return max(0.0, track.rolling_total), len(track.rolling)

    def _assign(self, tracks, detections):
        active = [i for i,t in enumerate(tracks) if t.initialized]
        if not active or not detections: return {}
        best, best_cost = {}, float('inf')
        for k in range(1, min(len(active),len(detections))+1):
            for tperm in permutations(active,k):
                for dperm in permutations(range(len(detections)),k):
                    mapping={}; cost=0; valid=True
                    for ti,di in zip(tperm,dperm):
                        t=tracks[ti]; d=detections[di]
                        dist=euclidean_distance(t.x,t.y,d.x,d.y)
                        if dist>self.max_match_distance: valid=False; break
                        mapping[ti]=di; cost+=dist
                    # Prefer assignments covering more existing tracks.
                    cost += (len(active)-k)*self.max_match_distance*0.8
                    if valid and cost<best_cost: best,best_cost=mapping,cost
        return best

    def update_well(self, well, detections, timestamp_s, registration_ok=True):
        tracks=self.tracks[well]; last=self.last_time[well]
        gap=None if last is None else timestamp_s-last
        self.last_time[well]=timestamp_s
        singles=[d for d in detections if d.estimated_fly_count==1]
        overlaps=[d for d in detections if d.estimated_fly_count>1]
        for d in singles: self.area_samples[well].append(d.area_px)
        results=[]
        overlap_slots=set(); overlap_info={}
        if overlaps:
            self.overlap_serial[well]+=1
            for oi,d in enumerate(overlaps,1):
                count=min(self.flies_per_well,d.estimated_fly_count)
                ranked=sorted(range(len(tracks)), key=lambda i: euclidean_distance(tracks[i].x,tracks[i].y,d.x,d.y)
                if tracks[i].initialized else 1e9)[:count]
                group=f"{well}_O{self.overlap_serial[well]:04d}_{oi}"
                for ti in ranked: overlap_slots.add(ti); overlap_info[ti]=(group,count,d)
            self.was_overlapping[well]=True
        elif self.was_overlapping[well]:
            for t in tracks: t.low_confidence_frames=self.low_after_split
            self.was_overlapping[well]=False

        assign=self._assign(tracks, singles)
        used=set(assign.values())
        # Initialize empty slots from unmatched single detections, left-to-right for repeatability.
        unused=[(i,d) for i,d in enumerate(singles) if i not in used]
        unused.sort(key=lambda p:(p[1].x,p[1].y))
        empty=[i for i,t in enumerate(tracks) if not t.initialized]
        for ti,(di,d) in zip(empty,unused): assign[ti]=di; used.add(di)

        for ti,t in enumerate(tracks):
            if ti in overlap_slots:
                group,count,d=overlap_info[ti]
                rolling,samples=self._roll(t,timestamp_s,None)
                results.append(TrackResult(well,t.fly_id,t.slot,None,None,None,None,d.area_px,d.threshold_value,
                                           "OVERLAP","UNKNOWN","LOW",group,count,None,None,rolling,samples,0.0))
                continue
            di=assign.get(ti)
            if di is None or not registration_ok or (gap is not None and gap>self.max_gap):
                rolling,samples=self._roll(t,timestamp_s,None)
                confidence="UNKNOWN" if not t.initialized else ("LOW" if t.low_confidence_frames else "MEDIUM")
                results.append(TrackResult(well,t.fly_id,t.slot,None,None,None,None,None,None,
                                           "UNKNOWN","UNKNOWN",confidence,"",0,None,None,rolling,samples,0.0))
                continue
            d=singles[di]
            raw=0.0 if not t.initialized else euclidean_distance(t.x,t.y,d.x,d.y)
            distance=apply_jitter_deadband(raw,self.jitter)
            first=not t.initialized
            t.initialized=True; t.x=d.x; t.y=d.y
            if distance>0 or t.immobile_since is None: t.immobile_since=timestamp_s
            immobile=max(0.0,timestamp_s-(t.immobile_since or timestamp_s))
            activity="ASLEEP" if immobile>=self.sleep_duration else ("AWAKE" if distance>0 else "INACTIVE")
            rolling,samples=self._roll(t,timestamp_s,distance)
            status="DETECTED" if first and not t.initialization_reported else ""
            if first: t.initialization_reported=True
            confidence="LOW" if t.low_confidence_frames>0 else "HIGH"
            if t.low_confidence_frames>0: t.low_confidence_frames-=1
            results.append(TrackResult(well,t.fly_id,t.slot,d.x,d.y,d.local_x,d.local_y,d.area_px,d.threshold_value,
                                       status,activity,confidence,"",1,raw,distance,rolling,samples,immobile))
        return results

    def update_all(self,detections_by_well,timestamp_s,registration_ok=True):
        out=[]
        for well in self.tracks:
            out.extend(self.update_well(well,detections_by_well.get(well,[]),timestamp_s,registration_ok))
        return out
