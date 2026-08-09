"""One persistent, wall-aware, confidence-aware fly track per well."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from sleep_analysis.fly_detection import FlyDetection
from sleep_analysis.movement import apply_jitter_deadband, euclidean_distance
from sleep_analysis.rolling_sleep import RollingMovement


def _point_near_box(x: float, y: float, box, margin: float) -> bool:
    bx, by, bw, bh = box
    return (bx-margin) <= x <= (bx+bw+margin) and (by-margin) <= y <= (by+bh+margin)


def _bbox_overlap_fraction(d: FlyDetection, box) -> float:
    bx, by, bw, bh = box
    ix1=max(d.bbox_x1,bx); iy1=max(d.bbox_y1,by)
    ix2=min(d.bbox_x2,bx+bw); iy2=min(d.bbox_y2,by+bh)
    if ix2 < ix1 or iy2 < iy1: return 0.0
    inter=(ix2-ix1+1)*(iy2-iy1+1)
    area=max(1,(d.bbox_x2-d.bbox_x1+1)*(d.bbox_y2-d.bbox_y1+1))
    return float(inter)/float(area)


@dataclass(frozen=True)
class TrackResult:
    well: str; fly_name: str; state: str; valid_tracking: bool; reason: str; detected: bool
    x_px: float|None; y_px: float|None; local_x_px: float|None; local_y_px: float|None
    area_px: int|None; threshold_value: int|None
    bbox_x1: int|None; bbox_y1: int|None; bbox_x2: int|None; bbox_y2: int|None
    center_gray: float|None; component_median_gray: float|None
    raw_distance_px: float|None; distance_px: float|None
    cumulative_raw_distance_px: float; cumulative_distance_px: float
    rolling_distance_px: float; rolling_samples: int; immobile_duration_seconds: float
    confidence: float = 0.0
    detection_stage: str = ""
    wall_mode: bool = False
    radial_fraction: float|None = None
    arrival_change: float = 0.0
    arrival_fraction: float = 0.0
    departure_change: float = 0.0


class SingleFlyTracker:
    def __init__(self, well_names:list[str], *, jitter_threshold_px:float,
                 rolling_window_seconds:float, sleep_duration_seconds:float,
                 max_position_jump_px:float, max_valid_sample_gap_seconds:float,
                 food_overlap_penalty:float=35.0, food_contact_max_jump_multiplier:float=1.35,
                 food_contact_search_radius_px:float=35.0, grayscale_identity_weight:float=0.30,
                 reacquire_jump_growth:float=0.75, reacquire_max_jump_px:float=220.0,
                 full_well_reacquire_after_misses:int=2,
                 confidence_min_for_model_update:float=0.62,
                 confidence_min_accept:float=0.30,
                 wall_mode_radial_fraction:float=0.72,
                 wall_mode_hold_frames:int=6,
                 wall_area_penalty_scale:float=0.25,
                 motion_prediction_weight:float=0.65,
                 appearance_update_alpha:float=0.06,
                 area_update_alpha:float=0.10,
                 wall_area_update_alpha:float=0.02,
                 frame_difference_weight:float=1.25,
                 departure_motion_threshold:float=4.0,
                 arrival_motion_threshold:float=3.0,
                 moving_old_position_penalty:float=30.0) -> None:
        self.jitter=float(jitter_threshold_px); self.sleep_seconds=float(sleep_duration_seconds)
        self.max_jump=float(max_position_jump_px); self.max_gap=float(max_valid_sample_gap_seconds)
        self.food_overlap_penalty=float(food_overlap_penalty)
        self.food_contact_jump_multiplier=float(food_contact_max_jump_multiplier)
        self.food_contact_search_radius=float(food_contact_search_radius_px)
        self.grayscale_identity_weight=float(grayscale_identity_weight)
        self.reacquire_jump_growth=float(reacquire_jump_growth); self.reacquire_max_jump=float(reacquire_max_jump_px)
        self.full_well_after=int(full_well_reacquire_after_misses)
        self.model_update_conf=float(confidence_min_for_model_update); self.min_accept_conf=float(confidence_min_accept)
        self.wall_radial=float(wall_mode_radial_fraction); self.wall_hold=int(wall_mode_hold_frames)
        self.wall_area_penalty_scale=float(wall_area_penalty_scale)
        self.motion_prediction_weight=float(motion_prediction_weight)
        self.gray_alpha=float(appearance_update_alpha); self.area_alpha=float(area_update_alpha)
        self.wall_area_alpha=float(wall_area_update_alpha)
        self.diff_weight=float(frame_difference_weight)
        self.departure_threshold=float(departure_motion_threshold)
        self.arrival_threshold=float(arrival_motion_threshold)
        self.moving_old_penalty=float(moving_old_position_penalty)
        self.data={well:dict(x=None,y=None,last_t=None,immobile=0.0,food_boxes=[],initialized=False,
                             initial_fly_bbox=None,expected_area=None,expected_gray=None,missed_frames=0,wall_mode_frames=0,
                             history=deque(maxlen=5), cumulative_raw_distance=0.0,cumulative_distance=0.0,
                             rolling=RollingMovement(rolling_window_seconds)) for well in well_names}

    def initialize_manual(self, well:str, fly_bbox, food_boxes=None)->None:
        st=self.data[well]; st['food_boxes']=list(food_boxes or []); st['initial_fly_bbox']=list(fly_bbox) if fly_bbox else None
        if fly_bbox:
            x,y,w,h=fly_bbox; st['x']=float(x)+float(w)/2; st['y']=float(y)+float(h)/2; st['initialized']=True

    def _predicted_position(self, st):
        hist=list(st['history'])
        if len(hist) < 2:
            return (st['x'], st['y'])
        x2,y2,t2=hist[-1]; x1,y1,t1=hist[-2]
        dt=max(1e-6,t2-t1)
        # One-frame constant-velocity prediction, clipped so a noisy frame cannot launch the search.
        vx=(x2-x1)/dt; vy=(y2-y1)/dt
        next_dt=min(2.0,max(0.25,dt))
        dx=max(-self.max_jump,min(self.max_jump,vx*next_dt))
        dy=max(-self.max_jump,min(self.max_jump,vy*next_dt))
        return (x2+dx,y2+dy)

    def tracking_hints(self)->dict[str,dict[str,float]]:
        hints={}
        for well,st in self.data.items():
            if st['x'] is None: continue
            missed=int(st['missed_frames'])
            radius=min(self.reacquire_max_jump,self.max_jump*(1+self.reacquire_jump_growth*missed))
            if missed >= self.full_well_after: radius=self.reacquire_max_jump
            pred=self._predicted_position(st)
            # x/y remain the last accepted centroid so the detector can measure
            # whether the fly departed that exact location between frames. The
            # prediction is supplied separately for ranking/search guidance.
            hints[well]={'x':float(st['x']),'y':float(st['y']),
                         'predicted_x':float(pred[0]),'predicted_y':float(pred[1]),
                         'search_radius_px':float(radius),
                         'wall_mode':1.0 if int(st['wall_mode_frames'])>0 else 0.0}
        return hints

    def _candidate_confidence(self, st, d:FlyDetection, distance:float, predicted_distance:float,
                              previous_near_food:bool, wall_mode:bool)->float:
        detector=max(0.0,min(1.0,float(d.candidate_score)/9.0))
        continuity=max(0.0,1.0-distance/max(1.0,self.max_jump*1.5))
        motion_cont=max(0.0,1.0-predicted_distance/max(1.0,self.max_jump*1.5))
        gray=0.65
        if st.get('expected_gray') is not None:
            gray=max(0.0,1.0-abs(float(d.component_median_gray)-float(st['expected_gray']))/55.0)
        area=0.65
        if st.get('expected_area'):
            ratio=max(float(d.area_px),float(st['expected_area']))/max(1.0,min(float(d.area_px),float(st['expected_area'])))
            area=max(0.0,1.0-(ratio-1.0)/(8.0 if wall_mode else 3.0))
        if previous_near_food: area=max(area,0.55)
        # Near the wall, continuity + darkness matter more than apparent blob size.
        if wall_mode:
            return max(0.0,min(1.0,0.30*detector+0.21*continuity+0.19*motion_cont+0.25*gray+0.05*area))
        return max(0.0,min(1.0,0.31*detector+0.22*continuity+0.13*motion_cont+0.22*gray+0.12*area))

    def _choose(self, well:str, candidates:list[FlyDetection]):
        if not candidates: return None,0.0,''
        st=self.data[well]
        if st['x'] is None:
            d=max(candidates,key=lambda z:z.candidate_score); return d,0.7,'initial'
        prev_food=any(_point_near_box(st['x'],st['y'],b,self.food_contact_search_radius) for b in st['food_boxes'])
        missed=int(st['missed_frames']); full_well=missed >= self.full_well_after
        expected_area=st.get('expected_area'); predicted=self._predicted_position(st)
        wall_mode=int(st['wall_mode_frames'])>0

        # Area is only a broad upper guard. A wall-climbing fly may look like a very small dot.
        plausible=candidates
        if expected_area and not prev_food:
            upper=max(float(expected_area)*(6.0 if wall_mode else 3.5),float(expected_area)+(90 if wall_mode else 45))
            plausible=[d for d in candidates if float(d.area_px)<=upper]
            if not plausible: return None,0.0,'no_plausible_blob'

        def cost(d):
            dist=euclidean_distance(st['x'],st['y'],d.x,d.y)
            pred_dist=euclidean_distance(predicted[0],predicted[1],d.x,d.y)
            candidate_wall = float(getattr(d,'radial_fraction',0.0)) >= self.wall_radial
            effective_wall = wall_mode or candidate_wall
            food=max((_bbox_overlap_fraction(d,b) for b in st['food_boxes']),default=0.0)
            penalty=0.0 if prev_food else self.food_overlap_penalty*food
            init_bonus=70.0*_bbox_overlap_fraction(d,st['initial_fly_bbox']) if st['last_t'] is None and st['initial_fly_bbox'] else 0.0
            gray_pen=0.0 if st.get('expected_gray') is None else self.grayscale_identity_weight*abs(float(d.component_median_gray)-float(st['expected_gray']))
            area_pen=0.0
            if expected_area:
                ratio=max(float(d.area_px),float(expected_area))/max(1.0,min(float(d.area_px),float(expected_area)))
                scale=self.wall_area_penalty_scale if effective_wall else 1.0
                area_pen=scale*min(45.0,max(0.0,ratio-1.8)*12.0)
            distance_weight=0.18 if full_well else (0.75 if effective_wall else 1.0)
            prediction_bonus=self.motion_prediction_weight*pred_dist

            # Per-well signed frame difference. If the old fly location became
            # brighter, the fly likely departed. Prefer a candidate whose current
            # pixels became darker (arrival) and penalize staying on an old dark
            # artifact near the previous centroid. When there is little departure
            # evidence, this term is weak so a genuinely sleeping fly can remain
            # stationary indefinitely.
            departure=float(getattr(d,'departure_change',0.0))
            arrival=float(getattr(d,'arrival_change',0.0))
            arrival_fraction=float(getattr(d,'arrival_fraction',0.0))
            movement_evidence=max(0.0,(departure-self.departure_threshold)/max(1.0,self.departure_threshold))
            arrival_reward=self.diff_weight*movement_evidence*(min(arrival,24.0)+10.0*arrival_fraction)
            old_position_penalty=0.0
            if movement_evidence>0 and dist <= max(self.jitter*2.0,8.0) and arrival < self.arrival_threshold:
                old_position_penalty=self.moving_old_penalty*min(2.0,movement_evidence)
            return (distance_weight*dist + prediction_bonus + penalty + gray_pen + area_pen
                    + old_position_penalty - arrival_reward
                    - min(14.0,2.0*d.candidate_score) - init_bonus)

        best=min(plausible,key=cost)
        dist=euclidean_distance(st['x'],st['y'],best.x,best.y)
        pred_dist=euclidean_distance(predicted[0],predicted[1],best.x,best.y)
        candidate_wall=float(getattr(best,'radial_fraction',0.0)) >= self.wall_radial
        effective_wall=wall_mode or candidate_wall
        allowed=min(self.reacquire_max_jump,self.max_jump*(1+self.reacquire_jump_growth*missed))
        if prev_food: allowed*=self.food_contact_jump_multiplier
        if effective_wall: allowed*=1.35
        if not full_well and dist>allowed: return None,0.0,'outside_search_radius'
        conf=self._candidate_confidence(st,best,dist,pred_dist,prev_food,effective_wall)
        threshold=self.min_accept_conf + (0.08 if full_well else 0.0)
        # A tiny wall dot can be accepted with slightly lower confidence if it follows the track.
        if effective_wall and pred_dist <= self.max_jump*0.75: threshold=max(0.22,threshold-0.05)
        if conf < threshold: return None,conf,'low_confidence_candidate'
        stage='full_well_reacquire' if full_well else ('expanded_search' if missed else 'local_search')
        # Mark frames where signed frame difference actively supported movement.
        if float(getattr(best,'departure_change',0.0)) >= self.departure_threshold and float(getattr(best,'arrival_change',0.0)) >= self.arrival_threshold:
            stage += '_frame_diff'
        if effective_wall: stage += '_wall'
        return best,conf,stage

    def update(self, well:str, candidates:list[FlyDetection], timestamp_s:float, registration_ok:bool)->TrackResult:
        st=self.data[well]; rolling=st['rolling']; rolling.advance(timestamp_s); fly=f'{well}_Fly'
        def missing(reason):
            st['missed_frames']=int(st['missed_frames'])+1
            if st['wall_mode_frames']>0: st['wall_mode_frames']=max(0,int(st['wall_mode_frames'])-1)
            return TrackResult(well,fly,'UNKNOWN',False,reason,False,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
                               float(st['cumulative_raw_distance']),float(st['cumulative_distance']),rolling.total,rolling.samples,st['immobile'],0.0,'missing',
                               int(st['wall_mode_frames'])>0,None)
        if not registration_ok: return missing('registration_failed')
        d,conf,stage=self._choose(well,candidates)
        if d is None: return missing(stage or 'fly_not_detected')
        px,py,pt=st['x'],st['y'],st['last_t']
        st['x'],st['y'],st['last_t']=d.x,d.y,timestamp_s; st['missed_frames']=0
        st['history'].append((float(d.x),float(d.y),float(timestamp_s)))
        is_wall=float(getattr(d,'radial_fraction',0.0)) >= self.wall_radial
        st['wall_mode_frames']=self.wall_hold if is_wall else max(0,int(st['wall_mode_frames'])-1)
        wall_mode=int(st['wall_mode_frames'])>0
        on_food=any(_bbox_overlap_fraction(d,b)>0 for b in st['food_boxes'])

        # Anti-drift + gradual reacquisition: grayscale learns slowly; apparent area learns
        # especially slowly in wall mode so a tiny dot does not redefine the fly's normal size.
        if (not on_food) and conf >= self.model_update_conf:
            if st.get('expected_area') is None:
                st['expected_area']=float(d.area_px)
            else:
                a=self.wall_area_alpha if wall_mode else self.area_alpha
                st['expected_area']=(1-a)*float(st['expected_area'])+a*float(d.area_px)
            if st.get('expected_gray') is None:
                st['expected_gray']=float(d.component_median_gray)
            else:
                st['expected_gray']=(1-self.gray_alpha)*float(st['expected_gray'])+self.gray_alpha*float(d.component_median_gray)

        if px is None or pt is None:
            st['immobile']=0.0; rolling.add(timestamp_s,0.0); reason='first_valid_position'; raw=dist=0.0
        else:
            dt=timestamp_s-pt
            if dt<=0 or dt>self.max_gap:
                st['immobile']=0.0; rolling.add(timestamp_s,0.0); reason='reacquired_after_gap'; raw=dist=0.0
            else:
                raw=euclidean_distance(px,py,d.x,d.y); dist=apply_jitter_deadband(raw,self.jitter)
                st['cumulative_raw_distance']+=raw; st['cumulative_distance']+=dist; rolling.add(timestamp_s,dist)
                st['immobile']=0.0 if dist>0 else st['immobile']+dt; reason='valid'
        state='ASLEEP' if st['immobile']>=self.sleep_seconds else 'AWAKE'
        return TrackResult(well,fly,state,True,reason,True,d.x,d.y,d.local_x,d.local_y,d.area_px,d.threshold_value,
                           d.bbox_x1,d.bbox_y1,d.bbox_x2,d.bbox_y2,d.center_gray,d.component_median_gray,raw,dist,
                           float(st['cumulative_raw_distance']),float(st['cumulative_distance']),rolling.total,rolling.samples,st['immobile'],conf,stage,
                           wall_mode,float(getattr(d,'radial_fraction',0.0)),
                           float(getattr(d,'arrival_change',0.0)),float(getattr(d,'arrival_fraction',0.0)),
                           float(getattr(d,'departure_change',0.0)))
