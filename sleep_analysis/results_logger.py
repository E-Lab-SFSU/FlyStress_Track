"""Three independent CSV outputs: positions, states, and well/frame diagnostics."""
from __future__ import annotations
import csv
from pathlib import Path

POSITION_FIELDS=["timestamp_iso","elapsed_seconds","image","frame_number","well","fly_id","fly_slot",
                 "x_px","y_px","well_relative_x_px","well_relative_y_px","area_px","raw_distance_px","distance_px"]
STATE_FIELDS=["timestamp_iso","elapsed_seconds","image","frame_number","well","fly_id","fly_slot",
              "observation_status","activity_state","identity_confidence","overlap_group","overlap_count",
              "rolling_distance_px","rolling_samples","immobile_duration_seconds"]
WELL_FIELDS=["timestamp_iso","elapsed_seconds","image","frame_number","well","configured_flies",
             "separate_detections","overlap_blobs","estimated_flies_visible","unknown_slots","threshold_value",
             "registration_succeeded","registration_score"]

class _Csv:
    def __init__(self,path,fields):
        path.parent.mkdir(parents=True,exist_ok=True); self.f=path.open("w",newline="",encoding="utf-8",buffering=1)
        self.w=csv.DictWriter(self.f,fieldnames=fields); self.w.writeheader()
    def write(self,row): self.w.writerow(row); self.f.flush()
    def close(self): self.f.close()

class ExperimentLoggers:
    def __init__(self,analysis_folder:Path):
        self.positions=_Csv(analysis_folder/"fly_positions.csv",POSITION_FIELDS)
        self.states=_Csv(analysis_folder/"fly_states.csv",STATE_FIELDS)
        self.wells=_Csv(analysis_folder/"well_information.csv",WELL_FIELDS)
    def close(self): self.positions.close(); self.states.close(); self.wells.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
