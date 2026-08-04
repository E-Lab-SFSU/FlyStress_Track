"""FlyStress Track: three flies per manually selected well.
Run with no path for live capture, or pass an experiment/image folder for offline analysis.
"""
from __future__ import annotations
import argparse,csv,math,platform,shutil,time
from datetime import datetime
from pathlib import Path
import cv2,numpy as np
import config
from live_views import LiveViewManager
from manual_well_calibration import calibrate as manual_calibrate
from sleep_analysis.fly_detection import detect_flies
from sleep_analysis.registration import register_pair
from sleep_analysis.multi_fly_tracker import PerWellMultiFlyTracker
from sleep_analysis.results_logger import ExperimentLoggers
IMAGE_EXTENSIONS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

def create_experiment_folder(root):
    root.mkdir(parents=True,exist_ok=True);n=1
    while (root/f"{config.EXPERIMENT_PREFIX}{n:05d}").exists():n+=1
    p=root/f"{config.EXPERIMENT_PREFIX}{n:05d}";p.mkdir();return p

def collect_images(folder):
    def num(p):
        d=''.join(c for c in p.stem if c.isdigit());return int(d) if d else -1
    images=sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),key=num)
    if len(images)<2:raise RuntimeError(f"At least two images required; found {len(images)} in {folder}")
    return images

def load_image(p):
    im=cv2.imread(str(p));
    if im is None:raise FileNotFoundError(p)
    return im

def save_image(p,im):p.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(p),im)

def load_wells(p):
    with p.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    wells=[dict(well=r['well'],row=int(float(r.get('row',0) or 0)),column=int(float(r.get('column',0) or 0)),x=int(float(r['x'])),y=int(float(r['y'])),radius=int(float(r['radius']))) for r in rows]
    if len(wells)!=config.EXPECTED_WELLS:raise RuntimeError(f"Expected {config.EXPECTED_WELLS} wells; found {len(wells)}")
    return wells

def resolve_wells(reference,plate_folder):
    plate_folder.mkdir(parents=True,exist_ok=True);csv_path=plate_folder/'plate_wells.csv'
    if config.REUSE_EXISTING_WELL_CALIBRATION and csv_path.is_file():return load_wells(csv_path)
    if config.MANUAL_WELL_CSV and Path(config.MANUAL_WELL_CSV).is_file():shutil.copy2(config.MANUAL_WELL_CSV,csv_path);return load_wells(csv_path)
    ref=plate_folder/'manual_calibration_reference.png';save_image(ref,reference)
    if not config.SHOW_WINDOWS:raise RuntimeError('Manual calibration required but SHOW_WINDOWS=False.')
    if not manual_calibrate(ref,csv_path,load_existing=False):raise RuntimeError('Calibration canceled.')
    return load_wells(csv_path)

def detection_settings():
    return dict(max_components=config.MAX_DETECTION_COMPONENTS_PER_WELL,mask_margin_px=config.WELL_MASK_MARGIN_PX,
                dark_percentile=config.FLY_DARK_PERCENTILE,threshold_offset=config.FLY_THRESHOLD_OFFSET,min_area_px=config.FLY_MIN_AREA_PX,
                max_single_area_px=config.FLY_MAX_SINGLE_AREA_PX,max_overlap_area_px=config.FLY_MAX_OVERLAP_AREA_PX,
                morph_kernel=config.FLY_MORPH_KERNEL,open_iterations=config.FLY_MORPH_OPEN_ITERATIONS,close_iterations=config.FLY_MORPH_CLOSE_ITERATIONS,
                use_clahe=config.DETECTION_USE_CLAHE,clahe_clip_limit=config.DETECTION_CLAHE_CLIP_LIMIT,clahe_tile_size=config.DETECTION_CLAHE_TILE_SIZE,
                overlap_two_multiplier=config.OVERLAP_TWO_FLY_MULTIPLIER,overlap_three_multiplier=config.OVERLAP_THREE_FLY_MULTIPLIER)

def create_tracker(wells):
    return PerWellMultiFlyTracker([str(w['well']) for w in wells],flies_per_well=config.FLIES_PER_WELL,
                                  max_match_distance_px=config.MAX_POSITION_JUMP_PX,jitter_threshold_px=config.JITTER_THRESHOLD_PX,
                                  rolling_window_seconds=config.ROLLING_WINDOW_SECONDS,sleep_duration_seconds=config.SLEEP_DURATION_SECONDS,
                                  max_valid_sample_gap_seconds=config.MAX_VALID_SAMPLE_GAP_SECONDS,
                                  low_confidence_frames_after_split=config.IDENTITY_LOW_CONFIDENCE_FRAMES_AFTER_SPLIT)

def annotate(image,wells,results):
    out=image.copy()
    for w in wells:cv2.circle(out,(w['x'],w['y']),w['radius'],(150,150,150),1)
    for r in results:
        if r.x_px is not None:
            c=(int(r.x_px),int(r.y_px)); color=(0,255,0) if r.activity_state=='AWAKE' else (0,0,255) if r.activity_state=='ASLEEP' else (0,255,255)
            cv2.circle(out,c,6,color,2);cv2.putText(out,f"{r.fly_id} {r.activity_state} {r.identity_confidence}",(c[0]+7,c[1]-7),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
        elif r.observation_status in ('OVERLAP','UNKNOWN'):
            w=next(x for x in wells if x['well']==r.well); cv2.putText(out,f"{r.fly_id}: {r.observation_status}",(w['x']-w['radius'],w['y']-w['radius']+14*r.fly_slot),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,165,255),1,cv2.LINE_AA)
    return out

def fmt(v,d=3):return '' if v is None else f"{v:.{d}f}"
def log_frame(loggers,results,detections,wells,*,timestamp,elapsed,image,frame,reg_ok,reg_score):
    for r in results:
        common=dict(timestamp_iso=timestamp,elapsed_seconds=f'{elapsed:.3f}',image=image,frame_number=frame,well=r.well,fly_id=r.fly_id,fly_slot=r.fly_slot)
        loggers.positions.write(common|dict(x_px=fmt(r.x_px),y_px=fmt(r.y_px),well_relative_x_px=fmt(r.local_x_px),well_relative_y_px=fmt(r.local_y_px),area_px='' if r.area_px is None else r.area_px,raw_distance_px=fmt(r.raw_distance_px),distance_px=fmt(r.distance_px)))
        loggers.states.write(common|dict(observation_status=r.observation_status,activity_state=r.activity_state,identity_confidence=r.identity_confidence,overlap_group=r.overlap_group,overlap_count=r.overlap_count,rolling_distance_px=f'{r.rolling_distance_px:.3f}',rolling_samples=r.rolling_samples,immobile_duration_seconds=f'{r.immobile_duration_seconds:.3f}'))
    bywell={w['well']:[r for r in results if r.well==w['well']] for w in wells}
    for w in wells:
        ds=detections.get(w['well'],[]); rr=bywell[w['well']]; thresh=next((d.threshold_value for d in ds),'')
        loggers.wells.write(dict(timestamp_iso=timestamp,elapsed_seconds=f'{elapsed:.3f}',image=image,frame_number=frame,well=w['well'],configured_flies=config.FLIES_PER_WELL,separate_detections=sum(d.estimated_fly_count==1 for d in ds),overlap_blobs=sum(d.estimated_fly_count>1 for d in ds),estimated_flies_visible=sum(d.estimated_fly_count for d in ds),unknown_slots=sum(r.observation_status=='UNKNOWN' for r in rr),threshold_value=thresh,registration_succeeded=reg_ok,registration_score='' if reg_score is None or not math.isfinite(reg_score) else f'{reg_score:.8f}'))

def output_folders(base):
    a=base/'analysis';d={'analysis':a,'registered':a/'registered','difference':a/'difference','binary':a/'difference_thresholded','masks':a/'fly_detection_masks','tracking':a/'tracked_fly_overlays'}
    for p in d.values():p.mkdir(parents=True,exist_ok=True)
    return d

def process_frame(frame,reference,tracker,wells,elapsed,first=False):
    if first:aligned=frame;ok=True;score=None;diff=np.zeros(frame.shape[:2],np.uint8);db=diff
    else:
        r=register_pair(reference,frame,motion_model=config.REGISTRATION_MOTION_MODEL,blur_kernel=config.REGISTRATION_BLUR_KERNEL,max_iterations=config.REGISTRATION_MAX_ITERATIONS,epsilon=config.REGISTRATION_EPSILON,difference_threshold=config.DIFFERENCE_THRESHOLD)
        aligned,ok,score,diff,db=r.aligned_bgr,r.succeeded,r.correlation,r.difference,r.thresholded_difference
    detections,mask=detect_flies(aligned,wells,area_hints=tracker.area_hints(),**detection_settings())
    results=tracker.update_all(detections,elapsed,ok);overlay=annotate(aligned,wells,results)
    return aligned,ok,score,diff,db,detections,mask,results,overlay

def analyze_offline(path):
    source=path.expanduser().resolve(); images_folder=source/'images' if (source/'images').is_dir() else source
    output_base=source if (source/'images').is_dir() else source/'FlyStress_analysis';plate=output_base/'plate'
    images=collect_images(images_folder);first=load_image(images[0]);wells=resolve_wells(first,plate);tracker=create_tracker(wells);folders=output_folders(output_base)
    views=LiveViewManager(first,config.DISPLAY_WIDTH) if config.SHOW_WINDOWS else None
    with ExperimentLoggers(folders['analysis']) as logs:
        reference=first
        for n,p in enumerate(images,1):
            frame=load_image(p);elapsed=(n-1)*config.CAPTURE_INTERVAL_SECONDS
            aligned,ok,score,diff,db,dets,mask,res,overlay=process_frame(frame,reference,tracker,wells,elapsed,n==1)
            log_frame(logs,res,dets,wells,timestamp='',elapsed=elapsed,image=p.name,frame=n,reg_ok=ok,reg_score=score)
            if n==1 or (config.storage_settings()['every_n_frames'] and n%int(config.storage_settings()['every_n_frames'])==0):
                save_image(folders['masks']/f'mask_{p.stem}.png',mask);save_image(folders['tracking']/f'tracked_{p.stem}.png',overlay)
                save_image(folders['difference']/f'difference_{p.stem}.png',diff);save_image(folders['binary']/f'binary_{p.stem}.png',db)
            print(f'[{n}/{len(images)}] {p.name} | tracks={sum(r.x_px is not None for r in res)}/{len(res)}')
            if views and views.show(aligned,overlay,'Offline analysis - q to stop',mask):break
    cv2.destroyAllWindows();print(f"Saved: {folders['analysis']}")

def open_camera():
    attempts=[(config.CAMERA_INDEX,cv2.CAP_DSHOW),(config.CAMERA_INDEX,cv2.CAP_ANY)] if platform.system()=='Windows' else [(config.CAMERA_DEVICE,cv2.CAP_V4L2),(config.CAMERA_INDEX,cv2.CAP_ANY)]
    for s,b in attempts:
        cap=cv2.VideoCapture(s,b)
        if cap.isOpened():return cap
        cap.release()
    return cv2.VideoCapture()

def live_analysis():
    exp=create_experiment_folder(Path(config.OUTPUT_ROOT).expanduser());images=exp/'images';plate=exp/'plate';images.mkdir();plate.mkdir();folders=output_folders(exp)
    cap=open_camera();
    if not cap.isOpened():raise RuntimeError('Could not open camera.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,config.CAMERA_WIDTH);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,config.CAMERA_HEIGHT);cap.set(cv2.CAP_PROP_FPS,config.CAMERA_FPS)
    time.sleep(config.CAMERA_WARMUP_SECONDS);ok,first=cap.read();
    if not ok:raise RuntimeError('Could not read camera.')
    wells=resolve_wells(first,plate);tracker=create_tracker(wells);views=LiveViewManager(first,config.DISPLAY_WIDTH) if config.SHOW_WINDOWS else None
    start=time.monotonic();next_sample=start;reference=first;n=0
    try:
        with ExperimentLoggers(folders['analysis']) as logs:
            while True:
                ok,frame=cap.read();
                if not ok:raise RuntimeError('Camera frame read failed.')
                now=time.monotonic()
                if now<next_sample:continue
                n+=1;elapsed=now-start;name=f'image{n:06d}{config.IMAGE_EXTENSION}';save_image(images/name,frame)
                aligned,rok,score,diff,db,dets,mask,res,overlay=process_frame(frame,reference,tracker,wells,elapsed,n==1)
                timestamp=datetime.now().astimezone().isoformat(timespec='milliseconds');log_frame(logs,res,dets,wells,timestamp=timestamp,elapsed=elapsed,image=name,frame=n,reg_ok=rok,reg_score=score)
                if views and views.show(aligned,overlay,'Live analysis - q to stop',mask):break
                next_sample=now+config.CAPTURE_INTERVAL_SECONDS
    finally:cap.release();cv2.destroyAllWindows();print(f'Experiment saved in: {exp}')

def main():
    p=argparse.ArgumentParser();p.add_argument('path',nargs='?',type=Path);a=p.parse_args();live_analysis() if a.path is None else analyze_offline(a.path)
if __name__=='__main__':main()
