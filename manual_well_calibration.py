"""Simple manual drag-circle calibration for A1-D8."""
from __future__ import annotations
import csv, math
from pathlib import Path
import cv2
import config

def calibrate(image_path:Path,output_csv:Path,*,load_existing=False,**_):
    image=cv2.imread(str(image_path))
    if image is None: raise FileNotFoundError(image_path)
    scale=min(1.0,config.MANUAL_CALIBRATION_DISPLAY_MAX_WIDTH/image.shape[1],config.MANUAL_CALIBRATION_DISPLAY_MAX_HEIGHT/image.shape[0])
    base=cv2.resize(image,None,fx=scale,fy=scale) if scale<1 else image.copy()
    wells=[]; start=None
    def callback(event,x,y,flags,param):
        nonlocal start
        ox,oy=int(round(x/scale)),int(round(y/scale))
        if event==cv2.EVENT_LBUTTONDOWN:start=(ox,oy)
        elif event==cv2.EVENT_LBUTTONUP and start and len(wells)<config.EXPECTED_WELLS:
            r=int(round(math.hypot(ox-start[0],oy-start[1])))
            if config.MANUAL_CALIBRATION_MIN_RADIUS<=r<=config.MANUAL_CALIBRATION_MAX_RADIUS:wells.append((start[0],start[1],r))
            start=None
        elif event==cv2.EVENT_RBUTTONDOWN and wells:wells.pop()
    name=config.MANUAL_CALIBRATION_WINDOW_NAME
    cv2.namedWindow(name,cv2.WINDOW_NORMAL); cv2.setMouseCallback(name,callback)
    saved=False
    while True:
        d=base.copy()
        for i,(x,y,r) in enumerate(wells):
            c=(int(x*scale),int(y*scale)); rr=int(r*scale); label=f"{chr(65+i//config.PLATE_COLUMNS)}{i%config.PLATE_COLUMNS+1}"
            cv2.circle(d,c,rr,(0,255,0),2); cv2.putText(d,label,(c[0]+5,c[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),2)
        cv2.putText(d,f"Draw wells {len(wells)}/{config.EXPECTED_WELLS} | S save | U undo | R reset | Q cancel",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        cv2.imshow(name,d); key=cv2.waitKey(20)&0xFF
        if key in (ord('q'),27):break
        if key==ord('u') and wells:wells.pop()
        if key==ord('r'):wells.clear()
        if key==ord('s') and len(wells)==config.EXPECTED_WELLS:
            output_csv.parent.mkdir(parents=True,exist_ok=True)
            with output_csv.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=['well','row','column','x','y','radius','diameter']);w.writeheader()
                for i,(x,y,r) in enumerate(wells):w.writerow(dict(well=f"{chr(65+i//config.PLATE_COLUMNS)}{i%config.PLATE_COLUMNS+1}",row=i//config.PLATE_COLUMNS+1,column=i%config.PLATE_COLUMNS+1,x=x,y=y,radius=r,diameter=2*r))
            saved=True;break
    cv2.destroyWindow(name);return saved
