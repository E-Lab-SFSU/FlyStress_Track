from __future__ import annotations
import cv2, numpy as np
class LiveViewManager:
    WINDOW_NAMES=("Background Image","Grayscale Image","Binary Image","Detect Image","Tracking Window")
    def __init__(self,background_bgr,display_width=640):
        self.background_bgr=background_bgr.copy(); self.display_width=display_width
        for n in self.WINDOW_NAMES: cv2.namedWindow(n,cv2.WINDOW_NORMAL)
    def _resize(self,image):
        h,w=image.shape[:2]
        if w<=self.display_width:return image
        s=self.display_width/w
        return cv2.resize(image,(self.display_width,int(h*s)),interpolation=cv2.INTER_AREA)
    @staticmethod
    def _text(image,text,line=0):
        y=30+26*line
        cv2.putText(image,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(image,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
    def show(self,frame,tracking_image,status,detection_mask=None):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        binary=np.zeros_like(gray) if detection_mask is None else detection_mask
        detect=frame.copy()
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h=cv2.boundingRect(c); cv2.rectangle(detect,(x,y),(x+w,y+h),(0,255,0),2)
        self._text(detect,f"Actual detector components: {len(contours)}")
        background=self.background_bgr.copy(); self._text(background,"First captured frame")
        tracking=frame.copy() if tracking_image is None else tracking_image.copy(); self._text(tracking,status)
        for n,img in zip(self.WINDOW_NAMES,(background,gray,binary,detect,tracking)):
            cv2.imshow(n,self._resize(img))
        return (cv2.waitKey(1)&0xFF)==ord('q')
