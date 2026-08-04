from __future__ import annotations
from dataclasses import dataclass
import cv2, numpy as np
@dataclass(frozen=True)
class RegistrationResult:
    aligned_bgr: np.ndarray; difference: np.ndarray; thresholded_difference: np.ndarray
    correlation: float; succeeded: bool

def register_pair(reference_bgr,current_bgr,*,motion_model='euclidean',blur_kernel=5,max_iterations=100,epsilon=1e-6,difference_threshold=18):
    if reference_bgr.shape!=current_bgr.shape: raise ValueError('Images must have identical dimensions.')
    rg=cv2.GaussianBlur(cv2.cvtColor(reference_bgr,cv2.COLOR_BGR2GRAY),(blur_kernel,blur_kernel),0)
    cg=cv2.GaussianBlur(cv2.cvtColor(current_bgr,cv2.COLOR_BGR2GRAY),(blur_kernel,blur_kernel),0)
    code={'translation':cv2.MOTION_TRANSLATION,'euclidean':cv2.MOTION_EUCLIDEAN,'affine':cv2.MOTION_AFFINE}[motion_model]
    warp=np.eye(2,3,dtype=np.float32); ok=True; corr=float('nan')
    try:
        corr,warp=cv2.findTransformECC(rg,cg,warp,code,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,max_iterations,epsilon),None,1)
        h,w=rg.shape
        aligned=cv2.warpAffine(current_bgr,warp,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
    except cv2.error:
        aligned=current_bgr.copy(); ok=False
    ag=cv2.GaussianBlur(cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY),(blur_kernel,blur_kernel),0)
    diff=cv2.absdiff(rg,ag); _,binary=cv2.threshold(diff,difference_threshold,255,cv2.THRESH_BINARY)
    return RegistrationResult(aligned,diff,binary,float(corr),ok)
