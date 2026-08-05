from __future__ import annotations
import cv2
import numpy as np

class LiveViewManager:
    WINDOW_NAMES = ("Reference Image", "Registered Image", "Binary Image",
                    "Detect Image", "Tracking Window")
    def __init__(self, reference_bgr, display_width=640):
        self.reference_bgr = reference_bgr.copy()
        self.display_width = display_width
        for name in self.WINDOW_NAMES:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    def _resize(self, image):
        h, w = image.shape[:2]
        if w <= self.display_width:
            return image
        scale = self.display_width / w
        return cv2.resize(image, (self.display_width, int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    @staticmethod
    def _text(image, text, line=0):
        y = 30 + 26 * line
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 255, 255), 2, cv2.LINE_AA)
    def show(self, frame, tracking_image, status, detection_mask=None):
        binary = np.zeros(frame.shape[:2], np.uint8) if detection_mask is None else detection_mask
        detect = frame.copy()
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(detect, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self._text(detect, f"Accepted detector components: {len(contours)}")
        reference = self.reference_bgr.copy()
        self._text(reference, "Median empty/background reference")
        registered = frame.copy()
        self._text(registered, "Registered experiment image")
        tracking = frame.copy() if tracking_image is None else tracking_image.copy()
        self._text(tracking, status)
        for name, image in zip(self.WINDOW_NAMES,
                               (reference, registered, binary, detect, tracking)):
            cv2.imshow(name, self._resize(image))
        return (cv2.waitKey(1) & 0xFF) == ord("q")
