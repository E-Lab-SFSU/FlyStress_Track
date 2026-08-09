from __future__ import annotations

import cv2
import numpy as np


class LiveViewManager:
    WINDOW_NAMES = (
        "Background Image",
        "Grayscale Image",
        "Binary Image",
        "Detect Image",
        "Tracking Window",
    )

    def __init__(self, background_bgr: np.ndarray, display_width: int = 640,
                 binary_threshold: int = 25, minimum_motion_area: int = 20) -> None:
        self.background_bgr = background_bgr.copy()
        self.background_gray = cv2.GaussianBlur(
            cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0
        )
        self.display_width = display_width
        self.binary_threshold = binary_threshold
        self.minimum_motion_area = minimum_motion_area
        for name in self.WINDOW_NAMES:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w <= self.display_width:
            return image
        scale = self.display_width / w
        return cv2.resize(image, (self.display_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _text(image: np.ndarray, text: str, line: int = 0) -> None:
        y = 30 + 26 * line
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

    def _legacy_process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Old whole-plate motion view, retained only as a fallback."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        difference = cv2.absdiff(self.background_gray, blurred)
        _, binary = cv2.threshold(difference, self.binary_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.dilate(binary, kernel, iterations=1)
        detect = frame.copy()
        count = 0
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < self.minimum_motion_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(detect, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1
        self._text(detect, f"Whole-plate motion regions: {count}")
        return gray, binary, detect

    def process(self, frame: np.ndarray, detection_mask: np.ndarray | None = None,
                wells: list[dict[str, object]] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the diagnostic windows.

        When a detector mask is supplied, Detect Image shows the *actual per-well
        candidate mask* used by the fly detector. It no longer displays unrelated
        whole-plate motion regions, which previously made the tracker look as if it
        were considering hundreds of global objects.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if detection_mask is None or detection_mask.shape != gray.shape:
            return self._legacy_process(frame)

        binary = detection_mask.copy()
        detect = frame.copy()
        total_regions = 0

        if wells:
            for well in wells:
                cx = int(float(well["x"]))
                cy = int(float(well["y"]))
                radius = int(float(well["radius"]))
                cv2.circle(detect, (cx, cy), radius, (130, 130, 130), 1)
                cv2.putText(detect, str(well["well"]), (cx - 12, cy - radius - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 1:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(detect, (x, y), (x + w, y + h), (0, 255, 0), 1)
            total_regions += 1

        self._text(detect, f"Per-well fly candidate regions: {total_regions}")
        self._text(detect, "Only candidates inside each fly's own well are considered", 1)
        return gray, binary, detect

    def show(self, frame: np.ndarray, tracking_image: np.ndarray | None,
             status: str, detection_mask: np.ndarray | None = None,
             wells: list[dict[str, object]] | None = None) -> bool:
        gray, binary, detect = self.process(frame, detection_mask=detection_mask, wells=wells)
        background = self.background_bgr.copy()
        self._text(background, "First captured frame")
        tracking = frame.copy() if tracking_image is None else tracking_image.copy()
        self._text(tracking, status)
        cv2.imshow("Background Image", self._resize(background))
        cv2.imshow("Grayscale Image", self._resize(gray))
        cv2.imshow("Binary Image", self._resize(binary))
        cv2.imshow("Detect Image", self._resize(detect))
        cv2.imshow("Tracking Window", self._resize(tracking))
        return (cv2.waitKey(1) & 0xFF) == ord("q")
