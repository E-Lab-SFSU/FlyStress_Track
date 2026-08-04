from __future__ import annotations

from typing import Any

import cv2
import numpy as np


# Window names
BACKGROUND_WINDOW = "Background Image"
GRAYSCALE_WINDOW = "Grayscale Image"
BINARY_WINDOW = "Binary Image"
DETECT_WINDOW = "Detect Image"
TRACKING_WINDOW = "Tracking Window"


class LiveViewManager:
    """
    Creates and updates diagnostic camera windows.

    These windows operate even when the plate wells have not been detected.
    """

    def __init__(
            self,
            background_frame: np.ndarray,
            *,
            display_width: int = 640,
            binary_threshold: int = 25,
            minimum_motion_area: int = 20,
    ) -> None:
        if background_frame is None:
            raise ValueError("background_frame cannot be None.")

        self.background_bgr = background_frame.copy()

        self.background_gray = cv2.cvtColor(
            self.background_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        self.background_gray = cv2.GaussianBlur(
            self.background_gray,
            (5, 5),
            0,
        )

        self.display_width = display_width
        self.binary_threshold = binary_threshold
        self.minimum_motion_area = minimum_motion_area

        self._create_windows()

    def _create_windows(self) -> None:
        """Create all five display windows."""

        window_names = [
            BACKGROUND_WINDOW,
            GRAYSCALE_WINDOW,
            BINARY_WINDOW,
            DETECT_WINDOW,
            TRACKING_WINDOW,
        ]

        for window_name in window_names:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def _resize_for_display(
            self,
            image: np.ndarray,
    ) -> np.ndarray:
        """
        Resize an image for display without changing the original image.
        """

        height, width = image.shape[:2]

        if width <= self.display_width:
            return image

        scale = self.display_width / width
        display_height = int(height * scale)

        return cv2.resize(
            image,
            (self.display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _add_status_text(
            image: np.ndarray,
            text: str,
            line_number: int = 0,
    ) -> None:
        """Draw readable status text onto an image."""

        y = 30 + line_number * 28

        # Black outline
        cv2.putText(
            image,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        # White foreground
        cv2.putText(
            image,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def create_processed_images(
            self,
            frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create grayscale, binary-motion, and motion-detection images.

        The binary image is created by comparing the current frame with the
        first frame captured from the camera.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        difference = cv2.absdiff(
            self.background_gray,
            gray_blurred,
        )

        _, binary = cv2.threshold(
            difference,
            self.binary_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        # Remove isolated camera-noise pixels.
        kernel = np.ones((3, 3), dtype=np.uint8)

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_DILATE,
            kernel,
            iterations=1,
        )

        detect_image = frame.copy()

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        accepted_contours = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.minimum_motion_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            cv2.rectangle(
                detect_image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            center_x = x + width // 2
            center_y = y + height // 2

            cv2.circle(
                detect_image,
                (center_x, center_y),
                4,
                (0, 0, 255),
                -1,
            )

            accepted_contours += 1

        self._add_status_text(
            detect_image,
            f"Motion regions: {accepted_contours}",
        )

        return gray, binary, detect_image

    @staticmethod
    def _draw_wells(
            image: np.ndarray,
            wells: list[dict[str, Any]] | None,
    ) -> None:
        """Draw well circles when well coordinates are available."""

        if not wells:
            return

        for well in wells:
            x = int(well["x"])
            y = int(well["y"])
            radius = int(well["radius"])
            label = str(well["well"])

            cv2.circle(
                image,
                (x, y),
                radius,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                label,
                (x - 18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def create_tracking_image(
            self,
            frame: np.ndarray,
            *,
            wells: list[dict[str, Any]] | None = None,
            tracks: list[dict[str, Any]] | None = None,
            well_detection_error: str | None = None,
    ) -> np.ndarray:
        """
        Create the tracking display.

        It remains available even when wells or fly tracks do not exist.
        """

        tracking_image = frame.copy()

        self._draw_wells(tracking_image, wells)

        if tracks:
            for track in tracks:
                x = track.get("x")
                y = track.get("y")

                if x is None or y is None:
                    continue

                x = int(round(float(x)))
                y = int(round(float(y)))

                name = str(track.get("fly_name", "Fly"))
                state = str(track.get("fly_state", "UNKNOWN"))

                if state == "AWAKE":
                    color = (0, 255, 0)
                elif state == "ASLEEP":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 255)

                cv2.circle(
                    tracking_image,
                    (x, y),
                    7,
                    color,
                    -1,
                )

                cv2.putText(
                    tracking_image,
                    f"{name}: {state}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            self._add_status_text(
                tracking_image,
                f"Active tracks: {len(tracks)}",
            )

        elif wells:
            self._add_status_text(
                tracking_image,
                "Wells detected - waiting for fly tracking",
            )

        else:
            self._add_status_text(
                tracking_image,
                "Tracking unavailable - wells not detected",
            )

            if well_detection_error:
                short_error = well_detection_error[:90]

                self._add_status_text(
                    tracking_image,
                    short_error,
                    line_number=1,
                )

        return tracking_image

    def show(
            self,
            frame: np.ndarray,
            *,
            wells: list[dict[str, Any]] | None = None,
            tracks: list[dict[str, Any]] | None = None,
            well_detection_error: str | None = None,
    ) -> bool:
        """
        Update all windows.

        Returns True when the user presses q.
        """

        gray, binary, detect_image = self.create_processed_images(frame)

        tracking_image = self.create_tracking_image(
            frame,
            wells=wells,
            tracks=tracks,
            well_detection_error=well_detection_error,
        )

        background_display = self.background_bgr.copy()

        self._add_status_text(
            background_display,
            "First camera frame",
        )

        cv2.imshow(
            BACKGROUND_WINDOW,
            self._resize_for_display(background_display),
        )

        cv2.imshow(
            GRAYSCALE_WINDOW,
            self._resize_for_display(gray),
        )

        cv2.imshow(
            BINARY_WINDOW,
            self._resize_for_display(binary),
        )

        cv2.imshow(
            DETECT_WINDOW,
            self._resize_for_display(detect_image),
        )

        cv2.imshow(
            TRACKING_WINDOW,
            self._resize_for_display(tracking_image),
        )

        key = cv2.waitKey(1) & 0xFF

        return key == ord("q")