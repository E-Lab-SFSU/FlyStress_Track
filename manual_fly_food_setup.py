"""Manual initial-fly and food annotation for an image sequence.

Wells continue to come from the existing plate_wells.csv calibration. This
module only adds one initial fly box and zero-or-more food boxes per well.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import config


def _fit(image: np.ndarray, max_w: int, max_h: int):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 0.999:
        return image.copy(), 1.0
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def _crop(image: np.ndarray, well: dict[str, object], pad_fraction: float = 0.12):
    cx, cy, r = int(well["x"]), int(well["y"]), int(well["radius"])
    pad = max(8, int(round(r * pad_fraction)))
    x1, y1 = max(0, cx-r-pad), max(0, cy-r-pad)
    x2, y2 = min(image.shape[1], cx+r+pad+1), min(image.shape[0], cy+r+pad+1)
    return image[y1:y2, x1:x2].copy(), x1, y1


def _binary_preview(frame: np.ndarray, well: dict[str, object]) -> np.ndarray:
    """Use the same dark-object idea as the existing detector for annotation only."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cx, cy, radius = int(well["x"]), int(well["y"]), int(well["radius"])
    usable = max(3, radius - int(config.WELL_MASK_MARGIN_PX))
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), usable, 255, -1)
    pixels = gray[mask > 0]
    out = np.zeros_like(gray)
    if not pixels.size:
        return out
    otsu, _ = cv2.threshold(pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    percentile = float(np.percentile(pixels, config.FLY_DARK_PERCENTILE))
    median = float(np.median(pixels)) - 5.0
    threshold = int(np.clip(min(otsu + config.FLY_THRESHOLD_OFFSET, percentile, median), 0, 255))
    out = cv2.inRange(gray, 0, threshold)
    return cv2.bitwise_and(out, mask)


def _select_one_box(frame: np.ndarray, well: dict[str, object]):
    crop, x0, y0 = _crop(frame, well)
    zoom = float(config.MANUAL_SUBJECT_ZOOM)
    shown = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    title = f"FLY {well['well']} - draw tight box, ENTER/SPACE accept, C cancel/redo"
    while True:
        roi = cv2.selectROI(title, shown, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        x, y, w, h = roi
        if w > 0 and h > 0:
            return [int(round(x/zoom))+x0, int(round(y/zoom))+y0,
                    max(1, int(round(w/zoom))), max(1, int(round(h/zoom)))]
        answer = input(f"No fly selected for {well['well']}. Retry? [Y/n]: ").strip().lower()
        if answer == "n":
            return None


def _select_food_boxes(frame: np.ndarray, well: dict[str, object]):
    binary = _binary_preview(frame, well)
    crop, x0, y0 = _crop(binary, well)
    zoom = float(config.MANUAL_SUBJECT_ZOOM)
    base = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), None,
                      fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    boxes: list[tuple[int,int,int,int]] = []
    dragging = False
    start = current = None

    def mouse(event, x, y, flags, param):
        nonlocal dragging, start, current
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging, start, current = True, (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False; current = (x, y)
            xa, xb = sorted((start[0], current[0])); ya, yb = sorted((start[1], current[1]))
            if xb-xa >= 3 and yb-ya >= 3:
                boxes.append((xa, ya, xb-xa, yb-ya))
            start = current = None
        elif event == cv2.EVENT_RBUTTONDOWN and boxes:
            boxes.pop()

    title = f"FOOD {well['well']} - binary view; drag boxes; ENTER done; U/right-click undo"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse)
    while True:
        view = base.copy()
        for x,y,w,h in boxes:
            cv2.rectangle(view, (x,y), (x+w,y+h), (0,165,255), 2)
        if dragging and start and current:
            cv2.rectangle(view, start, current, (0,255,255), 1)
        cv2.putText(view, f"Food boxes: {len(boxes)} (zero is OK)", (8,22),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13,10): break
        if key in (ord('u'),ord('U')) and boxes: boxes.pop()
        if key in (27,ord('q'),ord('Q')):
            cv2.destroyAllWindows(); raise SystemExit("Fly/food setup cancelled.")
    cv2.destroyWindow(title)
    return [[int(round(x/zoom))+x0, int(round(y/zoom))+y0,
             max(1,int(round(w/zoom))), max(1,int(round(h/zoom)))] for x,y,w,h in boxes]


def load_or_create(first_frame: np.ndarray, wells: list[dict[str, object]], plate_folder: Path):
    path = plate_folder / str(config.FLY_FOOD_SETUP_FILENAME)
    if config.REUSE_EXISTING_FLY_FOOD_SETUP and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("frame_width") == first_frame.shape[1] and data.get("frame_height") == first_frame.shape[0]:
            print(f"Reusing fly/food setup: {path}")
            return data
        print("Saved fly/food setup has different image dimensions; recreating it.")
    if not config.SHOW_WINDOWS:
        raise RuntimeError("Manual fly/food setup is required but SHOW_WINDOWS=False.")

    print("\nFLY SETUP: draw ONE tight box around the fly in each well on the FIRST image.")
    entries = {}
    for well in wells:
        name = str(well["well"])
        entries[name] = {"fly_bbox": _select_one_box(first_frame, well), "food_bboxes": []}
    print("\nFOOD SETUP: mark stationary food regions on the binary preview. Enter immediately for none.")
    for well in wells:
        name = str(well["well"])
        entries[name]["food_bboxes"] = _select_food_boxes(first_frame, well)

    data = {"version": 5, "frame_width": first_frame.shape[1], "frame_height": first_frame.shape[0],
            "wells": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved fly/food setup: {path}")
    return data
