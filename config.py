"""User-editable settings for FlyStress Track v1.0."""
from pathlib import Path

# Camera: Raspberry Pi 5 first; Windows compatible.
CAMERA_DEVICE = "/dev/video0"
CAMERA_INDEX = 1
CAMERA_WIDTH = 1600
CAMERA_HEIGHT = 1200
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"
CAMERA_WARMUP_SECONDS = 2.0
CAPTURE_INTERVAL_SECONDS = 1.0

# Output and input.
OUTPUT_ROOT = Path.home() / "FlyStress_Experiments"
EXPERIMENT_PREFIX = "exp"
IMAGE_EXTENSION = ".png"
SAVE_CAPTURED_IMAGES = True

# Plate: one fly per manually calibrated well.
PLATE_ROWS = 4
PLATE_COLUMNS = 8
EXPECTED_WELLS = PLATE_ROWS * PLATE_COLUMNS
REUSE_EXISTING_WELL_CALIBRATION = True
MANUAL_WELL_CSV = None  # Optional shared CSV; usually leave None.
MANUAL_CALIBRATION_IMAGE = r"C:\Users\chana\Downloads\exp00002\image00005.png"
MANUAL_CALIBRATION_OUTPUT = OUTPUT_ROOT / "exp00001" / "plate" / "plate_wells.csv"
MANUAL_CALIBRATION_DISPLAY_MAX_WIDTH = 1200
MANUAL_CALIBRATION_DISPLAY_MAX_HEIGHT = 850
MANUAL_CALIBRATION_MIN_RADIUS = 20
MANUAL_CALIBRATION_MAX_RADIUS = 220
MANUAL_CALIBRATION_WINDOW_NAME = "Manual Well Calibration"
MANUAL_CALIBRATION_PREVIEW_FILENAME = "plate_wells_manual_preview.png"

# Five diagnostic windows. Set False for headless/SSH runs.
SHOW_WINDOWS = True
DISPLAY_WIDTH = 640
BACKGROUND_BINARY_THRESHOLD = 25
MINIMUM_MOTION_AREA_PX = 20

# Registration and image difference.
REGISTRATION_MOTION_MODEL = "euclidean"  # translation, euclidean, affine
REGISTRATION_BLUR_KERNEL = 5
REGISTRATION_MAX_ITERATIONS = 100
REGISTRATION_EPSILON = 1e-6
DIFFERENCE_THRESHOLD = 18

# Single-fly detection inside each well.
WELL_MASK_MARGIN_PX = 8
FLY_DARK_PERCENTILE = 16.0
FLY_THRESHOLD_OFFSET = 4
FLY_MIN_AREA_PX = 12
FLY_MAX_AREA_PX = 600
FLY_MORPH_KERNEL = 3
FLY_MORPH_OPEN_ITERATIONS = 1
FLY_MORPH_CLOSE_ITERATIONS = 1
MAX_DETECTION_CANDIDATES_PER_WELL = 4
MAX_POSITION_JUMP_PX = 45.0

# Movement and sleep state.
JITTER_THRESHOLD_PX = 3.0
ROLLING_WINDOW_SECONDS = 300.0
SLEEP_DURATION_SECONDS = 300.0
MAX_VALID_SAMPLE_GAP_SECONDS = 2.5

# Storage. Every sampled frame is analyzed in every mode.
# full: all raw and diagnostic images
# balanced: all raw images; diagnostics every N frames
# minimal: all raw images and CSVs; no diagnostic image sets
STORAGE_MODE = "balanced"
DIAGNOSTIC_EVERY_N_FRAMES_BALANCED = 60
MIN_FREE_SPACE_GB = 5.0
STOP_WHEN_DISK_LOW = True


def storage_settings() -> dict[str, object]:
    mode = STORAGE_MODE.strip().lower()
    if mode == "full":
        return dict(save_registered=True, save_difference=True, save_binary=True,
                    save_detection_mask=True, save_tracking_overlay=True, every_n_frames=1)
    if mode == "balanced":
        return dict(save_registered=False, save_difference=True, save_binary=True,
                    save_detection_mask=False, save_tracking_overlay=True,
                    every_n_frames=DIAGNOSTIC_EVERY_N_FRAMES_BALANCED)
    if mode == "minimal":
        return dict(save_registered=False, save_difference=False, save_binary=False,
                    save_detection_mask=False, save_tracking_overlay=False, every_n_frames=0)
    raise ValueError("STORAGE_MODE must be 'full', 'balanced', or 'minimal'.")
MANUAL_CALIBRATION_LOAD_EXISTING = True
