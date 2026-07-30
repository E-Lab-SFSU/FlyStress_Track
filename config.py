from pathlib import Path

# Camera
CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 800
CAMERA_FPS = 30
CAPTURE_INTERVAL_SECONDS = 1.0
CAMERA_WARMUP_SECONDS = 2.0

# Plate: fixed 4 x 8 layout = 32 wells
PLATE_ROWS = 4
PLATE_COLUMNS = 8
EXPECTED_WELLS = PLATE_ROWS * PLATE_COLUMNS
CROP_PADDING_PX = 20

# Output
# Windows: ~/Downloads/FS_IMG
# Raspberry Pi/Linux: ~/FS_IMG
if (Path.home() / "Downloads").exists():
    OUTPUT_ROOT = Path.home() / "Downloads" / "FS_IMG"
else:
    OUTPUT_ROOT = Path.home() / "FS_IMG"

IMAGE_EXTENSION = ".png"
PREVIEW_WINDOW_NAME = "FlyStress Camera Preview"

# Registration and image-difference diagnostics
REGISTRATION_MOTION_MODEL = "euclidean"
REGISTRATION_BLUR_KERNEL = 5
REGISTRATION_MAX_ITERATIONS = 100
REGISTRATION_EPSILON = 1e-6
DIFFERENCE_THRESHOLD = 18

# Multi-fly detection inside each well
FLIES_PER_WELL_MAX = 4
WELL_MASK_MARGIN_PX = 8
FLY_DARK_PERCENTILE = 16.0
FLY_THRESHOLD_OFFSET = 4
FLY_MIN_AREA_PX = 12
FLY_MAX_AREA_PX = 450
FLY_MORPH_KERNEL = 3
FLY_MORPH_OPEN_ITERATIONS = 1
FLY_MORPH_CLOSE_ITERATIONS = 1

# Persistent per-well tracking
TRACK_MAX_MATCH_DISTANCE_PX = 35.0
TRACK_MAX_MISSED_FRAMES = 5

# Movement and sleep classification
JITTER_THRESHOLD_PX = 3.0
ROLLING_WINDOW_SECONDS = 300
AWAKE_THRESHOLD_PX = 20.0

from pathlib import Path

# ------------------------------------------------------------
# Raspberry Pi image capture
# ------------------------------------------------------------

# Camera index 0 normally corresponds to /dev/video0.
CAMERA_INDEX = 0

# Requested USB-camera capture mode.
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

# Allow exposure and white balance to stabilize before plate detection.
CAMERA_WARMUP_SECONDS = 2.0

# Image capture interval.
CAPTURE_INTERVAL_SECONDS = 1.0

# Saved image format.
IMAGE_EXTENSION = ".png"

# Preview window title.
PREVIEW_WINDOW_NAME = "Camera Preview"

# Plate configuration.
PLATE_ROWS = WELL_ROWS
PLATE_COLUMNS = WELL_COLS
EXPECTED_WELLS = PLATE_ROWS * PLATE_COLUMNS

# Extra space around the detected plate crop.
CROP_PADDING_PX = 20

# Experiment output directory.
OUTPUT_ROOT = Path.home() / "FlyStress_Experiments"
