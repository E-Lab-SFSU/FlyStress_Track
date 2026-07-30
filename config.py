from pathlib import Path

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 800
CAMERA_FPS = 30
CAPTURE_INTERVAL_SECONDS = 1.0
CAMERA_WARMUP_SECONDS = 2.0

# Plate
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
