"""FlyStress Track v2.1 settings: conservative three-fly tracking."""
from pathlib import Path

CAMERA_DEVICE = "/dev/video0"
CAMERA_INDEX = 1
CAMERA_WIDTH = 1600
CAMERA_HEIGHT = 1200
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"
CAMERA_WARMUP_SECONDS = 2.0
CAPTURE_INTERVAL_SECONDS = 1.0

OUTPUT_ROOT = Path.home() / "FlyStress_Experiments"
EXPERIMENT_PREFIX = "exp"
IMAGE_EXTENSION = ".png"
SAVE_CAPTURED_IMAGES = True

PLATE_ROWS = 4
PLATE_COLUMNS = 8
EXPECTED_WELLS = PLATE_ROWS * PLATE_COLUMNS
FLIES_PER_WELL = 3
REUSE_EXISTING_WELL_CALIBRATION = True
MANUAL_WELL_CSV = None
MANUAL_CALIBRATION_DISPLAY_MAX_WIDTH = 1200
MANUAL_CALIBRATION_DISPLAY_MAX_HEIGHT = 850
MANUAL_CALIBRATION_MIN_RADIUS = 10
MANUAL_CALIBRATION_MAX_RADIUS = 300
MANUAL_CALIBRATION_WINDOW_NAME = "Manual Well Calibration"
MANUAL_CALIBRATION_PREVIEW_FILENAME = "plate_wells_manual_preview.png"

SHOW_WINDOWS = True
DISPLAY_WIDTH = 900

# Background model. Empty-reference images are optional. If absent, a temporal
# median is built from the experiment images, which is usually preferable to a
# mismatched empty-plate image.
EMPTY_REFERENCE_FOLDER_NAME = "empty_reference"
BACKGROUND_MEDIAN_MAX_IMAGES = 80
BACKGROUND_MIN_IMAGES = 5
BACKGROUND_REGISTER_SAMPLES = True
BACKGROUND_CACHE_FILENAME = "median_background.png"
PREFER_TEMPORAL_EXPERIMENT_MEDIAN = True

# Preserve almost the whole well while avoiding the strongest selected rim.
WELL_INNER_MASK_SCALE = 0.96
WELL_EDGE_EXCLUSION_PX = 2

# Darker-than-median detector. Flies are expected to be darker than the local
# temporal background. Start at 6; increase if noise appears, decrease if flies
# are missed.
DIFFERENCE_BLUR_KERNEL = 3
BACKGROUND_DIFFERENCE_THRESHOLD = 6
DIFFERENCE_USE_OTSU_FLOOR = False
DIFFERENCE_OTSU_MIN_THRESHOLD = 4
DIFFERENCE_OTSU_MAX_THRESHOLD = 18

# Object geometry. These are intentionally conservative so plate details are
# not accepted as flies.
FLY_MIN_AREA_PX = 5
FLY_MAX_SINGLE_AREA_PX = 150
FLY_MAX_COMPONENT_AREA_PX = 420
FLY_MIN_FILL_RATIO = 0.12
FLY_MAX_ASPECT_RATIO = 6.0
FLY_MORPH_KERNEL = 3
FLY_MORPH_OPEN_ITERATIONS = 0
FLY_MORPH_CLOSE_ITERATIONS = 1
MAX_DETECTION_COMPONENTS_PER_WELL = 5

# Overlap is inferred only after enough ordinary single-fly samples have been
# learned for that well. Until then, ambiguous large blobs remain UNKNOWN.
ENABLE_OVERLAP_DETECTION = True
MIN_SINGLE_AREA_SAMPLES_FOR_OVERLAP = 18
OVERLAP_TWO_FLY_MULTIPLIER = 1.75
OVERLAP_THREE_FLY_MULTIPLIER = 2.80
OVERLAP_MAX_TOTAL_FLIES_PER_WELL = 3

REGISTRATION_MOTION_MODEL = "euclidean"
REGISTRATION_BLUR_KERNEL = 5
REGISTRATION_MAX_ITERATIONS = 100
REGISTRATION_EPSILON = 1e-6
DIFFERENCE_THRESHOLD = 18

MAX_POSITION_JUMP_PX = 65.0
JITTER_THRESHOLD_PX = 3.0
ROLLING_WINDOW_SECONDS = 300.0
SLEEP_DURATION_SECONDS = 300.0
MAX_VALID_SAMPLE_GAP_SECONDS = 2.5
IDENTITY_LOW_CONFIDENCE_FRAMES_AFTER_SPLIT = 8

STORAGE_MODE = "balanced"
DIAGNOSTIC_EVERY_N_FRAMES_BALANCED = 60

def storage_settings() -> dict[str, object]:
    mode = STORAGE_MODE.strip().lower()
    if mode == "full":
        return dict(save_registered=True, save_difference=True, save_binary=True,
                    save_detection_mask=True, save_tracking_overlay=True, every_n_frames=1)
    if mode == "balanced":
        return dict(save_registered=False, save_difference=True, save_binary=True,
                    save_detection_mask=True, save_tracking_overlay=True,
                    every_n_frames=DIAGNOSTIC_EVERY_N_FRAMES_BALANCED)
    if mode == "minimal":
        return dict(save_registered=False, save_difference=False, save_binary=False,
                    save_detection_mask=False, save_tracking_overlay=False, every_n_frames=0)
    raise ValueError("STORAGE_MODE must be 'full', 'balanced', or 'minimal'.")
