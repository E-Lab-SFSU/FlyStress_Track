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
OUTPUT_ROOT = Path.home() / "Downloads" / "FS_IMG"
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

# Optional empty-plate/background reference. The supplied reference image is
# included with this project. It is only used when its dimensions match the
# camera frames; otherwise the program warns and safely falls back to the
# normal manual-ROI detector.
USE_BACKGROUND_REFERENCE = True
BACKGROUND_REFERENCE_IMAGE = Path(__file__).resolve().parent / "background_reference.png"

# Single-fly detection inside each well.
WELL_MASK_MARGIN_PX = 10  # exclude more of the circular well wall/rim
FLY_DARK_PERCENTILE = 22.0
FLY_THRESHOLD_OFFSET = 8
FLY_MIN_AREA_PX = 12
FLY_MAX_AREA_PX = 600
FLY_MORPH_KERNEL = 3
FLY_MORPH_OPEN_ITERATIONS = 1
FLY_MORPH_CLOSE_ITERATIONS = 1
MAX_DETECTION_CANDIDATES_PER_WELL = 4
FLY_BACKGROUND_DIFFERENCE_THRESHOLD = 8
FLY_MOTION_DIFFERENCE_THRESHOLD = 6
FLY_MAX_ASPECT_RATIO = 4.0
FLY_MIN_FILL_RATIO = 0.15
FLY_EDGE_EXCLUSION_PX = 8

# Darkness gate. The well wall is visibly lighter than the flies, so cap the
# adaptive per-well threshold at this grayscale value. Lower values are stricter
# (fewer wall/shadow pixels); higher values admit lighter objects.
USE_FLY_DARKNESS_GATE = True
FLY_MAX_GRAY_VALUE = 135

MAX_POSITION_JUMP_PX = 45.0

# Dynamic fly-following / identity settings. The manual initialization box is
# only used on the first image. Thereafter the accepted fly centroid becomes
# the center of the next search. Grayscale similarity helps distinguish the
# fly from fixed dark artifacts, while the reacquisition radius expands after
# missed frames.
GRAYSCALE_IDENTITY_WEIGHT = 0.30
REACQUIRE_JUMP_GROWTH = 0.75
REACQUIRE_MAX_JUMP_PX = 220.0
TRACKING_BOX_HALF_SIZE_PX = 10
# Tracking overlay diagnostics. The crosshair is drawn at the exact (x, y)
# centroid written to fly_positions.csv and used for distance calculations.
# Food ROIs are hidden by default so fixed orange food boxes are not mistaken
# for fly-tracking boxes.
SHOW_FOOD_BOXES_IN_TRACKING_OVERLAY = False
DRAW_TRACKING_CROSSHAIR = True
TRACKING_CROSSHAIR_HALF_SIZE_PX = 7
SHOW_STEP_DISTANCE_IN_OVERLAY = True

# Robust per-well reacquisition. Thresholds are computed independently for each
# circular well. After this many missed frames the tracker may reacquire the fly
# anywhere inside that same well (never in a neighboring well).
FULL_WELL_REACQUIRE_AFTER_MISSES = 2
TRACKING_CONFIDENCE_MIN_FOR_MODEL_UPDATE = 0.62
TRACKING_CONFIDENCE_MIN_ACCEPT = 0.30
FLY_LOCAL_BACKGROUND_PERCENTILE = 70.0
FLY_MIN_CONTRAST_FROM_BACKGROUND = 12.0

# Wall-climbing behavior. Near the well wall a fly can appear as a very small
# dark dot. In this mode, blob size is treated as a weak cue, motion continuity
# and grayscale identity matter more, and morphology is less aggressive.
WALL_MODE_RADIAL_FRACTION = 0.72
WALL_MODE_HOLD_FRAMES = 6
WALL_EDGE_EXCLUSION_PX = 1
WALL_AREA_PENALTY_SCALE = 0.25
MOTION_PREDICTION_WEIGHT = 0.65
APPEARANCE_UPDATE_ALPHA = 0.06
AREA_UPDATE_ALPHA = 0.10
WALL_AREA_UPDATE_ALPHA = 0.02

# Consecutive-frame, per-well tracking. Signed image difference distinguishes
# where a fly arrived (current frame became darker) from the spot it departed
# (current frame became brighter). These cues only become strong when movement
# is visible, so a genuinely stationary/sleeping fly is not penalized.
FRAME_DIFFERENCE_WEIGHT = 1.25
DEPARTURE_MOTION_THRESHOLD = 4.0
ARRIVAL_MOTION_THRESHOLD = 3.0
MOVING_OLD_POSITION_PENALTY = 30.0

# Strong per-well temporal isolation. Consecutive-frame illumination is normalized
# separately inside every circular well before motion is used for candidate ranking.
# The arrival cue is never computed from the whole plate, so motion/reflections in
# A2 cannot create a candidate for B2 (and so on).
NORMALIZE_PER_WELL_ILLUMINATION = True
PER_WELL_ARRIVAL_THRESHOLD = 4.0
PER_WELL_ARRIVAL_MIN_CONTRAST = 6.0
PER_WELL_MOTION_DILATE_ITERATIONS = 1

# The Detect Image debug window now shows the actual per-well candidate mask used
# by fly detection instead of the old whole-plate background-motion contours.
SHOW_ACTUAL_PER_WELL_DETECTIONS = True

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

# Manual fly/food initialization for offline IMAGE-SEQUENCE analysis.
# These settings are additive; all existing detection, movement, sleep, storage,
# camera, and calibration settings above remain unchanged.
USE_MANUAL_FLY_FOOD_SETUP = True
REUSE_EXISTING_FLY_FOOD_SETUP = True
FLY_FOOD_SETUP_FILENAME = "fly_food_setup.json"
MANUAL_SUBJECT_DISPLAY_MAX_WIDTH = 1200
MANUAL_SUBJECT_DISPLAY_MAX_HEIGHT = 850
MANUAL_SUBJECT_ZOOM = 3.0

# Identity protection. The manually selected fly establishes the identity in
# each well. Later detections are chosen primarily by continuity with that fly.
INITIAL_FLY_BOX_PADDING_PX = 8
FOOD_OVERLAP_PENALTY = 35.0
FOOD_CONTACT_SEARCH_RADIUS_PX = 35
FOOD_CONTACT_MAX_JUMP_MULTIPLIER = 1.35
MANUAL_TRACKING_MIN_AREA_PX = 5
MANUAL_TRACKING_MAX_CANDIDATES_PER_WELL = 12
MANUAL_TRACKING_MAX_AREA_PX = 1600
MANUAL_TRACKING_RELAX_SHAPE_FILTER = True
