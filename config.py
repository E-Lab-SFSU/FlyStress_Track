# Copyright (c) 2025 Thomas Zimmerman — MIT License
"""
config.py
Central configuration file for the plankton tracking pipeline.
All tuning parameters should be modified here rather than in code.
"""

# Video / timing
FPS = 1
PLAYBACK_DELAY = 1   # milliseconds; 1 = real-time-ish, 0 = pause
HREZ = 640  # Display resolution
VREZ = 512

# Background modeling
BG_SAMPLES = 100   # number of frames used to build median background

# Circular detection mask
ENABLE_MASK = False      # True = apply circular ROI, False = full frame
MASK_XC = 320
MASK_YC = 256
MASK_RADIUS = 128

# Detection / thresholding
THRESH_METHOD = "fixed"   # "fixed" or "otsu"
THRESH = 25
MIN_A = 10
MAX_A = 824
MIN_WH = 2
MAX_WH = 52
MIN_FRAME_DISPLACEMENT_PX = 0.5
DETECT_WINDOW_FRAMES = 5
DETECT_MIN_COUNT = 3


# Tracking
MAX_HISTORY = 50        # stored centroid history (memory budget)
MAX_TRACK_DIST = 50
MAX_LOST_FRAMES = 10

# Motion / behavior
VELOCITY_WINDOW_FRAMES = 15    # window for averaged velocity
K_CONSECUTIVE_FRAMES = 3      # anti-flicker requirement
MIN_MOVEMENT_SPEED_PX_S = 8.0 # estimated from jitter + FPS
BALL_ASPECT_EPS = 0.25
NUM_CAP_LINES = 4
MIN_DISPLACEMENT_PX = 2.0   # net displacement gate for windowed velocity

# -------------------------
# Well Plates
# -------------------------
WELL_ROWS = 4
WELL_COLS = 8
FULL_WELL = WELL_ROWS * WELL_COLS

# Expected (E) flies & activity
E_FLY_PER_WELL = 2      #expected fly per well
E_TOTAL_FLIES = FULL_WELL * E_FLY_PER_WELL      # total number of expected flies

SLEEP_SEC = 300.0       # 5 minutes | time used to determine when fly is sleep
SLEEP_AMT = 50          # % | when x amount of flies sleep...
INACTIVE_RNG = 0.5        # pixels | range a fly can move and still be considered inactive

# well display
SHOW_OVERLAY = True
SHOW_LABELS = True
SHOW_ASSIGNMENT_BOUNDARY = False
                            # Well
WELL_TL = (285.0, 320.0)     # center of top-left physical well
WELL_TR = (1390.0, 290.0)    # center of top-right physical well
WELL_BL = (180.0, 850.0)     # center of bottom-left physical well
WELL_BR = (1375.0, 850.0)    # center of bottom-right physical well

WELL_DIAMETER = 150.0       # pixels
WELL_MARGIN = 2.0       # pixels | used to exclude well wells

# ------------------------------------------------------------
# Automatic well detection
# ------------------------------------------------------------

WELL_ROWS = 4
WELL_COLS = 8
FULL_WELL = WELL_ROWS * WELL_COLS

# Actual well-circle range in the reference image.
WELL_MIN_RADIUS = 45
WELL_MAX_RADIUS = 90

# Minimum spacing between well-circle candidates.
WELL_MIN_DISTANCE = 105

# Lower values detect more circles.
WELL_HOUGH_PARAM2 = 25.0

# Mounting-bolt detection.
BOLT_MIN_RADIUS = 70
BOLT_MAX_RADIUS = 145
BOLT_PADDING = 5

# Normalize all physical wells to one common radius.
NORMALIZE_WELL_RADII = True
WELL_RADIUS_TOLERANCE = 2

# Save calibration outputs.
SAVE_WELL_GEOMETRY_CSV = True
SAVE_WELL_DEBUG_IMAGE = True

# Output settings
SAVE_POSITION_CSV = True
SAVE_MASS_STATE_CSV = True
CSV_FLUSH_INTERVAL_FRAMES = 30

# Notes:
# - Velocity ramps smoothly over VELOCITY_WINDOW_FRAMES
# - Movement states: SWIM_FORWARD, SWIM_BACKWARD, ATTACHED
# - No IMMOBILE state (removed by background subtraction)
