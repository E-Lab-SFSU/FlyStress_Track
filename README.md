# Fly-Tracking-Pipeline
Modular video analysis pipeline for fly behavior studies. Performs detection, tracking, motion, and morphology analysis, and rule-based behavior classification (awake vs inactive). Determines when a fly is asleep Includes standalone tuning tools for reproducible ROI and threshold calibration.


---

## Design Principles

- **Separation of concerns**
  - Tuning ≠ tracking ≠ behavior analysis
- **Causal computation**
  - No future information used for velocity or state
- **Reproducibility**
  - All parameters recorded in per-run config manifests
- **Biological interpretability**
  - Motion + morphology jointly define behavioral state

---

## Intended Use

1. Tune detection parameters using standalone tools
2. Copy final values into `config.py`
3. Run the main pipeline on individual videos or directories
4. Perform higher-level behavioral analysis in post-processing

---

````markdown
---

## Running the Pipeline

### 1. Select video input (`run_tracker.py`)

Open `run_tracker.py` and choose **single-file** or **batch** mode.

**Single video (recommended for debugging):**
```python
RUN_SINGLE_FILE = True
VIDEO_PATH = r"C:/path/to/your/video.mov"
````

**Batch mode (process all videos in a directory):**

```python
RUN_SINGLE_FILE = False
VIDEO_DIR = r"C:/path/to/video_directory"
```

In batch mode, the pipeline will:

* iterate over all supported video files in the directory
* create a `csv/` subdirectory alongside the videos
* save one CSV file per video

---

### 2. Set analysis parameters (`config.py`)

Edit `config.py` to define detection, motion, and behavior parameters.

**ROI / Mask (from mask tuner):**

```python
ENABLE_MASK = True
XC = 320
YC = 256
RADIUS = 128
```

**Detection thresholds (from threshold tuner):**

```python
THRESH = 10
MIN_A = 10
MAX_A = 824
MIN_WH = 2
MAX_WH = 52
```

**Motion and behavior parameters:**

```python
FPS = 30
VELOCITY_WINDOW = 8
MIN_MOVEMENT_SPEED_PX_S = 15.0
MIN_DISPLACEMENT_PX = 2.0
```

---

### 3. Run the pipeline

From a terminal in the project directory:

```bash
python run_tracker.py
```

---

### 4. Runtime controls

* **ESC** or **q** — stop the current video
* **Ctrl + C** — stop batch processing

---

### Workflow summary

1. Tune mask and threshold using the standalone tools
2. Copy final values into `config.py`
3. Select video file or directory in `run_tracker.py`
4. Run the pipeline and analyze CSV outputs

---

```

---

### One-line takeaway
> This section cleanly documents **how to run the pipeline** without exposing internal implementation details.

If you want, I can also add:
- a **Quick Start** (5-line version),
- a **Troubleshooting** section,
- or a **Reproducibility note** for reviewers.
```

## License

MIT License (C) 2025, 2026 Thomas G. Zimmerman

