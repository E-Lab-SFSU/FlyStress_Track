# -*- coding: utf-8 -*-
"""
run_tracker.py

Entry point for running the plankton tracking pipeline in:
1) single-video mode, or
2) batch directory mode

CSV output:
For each video, a CSV is written into a subdirectory called "csv"
located in the same directory as the video file.
"""

# Copyright (c) 2025 Thomas Zimmerman — MIT License

import os
from pipeline import FlyPipeline

# -------------------------------------------------
# USER CONFIG
# -------------------------------------------------
RUN_SINGLE_FILE = True   # True = single file, False = batch directory


# --- Single-video mode (debugging) ---
VIDEO_PATH = r"C:\Users\chana\Videos\Screen Recordings\Fly_Test_Vid.mp4"               # <======= REPLACE WITH VIDEO FILE LOCATION FOR SINGLE VIDEO RUN

# --- Batch mode (production) ---
VIDEO_DIR = r"C:/Users/FOO/Videos/Test//"        # <======= REPLACE WITH VIDEO FILE DIRECTORY FOR BATCH RUN

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
SHOW = True        # show visualization windows or not

# -------------------------------------------------


def run_single_video(video_path):
    print(f"Processing single video:\n  {video_path}")

    pipeline = FlyPipeline(video_path, show=SHOW)
    pipeline.run()


def run_video_directory(video_dir):
    video_files = [
        f for f in sorted(os.listdir(video_dir))
        if f.lower().endswith(VIDEO_EXTS)
    ]

    if not video_files:
        print("No video files found in directory.")
        return

    print(f"Found {len(video_files)} videos in directory")

    for i, fname in enumerate(video_files, 1):
        full_path = os.path.join(video_dir, fname)

        print(f"\n[{i}/{len(video_files)}] Processing: {fname}")

        try:
            pipeline = FlyPipeline(full_path, show=SHOW)
            pipeline.run()
        except KeyboardInterrupt:
            print("Batch interrupted by user")
            break


def main():
    if RUN_SINGLE_FILE:
        if not VIDEO_PATH:
            raise ValueError("RUN_SINGLE_FILE=True but VIDEO_PATH is empty")
        if not os.path.isfile(VIDEO_PATH):
            raise FileNotFoundError(VIDEO_PATH)

        run_single_video(VIDEO_PATH)

    else:
        if not VIDEO_DIR:
            raise ValueError("RUN_SINGLE_FILE=False but VIDEO_DIR is empty")
        if not os.path.isdir(VIDEO_DIR):
            raise NotADirectoryError(VIDEO_DIR)

        run_video_directory(VIDEO_DIR)



if __name__ == "__main__":
    main()


