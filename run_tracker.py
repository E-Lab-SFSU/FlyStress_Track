# -*- coding: utf-8 -*-
# Copyright (c) 2025 Thomas Zimmerman — MIT License
"""
run_tracker.py

Entry point for running FlyPipeline on one video or every supported
video in a directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pipeline import FlyPipeline


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

RUN_SINGLE_FILE = True

VIDEO_PATH = Path(
    r"C:\Users\chana\Videos\Screen Recordings\Fly_Test_Vid.mp4"  ----->   # paste path to single video for analysis
)

VIDEO_DIR = Path(r"C:\Users\chana\Videos")                   ------->     # past path to directory (folder) containing 2 or more videos

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
SHOW_WINDOWS = True


# ---------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------

def run_single_video(video_path: Path) -> bool:
    """Run one video. Return True on success."""

    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        print(f"Video file not found: {video_path}")
        return False

    print(f"\nProcessing video:\n  {video_path}")

    try:
        FlyPipeline(str(video_path), show=SHOW_WINDOWS).run()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        return False
    except Exception as error:
        print(f"\nFailed to process {video_path.name}: {error}")
        return False

    return True


def find_videos(video_dir: Path) -> list[Path]:
    """Return supported video files directly inside a directory."""

    video_dir = video_dir.expanduser().resolve()
    if not video_dir.is_dir():
        raise NotADirectoryError(video_dir)

    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def run_video_directory(video_dir: Path) -> None:
    """Run all supported videos in a directory."""

    video_files = find_videos(video_dir)
    if not video_files:
        print(f"No supported videos found in: {video_dir.resolve()}")
        return

    print(f"Found {len(video_files)} video(s) in {video_dir.resolve()}")

    succeeded = 0
    failed = 0

    for index, video_path in enumerate(video_files, start=1):
        print(f"\n[{index}/{len(video_files)}] {video_path.name}")
        if run_single_video(video_path):
            succeeded += 1
        else:
            failed += 1

    print("\nBatch complete.")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")


def main() -> int:
    """Validate settings and start the requested run mode."""

    if RUN_SINGLE_FILE:
        return 0 if run_single_video(VIDEO_PATH) else 1

    try:
        run_video_directory(VIDEO_DIR)
    except NotADirectoryError:
        print(f"Video directory not found: {VIDEO_DIR.expanduser().resolve()}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
