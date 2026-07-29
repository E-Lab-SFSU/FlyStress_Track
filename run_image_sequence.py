# -*- coding: utf-8 -*-
"""
run_image_sequence.py

Entry point for the simplified one-image-per-second sleep pipeline.

Expects a directory of imageNNNNNN.png files (already cropped to the
48 wells, one image captured per second) and writes tracking + mass
state CSVs into an image_dir/csv/ subdirectory.
"""

from __future__ import annotations

import config
from image_pipeline import ImagePipeline


# Set this to the directory containing your image000001.png, ... sequence,
# or just edit config.IMAGE_DIR directly.
IMAGE_DIR = config.IMAGE_DIR
SHOW = False  # set True to preview alignment/detection/well overlay live


def main() -> None:
    pipeline = ImagePipeline(IMAGE_DIR, show=SHOW)
    pipeline.run()


if __name__ == "__main__":
    main()