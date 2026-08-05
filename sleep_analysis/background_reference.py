"""Build a temporal median or empty-plate background reference."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from sleep_analysis.registration import register_pair

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

def _sample_paths(paths: list[Path], max_images: int) -> list[Path]:
    if len(paths) <= max_images:
        return paths
    indexes = np.linspace(0, len(paths) - 1, max_images, dtype=int)
    return [paths[int(i)] for i in indexes]

def build_median_reference(paths: list[Path], *, max_images: int,
                           register_samples: bool,
                           registration_settings: dict) -> np.ndarray:
    paths = _sample_paths(paths, max_images)
    if not paths:
        raise RuntimeError("No images were supplied for the background reference.")
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise FileNotFoundError(paths[0])
    gray_stack: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None or image.shape != first.shape:
            continue
        if register_samples and path != paths[0]:
            result = register_pair(first, image, **registration_settings)
            image = result.aligned_bgr if result.succeeded else image
        gray_stack.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if not gray_stack:
        raise RuntimeError("No readable, same-sized images were available.")
    median_gray = np.median(np.stack(gray_stack, axis=0), axis=0).astype(np.uint8)
    return cv2.cvtColor(median_gray, cv2.COLOR_GRAY2BGR)

def resolve_reference(*, experiment_root: Path, experiment_images: list[Path],
                      empty_folder_name: str, cache_filename: str,
                      max_images: int, min_images: int,
                      register_samples: bool, registration_settings: dict,
                      prefer_temporal_experiment_median: bool = True) -> tuple[np.ndarray, str]:
    cache_path = experiment_root / "plate" / cache_filename
    if cache_path.is_file():
        cached = cv2.imread(str(cache_path))
        if cached is not None:
            return cached, f"cached reference: {cache_path}"

    empty_paths = image_files(experiment_root / empty_folder_name)
    if prefer_temporal_experiment_median or len(empty_paths) < min_images:
        source = experiment_images
        label = "temporal experiment median"
    else:
        source = empty_paths
        label = "empty-reference median"

    reference = build_median_reference(
        source, max_images=max_images, register_samples=register_samples,
        registration_settings=registration_settings)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(cache_path), reference):
        raise IOError(f"Could not save background reference: {cache_path}")
    return reference, label
