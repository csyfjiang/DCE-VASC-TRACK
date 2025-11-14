"""Utility functions: logging, image conversion, file handling."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import SimpleITK as sitk
import numpy as np


def setup_logging(log_file: str | Path = "ftv_generation.log", level: int = logging.INFO) -> None:
    """Configure root logger to write to *log_file*."""
    logging.basicConfig(
        filename=str(log_file),
        filemode="w",
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )


def array_to_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Convert a NumPy array to a SimpleITK image inheriting geometry from *reference*."""
    img = sitk.GetImageFromArray(array.astype(np.float32))
    img.CopyInformation(reference)  # copies origin, spacing, direction
    return img


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p