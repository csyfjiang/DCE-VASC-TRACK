"""Utilities: validators, angle generators, Parameters dataclass."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict


def manhattan_angles_2d(half: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if half:
        angles = [[0, 0, 1], [0, 1, 0]]
    else:
        angles = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0]]
    i, j, k = zip(*angles)
    return np.array(i), np.array(j), np.array(k)


def manhattan_angles_3d(half: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if half:
        angles = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    else:
        angles = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
    i, j, k = zip(*angles)
    return np.array(i), np.array(j), np.array(k)


def chebyshev_angles_2d(half: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = [-1, 0, 1]
    angles = []
    for x in vals:
        for y in vals:
            if not half and x == y == 0:
                continue
            if half and (x == -1 or (x == 0 and y <= 0)):
                continue
            angles.append([0, x, y])
    i, j, k = zip(*angles)
    return np.array(i), np.array(j), np.array(k)


def chebyshev_angles_3d(half: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = [-1, 0, 1]
    angles = []
    for x in vals:
        for y in vals:
            for z in vals:
                if not half and x == y == z == 0:
                    continue
                if half and (x == -1 or (x == 0 and y == -1) or (x == y == 0 and z <= 0)):
                    continue
                angles.append([x, y, z])
    i, j, k = zip(*angles)
    return np.array(i), np.array(j), np.array(k)


def float_validator(
    value: float | None,
    default: float,
    allow_none: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    lower_incl: bool = False,
    upper_incl: bool = False,
) -> float | None:
    if allow_none and value is None:
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return default
    if lower_bound is not None and (value < lower_bound or (not lower_incl and value == lower_bound)):
        return default
    if upper_bound is not None and (value > upper_bound or (not upper_incl and value == upper_bound)):
        return default
    return value


def int_validator(
    value: int | None,
    default: int,
    allow_none: bool = False,
    lower_bound: int | None = None,
    upper_bound: int | None = None,
    lower_incl: bool = False,
    upper_incl: bool = False,
) -> int | None:
    if allow_none and value is None:
        return None
    try:
        value = int(value)
    except (ValueError, TypeError):
        return default
    if lower_bound is not None and (value < lower_bound or (not lower_incl and value == lower_bound)):
        return default
    if upper_bound is not None and (value > upper_bound or (not upper_incl and value == upper_bound)):
        return default
    return value


def bool_validator(value: bool, default: bool) -> bool:
    return bool(value) if isinstance(value, bool) else default


def float_list_validator(
    value: List[float] | None,
    length: int,
    default_elem: float,
    allow_none: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    lower_incl: bool = False,
    upper_incl: bool = False,
) -> List[float]:
    if not isinstance(value, list):
        return [default_elem] * length
    validated = []
    for i, x in enumerate(value):
        if i >= length:
            break
        val = float_validator(x, default_elem, allow_none, lower_bound, upper_bound, lower_incl, upper_incl)
        if val is not None:
            validated.append(val)
    return validated


def string_list_validator(
    value: List[str],
    default: List[str],
    candidates: List[str] | None = None,
) -> List[str]:
    if not isinstance(value, list):
        return default
    validated = []
    for elem in value:
        try:
            elem = str(elem)
        except (ValueError, TypeError):
            continue
        if candidates is not None and elem not in candidates:
            continue
        validated.append(elem)
    return validated


@dataclass
class Parameters:
    """Configuration for preprocessing & feature extraction."""

    interpolation_resolution: List[float] = field(default_factory=lambda: [1.0, 1.0, 3.0])
    image_interpolator: str = "sitkLinear"
    mask_interpolator: str = "sitkNearestNeighbor"
    mask_partial_volume: float = 0.5
    padding_size: int = 0
    reseg_lower_bound: float | None = None
    reseg_upper_bound: float | None = None
    outlier_sigma: float | None = None
    bin_number: int = 32
    bin_size: float = 1.0
    image_intensity_rounding: bool = False
    discretization_lower_bound: float | None = None
    discretization_upper_bound: float | None = None
    ivh_bin_number: int = 32
    ivh_bin_size: float = 1.0
    feature_classes: List[str] = field(
        default_factory=lambda: [
            "Morphological",
            "Intensity statistical",
            "Local intensity",
            "Intensity volume histogram",
            "Intensity histogram",
            "GLCM",
            "GLDZM",
            "GLRLM",
            "GLSZM",
            "NGLDM",
            "NGTDM",
        ]
    )
    glcm_aggregations: List[str] = field(default_factory=lambda: ["3d_average"])
    gldzm_aggregations: List[str] = field(default_factory=lambda: ["3d_average"])
    glrlm_aggregations: List[str] = field(default_factory=lambda: ["3d_average"])
    glszm_aggregations: List[str] = field(default_factory=lambda: ["3d_average"])
    ngldm_aggregations: List[str] = field(default_factory=lambda: ["3d_average"])
    ngtdm_aggregations: List[str] = field(default_factory=lambda: ["3d_average"])

    def parameter_report(self) -> pd.DataFrame:
        """Generate IBSI-compliant report of parameters."""
        rows = [
            ["Image processing - image interpolation", "Interpolation method", "50a", "Interpolation algorithm", self.image_interpolator],
            ["Image processing - image interpolation", "Interpolation method", "50b", "Interpolation grid", "Align grid centers"],
            # ... add the full list from your original parameter_report()
        ]
        return pd.DataFrame(rows, columns=["Topic", "Subtopic", "Item ID", "Item Description", "Item value"])