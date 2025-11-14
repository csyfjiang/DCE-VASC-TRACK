"""Low-level utilities: cropping, padding, discretisation, etc."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from typing import Tuple, List, Optional


def mask_cropping(mask: sitk.Image, fixed_rel_margin: int = 0) -> sitk.Image:
    """Crop mask to the tightest bounding box of all non-zero labels."""
    lsif = sitk.LabelStatisticsImageFilter()
    lsif.Execute(mask, mask)
    labels = [l for l in lsif.GetLabels() if l != 0]
    if not labels:
        return mask

    boxes = [np.reshape(lsif.GetBoundingBox(l), (-1, 2)) for l in labels]
    lower = np.min([b[:, 0] for b in boxes], axis=0) - fixed_rel_margin
    upper = np.max([b[:, 1] for b in boxes], axis=0) + fixed_rel_margin

    size = np.array(mask.GetSize())
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, size - 1)

    crop = sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize((lower).astype(int).tolist())
    crop.SetUpperBoundaryCropSize((size - upper - 1).astype(int).tolist())
    return crop.Execute(mask)


def image_boundary(
    image: sitk.Image,
    boundary_image: Optional[sitk.Image] = None,
    transformer: Optional[sitk.Transform] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (origin_index, size) of the bounding box after optional transform."""
    if boundary_image is not None:
        origin = np.array(boundary_image.GetOrigin())
        end = np.array(boundary_image.TransformIndexToPhysicalPoint(boundary_image.GetSize()))
    else:
        origin = np.array(image.GetOrigin())
        end = np.array(image.TransformIndexToPhysicalPoint(image.GetSize()))

    # all 8 corners
    corners = np.array(np.meshgrid(*[origin, end])).T.reshape(-1, len(origin))
    pts = []
    for pt in corners:
        if transformer:
            c = np.array(transformer.GetCenter())
            M = np.array(transformer.GetMatrix()).reshape((len(origin), len(origin)))
            pt = np.dot(pt - c, M) + c
        pts.append(image.TransformPhysicalPointToContinuousIndex(pt))
    pts = np.round(pts).astype(int)
    return np.min(pts, axis=0), np.max(pts, axis=0) - np.min(pts, axis=0) + 1


def image_cropping(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    origin_idx, size = image_boundary(image, mask)
    img_size = np.array(image.GetSize())
    lower = np.maximum(origin_idx, 0)
    upper = np.maximum(img_size - (origin_idx + size), 0)

    crop = sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize(lower.astype(int).tolist())
    crop.SetUpperBoundaryCropSize(upper.astype(int).tolist())
    return crop.Execute(image)


def resegmentation(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    lower_bound: Optional[float] = None,
    higher_bound: Optional[float] = None,
    z_score: Optional[float] = None,
) -> np.ndarray:
    """Apply intensity-based resegmentation inside the VOI."""
    intensity_mask = mask_arr > 1
    values = image_arr[intensity_mask]

    if z_score is not None:
        mean, std = values.mean(), values.std()
        l = mean - z_score * std
        u = mean + z_score * std
        lower_bound = max(l, lower_bound) if lower_bound is not None else l
        higher_bound = min(u, higher_bound) if higher_bound is not None else u

    if lower_bound is not None:
        intensity_mask[image_arr < lower_bound] = False
    if higher_bound is not None:
        intensity_mask[image_arr > higher_bound] = False

    return (mask_arr > 0).astype(int) + intensity_mask.astype(int)


def intensity_discretization(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    bin_number: int = 32,
    bin_size: float = 1.0,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> Tuple[np.ndarray, int, float, float]:
    """Fixed-bin or fixed-width discretisation inside the mask."""
    values = image_arr[mask_arr] if mask_arr.any() else image_arr.flatten()
    lb = lower_bound if lower_bound is not None else values.min()
    ub = upper_bound if upper_bound is not None else values.max()

    if lb == ub:
        return np.ones_like(image_arr, dtype=int), 1, lb, ub

    if bin_number == 0:
        Ng = int(np.floor((ub - lb) / bin_size)) + 1
        disc = np.floor((image_arr - lb) / bin_size).astype(int)
    else:
        disc = np.floor(bin_number * (image_arr - lb) / (ub - lb)).astype(int)
        Ng = bin_number

    disc[disc >= Ng] = Ng - 1
    return disc, Ng, lb, ub


def image_padding(
    image: sitk.Image,
    mask: sitk.Image,
    padding_size: int,
    method: str,
    constant: int = 0,
    slice_wise: bool = False,
) -> Tuple[sitk.Image, sitk.Image, List[Tuple[int, int]]]:
    """Pad image & mask to guarantee a minimum distance from the VOI edge."""
    img_arr = sitk.GetArrayFromImage(image)
    msk_arr = sitk.GetArrayFromImage(mask)

    pad_width: List[Tuple[int, int]] = []
    origin_shift: List[int] = []

    for ax in range(img_arr.ndim):
        if slice_wise and ax == 0:
            pad_width.append((0, 0))
            origin_shift.append(0)
            continue

        idx = np.argwhere(msk_arr.sum(axis=ax) > 0)
        if idx.size == 0:
            pad_width.append((0, 0))
            origin_shift.append(0)
            continue

        min_dist = idx.min()
        max_dist = img_arr.shape[ax] - idx.max() - 1
        left = max(0, padding_size - min_dist + 1)
        right = max(0, padding_size - max_dist + 1)
        pad_width.append((left, right))
        origin_shift.append(-left)

    mode_map = {
        "Nearest value": "edge",
        "Constant value": "constant",
        "Periodic": "wrap",
        "Mirror": "reflect",
    }
    mode = mode_map[method]

    if method == "Constant value":
        img_padded = np.pad(img_arr, pad_width, mode=mode, constant_values=constant)
    else:
        img_padded = np.pad(img_arr, pad_width, mode=mode)
    msk_padded = np.pad(msk_arr, pad_width, mode="constant", constant_values=0)

    padded_img = sitk.GetImageFromArray(img_padded)
    padded_msk = sitk.GetImageFromArray(msk_padded)

    new_origin = image.TransformIndexToPhysicalPoint(origin_shift[::-1])
    padded_img.CopyInformation(image)
    padded_img.SetOrigin(new_origin)
    padded_msk.CopyInformation(mask)
    padded_msk.SetOrigin(new_origin)

    return padded_img, padded_msk, pad_width[::-1]