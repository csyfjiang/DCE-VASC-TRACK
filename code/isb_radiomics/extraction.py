"""Batch extraction pipelines (ISPY, IBSI, PyRadiomics wrappers)."""
from __future__ import annotations

import os
import time
from pathlib import Path
import tracemalloc
import psutil
import json

import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from radiomics.featureextractor import RadiomicsFeatureExtractor

from .core import FeatureCalculation
from .utils import Parameters
# Assuming you have a Preprocessing class from previous package
from dce_preprocessing.core import Preprocessing  # import if separate package


def image_data_compression(image: sitk.Image) -> tuple[sitk.Image, float, float]:
    """Compress image to uint16 for storage."""
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    max_val = stats.GetMaximum()
    min_val = stats.GetMinimum()
    rng = max_val - min_val
    slope = np.iinfo(np.uint16).max / rng
    intercept = min_val if max_val > 0 else max_val  # handle negatives
    if max_val < 0:
        slope = -slope
    image = (image - intercept) * slope
    return sitk.Cast(image, sitk.sitkUInt16), intercept, slope


def feature_extraction(
    image: sitk.Image,
    mask: sitk.Image,
    preprocessor: Preprocessing,
    feature_calc: FeatureCalculation,
    preprocess_dir: Path | None = None,
    extract_dir: Path | None = None,
    voxel_wise: bool = False,
    kernel_size: int = 5,
) -> pd.DataFrame | Dict[str, np.ndarray]:
    """Extract features (voxel-wise or ROI-based) post-preprocessing."""
    res = preprocessor.execute(image, mask, export_directory=preprocess_dir)
    if res is None:
        raise ValueError("Preprocessing failed.")
    filter_names, interp_imgs, reseg_masks = res

    if voxel_wise:
        maps: Dict[str, np.ndarray] = {}
        headers: Dict[str, tuple[float, float]] = {}
        for i, name in enumerate(filter_names):
            img_arr = sitk.GetArrayFromImage(interp_imgs[i])
            mask_arr = sitk.GetArrayFromImage(reseg_masks[i])
            res_arr = np.flip(img_arr.GetSpacing())
            feats = feature_calc.voxel_wise_feature_calculation(
                img_arr, mask_arr, kernel_size, res_arr, name
            )
            for feat_name, feat_map in feats.items():
                valid = feat_map[(mask_arr > 0)]
                min_val = valid.min() if valid.size else 0
                feat_map[~(mask_arr > 0)] = min_val
                feat_img = sitk.GetImageFromArray(feat_map)
                feat_img.CopyInformation(interp_imgs[i])
                compressed, intercept, slope = image_data_compression(feat_img)
                maps[feat_name] = sitk.GetArrayFromImage(compressed)
                headers[feat_name] = (intercept, slope)
                if extract_dir:
                    sitk.WriteImage(compressed, extract_dir / f"{feat_name.replace('.', '_')}.mha")

        header_df = pd.DataFrame.from_dict(headers, orient="index", columns=["Intercept", "Slope"])
        if extract_dir:
            header_df.to_csv(extract_dir / "feature_map_headers.csv")
        return maps

    else:
        if len(interp_imgs) == 1:
            img_arr = sitk.GetArrayFromImage(interp_imgs[0])
            mask_arr = sitk.GetArrayFromImage(reseg_masks[0])
            res_arr = np.flip(interp_imgs[0].GetSpacing())
            feats = feature_calc.individual_feature_calculation(img_arr, mask_arr, res_arr, filter_names[0])
        else:
            img_arr = np.array([sitk.GetArrayFromImage(img) for img in interp_imgs])
            mask_arr = np.array([sitk.GetArrayFromImage(msk) for msk in reseg_masks])
            res_arr = np.flip(interp_imgs[0].GetSpacing())
            feats = feature_calc.stack_feature_calculation(img_arr, mask_arr, res_arr, filter_names)

        if extract_dir:
            feats.to_csv(extract_dir / "feature_values.csv")
        return feats


def batch_feature_extraction(
    db_dir: Path,
    img_name: str,
    mask_name: str,
    params: Parameters,
    preprocess_dir: Path,
    extract_dir: Path,
    voxel_wise: bool = False,
    kernel_size: int = 5,
    mask_labels: List[int] | None = None,
    perturbation: bool = False,
) -> None:
    """Batch-process a dataset (e.g., ISPY2/ISPY1)."""
    tracemalloc.start()
    patients = sorted(db_dir.iterdir())
    combined_feats = {}
    preprocessor = Preprocessing(params)
    feature_calc = FeatureCalculation(params)

    for patient in tqdm(patients):
        if not patient.is_dir():
            continue
        img_path = patient / img_name
        mask_path = patient / mask_name
        if not (img_path.exists() and mask_path.exists()):
            continue

        img = sitk.ReadImage(str(img_path))
        mask = sitk.ReadImage(str(mask_path))

        patient_pre = preprocess_dir / patient.name
        patient_ext = extract_dir / patient.name
        patient_pre.mkdir(exist_ok=True)
        patient_ext.mkdir(exist_ok=True)

        feats = feature_extraction(
            img,
            mask,
            preprocessor,
            feature_calc,
            preprocess_dir=patient_pre,
            extract_dir=patient_ext,
            voxel_wise=voxel_wise,
            kernel_size=kernel_size,
        )
        combined_feats[patient.name] = feats

    # Save combined results
    combined_df = pd.concat(combined_feats, axis=1)  # adjust if voxel-wise (dict)
    combined_df.to_csv(extract_dir / "combined_features.csv")

    # Memory stats
    mem = tracemalloc.get_traced_memory()
    print(f"Memory: {mem[1] / 1024**2:.2f} MB peak")
    tracemalloc.stop()


# Wrappers for MIRP/PyRadiomics (add your implementations)
def mirp_batch_feature_extraction(...):
    pass  # implement as needed

def pyradiomics_batch_feature_extraction(...):
    pass  # implement as needed