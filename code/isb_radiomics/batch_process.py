"""Batch-process an entire patient folder (as used in your ISPY-2 pipeline)."""
import os
from pathlib import Path
from dce_preprocessing.core import Preprocessing
from dce_preprocessing.perturbation import PerturbationSpecification
import SimpleITK as sitk

ROOT = Path(r"Z:\Radiomics_Projects\ISPY-2\TE\cleaned_dataset_ISPY2")
OUT_ROOT = Path(r"Z:\Radiomics_Projects\ISPY-2\TE\preprocessed")

params = {
    "interpolation_resolution": [1, 1, 1],
    "image_interpolator": "LIN",
    "mask_interpolator": "NNB",
    "padding_size": 5,
    "reseg_lower_bound": -1000,
    "reseg_upper_bound": 400,
    "bin_number": 32,
    # … add the rest of your Parameters
}

proc = Preprocessing(params)  # type: ignore

for patient in ROOT.iterdir():
    if not patient.is_dir():
        continue
    img = sitk.ReadImage(str(patient / "early_contrast_DCE.mha"))
    mask = sitk.ReadImage(str(patient / "original_FTV_T0_mask.mha"))

    out_dir = OUT_ROOT / patient.name
    out_dir.mkdir(parents=True, exist_ok=True)

    proc.execute(img, mask, export_dir=out_dir)
    print(f"{patient.name} done")