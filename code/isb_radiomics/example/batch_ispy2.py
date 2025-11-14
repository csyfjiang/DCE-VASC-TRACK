"""Batch extraction for ISPY2 cohort."""
from pathlib import Path
from isb_radiomics.extraction import batch_feature_extraction
from isb_radiomics.utils import Parameters

DB_DIR = Path(r"Z:\Radiomics_Projects\ISPY-2\TE\cleaned_dataset_ISPY2")
PRE_DIR = Path(r"Z:\Radiomics_Projects\BreastPCR\ISPY2_image_data_preprocessing\early_DCE-MRI_T0")
EXT_DIR = Path(r"Z:\Radiomics_Projects\BreastPCR\ISPY2_image_data_feature_extraction\early_DCE-MRI_T0")

params = Parameters(
    interpolation_resolution=[1.0, 1.0, 3.0],
    bin_number=32,
    # ... add your full params
)

batch_feature_extraction(
    DB_DIR,
    "early_DCE-MRI_T0.mha",
    "original_FTV_T0_mask.mha",
    params,
    PRE_DIR,
    EXT_DIR,
)