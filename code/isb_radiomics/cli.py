"""CLI: extract-radiomics."""
import argparse
from pathlib import Path

import SimpleITK as sitk

from .extraction import batch_feature_extraction
from .utils import Parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Radiomics feature extraction (IBSI-compliant).")
    parser.add_argument("db_dir", type=Path, help="Database root (patient folders)")
    parser.add_argument("img_name", help="Image filename pattern (e.g., 'DCE_MRIT0_Phase_0.mha')")
    parser.add_argument("mask_name", help="Mask filename pattern (e.g., 'new_FTV_GLCM_SS_T0_mask.mha')")
    parser.add_argument("-p", "--preprocess-dir", type=Path, required=True, help="Preprocessing output dir")
    parser.add_argument("-e", "--extract-dir", type=Path, required=True, help="Feature extraction output dir")
    parser.add_argument("--params", type=Path, help="JSON params file (optional)")
    parser.add_argument("--voxel-wise", action="store_true", help="Voxel-wise extraction")
    parser.add_argument("--kernel-size", type=int, default=5, help="Kernel for voxel-wise")
    args = parser.parse_args()

    if args.params:
        with open(args.params, "r") as f:
            params_dict = json.load(f)
        params = Parameters(**params_dict)
    else:
        params = Parameters()  # defaults

    batch_feature_extraction(
        args.db_dir,
        args.img_name,
        args.mask_name,
        params,
        args.preprocess_dir,
        args.extract_dir,
        voxel_wise=args.voxel_wise,
        kernel_size=args.kernel_size,
    )
    print(f"Finished – results in {args.extract_dir}")


if __name__ == "__main__":
    main()