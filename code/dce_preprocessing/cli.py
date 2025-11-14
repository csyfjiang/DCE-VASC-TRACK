"""CLI entry point: `preprocess-dce`."""
import argparse
from pathlib import Path

import SimpleITK as sitk

from .core import Preprocessing
from ..infrastructure import Parameters


def _load_params(args) -> Parameters:
    # Minimal parameter object – extend as needed
    class P:
        pass

    p = P()
    for k, v in vars(args).items():
        setattr(p, k, v)
    return p  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="DCE-MRI preprocessing & perturbation")
    parser.add_argument("image", type=Path, help="Path to DCE image (.mha/.nii)")
    parser.add_argument("mask", type=Path, help="Path to VOI mask (.mha/.nii)")
    parser.add_argument("-o", "--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--perturb", action="store_true", help="Run perturbation mode")
    parser.add_argument("--times", type=int, default=5, help="Perturbation repetitions")
    # add any extra Parameters you need here
    args = parser.parse_args()

    img = sitk.ReadImage(str(args.image))
    msk = sitk.ReadImage(str(args.mask))
    args.out.mkdir(parents=True, exist_ok=True)

    params = _load_params(args)  # you can extend with a JSON/YAML config later
    proc = Preprocessing(params)

    if args.perturb:
        pert_spec = {
            "Perturbation times": args.times,
            "Rotation angles": [-15, -10, -5, 5, 10, 15],
            "Translation distances": [-2, 0, 2],
            "Contour randomization": {"Smoothing sigma": [10, 10, 10], "Intensity": [1, 1, 1]},
        }
        spec = PerturbationSpecification(pert_spec)
        proc.execute_perturbation(img, msk, spec, export_dir=args.out)
    else:
        proc.execute(img, msk, export_dir=args.out)

    print(f"Finished – results in {args.out}")


if __name__ == "__main__":
    main()