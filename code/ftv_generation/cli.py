"""Command-line interface."""
import argparse
from pathlib import Path

from .core import FTVGenerator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Functional Tumor Volume (FTV) masks from DCE-MRI."
    )
    parser.add_argument("dataset", help="Dataset name, e.g. ISPY2")
    parser.add_argument("patient_id", help="Patient identifier")
    parser.add_argument("time_point", help="MRI time-point, e.g. T0")
    parser.add_argument(
        "-d", "--data-dir", type=Path, help="Root data directory (optional)"
    )
    parser.add_argument(
        "-l", "--label", default="test", help="Suffix for output files"
    )

    args = parser.parse_args()

    gen = FTVGenerator(
        dataset_name=args.dataset,
        patient_id=args.patient_id,
        time_point=args.time_point,
        data_directory=args.data_dir,
    )
    gen.generate_ftv(label=args.label)
    print(f"FTV mask written to {gen.patient_dir}")


if __name__ == "__main__":
    main()