"""Core FTV generation logic."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import SimpleITK as sitk
import numpy as np
import pandas as pd

from .utils import array_to_image, setup_logging, ensure_dir


class FTVGenerator:
    """
    Generate Functional Tumor Volume (FTV) masks from DCE-MRI series.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'ISPY2'``).
    patient_id : str
        Patient identifier.
    time_point : str
        MRI time-point label (``'T0'``, ``'T1'``, …).
    data_directory : str | Path, optional
        Root folder containing cleaned data. If ``None`` the class will try to infer it.
    """

    def __init__(
        self,
        dataset_name: str,
        patient_id: str,
        time_point: str,
        data_directory: Optional[str | Path] = None,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.patient_id = str(patient_id)
        self.time_point = str(time_point)
        self.data_directory = Path(data_directory) if data_directory else Path("")
        self.patient_dir = self.data_directory / f"cleaned_dataset_{self.dataset_name}" / self.patient_id

        # default thresholds (minutes → seconds)
        self.early_contrast_time = 2.5 * 60
        self.late_contrast_time = 7.5 * 60
        self.pe_threshold = 70
        self.ser_threshold = 0

        setup_logging(self.patient_dir / "ftv_generation.log")
        self._load_time_info()
        self.pre_tag, self.early_tag, self.late_tag = self._identify_phases()

    # --------------------------------------------------------------------- #
    #                     TIME INFORMATION & PHASE SELECTION                #
    # --------------------------------------------------------------------- #
    def _load_time_info(self) -> None:
        """Read CSV with acquisition times and build ``self.time_dict``."""
        csv_path = (
            self.data_directory
            / "cleaned_time_information_ISPY2"
            / f"DCE_time_point_MRI{self.time_point}_list"
            / f"{self.patient_id}_time.csv"
        )
        if not csv_path.is_file():
            raise FileNotFoundError(f"Time info missing for {self.patient_id}")

        df = pd.read_csv(csv_path)
        self.time_dict: Dict[str, float] = dict(zip(df["name_tag"], df["acquisition_time_difference(s)"]))

    def _identify_phases(self) -> Tuple[str, str, str]:
        """Select pre-, early- and late-contrast phase tags based on timing."""
        tags = list(self.time_dict.keys())
        times = np.array(list(self.time_dict.values()))

        pre_tag = f"DCE_MRI{self.time_point}_Phase_{tags[0].split('_')[-1]}"
        early_idx = int(np.argmin(np.abs(times - self.early_contrast_time)))
        late_idx = int(np.argmin(np.abs(times - self.late_contrast_time)))

        early_tag = f"DCE_MRI{self.time_point}_Phase_{tags[early_idx].split('_')[-1]}"
        late_tag = f"DCE_MRI{self.time_point}_Phase_{tags[late_idx].split('_')[-1]}"

        # sanity checks & fallback to default ordering
        pre_tag, early_tag, late_tag, _ = self._sanity_check_labels(pre_tag, early_tag, late_tag)
        return pre_tag, early_tag, late_tag

    def _sanity_check_labels(
        self, pre: str, early: str, late: str
    ) -> Tuple[str, str, str, bool]:
        """Enforce logical ordering and fallback to 0/1/2 if only 3 phases exist."""
        redo = False

        if pre != f"DCE_MRI{self.time_point}_Phase_0":
            pre = f"DCE_MRI{self.time_point}_Phase_0"
            redo = True

        if early == f"DCE_MRI{self.time_point}_Phase_0":
            early = f"DCE_MRI{self.time_point}_Phase_1"
            redo = True

        if late == early:
            n = int(early.split("_")[-1]) + 1
            late = f"DCE_MRI{self.time_point}_Phase_{n}"
            redo = True

        # If exactly three phases exist → force 0/1/2
        if len(list(self.patient_dir.glob(f"DCE_MRI{self.time_point}_Phase*.mha"))) == 3:
            pre, early, late = (
                f"DCE_MRI{self.time_point}_Phase_0",
                f"DCE_MRI{self.time_point}_Phase_1",
                f"DCE_MRI{self.time_point}_Phase_2",
            )
            redo = True

        return pre, early, late, redo

    # --------------------------------------------------------------------- #
    #                           IMAGE LOADING & ALIGNMENT                   #
    # --------------------------------------------------------------------- #
    def _load_image(self, tag: str) -> sitk.Image:
        path = self.patient_dir / f"{tag}.mha"
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")
        return sitk.Cast(sitk.ReadImage(str(path)), sitk.sitkFloat64)

    def _align_to_reference(self, img: sitk.Image, ref: sitk.Image) -> sitk.Image:
        """Resample *img* to match geometry of *ref* (nearest-neighbor for masks)."""
        if (
            img.GetSize() != ref.GetSize()
            or img.GetSpacing() != ref.GetSpacing()
            or img.GetOrigin() != ref.GetOrigin()
            or img.GetDirection() != ref.GetDirection()
        ):
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            img = resampler.Execute(img)
            img = sitk.Cast(img, sitk.sitkUInt8)
        return img

    # --------------------------------------------------------------------- #
    #                            MAIN FTV PIPELINE                          #
    # --------------------------------------------------------------------- #
    def generate_ftv(self, label: str = "test") -> sitk.Image:
        """
        Compute the final FTV mask and write intermediate files.

        Parameters
        ----------
        label : str
            Suffix attached to output filenames.

        Returns
        -------
        sitk.Image
            Binary FTV mask (UInt8).
        """
        # ---- load DCE phases ------------------------------------------------
        S0 = self._load_image(self.pre_tag)
        S1 = self._load_image(self.early_tag)
        S2 = self._load_image(self.late_tag)

        S0_arr = sitk.GetArrayFromImage(S0)
        S1_arr = sitk.GetArrayFromImage(S1)
        S2_arr = sitk.GetArrayFromImage(S2)

        # ---- VOI (bounding-box mask) ---------------------------------------
        voi_path = self.patient_dir / f"bounding_box_{self.time_point}_mask.mha"
        voi = sitk.ReadImage(str(voi_path))
        voi = self._align_to_reference(voi, S0)
        sitk.WriteImage(voi, str(self.patient_dir / f"VOI_mask_{self.time_point}.mha"))

        # ---- 1) Percent Enhancement (PE) ------------------------------------
        pe_num = (S1_arr - S0_arr) * 100.0
        pe_den = S0_arr.astype(np.float64)
        pe_map = np.divide(pe_num, pe_den, out=np.zeros_like(pe_num), where=pe_den != 0)
        pe_img = array_to_image(pe_map, S0)
        sitk.WriteImage(pe_img, str(self.patient_dir / f"PE_{self.time_point}.mha"))

        pe_mask = pe_img >= self.pe_threshold

        # ---- 2) Signal Enhancement Ratio (SER) -----------------------------
        ser_num = S1_arr - S0_arr
        ser_den = S2_arr - S0_arr
        ser_map = np.divide(ser_num, ser_den, out=np.zeros_like(ser_num), where=ser_den != 0)
        ser_img = array_to_image(ser_map, S0)
        sitk.WriteImage(ser_img, str(self.patient_dir / f"SER_{self.time_point}.mha"))

        ser_mask = ser_img >= self.ser_threshold

        # ---- 3) Noise filtering (5th percentile of pre-contrast in VOI) -----
        voi_vals = S0_arr[sitk.GetArrayFromImage(voi) > 0.5]
        noise_thr = np.percentile(voi_vals, 5) if voi_vals.size else 0.0
        noise_mask = S0 > noise_thr
        sitk.WriteImage(noise_mask, str(self.patient_dir / "noise_mask_all.mha"))

        # ---- Morphological cleaning -----------------------------------------
        opening = sitk.BinaryMorphologicalOpeningImageFilter()
        opening.SetKernelRadius(1)

        pe_mask = opening.Execute(pe_mask)
        ser_mask = opening.Execute(ser_mask)

        # ---- Final FTV ------------------------------------------------------
        ftv = voi * pe_mask * ser_mask * noise_mask
        out_path = self.patient_dir / f"new_FTV_{label}_{self.time_point}_mask.mha"
        sitk.WriteImage(ftv, str(out_path))
        return ftv

    # --------------------------------------------------------------------- #
    #                     ORIGINAL SEGMENTATION → FTV (legacy)            #
    # --------------------------------------------------------------------- #
    def export_original_ftv(self) -> None:
        """Convert the provided segmentation (label 0) to a binary FTV mask."""
        seg_path = self.patient_dir / f"segmentation_MRI{self.time_point}.mha"
        if not seg_path.is_file():
            raise FileNotFoundError(f"Segmentation missing: {seg_path}")

        mask = sitk.Cast(sitk.ReadImage(str(seg_path)) == 0, sitk.sitkUInt8)
        sitk.WriteImage(mask, str(self.patient_dir / "original_FTV.mha"))