"""Core feature calculation & aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict

from .utils import Parameters
from .FeatureClasses.glcm import GLCM
from .FeatureClasses.gldzm import GLDZM
from .FeatureClasses.glrlm import GLRLM
from .FeatureClasses.glszm import GLSZM
from .FeatureClasses.intensity_histogram import IntensityHistogram
from .FeatureClasses.intensity_statistical import IntensityStatistical
from .FeatureClasses.intensity_volume_histogram import IntensityVolumeHistogram
from .FeatureClasses.local_intensity import LocalIntensity
from .FeatureClasses.morphological import Morphological
from .FeatureClasses.ngldm import NGLDM
from .FeatureClasses.ngtdm import NGTDM
from .utils import intensity_discretization  # reuse from preprocessing if needed


class FeatureCalculation:
    """
    Aggregate & compute radiomics features (IBSI-compliant).

    Parameters
    ----------
    params : Parameters
        Configuration (resolutions, binning, feature classes, etc.).
    """

    def __init__(self, params: Parameters):
        self.params = params
        self.calculators: Dict[str, object] = {
            "Morphological": Morphological(),
            "Intensity statistical": IntensityStatistical(),
            "Local intensity": LocalIntensity(),
            "Intensity volume histogram": IntensityVolumeHistogram(
                bin_number=params.ivh_bin_number,
                bin_size=params.ivh_bin_size,
                lower_bound=params.reseg_lower_bound,
                upper_bound=params.reseg_upper_bound,
            ),
            "Intensity histogram": IntensityHistogram(),
            "GLCM": GLCM(aggregation_names=params.glcm_aggregations),
            "GLDZM": GLDZM(aggregation_names=params.gldzm_aggregations),
            "GLRLM": GLRLM(aggregation_names=params.glrlm_aggregations),
            "GLSZM": GLSZM(aggregation_names=params.glszm_aggregations),
            "NGLDM": NGLDM(aggregation_names=params.ngldm_aggregations),
            "NGTDM": NGTDM(aggregation_names=params.ngtdm_aggregations),
        }

    def individual_feature_calculation(
        self,
        interp_img_arr: np.ndarray,
        mask_arr: np.ndarray,
        resolution: np.ndarray,
        filter_name: str,
    ) -> pd.DataFrame:
        """Compute features for a single image/mask (ROI-based)."""
        lower, upper = (
            (self.params.reseg_lower_bound, self.params.reseg_upper_bound)
            if filter_name == "Original"
            else (None, None)
        )
        disc_arr, Ng, lb, ub = intensity_discretization(
            interp_img_arr,
            mask_arr > 1,
            bin_number=self.params.bin_number,
            bin_size=self.params.bin_size,
            lower_bound=lower,
            upper_bound=upper,
        )

        interp_img_arr = interp_img_arr.astype(np.float64)
        disc_arr = disc_arr.astype(np.uint32)
        morph_mask = (mask_arr > 0).astype(np.uint8)
        inten_mask = (mask_arr > 1).astype(np.uint8)

        tables: List[pd.DataFrame] = []
        for cls in self.params.feature_classes:
            if cls in self.calculators:
                table = self.calculators[cls].calculate(
                    interp_img_arr,
                    disc_arr,
                    morph_mask,
                    inten_mask,
                    Ng,
                    resolution=resolution,
                )
                table.index = [(filter_name, idx) for idx in table.index]
                tables.append(table)

        return pd.concat(tables, axis=0)

    def voxel_wise_feature_calculation(
        self,
        interp_img_arr: np.ndarray,
        mask_arr: np.ndarray,
        kernel_size: int,
        resolution: np.ndarray,
        filter_name: str,
    ) -> Dict[str, np.ndarray]:
        """Compute voxel-wise feature maps."""
        lower, upper = (
            (self.params.reseg_lower_bound, self.params.reseg_upper_bound)
            if filter_name == "Original"
            else (None, None)
        )
        disc_arr, Ng, lb, ub = intensity_discretization(
            interp_img_arr,
            mask_arr > 1,
            bin_number=self.params.bin_number,
            bin_size=self.params.bin_size,
            lower_bound=lower,
            upper_bound=upper,
        )

        interp_img_arr = interp_img_arr.astype(np.float64)
        disc_arr = disc_arr.astype(np.uint32)
        morph_mask = (mask_arr > 0).astype(np.uint8)
        inten_mask = (mask_arr > 1).astype(np.uint8)

        feature_maps: Dict[str, np.ndarray] = {}
        for cls in self.params.feature_classes:
            if cls in self.calculators:
                table = self.calculators[cls].calculate_kernel(
                    interp_img_arr,
                    disc_arr,
                    morph_mask,
                    inten_mask,
                    Ng,
                    kernel_size,
                    resolution=resolution,
                )
                for name, values in table.values():
                    feature_maps[(filter_name, name)] = values

        return feature_maps

    def stack_feature_calculation(
        self,
        interp_img_arr: np.ndarray,
        mask_arr: np.ndarray,
        resolution: np.ndarray,
        filter_names: List[str],
        sample_index: List[int] | None = None,
    ) -> pd.DataFrame:
        """Compute features for a stack of filtered images."""
        disc_arrs: List[np.ndarray] = []
        Ngs: List[int] = []
        for i in range(interp_img_arr.shape[0]):
            lower, upper = (
                (self.params.reseg_lower_bound, self.params.reseg_upper_bound)
                if filter_names[i] == "Original"
                else (None, None)
            )
            disc, Ng, lb, ub = intensity_discretization(
                interp_img_arr[i],
                mask_arr[i] > 1,
                bin_number=self.params.bin_number,
                bin_size=self.params.bin_size,
                lower_bound=lower,
                upper_bound=upper,
            )
            disc_arrs.append(disc)
            Ngs.append(Ng)

        disc_arrs = np.array(disc_arrs, dtype=np.uint32)
        Ngs = np.array(Ngs)
        interp_img_arr = interp_img_arr.astype(np.float64)
        morph_mask = (mask_arr > 0).astype(np.uint8)
        inten_mask = (mask_arr > 1).astype(np.uint8)

        tables: List[pd.DataFrame] = []
        for cls in self.params.feature_classes:
            if cls in self.calculators:
                table = self.calculators[cls].calculate_stack(
                    interp_img_arr,
                    disc_arrs,
                    morph_mask,
                    inten_mask,
                    Ngs,
                    resolution=resolution,
                )
                tables.append(table)

        df = pd.concat(tables, axis=1)
        if sample_index is None:
            df.index = filter_names
        else:
            df.index = pd.MultiIndex.from_tuples(list(zip(filter_names, sample_index)))
        return df