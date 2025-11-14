"""Core preprocessing pipeline (cropping → interpolation → resegmentation → discretisation)."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import SimpleITK as sitk
import numpy as np
import pandas as pd

from .utils import (
    mask_cropping,
    image_cropping,
    resegmentation,
    intensity_discretization,
    image_padding,
)
from .perturbation import PerturbationSpecification, ContourRandomization, ImagePerturbations
from ..infrastructure import Parameters  # keep your original Parameters dataclass


class Preprocessing:
    """Full preprocessing + optional perturbation pipeline."""

    def __init__(self, params: Parameters):
        self.params = params
        self.image_interp = {
            "NNB": sitk.sitkNearestNeighbor,
            "LIN": sitk.sitkLinear,
        }.get(params.image_interpolator, sitk.sitkBSpline)

        self.mask_interp = {
            "LIN": sitk.sitkLinear,
            "CSI": sitk.sitkBSpline,
        }.get(params.mask_interpolator, sitk.sitkNearestNeighbor)

    # --------------------------------------------------------------------- #
    #                         SINGLE-VOLUME PIPELINE                         #
    # --------------------------------------------------------------------- #
    def execute(
        self,
        image: sitk.Image,
        mask: sitk.Image,
        export_dir: Optional[Path] = None,
    ) -> Optional[Tuple[List[str], List[sitk.Image], List[sitk.Image]]]:
        """Run cropping → interpolation → resegmentation → filtering."""
        if not np.any(sitk.GetArrayFromImage(mask)):
            print("Empty mask – aborting.")
            return None

        mask = mask_cropping(mask, self.params.padding_size or 0)
        cropped_img = image_cropping(image, mask)

        interp_img, interp_mask = self._interpolate(image, cropped_img, mask, export_dir=export_dir)

        # resegmentation
        img_arr = sitk.GetArrayFromImage(interp_img)
        msk_arr = sitk.GetArrayFromImage(interp_mask)
        msk_arr = resegmentation(
            img_arr,
            msk_arr,
            lower_bound=self.params.reseg_lower_bound,
            higher_bound=self.params.reseg_upper_bound,
            z_score=self.params.outlier_sigma,
        )
        resegmented = sitk.GetImageFromArray(msk_arr.astype(np.uint8))
        resegmented.CopyInformation(interp_mask)

        if export_dir:
            sitk.WriteImage(interp_img, export_dir / "interpolated_image.mha")
            sitk.WriteImage(resegmented, export_dir / "resegmented_mask.mha")

        # optional image filtering (your `image_filtering` function)
        from .image_filtering import image_filtering  # lazy import

        filter_names, filtered_imgs = image_filtering(interp_img, resegmented, self.params)
        masks_out = [resegmented] * len(filter_names)
        return filter_names, filtered_imgs, masks_out

    # --------------------------------------------------------------------- #
    #                         PERTURBATION PIPELINE                         #
    # --------------------------------------------------------------------- #
    def execute_perturbation(
        self,
        image: sitk.Image,
        mask: sitk.Image,
        pert_spec: PerturbationSpecification,
        export_dir: Optional[Path] = None,
    ) -> Optional[Tuple[List[str], List[sitk.Image], List[sitk.Image]]]:
        """Generate *perturbation_times* perturbed versions + preprocessing."""
        if not np.any(sitk.GetArrayFromImage(mask)):
            return None

        margin = self.params.padding_size or 0
        mask = mask_cropping(mask, margin)
        cropped = image_cropping(image, mask)

        contour = ContourRandomization(mask, pert_spec.contour_params)
        pert = ImagePerturbations(image, mask)

        idx = pert_spec.shuffle_index()
        specs = [pert_spec.specs[i] for i in idx]

        transforms, origins, max_size, pert_record = pert.execute(specs)

        all_names: List[str] = []
        all_imgs: List[sitk.Image] = []
        all_masks: List[sitk.Image] = []

        contour_records = []

        for i, (tx, origin) in enumerate(zip(transforms, origins)):
            warped_mask = contour.execute(seed=idx[i])
            if export_dir:
                dsc, shd = contour_similarities(mask, warped_mask)
                contour_records.append(pd.Series([dsc, shd], index=["DSC", "SHD"]))
                sitk.WriteImage(warped_mask, export_dir / f"pert_{i}" / "warped_mask.mha")

            sub_dir = export_dir / f"pert_{i}" if export_dir else None
            boundary_img = sitk.Image(max_size.tolist()[::-1], sitk.sitkUInt8)
            boundary_img.SetOrigin(origin.tolist())
            boundary_img.SetDirection(image.GetDirection())

            res = self._interpolate(
                image,
                cropped,
                warped_mask,
                transformer=tx,
                boundary_image=boundary_img,
                export_dir=sub_dir,
            )
            if not res:
                continue
            interp_img, interp_mask = res

            from .image_filtering import image_filtering

            names, imgs = image_filtering(interp_img, interp_mask, self.params)
            all_names.extend(names)
            all_imgs.extend(imgs)
            all_masks.extend([interp_mask] * len(names))

        if export_dir:
            pd.concat(contour_records, axis=1).T.to_csv(export_dir / "contour_records.csv")
            pert_record.to_csv(export_dir / "perturbation_record.csv")

        return all_names, all_imgs, all_masks

    # --------------------------------------------------------------------- #
    #                         INTERNAL INTERPOLATION                        #
    # --------------------------------------------------------------------- #
    def _interpolate(
        self,
        image: sitk.Image,
        cropped_img: sitk.Image,
        mask: sitk.Image,
        transformer: Optional[sitk.Transform] = None,
        boundary_image: Optional[sitk.Image] = None,
        export_dir: Optional[Path] = None,
    ) -> Tuple[sitk.Image, sitk.Image]:
        """Core interpolation (3-D or slice-wise)."""
        if self.params.slice_wise:
            # ---- slice-wise interpolation (keeps original z-spacing) ----
            n_slices = cropped_img.GetSize()[-1]
            slices_img, slices_mask = [], []
            origin3d = None
            spacing3d = None

            for z in range(n_slices):
                loc = cropped_img.TransformIndexToPhysicalPoint([0, 0, z])
                z_full = int(image.TransformPhysicalPointToIndex(loc)[-1])
                img2d = image[:, :, z_full]
                crop2d = cropped_img[:, :, z]
                msk2d = mask[:, :, z]

                res = self._interpolate_2d(
                    img2d,
                    crop2d,
                    msk2d,
                    transformer,
                    boundary_image,
                )
                img2d_res, msk2d_res = res
                if origin3d is None:
                    origin3d = list(img2d_res.GetOrigin()) + [loc[-1]]
                if spacing3d is None:
                    spacing3d = list(img2d_res.GetSpacing())
                slices_img.append(img2d_res)
                slices_mask.append(msk2d_res)

            spacing3d.append(cropped_img.GetSpacing()[-1])
            interp_img = sitk.JoinSeries(slices_img)
            interp_mask = sitk.JoinSeries(slices_mask)
            interp_img.SetOrigin(origin3d)
            interp_img.SetSpacing(spacing3d)
            interp_img.SetDirection(cropped_img.GetDirection())
            interp_mask.CopyInformation(interp_img)
        else:
            interp_img, interp_mask = self._interpolate_2d(
                image, cropped_img, mask, transformer, boundary_image
            )

        if export_dir:
            sitk.WriteImage(interp_img, export_dir / "interpolated_image.mha")
            sitk.WriteImage(interp_mask, export_dir / "interpolated_mask.mha")

        return interp_img, interp_mask

    def _interpolate_2d(
        self,
        image: sitk.Image,
        cropped_img: sitk.Image,
        mask: sitk.Image,
        transformer: Optional[sitk.Transform],
        boundary_image: Optional[sitk.Image],
    ) -> Tuple[sitk.Image, sitk.Image]:
        """Low-level interpolation with user-specified resolution."""
        orig_sz = np.array(image.GetSize())
        orig_sp = np.array(image.GetSpacing())
        new_sp = [
            r if r is not None else orig_sp[i]
            for i, r in enumerate(self.params.interpolation_resolution)
        ]
        ratio = orig_sp / new_sp
        new_sz = np.ceil(orig_sz * ratio).astype(int)

        # bounding box on new grid
        if boundary_image is not None:
            o, s = image_boundary(image, boundary_image, transformer)
        else:
            o, s = image_boundary(image, mask, transformer)

        new_origin_rel = (orig_sz - 1 - ratio * (new_sz - 1)) / 2
        new_origin_rel += np.floor(o / ratio) * ratio
        new_sz = np.ceil(s / ratio).astype(int)

        new_origin_phys = image.TransformContinuousIndexToPhysicalPoint(new_origin_rel.tolist())

        res = sitk.ResampleImageFilter()
        res.SetOutputOrigin(new_origin_phys)
        res.SetOutputSpacing(new_sp.tolist())
        res.SetOutputDirection(image.GetDirection())
        res.SetSize(new_sz.tolist())
        res.SetDefaultPixelValue(0)
        if transformer:
            res.SetTransform(transformer)

        res.SetInterpolator(self.image_interp)
        img_res = res.Execute(sitk.Cast(cropped_img if cropped_img else image, sitk.sitkFloat32))

        res.SetInterpolator(self.mask_interp)
        msk_res = res.Execute(sitk.Cast(mask, sitk.sitkFloat32))

        # keep only voxels with >= partial_volume
        msk_res = msk_res >= self.params.mask_partial_volume

        if self.params.image_intensity_rounding:
            img_res = sitk.Round(img_res)
            img_res = sitk.Cast(img_res, sitk.sitkInt32)

        return img_res, msk_res