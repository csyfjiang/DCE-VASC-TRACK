"""Perturbation framework (rotation, translation, contour randomisation)."""
from __future__ import annotations

import math
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import SimpleITK as sitk
import pandas as pd
import scipy.ndimage as ndi
from scipy import ndimage
from skimage.filters import gaussian

from .utils import contour_similarities  # reuse Dice / Hausdorff


class PerturbationSpecification:
    ROTATION_ANGLE = 0
    RELATIVE_TRANSLATION_DISTANCES = 2
    NOISE_ADDITION = 4

    def __init__(self, perturbation_parameters: Dict):
        self.params = perturbation_parameters
        self.contour_params = perturbation_parameters.get("Contour randomization")
        self.times = perturbation_parameters.get("Perturbation times", 5)

        specs: List[Dict] = [{}]

        for key, val in perturbation_parameters.items():
            if key == "Rotation angles" and val:
                specs = [
                    {**s, self.ROTATION_ANGLE: a}
                    for s in specs
                    for a in val
                ]
            elif key == "Translation distances" and isinstance(val, list):
                specs = [
                    {**s, self.RELATIVE_TRANSLATION_DISTANCES: [dx, dy, dz]}
                    for s in specs
                    for dx in val
                    for dy in val
                    for dz in val
                ]
            elif key == "Noise addition" and isinstance(val, dict):
                levels = val.get("levels", [])
                specs = [
                    {**s, self.NOISE_ADDITION: lvl}
                    for s in specs
                    for lvl in levels
                ]

        self.specs = specs

    def shuffle_index(self, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.choice(len(self.specs), self.times, replace=True)


class ContourRandomization:
    def __init__(self, mask: sitk.Image, params: Optional[Dict]):
        self.mask = mask
        self.params = params
        if params is None:
            return
        self.arr = sitk.GetArrayFromImage(mask)
        self.spacing = np.array(mask.GetSpacing())
        self.sigma = params.get("Smoothing sigma", [10, 10, 10])
        self.intensity = params.get("Intensity", [1, 1, 1])
        self.shape = self.arr.shape

    def execute(self, seed: Optional[int] = None) -> sitk.Image:
        if self.params is None:
            return self.mask

        warped = np.zeros_like(self.arr)
        dim = len(self.shape)
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        for _ in range(10):  # try up to 10 random fields
            mask_itk = sitk.GetImageFromArray(self.arr)
            resampler.SetReferenceImage(mask_itk)

            rng = np.random.RandomState(seed)
            fields = []
            for d in range(dim):
                rs = np.random.RandomState(rng.randint(1000))
                size = self.shape[0] if d == dim - 1 else self.shape
                field = rs.uniform(-1, 1, size)
                field = gaussian(field, self.sigma[d], mode="wrap")
                field /= np.sqrt(np.mean(field ** 2)) * self.intensity[d]
                fields.append(sitk.GetImageFromArray(field))

            disp = sitk.Compose(fields)
            tx = sitk.DisplacementFieldTransform(dim)
            tx.SetDisplacementField(disp)
            resampler.SetTransform(tx)
            warped_itk = resampler.Execute(mask_itk)

            if np.sum(sitk.GetArrayFromImage(warped_itk)) > 0:
                warped_itk.CopyInformation(self.mask)
                return warped_itk

        return self.mask  # fallback if all trials empty


class ImagePerturbations:
    def __init__(self, image: sitk.Image, mask: sitk.Image):
        self.image = image
        self.mask = mask
        self.center = image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2)

    def execute(
        self, specs: List[Dict]
    ) -> Tuple[List[sitk.Transform], List[np.ndarray], np.ndarray, pd.DataFrame]:
        transforms: List[sitk.Transform] = []
        origins: List[np.ndarray] = []
        sizes: List[np.ndarray] = []
        records: List[pd.Series] = []

        for spec in specs:
            tx, rec_tx = self._translation(spec.get(PerturbationSpecification.RELATIVE_TRANSLATION_DISTANCES))
            rot, rec_rot = self._rotation(spec.get(PerturbationSpecification.ROTATION_ANGLE))
            composite = sitk.CompositeTransform([tx, rot]) if tx and rot else (tx or rot)

            origin, size = image_boundary(self.image, self.mask, composite)
            origins.append(origin)
            sizes.append(size)

            transforms.append(composite)
            records.append(pd.concat([rec_tx, rec_rot]))

        record_df = pd.concat(records, axis=1).T
        max_size = np.max(sizes, axis=0)
        return transforms, origins, max_size, record_df

    def _translation(self, rel: Optional[List[int]]) -> Tuple[Optional[sitk.AffineTransform], pd.Series]:
        if not rel:
            return None, pd.Series()
        sp = np.array(self.image.GetSpacing())
        abs_trans = self.image.TransformContinuousIndexToPhysicalPoint(np.array(rel) * sp)
        abs_trans -= np.array(self.image.GetOrigin())
        rec = pd.Series(
            rel + abs_trans.tolist(),
            index=["RelTx0", "RelTx1", "RelTx2", "AbsTxX", "AbsTxY", "AbsTxZ"],
        )
        tx = sitk.AffineTransform(3)
        tx.SetTranslation(abs_trans)
        return tx, rec

    def _rotation(self, angle: Optional[float]) -> Tuple[Optional[sitk.AffineTransform], pd.Series]:
        if angle is None:
            return None, pd.Series()
        rot = sitk.AffineTransform(3)
        rot.Rotate(0, 1, math.radians(angle))
        rot.SetCenter(self.center)
        rec = pd.Series([angle] + list(self.center), index=["RotAngle", "RotCx", "RotCy", "RotCz"])
        return rot, rec