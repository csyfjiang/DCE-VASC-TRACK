import pytest
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from dce_preprocessing.core import Preprocessing
from ..infrastructure import Parameters

@pytest.fixture
def dummy():
    img = sitk.Image([64, 64, 32], sitk.sitkFloat32)
    arr = np.random.rand(32, 64, 64).astype(np.float32) * 1000
    sitk.GetArrayFromImage(img)[:] = arr.transpose(2, 1, 0)

    mask = sitk.Image([64, 64, 32], sitk.sitkUInt8)
    m = np.zeros((32, 64, 64), dtype=np.uint8)
    m[10:20, 20:40, 20:40] = 1
    sitk.GetArrayFromImage(mask)[:] = m.transpose(2, 1, 0)
    return img, mask

def test_preprocessing(dummy, tmp_path):
    img, mask = dummy
    class P:
        interpolation_resolution = [2, 2, 2]
        image_interpolator = "LIN"
        mask_interpolator = "NNB"
        padding_size = 3
        reseg_lower_bound = None
        reseg_upper_bound = None
        outlier_sigma = None
        bin_number = 32
        image_intensity_rounding = False
        mask_partial_volume = 0.5

    proc = Preprocessing(P())
    res = proc.execute(img, mask, export_dir=tmp_path)
    assert res is not None
    names, imgs, msks = res
    assert len(names) > 0
    assert all(i.GetSize() == imgs[0].GetSize() for i in imgs)