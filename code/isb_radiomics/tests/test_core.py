import pytest
import numpy as np
import SimpleITK as sitk
from isb_radiomics.core import FeatureCalculation
from isb_radiomics.utils import Parameters

@pytest.fixture
def dummy():
    img = sitk.Image([32, 32, 16], sitk.sitkFloat64)
    arr = np.random.rand(16, 32, 32).astype(np.float64) * 1000
    sitk.GetArrayFromImage(img)[:] = arr.transpose(2, 1, 0)

    mask = sitk.Image([32, 32, 16], sitk.sitkUInt8)
    m = np.zeros((16, 32, 32), dtype=np.uint8)
    m[5:10, 10:20, 10:20] = 2  # intensity mask
    sitk.GetArrayFromImage(mask)[:] = m.transpose(2, 1, 0)
    return img, mask

def test_individual_extraction(dummy):
    img, mask = dummy
    params = Parameters(feature_classes=["Intensity statistical", "GLCM"])
    calc = FeatureCalculation(params)
    img_arr = sitk.GetArrayFromImage(img)
    mask_arr = sitk.GetArrayFromImage(mask)
    res = np.flip(img.GetSpacing())
    feats = calc.individual_feature_calculation(img_arr, mask_arr, res, "Original")
    assert isinstance(feats, pd.DataFrame)
    assert len(feats) > 0