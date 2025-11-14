import pytest
import tempfile
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from ..core import FTVGenerator


@pytest.fixture
def dummy_data():
    """Create a minimal synthetic patient folder."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        patient_dir = root / "cleaned_dataset_ISPY2" / "P001"
        patient_dir.mkdir(parents=True)

        # dummy time CSV
        time_csv = (
            root
            / "cleaned_time_information_ISPY2"
            / "DCE_time_point_MRI_T0_list"
            / "P001_time.csv"
        )
        time_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "name_tag": ["DCE_MRI_T0_Phase_0", "DCE_MRI_T0_Phase_1", "DCE_MRI_T0_Phase_2"],
                "acquisition_time_difference(s)": [0, 150, 450],
            }
        ).to_csv(time_csv)

        # dummy images (3×64×64×64)
        shape = (64, 64, 64)
        for tag in ["DCE_MRI_T0_Phase_0", "DCE_MRI_T0_Phase_1", "DCE_MRI_T0_Phase_2"]:
            img = sitk.Image(shape, sitk.sitkFloat32)
            arr = np.random.rand(*shape).astype(np.float32) * 1000
            sitk.GetArrayFromImage(img)[:] = arr
            sitk.WriteImage(img, str(patient_dir / f"{tag}.mha"))

        # dummy VOI mask (all 1 inside a small box)
        voi = sitk.Image(shape, sitk.sitkUInt8)
        voi_arr = np.zeros(shape, dtype=np.uint8)
        voi_arr[20:40, 20:40, 20:40] = 1
        sitk.GetArrayFromImage(voi)[:] = voi_arr
        sitk.WriteImage(voi, str(patient_dir / "bounding_box_T0_mask.mha"))

        yield root, "P001", "T0"


def test_ftv_generation(dummy_data):
    root, pid, tp = dummy_data
    gen = FTVGenerator(dataset_name="ISPY2", patient_id=pid, time_point=tp, data_directory=root)
    mask = gen.generate_ftv(label="unit")
    arr = sitk.GetArrayFromImage(mask)
    assert arr.sum() > 0, "FTV mask is empty"
    assert (arr == 0).all() or (arr == 1).all(), "Mask must be binary"