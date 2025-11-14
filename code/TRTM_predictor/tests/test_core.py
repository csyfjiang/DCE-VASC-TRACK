"""
Unit tests for the TRTM predictor.
Uses the real pickled model and validation CSVs.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from trtm_predictor.core import trtm_prediction
from trtm_predictor.utils import delong_power_paired_auc


# ----------------------------------------------------------------------
# Fixtures – point to the *real* data that ships with the package
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def package_dir() -> Path:
    return Path(__file__).resolve().parents[1]   # trtm_predictor/

@pytest.fixture(scope="session")
def model_path(package_dir) -> Path:
    return package_dir / "data" / "TRTM_model.pkl"

@pytest.fixture(scope="session")
def data_path(package_dir) -> Path:
    return package_dir / "data" / "model_validation_X.csv"

@pytest.fixture(scope="session")
def outcome_path(package_dir) -> Path:
    return package_dir / "data" / "model_validation_y.csv"


# ----------------------------------------------------------------------
# 1. Prediction test – loads your model and checks shape / AUC
# ----------------------------------------------------------------------
def test_trtm_prediction_real_model(
    model_path, data_path, outcome_path
):
    # ---- load data ---------------------------------------------------
    X = pd.read_csv(data_path, index_col=0)
    y = pd.read_csv(outcome_path, index_col=0)["pCR"]

    # ---- run prediction ----------------------------------------------
    probs, auc = trtm_prediction(
        model_path=model_path,
        data_path=data_path,
        outcome_path=outcome_path,
    )

    # ---- basic sanity checks -----------------------------------------
    assert isinstance(probs, pd.Series)
    assert probs.index.equals(X.index)
    assert probs.between(0, 1).all()

    assert auc is not None
    assert 0.5 < auc <= 1.0

    # ---- reproducibility check (same AUC as in the manuscript) -----
    # You reported AUC ≈ 0.81 on this exact validation set
    assert abs(auc - 0.81) < 0.03


# ----------------------------------------------------------------------
# 2. DeLong power test – reproduces the 87.3 % power you quoted
# ----------------------------------------------------------------------
def test_delong_power():
    # Parameters from your manuscript / previous calculations
    power = delong_power_paired_auc(
        auc1=0.8,          # benchmark AUC
        auc2=0.81,         # TRTM AUC
        n_cases=45,        # pCR events
        n_controls=113,    # non-pCR
        alpha=0.05,
    )
    # The function prints a summary; we only check the numeric result
    assert abs(power - 0.873) < 0.005   # 87.3 % plus/minus rounding