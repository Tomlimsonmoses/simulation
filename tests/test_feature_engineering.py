"""
Unit tests for the feature engineering pipeline.

Tests:
  - Output shape is exactly 32 (not any other value)
  - No NaN / Inf values in any feature
  - No duplicate columns (correlation < 1.0 for every pair)
  - Organ selected matches drug's route (not always liver)
  - Pre-computed PK features (k_el, C_ss, AUC) are positive and clipped
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Make train.py importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.training.train import create_features, ROUTE_TO_ORGAN   # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')


@pytest.fixture(scope="module")
def data():
    drug_data    = pd.read_csv(os.path.join(DATA_DIR, 'drug_properties.csv'))
    organ_data   = pd.read_csv(os.path.join(DATA_DIR, 'organ_network.csv'))
    clinical_data = pd.read_csv(os.path.join(DATA_DIR, 'clinical_trials.csv'))
    disease_data  = pd.read_csv(os.path.join(DATA_DIR, 'disease_data.csv'))
    return drug_data, organ_data, clinical_data, disease_data


def _build_n(data, n=50):
    drug_data, organ_data, clinical_data, disease_data = data
    X, y = [], []
    for i in range(n):
        feats, targets = create_features(i, drug_data, organ_data, clinical_data, disease_data)
        X.append(feats)
        y.append(targets)
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Feature dimension
# ---------------------------------------------------------------------------

def test_feature_dimension(data):
    X, _ = _build_n(data, n=10)
    assert X.shape[1] == 32, f"Expected 32 features, got {X.shape[1]}"


# ---------------------------------------------------------------------------
# No NaN / Inf
# ---------------------------------------------------------------------------

def test_no_nan(data):
    X, y = _build_n(data, n=50)
    assert not np.any(np.isnan(X)), "Feature matrix contains NaN"
    assert not np.any(np.isnan(y)), "Target matrix contains NaN"


def test_no_inf(data):
    X, _ = _build_n(data, n=50)
    assert not np.any(np.isinf(X)), "Feature matrix contains Inf"


# ---------------------------------------------------------------------------
# No perfectly duplicate columns
# ---------------------------------------------------------------------------

def test_no_duplicate_columns(data):
    """No two columns should be perfectly correlated (r == 1.0)."""
    X, _ = _build_n(data, n=100)
    # Only check columns with sufficient variance
    stds = X.std(axis=0)
    non_const_cols = np.where(stds > 1e-4)[0]
    X_var = X[:, non_const_cols]
    corr = np.corrcoef(X_var.T)
    np.fill_diagonal(corr, 0.0)
    # Use nanmax so NaN from constant columns don't cause spurious failures
    max_corr = float(np.nanmax(np.abs(corr)))
    assert max_corr < 0.9999, (
        f"Found near-perfect duplicate columns (max |r| = {max_corr:.6f})"
    )


# ---------------------------------------------------------------------------
# Pre-computed PK features
# ---------------------------------------------------------------------------

def test_pk_features_positive(data):
    """k_el, C_ss, AUC_24 values must be positive (before clipping they come from
    clearance/Vd, dose/clearance — all positive quantities)."""
    X, _ = _build_n(data, n=50)
    pk_feats = X[:, 29:32]   # last 3 features
    assert np.all(pk_feats >= 0.0), "Pre-computed PK features should be non-negative"


def test_pk_features_clipped(data):
    """k_el clipped at 5, C_ss/1000 at 10, AUC/10000 at 10."""
    X, _ = _build_n(data, n=50)
    assert X[:, 29].max() <= 5.0,  "k_el exceeds clip bound of 5"
    assert X[:, 30].max() <= 10.0, "C_ss normalised exceeds clip bound of 10"
    assert X[:, 31].max() <= 10.0, "AUC normalised exceeds clip bound of 10"


# ---------------------------------------------------------------------------
# Organ routing
# ---------------------------------------------------------------------------

def test_oral_drug_maps_to_liver(data):
    """For an oral drug, the organ features should match the liver row."""
    drug_data, organ_data, clinical_data, disease_data = data
    # Find an oral drug sample
    oral_idx = clinical_data[
        clinical_data['drug_name'].isin(
            drug_data[drug_data['administration_route'] == 'oral']['drug_name']
        )
    ].index[0]

    # Liver row (organ blood_flow = 1500 in generator)
    liver = organ_data[organ_data['organ_name'] == 'liver'].iloc[0]
    expected_bf_feat = liver['blood_flow'] / 2000.0

    feats, _ = create_features(oral_idx, drug_data, organ_data, clinical_data, disease_data)
    # Feature index 11 is organ blood_flow / 2000
    assert abs(feats[11] - expected_bf_feat) < 1e-5, (
        f"Oral drug should use liver blood_flow feature: "
        f"expected {expected_bf_feat:.5f}, got {feats[11]:.5f}"
    )


def test_iv_drug_maps_to_kidneys(data):
    """For an IV drug, the organ features should match the kidneys row."""
    drug_data, organ_data, clinical_data, disease_data = data
    iv_drugs = drug_data[drug_data['administration_route'] == 'intravenous']['drug_name']
    if len(iv_drugs) == 0:
        pytest.skip("No intravenous drugs in dataset")

    iv_idx = clinical_data[clinical_data['drug_name'].isin(iv_drugs)].index
    if len(iv_idx) == 0:
        pytest.skip("No clinical samples for intravenous drugs")

    kidneys = organ_data[organ_data['organ_name'] == 'kidneys'].iloc[0]
    expected_bf_feat = kidneys['blood_flow'] / 2000.0

    feats, _ = create_features(iv_idx[0], drug_data, organ_data, clinical_data, disease_data)
    assert abs(feats[11] - expected_bf_feat) < 1e-5, (
        f"IV drug should use kidneys blood_flow feature: "
        f"expected {expected_bf_feat:.5f}, got {feats[11]:.5f}"
    )
