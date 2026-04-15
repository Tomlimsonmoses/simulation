"""
Feature engineering for PK-PD MLP models.

Single source of truth for the 32-dimensional feature vector used by all
training, evaluation, and checkpoint-loading code.

Feature layout (32 total):
  Drug        (10): MW, logP, bioavailability, protein_binding, clearance,
                    Vd, t½, is_small_molecule, is_oral, dose
  Organ       (10): volume, blood_flow, metabolic_rate, health_status, pH,
                    temperature, water_content, lipid_content, protein_content,
                    disease_impact
  Patient      (9): age, weight, sex, Caucasian, Asian, African, Hispanic,
                    time_point, adverse_event
  PK computed  (3): k_el, C_max_single_dose, AUC_24
"""

import numpy as np

# Map drug administration route → primary metabolic organ
ROUTE_TO_ORGAN = {
    "oral":           "liver",
    "sublingual":     "liver",
    "intravenous":    "kidneys",
    "intramuscular":  "muscle",
    "subcutaneous":   "skin",
    "inhalation":     "lungs",
}

FEATURE_DIM = 32  # fixed contract consumed by MechanisticPKPDEngine checkpoints


def select_organ(organ_data, route: str):
    """Return the organ row matching the drug's primary metabolic pathway."""
    target = ROUTE_TO_ORGAN.get(str(route).lower(), "liver")
    match = organ_data[organ_data["organ_name"] == target]
    return match.iloc[0] if len(match) > 0 else organ_data.iloc[0]


def create_features(sample_idx, drug_data, organ_data, clinical_data, disease_data):
    """Build a (32,) float32 feature vector and (2,) target vector for one sample.

    Returns
    -------
    features : np.ndarray  shape (32,)
    targets  : np.ndarray  shape (2,)  — [pk_target, pd_target]
    """
    clinical_sample = clinical_data.iloc[sample_idx % len(clinical_data)]
    drug_name = clinical_sample["drug_name"]

    drug_info = drug_data[drug_data["drug_name"] == drug_name]
    drug_info = drug_info.iloc[0] if len(drug_info) > 0 else drug_data.iloc[0]

    route  = drug_info["administration_route"]
    organ  = select_organ(organ_data, route)

    organ_diseases  = disease_data[disease_data["affected_organ"] == organ["organ_name"]]
    disease_impact  = float(organ_diseases["organ_impact_score"].mean()) if len(organ_diseases) > 0 else 0.0

    clearance = float(drug_info["clearance_rate"])
    vd        = float(drug_info["volume_distribution"])
    bioavail  = float(drug_info["bioavailability"])
    dose      = float(clinical_sample["dose"])
    k_el      = clearance / max(vd, 1e-6)
    c_max     = dose * bioavail / max(vd, 1e-6)
    auc_24    = dose * bioavail / max(clearance, 1e-6)

    features = [
        # Drug features (10)
        drug_info["molecular_weight"] / 1000.0,
        drug_info["logp"] / 10.0,
        bioavail,
        drug_info["protein_binding"],
        clearance / 5.0,
        vd / 10.0,
        drug_info["half_life"] / 100.0,
        1.0 if drug_info["drug_class"] == "small_molecule" else 0.0,
        1.0 if route == "oral" else 0.0,
        dose / 1000.0,

        # Organ features (10)
        organ["volume"] / 50.0,
        organ["blood_flow"] / 2000.0,
        organ["metabolic_rate"],
        organ["health_status"],
        organ["ph"] / 14.0,
        (organ["temperature"] - 35.0) / 5.0,
        organ["water_content"],
        organ["lipid_content"],
        organ["protein_content"],
        disease_impact,

        # Patient features (9)
        clinical_sample["age"] / 100.0,
        clinical_sample["weight"] / 200.0,
        1.0 if clinical_sample["sex"] == "M" else 0.0,
        1.0 if clinical_sample["ethnicity"] == "Caucasian" else 0.0,
        1.0 if clinical_sample["ethnicity"] == "Asian" else 0.0,
        1.0 if clinical_sample["ethnicity"] == "African" else 0.0,
        1.0 if clinical_sample["ethnicity"] == "Hispanic" else 0.0,
        clinical_sample.get("time_point", 0.0) / 24.0,
        1.0 if clinical_sample.get("adverse_event", False) else 0.0,

        # Pre-computed PK features (3)
        np.clip(k_el, 0.0, 5.0),
        np.clip(c_max / 1000.0, 0.0, 10.0),
        np.clip(auc_24 / 10000.0, 0.0, 10.0),
    ]

    pk_target = clinical_sample.get("concentration", 0.0) / 1000.0
    pd_target = clinical_sample.get("response_value",  0.0) / 100.0

    return (
        np.array(features, dtype=np.float32),
        np.array([pk_target, pd_target], dtype=np.float32),
    )


def load_raw_data(data_dir: str):
    """Load the four CSV files used by training/evaluation. Returns (drug, organ, clinical, disease)."""
    import os
    import pandas as pd
    return (
        pd.read_csv(os.path.join(data_dir, "drug_properties.csv")),
        pd.read_csv(os.path.join(data_dir, "organ_network.csv")),
        pd.read_csv(os.path.join(data_dir, "clinical_trials.csv")),
        pd.read_csv(os.path.join(data_dir, "disease_data.csv")),
    )


def build_dataset(data_dir: str, max_samples: int = 1000):
    """Build X, y arrays from raw CSVs. Returns (X, y) as float32 numpy arrays."""
    drug_data, organ_data, clinical_data, disease_data = load_raw_data(data_dir)
    n = min(max_samples, len(clinical_data))
    X, y = [], []
    for i in range(n):
        try:
            feats, targets = create_features(i, drug_data, organ_data, clinical_data, disease_data)
            X.append(feats)
            y.append(targets)
        except Exception:
            continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
