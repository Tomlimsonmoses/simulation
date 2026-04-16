"""
Data Loader
===========
Central hub for loading all application data from CSV / JSON files.

Pattern — "load or seed":
  1. Check if the target file exists in data/raw/ (or data/raw/simulation/).
  2. If found, parse and return its contents.
  3. If missing, write default (seed) values to the file, then return them.

This means hardcoded values live here only as *seed defaults*.  After the
first run, every data file is on disk and can be edited without touching
source code.  Code never owns the canonical data — the files do.

Directory layout
----------------
data/
  raw/
    server_drugs.csv          -- DrugProperties for 15 clinical drugs
    server_diseases.json      -- DiseaseState for 15 disease profiles
    side_effects.csv          -- SideEffect library keyed by drug
    drug_disease_map.csv      -- disease → recommended drug mappings
    drug_interactions.csv     -- known drug–drug incompatibilities
    drug_organ_effects.csv    -- per-drug organ-level effect magnitudes
    drug_receptors.csv        -- receptor binding data
    simulation/               -- ODE parameter configs (one JSON per engine)
      biological_cascade_config.json
      organ_damage_config.json
      long_term_adaptation_config.json
      analytical_pk_params.json
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_SRC_DIR = _HERE.parent
_PROJECT_DIR = _SRC_DIR.parent
RAW_DIR = _PROJECT_DIR / "data" / "raw"
SIM_DIR = RAW_DIR / "simulation"


# ---------------------------------------------------------------------------
# Low-level I/O helpers
# ---------------------------------------------------------------------------

def _load_or_seed_csv(path: Path, seed_rows: List[Dict]) -> List[Dict]:
    """Return CSV rows from *path*; create it from *seed_rows* if absent."""
    if path.exists():
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        except Exception as exc:
            logger.warning("Could not read %s (%s) — using seed defaults", path.name, exc)

    if not seed_rows:
        return []

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(seed_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)
    logger.info("Seeded data file: %s", path.relative_to(_PROJECT_DIR))
    return seed_rows


def _load_or_seed_json(path: Path, seed_data: Any) -> Any:
    """Return parsed JSON from *path*; create it from *seed_data* if absent."""
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Could not read %s (%s) — using seed defaults", path.name, exc)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(seed_data, fh, indent=2, ensure_ascii=False)
    logger.info("Seeded data file: %s", path.relative_to(_PROJECT_DIR))
    return seed_data


# ===========================================================================
# ── Seed data (canonical defaults — edit the generated files instead) ──────
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Clinical drug properties (15 drugs)
#    Columns reflect DrugProperties dataclass in app/server.py
#    clearance: L/h   volume_distribution: L   half_life: h
# ---------------------------------------------------------------------------
_SEED_DRUGS: List[Dict] = [
    {"drug_name": "Acetaminophen", "molecular_weight": 151.16, "logp": 1.0,  "bioavailability": 0.88, "half_life": 2.5,   "clearance": 21.0,  "volume_distribution": 42.0,   "dose_mg": 500.0,  "administration_route": "oral", "dosing_frequency": "qid",   "protein_binding": 0.10, "therapeutic_window_lo": 10.0, "therapeutic_window_hi": 50.0},
    {"drug_name": "Ibuprofen",     "molecular_weight": 206.29, "logp": 3.5,  "bioavailability": 0.80, "half_life": 2.0,   "clearance":  6.3,  "volume_distribution":  9.8,   "dose_mg": 400.0,  "administration_route": "oral", "dosing_frequency": "tid",   "protein_binding": 0.99, "therapeutic_window_lo": 10.0, "therapeutic_window_hi": 40.0},
    {"drug_name": "Naproxen",      "molecular_weight": 230.26, "logp": 3.2,  "bioavailability": 0.95, "half_life": 17.0,  "clearance":  0.42, "volume_distribution":  9.0,   "dose_mg": 500.0,  "administration_route": "oral", "dosing_frequency": "bid",   "protein_binding": 0.99, "therapeutic_window_lo": 20.0, "therapeutic_window_hi": 80.0},
    {"drug_name": "Lisinopril",    "molecular_weight": 405.49, "logp": -0.5, "bioavailability": 0.25, "half_life": 12.0,  "clearance":  2.1,  "volume_distribution": 120.0,  "dose_mg": 10.0,   "administration_route": "oral", "dosing_frequency": "daily", "protein_binding": 0.00, "therapeutic_window_lo":  5.0, "therapeutic_window_hi": 40.0},
    {"drug_name": "Metoprolol",    "molecular_weight": 267.37, "logp": 1.9,  "bioavailability": 0.50, "half_life": 3.5,   "clearance": 14.7,  "volume_distribution": 280.0,  "dose_mg": 50.0,   "administration_route": "oral", "dosing_frequency": "bid",   "protein_binding": 0.12, "therapeutic_window_lo": 30.0, "therapeutic_window_hi": 200.0},
    {"drug_name": "Atorvastatin",  "molecular_weight": 558.64, "logp": 6.4,  "bioavailability": 0.14, "half_life": 14.0,  "clearance": 12.6,  "volume_distribution": 381.0,  "dose_mg": 20.0,   "administration_route": "oral", "dosing_frequency": "daily", "protein_binding": 0.98, "therapeutic_window_lo": 10.0, "therapeutic_window_hi": 80.0},
    {"drug_name": "Amlodipine",    "molecular_weight": 408.88, "logp": 3.0,  "bioavailability": 0.64, "half_life": 35.0,  "clearance":  4.2,  "volume_distribution": 840.0,  "dose_mg": 5.0,    "administration_route": "oral", "dosing_frequency": "daily", "protein_binding": 0.97, "therapeutic_window_lo":  5.0, "therapeutic_window_hi": 15.0},
    {"drug_name": "Metformin",     "molecular_weight": 129.16, "logp": -6.1, "bioavailability": 0.55, "half_life": 6.0,   "clearance": 35.0,  "volume_distribution": 126.0,  "dose_mg": 850.0,  "administration_route": "oral", "dosing_frequency": "bid",   "protein_binding": 0.00, "therapeutic_window_lo":  0.5, "therapeutic_window_hi":  2.5},
    {"drug_name": "Glipizide",     "molecular_weight": 445.54, "logp": 1.9,  "bioavailability": 0.90, "half_life": 7.0,   "clearance":  1.4,  "volume_distribution": 14.0,   "dose_mg": 5.0,    "administration_route": "oral", "dosing_frequency": "daily", "protein_binding": 0.98, "therapeutic_window_lo": 50.0, "therapeutic_window_hi": 150.0},
    {"drug_name": "Amoxicillin",   "molecular_weight": 365.40, "logp": -2.5, "bioavailability": 0.75, "half_life": 1.5,   "clearance": 14.0,  "volume_distribution": 21.0,   "dose_mg": 500.0,  "administration_route": "oral", "dosing_frequency": "tid",   "protein_binding": 0.18, "therapeutic_window_lo":  1.0, "therapeutic_window_hi":  8.0},
    {"drug_name": "Ciprofloxacin", "molecular_weight": 331.35, "logp": 0.3,  "bioavailability": 0.70, "half_life": 4.0,   "clearance": 37.8,  "volume_distribution": 210.0,  "dose_mg": 500.0,  "administration_route": "oral", "dosing_frequency": "bid",   "protein_binding": 0.30, "therapeutic_window_lo":  0.5, "therapeutic_window_hi":  4.0},
    {"drug_name": "Sertraline",    "molecular_weight": 306.23, "logp": 5.1,  "bioavailability": 0.44, "half_life": 26.0,  "clearance":  1.4,  "volume_distribution": 273.0,  "dose_mg": 50.0,   "administration_route": "oral", "dosing_frequency": "daily", "protein_binding": 0.98, "therapeutic_window_lo": 10.0, "therapeutic_window_hi": 150.0},
    {"drug_name": "Lorazepam",     "molecular_weight": 321.16, "logp": 2.4,  "bioavailability": 0.85, "half_life": 14.0,  "clearance":  1.26, "volume_distribution": 121.0,  "dose_mg": 1.0,    "administration_route": "oral", "dosing_frequency": "bid",   "protein_binding": 0.91, "therapeutic_window_lo": 10.0, "therapeutic_window_hi": 60.0},
    {"drug_name": "Levothyroxine", "molecular_weight": 776.87, "logp": 3.5,  "bioavailability": 0.70, "half_life": 168.0, "clearance":  0.105,"volume_distribution": 10.5,   "dose_mg": 0.1,    "administration_route": "oral", "dosing_frequency": "daily", "protein_binding": 0.999,"therapeutic_window_lo":  0.5, "therapeutic_window_hi":  2.0},
    {"drug_name": "Prednisone",    "molecular_weight": 358.43, "logp": 1.5,  "bioavailability": 0.70, "half_life": 3.5,   "clearance":  3.57, "volume_distribution": 35.0,   "dose_mg": 20.0,   "administration_route": "oral", "dosing_frequency": "daily", "protein_binding": 0.70, "therapeutic_window_lo": 10.0, "therapeutic_window_hi": 100.0},
]

# ---------------------------------------------------------------------------
# 2. Disease profiles (as JSON — nested structure: biomarkers dict, organs list)
# ---------------------------------------------------------------------------
_SEED_DISEASES: Dict = {
    "Hypertension":             {"severity": 0.55, "progression_rate": 0.004, "affected_organs": ["heart","kidneys","brain"],                       "biomarkers": {"systolic_bp": 150.0, "diastolic_bp": 95.0, "heart_rate": 82.0}},
    "Heart Failure (Mild)":     {"severity": 0.40, "progression_rate": 0.006, "affected_organs": ["heart","lungs","kidneys"],                       "biomarkers": {"ejection_fraction": 50.0, "bnp": 250.0, "heart_rate": 90.0}},
    "Heart Failure (Moderate)": {"severity": 0.65, "progression_rate": 0.009, "affected_organs": ["heart","lungs","kidneys","liver"],               "biomarkers": {"ejection_fraction": 35.0, "bnp": 600.0, "heart_rate": 100.0}},
    "Heart Failure (Severe)":   {"severity": 0.85, "progression_rate": 0.015, "affected_organs": ["heart","lungs","kidneys","liver","brain"],       "biomarkers": {"ejection_fraction": 20.0, "bnp": 1200.0, "heart_rate": 115.0}},
    "Type 2 Diabetes (Mild)":   {"severity": 0.35, "progression_rate": 0.005, "affected_organs": ["pancreas","liver"],                             "biomarkers": {"fasting_glucose": 130.0, "hba1c": 7.2, "insulin_resistance": 1.5}},
    "Type 2 Diabetes (Moderate)":{"severity":0.55, "progression_rate": 0.008, "affected_organs": ["pancreas","liver","kidneys"],                   "biomarkers": {"fasting_glucose": 160.0, "hba1c": 8.5, "insulin_resistance": 2.5}},
    "Type 2 Diabetes (Severe)": {"severity": 0.75, "progression_rate": 0.012, "affected_organs": ["pancreas","liver","kidneys","heart"],           "biomarkers": {"fasting_glucose": 200.0, "hba1c": 10.5, "insulin_resistance": 4.0}},
    "Type 2 Diabetes":          {"severity": 0.55, "progression_rate": 0.008, "affected_organs": ["pancreas","liver","kidneys"],                   "biomarkers": {"fasting_glucose": 160.0, "hba1c": 8.5, "insulin_resistance": 2.5}},
    "COPD (Moderate)":          {"severity": 0.55, "progression_rate": 0.007, "affected_organs": ["lungs","heart"],                               "biomarkers": {"fev1_pct": 58.0, "o2_saturation": 92.0, "fvc": 70.0}},
    "COPD (Severe)":            {"severity": 0.80, "progression_rate": 0.012, "affected_organs": ["lungs","heart","muscle"],                      "biomarkers": {"fev1_pct": 33.0, "o2_saturation": 87.0, "fvc": 55.0}},
    "Chronic Kidney Disease":   {"severity": 0.50, "progression_rate": 0.006, "affected_organs": ["kidneys","heart","brain"],                     "biomarkers": {"gfr": 40.0, "creatinine": 2.2, "urea": 25.0}},
    "Arthritis":                {"severity": 0.45, "progression_rate": 0.003, "affected_organs": ["joints","immune_system"],                      "biomarkers": {"crp": 8.0, "esr": 45.0, "pain_score": 6.0}},
    "Depression":               {"severity": 0.50, "progression_rate": 0.004, "affected_organs": ["brain"],                                      "biomarkers": {"cortisol": 25.0, "serotonin_index": 0.5, "sleep_quality": 0.4}},
    "Anxiety Disorder":         {"severity": 0.45, "progression_rate": 0.004, "affected_organs": ["brain","heart"],                              "biomarkers": {"cortisol": 28.0, "heart_rate": 90.0, "gaba_index": 0.6}},
    "Hypothyroidism":           {"severity": 0.40, "progression_rate": 0.003, "affected_organs": ["heart","brain","muscle"],                     "biomarkers": {"tsh": 8.0, "t4_free": 0.6, "metabolic_rate": 0.75}},
    "Bacterial Infection":      {"severity": 0.60, "progression_rate": 0.020, "affected_organs": ["immune_system","gut","liver"],                 "biomarkers": {"wbc": 14.0, "crp": 50.0, "temperature": 38.8}},
    "Autoimmune Disease":       {"severity": 0.60, "progression_rate": 0.006, "affected_organs": ["immune_system","joints","kidneys"],            "biomarkers": {"crp": 20.0, "esr": 55.0, "wbc": 11.0}},
    "Major Depressive Disorder":{"severity": 0.60, "progression_rate": 0.004, "affected_organs": ["brain"],                                      "biomarkers": {"cortisol": 28.0, "serotonin_index": 0.35, "sleep_quality": 0.3}},
}

# ---------------------------------------------------------------------------
# 3. Side effects — flat rows grouped by drug_name
# ---------------------------------------------------------------------------
_SEED_SIDE_EFFECTS: List[Dict] = [
    # Acetaminophen
    {"drug_name": "Acetaminophen", "effect_name": "Liver Toxicity",     "severity": "high",   "onset_hours": 24.0, "duration_hours": 48.0,  "probability": 0.05,  "description": "Elevated liver enzymes; hepatotoxicity at supratherapeutic doses"},
    {"drug_name": "Acetaminophen", "effect_name": "Nausea",             "severity": "low",    "onset_hours":  1.0, "duration_hours":  2.0,  "probability": 0.12,  "description": "Mild gastrointestinal discomfort"},
    {"drug_name": "Acetaminophen", "effect_name": "Allergic Reaction",  "severity": "medium", "onset_hours":  0.5, "duration_hours":  4.0,  "probability": 0.02,  "description": "Skin rash, urticaria, potential anaphylaxis"},
    # Ibuprofen
    {"drug_name": "Ibuprofen",     "effect_name": "GI Bleeding",        "severity": "high",   "onset_hours": 12.0, "duration_hours": 24.0,  "probability": 0.07,  "description": "Gastric ulcers and gastrointestinal bleeding"},
    {"drug_name": "Ibuprofen",     "effect_name": "Kidney Damage",      "severity": "medium", "onset_hours": 48.0, "duration_hours": 72.0,  "probability": 0.05,  "description": "Reduced renal perfusion; acute kidney injury risk"},
    {"drug_name": "Ibuprofen",     "effect_name": "Headache",           "severity": "low",    "onset_hours":  2.0, "duration_hours":  4.0,  "probability": 0.18,  "description": "Mild to moderate headache"},
    # Naproxen
    {"drug_name": "Naproxen",      "effect_name": "GI Irritation",      "severity": "medium", "onset_hours":  4.0, "duration_hours": 24.0,  "probability": 0.12,  "description": "Abdominal pain, dyspepsia, peptic ulceration"},
    {"drug_name": "Naproxen",      "effect_name": "Fluid Retention",    "severity": "medium", "onset_hours": 24.0, "duration_hours": 48.0,  "probability": 0.08,  "description": "Oedema due to renal prostaglandin inhibition"},
    {"drug_name": "Naproxen",      "effect_name": "Kidney Damage",      "severity": "medium", "onset_hours": 72.0, "duration_hours": 96.0,  "probability": 0.04,  "description": "Impaired renal function with prolonged use"},
    # Lisinopril
    {"drug_name": "Lisinopril",    "effect_name": "Angioedema",         "severity": "high",   "onset_hours":  2.0, "duration_hours": 24.0,  "probability": 0.01,  "description": "Swelling of face/lips/tongue — medical emergency"},
    {"drug_name": "Lisinopril",    "effect_name": "Dry Cough",          "severity": "low",    "onset_hours": 24.0, "duration_hours": 168.0, "probability": 0.20,  "description": "Persistent bradykinin-mediated dry cough"},
    {"drug_name": "Lisinopril",    "effect_name": "Hyperkalemia",       "severity": "medium", "onset_hours": 48.0, "duration_hours": 72.0,  "probability": 0.05,  "description": "Elevated potassium; risk in renal impairment"},
    {"drug_name": "Lisinopril",    "effect_name": "Hypotension",        "severity": "medium", "onset_hours":  1.0, "duration_hours":  4.0,  "probability": 0.08,  "description": "First-dose hypotension"},
    # Metoprolol
    {"drug_name": "Metoprolol",    "effect_name": "Bradycardia",        "severity": "medium", "onset_hours":  2.0, "duration_hours":  8.0,  "probability": 0.10,  "description": "Heart rate below 50 bpm"},
    {"drug_name": "Metoprolol",    "effect_name": "Fatigue",            "severity": "low",    "onset_hours": 24.0, "duration_hours": 72.0,  "probability": 0.25,  "description": "Reduced exercise tolerance, lethargy"},
    {"drug_name": "Metoprolol",    "effect_name": "Bronchospasm",       "severity": "high",   "onset_hours":  1.0, "duration_hours":  6.0,  "probability": 0.04,  "description": "Beta-2 mediated bronchoconstriction — caution in asthma/COPD"},
    {"drug_name": "Metoprolol",    "effect_name": "Cold Extremities",   "severity": "low",    "onset_hours": 12.0, "duration_hours": 48.0,  "probability": 0.15,  "description": "Peripheral vasoconstriction"},
    # Atorvastatin
    {"drug_name": "Atorvastatin",  "effect_name": "Myopathy",           "severity": "medium", "onset_hours": 168.0,"duration_hours": 336.0, "probability": 0.05,  "description": "Muscle pain, weakness; check CK if symptomatic"},
    {"drug_name": "Atorvastatin",  "effect_name": "Liver Enzyme Elev.", "severity": "medium", "onset_hours": 48.0, "duration_hours": 96.0,  "probability": 0.03,  "description": "Elevated ALT/AST; monitor LFTs"},
    {"drug_name": "Atorvastatin",  "effect_name": "Rhabdomyolysis",     "severity": "high",   "onset_hours": 336.0,"duration_hours": 168.0, "probability": 0.005, "description": "Severe muscle breakdown; risk increases with interactions"},
    # Amlodipine
    {"drug_name": "Amlodipine",    "effect_name": "Peripheral Oedema",  "severity": "low",    "onset_hours": 24.0, "duration_hours": 72.0,  "probability": 0.20,  "description": "Ankle swelling due to vasodilation"},
    {"drug_name": "Amlodipine",    "effect_name": "Flushing",           "severity": "low",    "onset_hours":  1.0, "duration_hours":  4.0,  "probability": 0.12,  "description": "Facial flushing and warmth"},
    {"drug_name": "Amlodipine",    "effect_name": "Reflex Tachycardia", "severity": "medium", "onset_hours":  2.0, "duration_hours":  6.0,  "probability": 0.08,  "description": "Compensatory heart rate increase"},
    # Metformin
    {"drug_name": "Metformin",     "effect_name": "Lactic Acidosis",    "severity": "high",   "onset_hours": 72.0, "duration_hours": 48.0,  "probability": 0.01,  "description": "Rare but serious; risk increased in renal impairment"},
    {"drug_name": "Metformin",     "effect_name": "GI Distress",        "severity": "medium", "onset_hours":  1.0, "duration_hours": 24.0,  "probability": 0.30,  "description": "Diarrhoea, nausea, abdominal discomfort — take with food"},
    {"drug_name": "Metformin",     "effect_name": "Vitamin B12 Defic.", "severity": "low",    "onset_hours": 720.0,"duration_hours": 168.0, "probability": 0.10,  "description": "Long-term B12 malabsorption; monitor annually"},
    # Glipizide
    {"drug_name": "Glipizide",     "effect_name": "Hypoglycaemia",      "severity": "high",   "onset_hours":  2.0, "duration_hours":  6.0,  "probability": 0.12,  "description": "Blood glucose <3.5 mmol/L; dizziness, sweating, confusion"},
    {"drug_name": "Glipizide",     "effect_name": "Weight Gain",        "severity": "low",    "onset_hours": 168.0,"duration_hours": 336.0, "probability": 0.20,  "description": "Insulin secretagogue effect promotes weight gain"},
    {"drug_name": "Glipizide",     "effect_name": "Nausea",             "severity": "low",    "onset_hours":  1.0, "duration_hours":  4.0,  "probability": 0.08,  "description": "Mild GI discomfort"},
    # Amoxicillin
    {"drug_name": "Amoxicillin",   "effect_name": "Allergic Reaction",  "severity": "high",   "onset_hours":  0.5, "duration_hours":  6.0,  "probability": 0.05,  "description": "Penicillin hypersensitivity; anaphylaxis risk"},
    {"drug_name": "Amoxicillin",   "effect_name": "Diarrhoea",          "severity": "low",    "onset_hours":  4.0, "duration_hours": 24.0,  "probability": 0.20,  "description": "Antibiotic-associated GI disruption"},
    {"drug_name": "Amoxicillin",   "effect_name": "C.diff Infection",   "severity": "high",   "onset_hours": 72.0, "duration_hours": 168.0, "probability": 0.03,  "description": "Dysbiosis leading to C. diff colitis"},
    # Ciprofloxacin
    {"drug_name": "Ciprofloxacin", "effect_name": "Tendinopathy",       "severity": "medium", "onset_hours": 48.0, "duration_hours": 96.0,  "probability": 0.04,  "description": "Tendon inflammation/rupture, especially Achilles"},
    {"drug_name": "Ciprofloxacin", "effect_name": "CNS Effects",        "severity": "medium", "onset_hours":  4.0, "duration_hours": 12.0,  "probability": 0.06,  "description": "Headache, dizziness, seizure risk in susceptible patients"},
    {"drug_name": "Ciprofloxacin", "effect_name": "QT Prolongation",    "severity": "high",   "onset_hours":  4.0, "duration_hours": 12.0,  "probability": 0.03,  "description": "Cardiac arrhythmia risk"},
    # Sertraline
    {"drug_name": "Sertraline",    "effect_name": "Nausea",             "severity": "low",    "onset_hours":  4.0, "duration_hours": 24.0,  "probability": 0.25,  "description": "Common early-treatment GI symptom — usually resolves"},
    {"drug_name": "Sertraline",    "effect_name": "Insomnia",           "severity": "low",    "onset_hours": 12.0, "duration_hours": 72.0,  "probability": 0.15,  "description": "Sleep disturbance, vivid dreams"},
    {"drug_name": "Sertraline",    "effect_name": "Serotonin Syndrome", "severity": "high",   "onset_hours":  2.0, "duration_hours": 12.0,  "probability": 0.01,  "description": "Hyperthermia, agitation, clonus — risk with MAOIs"},
    # Lorazepam
    {"drug_name": "Lorazepam",     "effect_name": "Sedation",           "severity": "medium", "onset_hours":  1.0, "duration_hours":  6.0,  "probability": 0.40,  "description": "CNS depression; impairs driving/coordination"},
    {"drug_name": "Lorazepam",     "effect_name": "Respiratory Depr.",  "severity": "high",   "onset_hours":  1.0, "duration_hours":  6.0,  "probability": 0.03,  "description": "Reduced respiratory drive; additive with opioids/alcohol"},
    {"drug_name": "Lorazepam",     "effect_name": "Dependence",         "severity": "high",   "onset_hours": 336.0,"duration_hours": 672.0, "probability": 0.15,  "description": "Physical and psychological dependence with prolonged use"},
    # Levothyroxine
    {"drug_name": "Levothyroxine", "effect_name": "Palpitations",       "severity": "medium", "onset_hours": 24.0, "duration_hours": 48.0,  "probability": 0.08,  "description": "Supraphysiological dosing causes tachyarrhythmia"},
    {"drug_name": "Levothyroxine", "effect_name": "Insomnia",           "severity": "low",    "onset_hours": 24.0, "duration_hours": 72.0,  "probability": 0.10,  "description": "CNS stimulation at high doses"},
    {"drug_name": "Levothyroxine", "effect_name": "Chest Pain",         "severity": "high",   "onset_hours": 12.0, "duration_hours": 24.0,  "probability": 0.02,  "description": "Angina risk in ischaemic heart disease patients"},
    # Prednisone
    {"drug_name": "Prednisone",    "effect_name": "Hyperglycaemia",     "severity": "medium", "onset_hours":  2.0, "duration_hours": 24.0,  "probability": 0.30,  "description": "Steroid-induced glucose elevation"},
    {"drug_name": "Prednisone",    "effect_name": "Immunosuppression",  "severity": "high",   "onset_hours": 24.0, "duration_hours": 72.0,  "probability": 0.50,  "description": "Increased infection risk"},
    {"drug_name": "Prednisone",    "effect_name": "GI Ulceration",      "severity": "medium", "onset_hours": 12.0, "duration_hours": 48.0,  "probability": 0.10,  "description": "Gastric irritation; take with food or PPI cover"},
    {"drug_name": "Prednisone",    "effect_name": "Osteoporosis",       "severity": "medium", "onset_hours": 720.0,"duration_hours": 336.0, "probability": 0.20,  "description": "Long-term bone density loss"},
]

# ---------------------------------------------------------------------------
# 4. Drug–disease treatment map
# ---------------------------------------------------------------------------
_SEED_DRUG_DISEASE_MAP: List[Dict] = [
    {"disease_name": "Hypertension",              "drug_name": "Lisinopril"},
    {"disease_name": "Hypertension",              "drug_name": "Metoprolol"},
    {"disease_name": "Hypertension",              "drug_name": "Amlodipine"},
    {"disease_name": "Hypertension",              "drug_name": "Atorvastatin"},
    {"disease_name": "Heart Failure (Mild)",      "drug_name": "Lisinopril"},
    {"disease_name": "Heart Failure (Mild)",      "drug_name": "Metoprolol"},
    {"disease_name": "Heart Failure (Mild)",      "drug_name": "Atorvastatin"},
    {"disease_name": "Heart Failure (Moderate)",  "drug_name": "Lisinopril"},
    {"disease_name": "Heart Failure (Moderate)",  "drug_name": "Metoprolol"},
    {"disease_name": "Heart Failure (Moderate)",  "drug_name": "Atorvastatin"},
    {"disease_name": "Heart Failure (Severe)",    "drug_name": "Lisinopril"},
    {"disease_name": "Heart Failure (Severe)",    "drug_name": "Metoprolol"},
    {"disease_name": "Type 2 Diabetes (Mild)",    "drug_name": "Metformin"},
    {"disease_name": "Type 2 Diabetes (Mild)",    "drug_name": "Glipizide"},
    {"disease_name": "Type 2 Diabetes (Moderate)","drug_name": "Metformin"},
    {"disease_name": "Type 2 Diabetes (Moderate)","drug_name": "Glipizide"},
    {"disease_name": "Type 2 Diabetes (Severe)",  "drug_name": "Metformin"},
    {"disease_name": "Type 2 Diabetes (Severe)",  "drug_name": "Glipizide"},
    {"disease_name": "Type 2 Diabetes",           "drug_name": "Metformin"},
    {"disease_name": "Type 2 Diabetes",           "drug_name": "Glipizide"},
    {"disease_name": "COPD (Moderate)",           "drug_name": "Prednisone"},
    {"disease_name": "COPD (Severe)",             "drug_name": "Prednisone"},
    {"disease_name": "Chronic Kidney Disease",    "drug_name": "Lisinopril"},
    {"disease_name": "Chronic Kidney Disease",    "drug_name": "Amlodipine"},
    {"disease_name": "Arthritis",                 "drug_name": "Ibuprofen"},
    {"disease_name": "Arthritis",                 "drug_name": "Naproxen"},
    {"disease_name": "Arthritis",                 "drug_name": "Prednisone"},
    {"disease_name": "Arthritis",                 "drug_name": "Acetaminophen"},
    {"disease_name": "Depression",                "drug_name": "Sertraline"},
    {"disease_name": "Anxiety Disorder",          "drug_name": "Sertraline"},
    {"disease_name": "Anxiety Disorder",          "drug_name": "Lorazepam"},
    {"disease_name": "Hypothyroidism",            "drug_name": "Levothyroxine"},
    {"disease_name": "Bacterial Infection",       "drug_name": "Amoxicillin"},
    {"disease_name": "Bacterial Infection",       "drug_name": "Ciprofloxacin"},
    {"disease_name": "Autoimmune Disease",        "drug_name": "Prednisone"},
    {"disease_name": "Major Depressive Disorder", "drug_name": "Sertraline"},
]

# ---------------------------------------------------------------------------
# 5. Drug incompatibilities / interactions
# ---------------------------------------------------------------------------
_SEED_DRUG_INTERACTIONS: List[Dict] = [
    {"drug1": "Ibuprofen",    "drug2": "Lisinopril",  "severity": "high",   "description": "NSAIDs blunt ACE inhibitor efficacy and increase renal injury risk"},
    {"drug1": "Naproxen",     "drug2": "Lisinopril",  "severity": "high",   "description": "NSAIDs blunt ACE inhibitor efficacy and increase renal injury risk"},
    {"drug1": "Ibuprofen",    "drug2": "Metformin",   "severity": "high",   "description": "NSAIDs may increase metformin accumulation → lactic acidosis"},
    {"drug1": "Metoprolol",   "drug2": "Amlodipine",  "severity": "medium", "description": "Additive negative chronotropy — risk of excessive bradycardia"},
    {"drug1": "Sertraline",   "drug2": "Lorazepam",   "severity": "medium", "description": "Enhanced CNS depression — monitor respiratory function"},
    {"drug1": "Prednisone",   "drug2": "Ibuprofen",   "severity": "high",   "description": "Additive risk of GI ulceration and bleeding"},
    {"drug1": "Prednisone",   "drug2": "Naproxen",    "severity": "high",   "description": "Additive risk of GI ulceration and bleeding"},
    {"drug1": "Levothyroxine","drug2": "Metformin",   "severity": "low",    "description": "Metformin may reduce levothyroxine absorption — stagger doses"},
    {"drug1": "Ciprofloxacin","drug2": "Metformin",   "severity": "medium", "description": "Ciprofloxacin inhibits renal tubular transport of metformin"},
    {"drug1": "Glipizide",    "drug2": "Ibuprofen",   "severity": "medium", "description": "NSAIDs may potentiate hypoglycaemic effect of sulphonylureas"},
]

# ---------------------------------------------------------------------------
# 6. Drug–organ effect magnitudes  (positive = beneficial, negative = harmful)
# ---------------------------------------------------------------------------
_SEED_DRUG_ORGAN_EFFECTS: List[Dict] = [
    {"drug_name": "Acetaminophen", "organ": "liver",         "effect_magnitude": -0.15},
    {"drug_name": "Ibuprofen",     "organ": "kidneys",       "effect_magnitude": -0.12},
    {"drug_name": "Ibuprofen",     "organ": "gut",           "effect_magnitude": -0.10},
    {"drug_name": "Ibuprofen",     "organ": "heart",         "effect_magnitude": -0.04},
    {"drug_name": "Naproxen",      "organ": "kidneys",       "effect_magnitude": -0.10},
    {"drug_name": "Naproxen",      "organ": "gut",           "effect_magnitude": -0.12},
    {"drug_name": "Lisinopril",    "organ": "heart",         "effect_magnitude":  0.15},
    {"drug_name": "Lisinopril",    "organ": "kidneys",       "effect_magnitude":  0.10},
    {"drug_name": "Metoprolol",    "organ": "heart",         "effect_magnitude":  0.18},
    {"drug_name": "Metoprolol",    "organ": "lungs",         "effect_magnitude": -0.06},
    {"drug_name": "Metoprolol",    "organ": "brain",         "effect_magnitude":  0.05},
    {"drug_name": "Atorvastatin",  "organ": "liver",         "effect_magnitude": -0.06},
    {"drug_name": "Atorvastatin",  "organ": "muscle",        "effect_magnitude": -0.08},
    {"drug_name": "Atorvastatin",  "organ": "heart",         "effect_magnitude":  0.10},
    {"drug_name": "Amlodipine",    "organ": "heart",         "effect_magnitude":  0.12},
    {"drug_name": "Amlodipine",    "organ": "brain",         "effect_magnitude":  0.08},
    {"drug_name": "Metformin",     "organ": "liver",         "effect_magnitude":  0.10},
    {"drug_name": "Metformin",     "organ": "kidneys",       "effect_magnitude": -0.04},
    {"drug_name": "Glipizide",     "organ": "pancreas",      "effect_magnitude":  0.15},
    {"drug_name": "Prednisone",    "organ": "immune_system", "effect_magnitude": -0.20},
    {"drug_name": "Prednisone",    "organ": "gut",           "effect_magnitude": -0.08},
    {"drug_name": "Prednisone",    "organ": "lungs",         "effect_magnitude":  0.12},
    {"drug_name": "Amoxicillin",   "organ": "gut",           "effect_magnitude": -0.06},
    {"drug_name": "Ciprofloxacin", "organ": "gut",           "effect_magnitude": -0.08},
    {"drug_name": "Ciprofloxacin", "organ": "muscle",        "effect_magnitude": -0.05},
    {"drug_name": "Sertraline",    "organ": "brain",         "effect_magnitude":  0.12},
    {"drug_name": "Sertraline",    "organ": "gut",           "effect_magnitude": -0.04},
    {"drug_name": "Lorazepam",     "organ": "brain",         "effect_magnitude":  0.08},
    {"drug_name": "Lorazepam",     "organ": "lungs",         "effect_magnitude": -0.08},
    {"drug_name": "Levothyroxine", "organ": "heart",         "effect_magnitude":  0.08},
    {"drug_name": "Levothyroxine", "organ": "brain",         "effect_magnitude":  0.10},
    {"drug_name": "Levothyroxine", "organ": "muscle",        "effect_magnitude":  0.06},
]

# ---------------------------------------------------------------------------
# 7. Drug receptor binding data
# ---------------------------------------------------------------------------
_SEED_DRUG_RECEPTORS: List[Dict] = [
    {"drug_name": "Lisinopril",    "receptor": "ACE",                "effect_type": "inhibition",   "organ_affinity": "heart|kidneys|lungs"},
    {"drug_name": "Metoprolol",    "receptor": "β1-Adrenergic",      "effect_type": "blockade",     "organ_affinity": "heart|brain"},
    {"drug_name": "Ibuprofen",     "receptor": "COX-1",              "effect_type": "inhibition",   "organ_affinity": "gut|kidneys"},
    {"drug_name": "Ibuprofen",     "receptor": "COX-2",              "effect_type": "inhibition",   "organ_affinity": "joints|immune_system"},
    {"drug_name": "Naproxen",      "receptor": "COX-1",              "effect_type": "inhibition",   "organ_affinity": "gut"},
    {"drug_name": "Naproxen",      "receptor": "COX-2",              "effect_type": "inhibition",   "organ_affinity": "joints"},
    {"drug_name": "Atorvastatin",  "receptor": "HMG-CoA Reductase",  "effect_type": "inhibition",   "organ_affinity": "liver"},
    {"drug_name": "Amlodipine",    "receptor": "L-Ca2+ Channel",     "effect_type": "blockade",     "organ_affinity": "heart|brain"},
    {"drug_name": "Metformin",     "receptor": "AMPK",               "effect_type": "activation",   "organ_affinity": "liver|muscle"},
    {"drug_name": "Glipizide",     "receptor": "KATP (SUR1)",        "effect_type": "closure",      "organ_affinity": "pancreas"},
    {"drug_name": "Amoxicillin",   "receptor": "PBP",                "effect_type": "inhibition",   "organ_affinity": "immune_system|gut"},
    {"drug_name": "Ciprofloxacin", "receptor": "DNA Gyrase",         "effect_type": "inhibition",   "organ_affinity": "immune_system"},
    {"drug_name": "Sertraline",    "receptor": "SERT",               "effect_type": "inhibition",   "organ_affinity": "brain"},
    {"drug_name": "Lorazepam",     "receptor": "GABA-A",             "effect_type": "potentiation", "organ_affinity": "brain"},
    {"drug_name": "Levothyroxine", "receptor": "TR-a/b",             "effect_type": "agonism",      "organ_affinity": "heart|brain|muscle"},
    {"drug_name": "Prednisone",    "receptor": "GR (Glucocorticoid)","effect_type": "agonism",      "organ_affinity": "immune_system|lungs"},
    {"drug_name": "Acetaminophen", "receptor": "COX-3 / TRPV1",     "effect_type": "inhibition",   "organ_affinity": "brain|liver"},
]


# ===========================================================================
# ── Public DataLoader class ──────────────────────────────────────────────━━
# ===========================================================================

class DataLoader:
    """Load application data from files; seed missing files from defaults."""

    # ── Drug properties ──────────────────────────────────────────────────────
    def load_drugs(self) -> List[Dict]:
        return _load_or_seed_csv(RAW_DIR / "server_drugs.csv", _SEED_DRUGS)

    # ── Disease profiles ─────────────────────────────────────────────────────
    def load_diseases(self) -> Dict:
        return _load_or_seed_json(RAW_DIR / "server_diseases.json", _SEED_DISEASES)

    # ── Side effects ─────────────────────────────────────────────────────────
    def load_side_effects(self) -> List[Dict]:
        return _load_or_seed_csv(RAW_DIR / "side_effects.csv", _SEED_SIDE_EFFECTS)

    # ── Drug–disease treatment map ───────────────────────────────────────────
    def load_drug_disease_map(self) -> List[Dict]:
        return _load_or_seed_csv(RAW_DIR / "drug_disease_map.csv", _SEED_DRUG_DISEASE_MAP)

    # ── Drug incompatibilities ───────────────────────────────────────────────
    def load_drug_interactions(self) -> List[Dict]:
        return _load_or_seed_csv(RAW_DIR / "drug_interactions.csv", _SEED_DRUG_INTERACTIONS)

    # ── Drug–organ effects ───────────────────────────────────────────────────
    def load_drug_organ_effects(self) -> List[Dict]:
        return _load_or_seed_csv(RAW_DIR / "drug_organ_effects.csv", _SEED_DRUG_ORGAN_EFFECTS)

    # ── Drug receptor binding ────────────────────────────────────────────────
    def load_drug_receptors(self) -> List[Dict]:
        return _load_or_seed_csv(RAW_DIR / "drug_receptors.csv", _SEED_DRUG_RECEPTORS)

    # ── Simulation configs ───────────────────────────────────────────────────
    def load_simulation_config(self, name: str, seed_data: Dict) -> Dict:
        """Load a named simulation JSON config (or seed it)."""
        return _load_or_seed_json(SIM_DIR / f"{name}.json", seed_data)


# Singleton
_data_loader: Optional[DataLoader] = None


def get_data_loader() -> DataLoader:
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader
