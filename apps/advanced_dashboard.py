"""
Professional PK-PD Biological System Simulation App
===================================================

Advanced real-time simulation of disease progression and drug effects with
comprehensive side effect monitoring and full configurability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sys
import os
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
import random
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'data'))

from models.mlp_architectures import SimpleMLP

# ---------------------------------------------------------------------------
# ML Model helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'simple_mlp_model.pth')


@st.cache_resource
def load_ml_model():
    """Load the trained SimpleMLP checkpoint once per server session."""
    if not os.path.exists(_CHECKPOINT_PATH):
        return None
    device = torch.device('cpu')
    model = SimpleMLP(input_dim=32, hidden_dim=128, output_dim=2)
    try:
        state = torch.load(_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state)
    except RuntimeError:
        # Checkpoint was trained with old input_dim=30 — signal incompatibility
        return None
    model.eval()
    return model


def _enable_dropout(model):
    """Switch all Dropout layers to train-mode so MC sampling is active."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def mc_predict(model, feature_vec: np.ndarray, n_samples: int = 50):
    """Return (mean_pk, std_pk, mean_pd, std_pd) via Monte Carlo Dropout."""
    _enable_dropout(model)
    x = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x).numpy()[0]   # [pk, pd]
            preds.append(out)
    preds = np.array(preds)  # (n_samples, 2)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    model.eval()  # restore eval-mode dropouts
    return float(mean[0]), float(std[0]), float(mean[1]), float(std[1])


def build_ml_feature(drug_props, bio_params, t: float, organ_data_row=None) -> np.ndarray:
    """Build the 32-dim feature vector for the dashboard's drug/patient config.

    Matches the layout in scripts/train.py create_features():
      Drug (10) | Organ (10) | Patient (9) | PK computed (3)
    """
    dose = float(drug_props.dose_mg)
    bioavail = float(drug_props.bioavailability)
    cl = float(drug_props.clearance)
    vd = float(drug_props.volume_distribution)
    route = getattr(drug_props, 'route', 'oral')

    tau = 12.0 if getattr(drug_props, 'dosing_frequency', 'daily') == 'bid' else 24.0
    k_el = cl / max(vd, 1e-6)
    c_max = dose * bioavail / max(vd, 1e-6)
    auc_24 = dose * bioavail / max(cl, 1e-6)

    # Organ placeholder (liver-like defaults if no row provided)
    if organ_data_row is None:
        org_vol, org_bf, org_mr, org_hs = 1.575, 1500.0, 1.16, 1.0
        org_ph, org_temp, org_wc, org_lc, org_pc = 7.4, 37.5, 0.71, 0.05, 0.24
        disease_impact = 0.0
    else:
        org = organ_data_row
        org_vol = float(org['volume'])
        org_bf = float(org['blood_flow'])
        org_mr = float(org['metabolic_rate'])
        org_hs = float(org['health_status'])
        org_ph = float(org['ph'])
        org_temp = float(org['temperature'])
        org_wc = float(org['water_content'])
        org_lc = float(org['lipid_content'])
        org_pc = float(org['protein_content'])
        disease_impact = 0.0

    features = [
        # Drug (10)
        getattr(drug_props, 'molecular_weight', 300.0) / 1000.0,
        getattr(drug_props, 'logp', 1.0) / 10.0,
        bioavail,
        float(drug_props.protein_binding),
        cl / 5.0,
        vd / 10.0,
        float(drug_props.half_life) / 100.0,
        1.0,                                          # small molecule assumed
        1.0 if route == 'oral' else 0.0,
        dose / 1000.0,
        # Organ (10)
        org_vol / 50.0, org_bf / 2000.0, org_mr, org_hs,
        org_ph / 14.0, (org_temp - 35.0) / 5.0, org_wc, org_lc, org_pc,
        disease_impact,
        # Patient (9)
        float(bio_params.age) / 100.0,
        float(bio_params.weight) / 200.0,
        float(bio_params.sex),
        0.0, 0.0, 0.0, 0.0,    # ethnicity one-hot (unknown)
        t / 24.0,
        0.0,                    # no adverse event flag at prediction time
        # PK computed (3)
        np.clip(k_el, 0.0, 5.0),
        np.clip(c_max / 1000.0, 0.0, 10.0),
        np.clip(auc_24 / 10000.0, 0.0, 10.0),
    ]
    return np.array(features, dtype=np.float32)

# Page Configuration
st.set_page_config(
    page_title="PK-PD Biological System Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .simulation-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .alert-high {
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff4444;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #ffa726, #ffb74d);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #66bb6a, #81c784);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-normal { background: #4caf50; }
    .status-warning { background: #ff9800; }
    .status-critical { background: #f44336; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class BiologicalParameters:
    """Biological system parameters"""
    age: float = 65.0
    weight: float = 70.0
    sex: int = 1  # 0: female, 1: male
    kidney_function: float = 0.8  # 0-1 scale
    liver_function: float = 0.9  # 0-1 scale
    heart_rate: float = 72.0
    blood_pressure: Tuple[float, float] = (120.0, 80.0)
    metabolic_rate: float = 1.0  # baseline multiplier

@dataclass
class DrugProperties:
    """Drug properties and administration"""
    name: str = ""
    molecular_weight: float = 300.0
    bioavailability: float = 0.7
    half_life: float = 8.0
    clearance: float = 5.0
    volume_distribution: float = 50.0
    dose_mg: float = 100.0
    administration_route: str = "oral"
    dosing_frequency: str = "daily"
    protein_binding: float = 0.8
    therapeutic_window: Tuple[float, float] = (10.0, 50.0)

@dataclass
class DiseaseState:
    """Disease progression state"""
    name: str = ""
    severity: float = 0.5  # 0-1 scale
    progression_rate: float = 0.01  # per hour
    affected_organs: List[str] = None
    biomarkers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.affected_organs is None:
            self.affected_organs = []
        if self.biomarkers is None:
            self.biomarkers = {}

class SideEffect:
    """Side effect definition"""
    def __init__(self, name: str, severity: str, onset_time: float, 
                 duration: float, probability: float, description: str):
        self.name = name
        self.severity = severity  # low, medium, high
        self.onset_time = onset_time  # hours after administration
        self.duration = duration  # hours
        self.probability = probability  # 0-1
        self.description = description
        self.active = False
        self.start_time = None
        self.end_time = None

class BiologicalSystemSimulator:
    """Advanced biological system simulator"""
    
    def __init__(self):
        self.drug_database = self._load_drug_database()
        self.disease_database = self._load_disease_database()
        self.side_effects_database = self._load_side_effects_database()
        self.drug_disease_map = self._load_drug_disease_map()
        self.drug_incompatibilities = self._load_drug_incompatibilities()
        self.organ_drug_effects = self._load_organ_drug_effects()
        self.current_time = 0.0
        self.simulation_running = False
        
    def _load_drug_database(self) -> Dict[str, DrugProperties]:
        """Load comprehensive drug database (15 clinical drugs)"""
        return {
            "Acetaminophen": DrugProperties(
                name="Acetaminophen", molecular_weight=151.16, bioavailability=0.88,
                half_life=2.5, clearance=21.0, volume_distribution=42.0,
                dose_mg=500.0, administration_route="oral", dosing_frequency="qid",
                protein_binding=0.10, therapeutic_window=(10.0, 50.0)
            ),
            "Ibuprofen": DrugProperties(
                name="Ibuprofen", molecular_weight=206.29, bioavailability=0.80,
                half_life=2.0, clearance=6.3, volume_distribution=9.8,
                dose_mg=400.0, administration_route="oral", dosing_frequency="tid",
                protein_binding=0.99, therapeutic_window=(10.0, 40.0)
            ),
            "Naproxen": DrugProperties(
                name="Naproxen", molecular_weight=230.26, bioavailability=0.95,
                half_life=17.0, clearance=0.42, volume_distribution=9.0,
                dose_mg=500.0, administration_route="oral", dosing_frequency="bid",
                protein_binding=0.99, therapeutic_window=(20.0, 80.0)
            ),
            "Lisinopril": DrugProperties(
                name="Lisinopril", molecular_weight=405.49, bioavailability=0.25,
                half_life=12.0, clearance=2.1, volume_distribution=120.0,
                dose_mg=10.0, administration_route="oral", dosing_frequency="daily",
                protein_binding=0.0, therapeutic_window=(5.0, 40.0)
            ),
            "Metoprolol": DrugProperties(
                name="Metoprolol", molecular_weight=267.37, bioavailability=0.50,
                half_life=3.5, clearance=14.7, volume_distribution=280.0,
                dose_mg=50.0, administration_route="oral", dosing_frequency="bid",
                protein_binding=0.12, therapeutic_window=(30.0, 200.0)
            ),
            "Atorvastatin": DrugProperties(
                name="Atorvastatin", molecular_weight=558.64, bioavailability=0.14,
                half_life=14.0, clearance=12.6, volume_distribution=381.0,
                dose_mg=20.0, administration_route="oral", dosing_frequency="daily",
                protein_binding=0.98, therapeutic_window=(10.0, 80.0)
            ),
            "Amlodipine": DrugProperties(
                name="Amlodipine", molecular_weight=408.88, bioavailability=0.64,
                half_life=35.0, clearance=4.2, volume_distribution=840.0,
                dose_mg=5.0, administration_route="oral", dosing_frequency="daily",
                protein_binding=0.97, therapeutic_window=(5.0, 15.0)
            ),
            "Metformin": DrugProperties(
                name="Metformin", molecular_weight=129.16, bioavailability=0.55,
                half_life=6.0, clearance=35.0, volume_distribution=126.0,
                dose_mg=850.0, administration_route="oral", dosing_frequency="bid",
                protein_binding=0.0, therapeutic_window=(0.5, 2.5)
            ),
            "Glipizide": DrugProperties(
                name="Glipizide", molecular_weight=445.54, bioavailability=0.90,
                half_life=7.0, clearance=1.4, volume_distribution=14.0,
                dose_mg=5.0, administration_route="oral", dosing_frequency="daily",
                protein_binding=0.98, therapeutic_window=(50.0, 150.0)
            ),
            "Amoxicillin": DrugProperties(
                name="Amoxicillin", molecular_weight=365.40, bioavailability=0.75,
                half_life=1.5, clearance=14.0, volume_distribution=21.0,
                dose_mg=500.0, administration_route="oral", dosing_frequency="tid",
                protein_binding=0.18, therapeutic_window=(1.0, 8.0)
            ),
            "Ciprofloxacin": DrugProperties(
                name="Ciprofloxacin", molecular_weight=331.35, bioavailability=0.70,
                half_life=4.0, clearance=37.8, volume_distribution=210.0,
                dose_mg=500.0, administration_route="oral", dosing_frequency="bid",
                protein_binding=0.30, therapeutic_window=(0.5, 4.0)
            ),
            "Sertraline": DrugProperties(
                name="Sertraline", molecular_weight=306.23, bioavailability=0.44,
                half_life=26.0, clearance=1.4, volume_distribution=273.0,
                dose_mg=50.0, administration_route="oral", dosing_frequency="daily",
                protein_binding=0.98, therapeutic_window=(10.0, 150.0)
            ),
            "Lorazepam": DrugProperties(
                name="Lorazepam", molecular_weight=321.16, bioavailability=0.85,
                half_life=14.0, clearance=1.26, volume_distribution=121.0,
                dose_mg=1.0, administration_route="oral", dosing_frequency="bid",
                protein_binding=0.91, therapeutic_window=(10.0, 60.0)
            ),
            "Levothyroxine": DrugProperties(
                name="Levothyroxine", molecular_weight=776.87, bioavailability=0.70,
                half_life=168.0, clearance=0.105, volume_distribution=10.5,
                dose_mg=0.1, administration_route="oral", dosing_frequency="daily",
                protein_binding=0.999, therapeutic_window=(0.5, 2.0)
            ),
            "Prednisone": DrugProperties(
                name="Prednisone", molecular_weight=358.43, bioavailability=0.70,
                half_life=3.5, clearance=3.57, volume_distribution=35.0,
                dose_mg=20.0, administration_route="oral", dosing_frequency="daily",
                protein_binding=0.70, therapeutic_window=(10.0, 100.0)
            ),
        }
    
    def _load_disease_database(self) -> Dict[str, DiseaseState]:
        """Load comprehensive disease database (12 named clinical configurations)"""
        return {
            "Hypertension": DiseaseState(
                name="Hypertension", severity=0.55, progression_rate=0.004,
                affected_organs=["heart", "kidneys", "brain"],
                biomarkers={"systolic_bp": 150.0, "diastolic_bp": 95.0, "heart_rate": 82.0}
            ),
            "Heart Failure (Mild)": DiseaseState(
                name="Heart Failure (Mild)", severity=0.40, progression_rate=0.006,
                affected_organs=["heart", "lungs", "kidneys"],
                biomarkers={"ejection_fraction": 50.0, "bnp": 250.0, "heart_rate": 90.0}
            ),
            "Heart Failure (Moderate)": DiseaseState(
                name="Heart Failure (Moderate)", severity=0.65, progression_rate=0.009,
                affected_organs=["heart", "lungs", "kidneys", "liver"],
                biomarkers={"ejection_fraction": 35.0, "bnp": 600.0, "heart_rate": 100.0}
            ),
            "Heart Failure (Severe)": DiseaseState(
                name="Heart Failure (Severe)", severity=0.85, progression_rate=0.015,
                affected_organs=["heart", "lungs", "kidneys", "liver", "brain"],
                biomarkers={"ejection_fraction": 20.0, "bnp": 1200.0, "heart_rate": 115.0}
            ),
            "Type 2 Diabetes (Mild)": DiseaseState(
                name="Type 2 Diabetes (Mild)", severity=0.35, progression_rate=0.005,
                affected_organs=["pancreas", "liver"],
                biomarkers={"fasting_glucose": 130.0, "hba1c": 7.2, "insulin_resistance": 1.5}
            ),
            "Type 2 Diabetes (Moderate)": DiseaseState(
                name="Type 2 Diabetes (Moderate)", severity=0.55, progression_rate=0.008,
                affected_organs=["pancreas", "liver", "kidneys"],
                biomarkers={"fasting_glucose": 160.0, "hba1c": 8.5, "insulin_resistance": 2.5}
            ),
            "Type 2 Diabetes (Severe)": DiseaseState(
                name="Type 2 Diabetes (Severe)", severity=0.75, progression_rate=0.012,
                affected_organs=["pancreas", "liver", "kidneys", "heart"],
                biomarkers={"fasting_glucose": 200.0, "hba1c": 10.5, "insulin_resistance": 4.0}
            ),
            "COPD (Moderate)": DiseaseState(
                name="COPD (Moderate)", severity=0.55, progression_rate=0.007,
                affected_organs=["lungs", "heart"],
                biomarkers={"fev1_pct": 58.0, "o2_saturation": 92.0, "fvc": 70.0}
            ),
            "COPD (Severe)": DiseaseState(
                name="COPD (Severe)", severity=0.80, progression_rate=0.012,
                affected_organs=["lungs", "heart", "muscle"],
                biomarkers={"fev1_pct": 33.0, "o2_saturation": 87.0, "fvc": 55.0}
            ),
            "Chronic Kidney Disease": DiseaseState(
                name="Chronic Kidney Disease", severity=0.50, progression_rate=0.006,
                affected_organs=["kidneys", "heart", "brain"],
                biomarkers={"gfr": 40.0, "creatinine": 2.2, "urea": 25.0}
            ),
            "Arthritis": DiseaseState(
                name="Arthritis", severity=0.45, progression_rate=0.003,
                affected_organs=["joints", "immune_system"],
                biomarkers={"crp": 8.0, "esr": 45.0, "pain_score": 6.0}
            ),
            "Depression": DiseaseState(
                name="Depression", severity=0.50, progression_rate=0.004,
                affected_organs=["brain"],
                biomarkers={"cortisol": 25.0, "serotonin_index": 0.5, "sleep_quality": 0.4}
            ),
            "Anxiety Disorder": DiseaseState(
                name="Anxiety Disorder", severity=0.45, progression_rate=0.004,
                affected_organs=["brain", "heart"],
                biomarkers={"cortisol": 28.0, "heart_rate": 90.0, "gaba_index": 0.6}
            ),
            "Hypothyroidism": DiseaseState(
                name="Hypothyroidism", severity=0.40, progression_rate=0.003,
                affected_organs=["heart", "brain", "muscle"],
                biomarkers={"tsh": 8.0, "t4_free": 0.6, "metabolic_rate": 0.75}
            ),
            "Bacterial Infection": DiseaseState(
                name="Bacterial Infection", severity=0.60, progression_rate=0.020,
                affected_organs=["immune_system", "gut", "liver"],
                biomarkers={"wbc": 14.0, "crp": 50.0, "temperature": 38.8}
            ),
        }
    
    def _load_side_effects_database(self) -> Dict[str, List[SideEffect]]:
        """Load comprehensive side effects database for all 15 drugs"""
        return {
            "Acetaminophen": [
                SideEffect("Liver Toxicity", "high", 24.0, 48.0, 0.05,
                           "Elevated liver enzymes; hepatotoxicity at supratherapeutic doses"),
                SideEffect("Nausea", "low", 1.0, 2.0, 0.12,
                           "Mild gastrointestinal discomfort"),
                SideEffect("Allergic Reaction", "medium", 0.5, 4.0, 0.02,
                           "Skin rash, urticaria, potential anaphylaxis"),
            ],
            "Ibuprofen": [
                SideEffect("GI Bleeding", "high", 12.0, 24.0, 0.07,
                           "Gastric ulcers and gastrointestinal bleeding"),
                SideEffect("Kidney Damage", "medium", 48.0, 72.0, 0.05,
                           "Reduced renal perfusion; acute kidney injury risk"),
                SideEffect("Headache", "low", 2.0, 4.0, 0.18,
                           "Mild to moderate headache"),
            ],
            "Naproxen": [
                SideEffect("GI Irritation", "medium", 4.0, 24.0, 0.12,
                           "Abdominal pain, dyspepsia, peptic ulceration"),
                SideEffect("Fluid Retention", "medium", 24.0, 48.0, 0.08,
                           "Oedema due to renal prostaglandin inhibition"),
                SideEffect("Kidney Damage", "medium", 72.0, 96.0, 0.04,
                           "Impaired renal function with prolonged use"),
            ],
            "Lisinopril": [
                SideEffect("Angioedema", "high", 2.0, 24.0, 0.01,
                           "Swelling of face/lips/tongue — medical emergency"),
                SideEffect("Dry Cough", "low", 24.0, 168.0, 0.20,
                           "Persistent bradykinin-mediated dry cough"),
                SideEffect("Hyperkalemia", "medium", 48.0, 72.0, 0.05,
                           "Elevated potassium; risk in renal impairment"),
                SideEffect("Hypotension", "medium", 1.0, 4.0, 0.08,
                           "First-dose hypotension, particularly in hypovolaemic patients"),
            ],
            "Metoprolol": [
                SideEffect("Bradycardia", "medium", 2.0, 8.0, 0.10,
                           "Heart rate below 50 bpm"),
                SideEffect("Fatigue", "low", 24.0, 72.0, 0.25,
                           "Reduced exercise tolerance, lethargy"),
                SideEffect("Bronchospasm", "high", 1.0, 6.0, 0.04,
                           "Beta-2 mediated bronchoconstriction — caution in asthma/COPD"),
                SideEffect("Cold Extremities", "low", 12.0, 48.0, 0.15,
                           "Peripheral vasoconstriction"),
            ],
            "Atorvastatin": [
                SideEffect("Myopathy", "medium", 168.0, 336.0, 0.05,
                           "Muscle pain, weakness; check CK if symptomatic"),
                SideEffect("Liver Enzyme Elevation", "medium", 48.0, 96.0, 0.03,
                           "Elevated ALT/AST; monitor LFTs"),
                SideEffect("Rhabdomyolysis", "high", 336.0, 168.0, 0.005,
                           "Severe muscle breakdown; risk increases with interactions"),
            ],
            "Amlodipine": [
                SideEffect("Peripheral Oedema", "low", 24.0, 72.0, 0.20,
                           "Ankle swelling due to vasodilation"),
                SideEffect("Flushing", "low", 1.0, 4.0, 0.12,
                           "Facial flushing and warmth"),
                SideEffect("Reflex Tachycardia", "medium", 2.0, 6.0, 0.08,
                           "Compensatory heart rate increase"),
            ],
            "Metformin": [
                SideEffect("Lactic Acidosis", "high", 72.0, 48.0, 0.01,
                           "Rare but serious; risk increased in renal impairment"),
                SideEffect("GI Distress", "medium", 1.0, 24.0, 0.30,
                           "Diarrhoea, nausea, abdominal discomfort — take with food"),
                SideEffect("Vitamin B12 Deficiency", "low", 720.0, 168.0, 0.10,
                           "Long-term B12 malabsorption; monitor annually"),
            ],
            "Glipizide": [
                SideEffect("Hypoglycaemia", "high", 2.0, 6.0, 0.12,
                           "Blood glucose <3.5 mmol/L; dizziness, sweating, confusion"),
                SideEffect("Weight Gain", "low", 168.0, 336.0, 0.20,
                           "Insulin secretagogue effect promotes weight gain"),
                SideEffect("Nausea", "low", 1.0, 4.0, 0.08,
                           "Mild GI discomfort"),
            ],
            "Amoxicillin": [
                SideEffect("Allergic Reaction", "high", 0.5, 6.0, 0.05,
                           "Penicillin hypersensitivity; anaphylaxis risk"),
                SideEffect("Diarrhoea", "low", 4.0, 24.0, 0.20,
                           "Antibiotic-associated GI disruption"),
                SideEffect("C. difficile Infection", "high", 72.0, 168.0, 0.03,
                           "Dysbiosis leading to C. diff colitis"),
            ],
            "Ciprofloxacin": [
                SideEffect("Tendinopathy", "medium", 48.0, 96.0, 0.04,
                           "Tendon inflammation/rupture, especially Achilles"),
                SideEffect("CNS Effects", "medium", 4.0, 12.0, 0.06,
                           "Headache, dizziness, seizure risk in susceptible patients"),
                SideEffect("QT Prolongation", "high", 4.0, 12.0, 0.03,
                           "Cardiac arrhythmia risk; avoid with other QT-prolonging drugs"),
            ],
            "Sertraline": [
                SideEffect("Nausea", "low", 4.0, 24.0, 0.25,
                           "Common early-treatment GI symptom — usually resolves"),
                SideEffect("Insomnia", "low", 12.0, 72.0, 0.15,
                           "Sleep disturbance, vivid dreams"),
                SideEffect("Serotonin Syndrome", "high", 2.0, 12.0, 0.01,
                           "Hyperthermia, agitation, clonus — risk with MAOIs"),
            ],
            "Lorazepam": [
                SideEffect("Sedation", "medium", 1.0, 6.0, 0.40,
                           "CNS depression; impairs driving/coordination"),
                SideEffect("Respiratory Depression", "high", 1.0, 6.0, 0.03,
                           "Reduced respiratory drive; additive with opioids/alcohol"),
                SideEffect("Dependence", "high", 336.0, 672.0, 0.15,
                           "Physical and psychological dependence with prolonged use"),
            ],
            "Levothyroxine": [
                SideEffect("Palpitations", "medium", 24.0, 48.0, 0.08,
                           "Supraphysiological dosing causes tachyarrhythmia"),
                SideEffect("Insomnia", "low", 24.0, 72.0, 0.10,
                           "CNS stimulation at high doses"),
                SideEffect("Chest Pain", "high", 12.0, 24.0, 0.02,
                           "Angina risk in ischaemic heart disease patients"),
            ],
            "Prednisone": [
                SideEffect("Hyperglycaemia", "medium", 2.0, 24.0, 0.30,
                           "Steroid-induced glucose elevation"),
                SideEffect("Immunosuppression", "high", 24.0, 72.0, 0.50,
                           "Increased infection risk"),
                SideEffect("GI Ulceration", "medium", 12.0, 48.0, 0.10,
                           "Gastric irritation; take with food or PPI cover"),
                SideEffect("Osteoporosis", "medium", 720.0, 336.0, 0.20,
                           "Long-term bone density loss; consider bisphosphonate"),
            ],
        }

    def _load_drug_disease_map(self) -> Dict[str, List[str]]:
        """Return mapping of disease name → list of indicated drugs"""
        return {
            "Hypertension":              ["Lisinopril", "Metoprolol", "Amlodipine", "Atorvastatin"],
            "Heart Failure (Mild)":      ["Lisinopril", "Metoprolol", "Atorvastatin"],
            "Heart Failure (Moderate)":  ["Lisinopril", "Metoprolol", "Atorvastatin"],
            "Heart Failure (Severe)":    ["Lisinopril", "Metoprolol"],
            "Type 2 Diabetes (Mild)":    ["Metformin", "Glipizide"],
            "Type 2 Diabetes (Moderate)":["Metformin", "Glipizide"],
            "Type 2 Diabetes (Severe)":  ["Metformin", "Glipizide"],
            "COPD (Moderate)":           ["Prednisone"],
            "COPD (Severe)":             ["Prednisone"],
            "Chronic Kidney Disease":    ["Lisinopril", "Amlodipine"],
            "Arthritis":                 ["Ibuprofen", "Naproxen", "Prednisone", "Acetaminophen"],
            "Depression":                ["Sertraline"],
            "Anxiety Disorder":          ["Sertraline", "Lorazepam"],
            "Hypothyroidism":            ["Levothyroxine"],
            "Bacterial Infection":       ["Amoxicillin", "Ciprofloxacin"],
        }

    def _load_drug_incompatibilities(self) -> Dict[tuple, str]:
        """Return pairs of drugs that have clinically significant interactions"""
        return {
            ("Ibuprofen",    "Lisinopril"):  "NSAIDs blunt ACE inhibitor efficacy and increase renal injury risk",
            ("Naproxen",     "Lisinopril"):  "NSAIDs blunt ACE inhibitor efficacy and increase renal injury risk",
            ("Ibuprofen",    "Metformin"):   "NSAIDs may increase metformin accumulation → lactic acidosis",
            ("Metoprolol",   "Amlodipine"):  "Additive negative chronotropy — risk of excessive bradycardia",
            ("Sertraline",   "Lorazepam"):   "Enhanced CNS depression — monitor respiratory function",
            ("Prednisone",   "Ibuprofen"):   "Additive risk of GI ulceration and bleeding",
            ("Prednisone",   "Naproxen"):    "Additive risk of GI ulceration and bleeding",
            ("Levothyroxine","Metformin"):   "Metformin may reduce levothyroxine absorption — stagger doses",
            ("Ciprofloxacin","Metformin"):   "Ciprofloxacin inhibits renal tubular transport of metformin",
            ("Glipizide",    "Ibuprofen"):   "NSAIDs may potentiate hypoglycaemic effect of sulphonylureas",
        }

    def _load_organ_drug_effects(self) -> Dict[str, Dict[str, float]]:
        """Return per-organ effect magnitudes at therapeutic concentration.
        Positive = benefit (improves organ function).
        Negative = harm (impairs organ function).
        Values represent fractional health change at 1× upper therapeutic window."""
        return {
            "Acetaminophen": {"liver": -0.15},
            "Ibuprofen":     {"kidneys": -0.12, "gut": -0.10, "heart": -0.04},
            "Naproxen":      {"kidneys": -0.10, "gut": -0.12},
            "Lisinopril":    {"heart":   +0.15, "kidneys": +0.10},
            "Metoprolol":    {"heart":   +0.18, "lungs": -0.06, "brain": +0.05},
            "Atorvastatin":  {"liver":   -0.06, "muscle": -0.08, "heart": +0.10},
            "Amlodipine":    {"heart":   +0.12, "brain":  +0.08},
            "Metformin":     {"liver":   +0.10, "kidneys": -0.04},
            "Glipizide":     {"pancreas":+0.15},
            "Prednisone":    {"immune_system": -0.20, "gut": -0.08, "lungs": +0.12},
            "Amoxicillin":   {"gut":     -0.06},
            "Ciprofloxacin": {"gut":     -0.08, "muscle": -0.05},
            "Sertraline":    {"brain":   +0.12, "gut": -0.04},
            "Lorazepam":     {"brain":   +0.08, "lungs": -0.08},
            "Levothyroxine": {"heart":   +0.08, "brain": +0.10, "muscle": +0.06},
        }
    
    def calculate_drug_concentration(self, drug: DrugProperties, time_hours: float, 
                                    bio_params: BiologicalParameters) -> float:
        """Calculate drug concentration at given time"""
        # Adjust for biological parameters
        adjusted_clearance = drug.clearance * bio_params.kidney_function * bio_params.liver_function
        adjusted_volume = drug.volume_distribution * (bio_params.weight / 70.0)
        
        # Multiple dosing consideration
        if drug.dosing_frequency == "daily":
            doses = int(time_hours / 24) + 1
        elif drug.dosing_frequency == "bid":
            doses = int(time_hours / 12) + 1
        else:
            doses = 1
        
        concentration = 0
        for dose in range(doses):
            dose_time = dose * (24 if drug.dosing_frequency == "daily" else 12)
            if dose_time <= time_hours:
                time_since_dose = time_hours - dose_time
                dose_concentration = (drug.dose_mg * drug.bioavailability / adjusted_volume)
                concentration += dose_concentration * np.exp(-adjusted_clearance * time_since_dose / adjusted_volume)
        
        return concentration
    
    def simulate_disease_progression(self, disease: DiseaseState, time_hours: float, 
                                   drug_effects: Dict[str, float]) -> DiseaseState:
        """Simulate disease progression with drug effects"""
        new_disease = DiseaseState(
            name=disease.name,
            severity=disease.severity,
            progression_rate=disease.progression_rate,
            affected_organs=disease.affected_organs.copy(),
            biomarkers=disease.biomarkers.copy()
        )
        
        # Natural progression
        new_disease.severity += disease.progression_rate * time_hours
        new_disease.severity = min(1.0, max(0.0, new_disease.severity))
        
        # Drug effects (simplified)
        for drug_name, effect in drug_effects.items():
            if effect > 0:  # Therapeutic effect
                new_disease.severity *= (1 - effect * 0.3)  # 30% reduction max
        
        # Update biomarkers based on severity
        for biomarker, base_value in disease.biomarkers.items():
            if biomarker == "blood_pressure":
                new_disease.biomarkers[biomarker] = base_value * (1 + new_disease.severity * 0.3)
            elif biomarker == "glucose":
                new_disease.biomarkers[biomarker] = base_value * (1 + new_disease.severity * 0.5)
            else:
                new_disease.biomarkers[biomarker] = base_value * (1 + new_disease.severity * 0.2)
        
        return new_disease
    
    def check_side_effects(self, drug_name: str, time_hours: float,
                           drug_concentration: float = 0.0) -> List[SideEffect]:
        """Check for side effects at given time using concentration-dependent probability.

        The base `effect.probability` is scaled by a sigmoid centred at 1.5× the
        upper therapeutic threshold, so higher concentrations strongly increase the
        risk of triggering side effects while sub-therapeutic concentrations have
        near-zero probability.
        """
        if drug_name not in self.side_effects_database:
            return []

        # Look up therapeutic window upper bound for the drug
        if drug_name in self.drug_database:
            tox_threshold = float(self.drug_database[drug_name].therapeutic_window[1]) * 1.5
        else:
            tox_threshold = 100.0   # safe default (mg/L)

        # Concentration-dependent multiplier: sigmoid centred at tox_threshold
        if tox_threshold > 0 and drug_concentration > 0:
            z = 5.0 * (drug_concentration - tox_threshold) / tox_threshold
            conc_multiplier = 1.0 / (1.0 + np.exp(-z))
        else:
            conc_multiplier = effect_prob_fallback = 0.5  # no concentration info

        active_effects = []
        for effect in self.side_effects_database[drug_name]:
            if not effect.active and time_hours >= effect.onset_time:
                # Base probability scaled by concentration-dependent multiplier
                effective_prob = effect.probability * max(conc_multiplier, 0.01)
                if random.random() < effective_prob:
                    effect.active = True
                    effect.start_time = time_hours
                    effect.end_time = time_hours + effect.duration
                    active_effects.append(effect)
            elif effect.active and time_hours >= effect.end_time:
                effect.active = False
            elif effect.active:
                active_effects.append(effect)

        return active_effects
    
    def get_system_status(self, bio_params: BiologicalParameters,
                         drug_concentrations: Dict[str, float],
                         disease_states: Dict[str, DiseaseState],
                         side_effects: List[SideEffect]) -> Dict[str, any]:
        """Get overall system status across all 12 organs with real drug effects"""
        ALL_ORGANS = [
            "brain", "heart", "liver", "kidneys", "lungs",
            "muscle", "adipose", "gut", "skin",
            "pancreas", "immune_system", "joints",
        ]

        organ_health: Dict[str, float] = {}
        organ_drug_breakdown: Dict[str, Dict[str, float]] = {o: {} for o in ALL_ORGANS}

        for organ in ALL_ORGANS:
            health = 1.0

            # Baseline impairment from patient physiology
            if organ == "kidneys":
                health *= bio_params.kidney_function
            elif organ == "liver":
                health *= bio_params.liver_function

            # Disease-induced impairment
            for disease in disease_states.values():
                if organ in disease.affected_organs:
                    health -= disease.severity * 0.30

            # Drug effects (beneficial or harmful), scaled by conc/therapeutic window
            for drug_name, conc in drug_concentrations.items():
                if drug_name in self.organ_drug_effects and organ in self.organ_drug_effects[drug_name]:
                    t_win_hi = self.drug_database[drug_name].therapeutic_window[1] if drug_name in self.drug_database else 50.0
                    conc_ratio = conc / (t_win_hi + 1e-10)
                    delta = self.organ_drug_effects[drug_name][organ] * conc_ratio
                    health += delta
                    organ_drug_breakdown[organ][drug_name] = delta

            organ_health[organ] = max(0.0, min(1.2, health))

        overall_health = float(np.mean(list(organ_health.values())))

        warnings: List[str] = []
        recommendations: List[str] = []

        # Side-effect warnings
        for effect in side_effects:
            if effect.severity == "high":
                warnings.append(f"HIGH: {effect.name} — {effect.description}")
            elif effect.severity == "medium":
                warnings.append(f"MEDIUM: {effect.name} — {effect.description}")

        # Drug interaction warnings
        selected = list(drug_concentrations.keys())
        for i, d1 in enumerate(selected):
            for d2 in selected[i + 1:]:
                key = (d1, d2) if (d1, d2) in self.drug_incompatibilities else (d2, d1)
                if key in self.drug_incompatibilities:
                    warnings.append(f"INTERACTION: {d1} + {d2} — {self.drug_incompatibilities[key]}")

        # Organ-based recommendations
        for organ, health in organ_health.items():
            if health < 0.60:
                recommendations.append(f"Closely monitor {organ.replace('_', ' ').title()} function (health: {health:.0%})")
            elif health < 0.75:
                recommendations.append(f"Watch {organ.replace('_', ' ').title()} function (health: {health:.0%})")

        # Renal/hepatic dosing cautions
        if bio_params.kidney_function < 0.50:
            for drug in ["Metformin", "Ciprofloxacin", "Amoxicillin"]:
                if drug in drug_concentrations:
                    recommendations.append(f"Reduce {drug} dose — GFR severely reduced (kidney function {bio_params.kidney_function:.0%})")
        if bio_params.liver_function < 0.50:
            for drug in ["Acetaminophen", "Atorvastatin", "Sertraline"]:
                if drug in drug_concentrations:
                    recommendations.append(f"Use {drug} with caution — hepatic impairment (liver function {bio_params.liver_function:.0%})")

        return {
            "overall_health":      overall_health,
            "organ_status":        organ_health,
            "organ_drug_effects":  organ_drug_breakdown,
            "warnings":            warnings,
            "recommendations":     recommendations,
        }

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'simulator' not in st.session_state:
        st.session_state.simulator = BiologicalSystemSimulator()
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = {
            'time_points': [],
            'drug_concentrations': {},
            'disease_severity': {},
            'side_effects': [],
            'system_status': [],
            'ml_pk_mean':  {},
            'ml_pk_upper': {},
            'ml_pk_lower': {},
            'ml_active':   False,
        }
    if 'current_config' not in st.session_state:
        st.session_state.current_config = {
            'bio_params':          BiologicalParameters(),
            'selected_drugs':      [],
            'selected_diseases':   [],
            'drug_configs':        {},
            'disease_configs':     {},
            'simulation_duration': 72.0,
        }

def render_configuration_panel():
    """Render configuration panel — diseases first, then filtered drugs"""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🔧 System Configuration")
    simulator = st.session_state.simulator

    # ── Biological Parameters ────────────────────────────────────────
    with st.expander("🧬 Biological Parameters", expanded=True):
        bio_params = st.session_state.current_config['bio_params']
        col1, col2 = st.columns(2)
        with col1:
            bio_params.age           = st.slider("Age (yrs)", 18, 100, int(bio_params.age))
            bio_params.weight        = st.slider("Weight (kg)", 40, 160, int(bio_params.weight))
            bio_params.sex           = st.selectbox("Sex", [("Male", 1), ("Female", 0)],
                                                     index=0 if bio_params.sex == 1 else 1)[1]
        with col2:
            bio_params.kidney_function = st.slider("Kidney Function", 0.0, 1.0,
                                                    bio_params.kidney_function, step=0.05)
            bio_params.liver_function  = st.slider("Liver Function",  0.0, 1.0,
                                                    bio_params.liver_function,  step=0.05)
            bio_params.metabolic_rate  = st.slider("Metabolic Rate",  0.5, 2.0,
                                                    bio_params.metabolic_rate,  step=0.05)

    # ── Disease Selection (drives drug filter) ───────────────────────
    with st.expander("🏥 Disease States", expanded=True):
        available_diseases = list(simulator.disease_database.keys())
        selected_diseases = st.multiselect(
            "Select active diseases",
            available_diseases,
            default=st.session_state.current_config.get('selected_diseases', []),
            help="Disease selection filters the available medication list below."
        )

        disease_configs: Dict[str, dict] = {}
        for disease in selected_diseases:
            with st.container():
                st.markdown(f"**{disease}**")
                ds = simulator.disease_database[disease]
                sev = st.slider(f"Severity (0–1)", 0.0, 1.0, ds.severity,
                                step=0.05, key=f"dis_sev_{disease}")
                disease_configs[disease] = {"severity": sev}
                # Show what organs are affected
                if ds.affected_organs:
                    st.caption(f"Affected organs: {', '.join(o.replace('_',' ').title() for o in ds.affected_organs)}")
                st.divider()

    # ── Drug Selection (filtered by selected diseases) ───────────────
    with st.expander("💊 Medication", expanded=True):
        # Build candidate set from disease map
        indicated_drugs: List[str] = []
        for disease in selected_diseases:
            indicated_drugs.extend(simulator.drug_disease_map.get(disease, []))
        indicated_drugs = sorted(set(indicated_drugs))

        if indicated_drugs:
            st.markdown(f"**Indicated drugs for selected diseases ({len(indicated_drugs)}):**")
            drug_options = indicated_drugs
        else:
            st.info("Select diseases above to see indicated medications, or choose manually from all drugs.")
            drug_options = sorted(simulator.drug_database.keys())

        # Always allow user to expand to full drug list
        show_all = st.checkbox("Show all 15 drugs (override disease filter)",
                               value=not bool(indicated_drugs))
        if show_all:
            drug_options = sorted(simulator.drug_database.keys())

        # Default previous selection but only from valid options
        prev_drugs = [d for d in st.session_state.current_config.get('selected_drugs', [])
                      if d in drug_options]
        selected_drugs = st.multiselect("Select drugs", drug_options, default=prev_drugs)

        # ── Compatibility check ──────────────────────────────────────
        if len(selected_drugs) > 1:
            interactions_found = []
            for i, d1 in enumerate(selected_drugs):
                for d2 in selected_drugs[i + 1:]:
                    key  = (d1, d2) if (d1, d2) in simulator.drug_incompatibilities else (d2, d1)
                    msg  = simulator.drug_incompatibilities.get(key)
                    if msg:
                        interactions_found.append(f"⚠️ **{d1} + {d2}:** {msg}")
            if interactions_found:
                st.markdown("**Drug Interactions Detected:**")
                for w in interactions_found:
                    st.warning(w)

        # ── Contraindications from bio_params ───────────────────────
        contras = []
        if bio_params.kidney_function < 0.40:
            for d in ["Metformin", "Ciprofloxacin", "Amoxicillin"]:
                if d in selected_drugs:
                    contras.append(f"🚫 **{d}** — contraindicated with GFR <50% (current: {bio_params.kidney_function:.0%})")
        if bio_params.liver_function < 0.40:
            for d in ["Acetaminophen", "Atorvastatin", "Sertraline"]:
                if d in selected_drugs:
                    contras.append(f"🚫 **{d}** — use with extreme caution in severe hepatic impairment")
        for c in contras:
            st.error(c)

        # ── Drug dose configuration ──────────────────────────────────
        drug_configs: Dict[str, dict] = {}
        for drug in selected_drugs:
            with st.container():
                st.markdown(f"**{drug}**")
                dp = simulator.drug_database[drug]
                col1, col2 = st.columns(2)
                with col1:
                    dose = st.number_input(f"Dose (mg)", min_value=0.01,
                                           max_value=2000.0, value=float(dp.dose_mg),
                                           step=1.0, key=f"dose_{drug}")
                with col2:
                    freq = st.selectbox("Frequency",
                                        ["daily", "bid", "tid", "qid"],
                                        index=["daily","bid","tid","qid"].index(dp.dosing_frequency)
                                              if dp.dosing_frequency in ["daily","bid","tid","qid"] else 0,
                                        key=f"freq_{drug}")
                # Show organ targets
                if drug in simulator.organ_drug_effects:
                    beneficial = [f"{o.replace('_',' ').title()} ↑" for o, v in simulator.organ_drug_effects[drug].items() if v > 0]
                    harmful    = [f"{o.replace('_',' ').title()} ↓" for o, v in simulator.organ_drug_effects[drug].items() if v < 0]
                    if beneficial:
                        st.caption("Benefit: " + ", ".join(beneficial))
                    if harmful:
                        st.caption("Caution: " + ", ".join(harmful))
                drug_configs[drug] = {"dose": dose, "frequency": freq}
                st.divider()

    # ── Simulation Parameters ─────────────────────────────────────────
    st.markdown("**Simulation Parameters**")
    duration = st.slider("Duration (hours)", 24, 168,
                         int(st.session_state.current_config['simulation_duration']))

    # Update session state
    st.session_state.current_config.update({
        'selected_drugs':    selected_drugs,
        'selected_diseases': selected_diseases,
        'drug_configs':      drug_configs,
        'disease_configs':   disease_configs,
        'simulation_duration': float(duration),
    })

    st.markdown('</div>', unsafe_allow_html=True)

def render_simulation_control():
    """Render simulation control panel"""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🚀 Simulation Control")

    # ML model toggle
    ml_available = os.path.exists(_CHECKPOINT_PATH)
    ml_label = "🤖 Use ML Model" if ml_available else "🤖 ML Model (train first)"
    use_ml = st.checkbox(ml_label, value=st.session_state.get('use_ml_model', False),
                         disabled=not ml_available,
                         help="Overlay SimpleMLP predictions with ±1σ confidence bands on the "
                              "concentration chart. Requires checkpoints/simple_mlp_model.pth.")
    st.session_state.use_ml_model = use_ml
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start Simulation", type="primary"):
            run_simulation()
    
    with col2:
        if st.button("🔄 Reset Simulation"):
            reset_simulation()
    
    # Simulation status
    if st.session_state.get('simulation_running', False):
        st.markdown('<div class="status-indicator status-normal"></div>Simulation Running', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-indicator status-warning"></div>Simulation Stopped', 
                   unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def run_simulation():
    """Run the biological system simulation"""
    st.session_state.simulation_running = True

    # Clear previous data
    st.session_state.simulation_data = {
        'time_points': [],
        'drug_concentrations': {},
        'disease_severity': {},
        'side_effects': [],
        'system_status': [],
        'ml_pk_mean': {},    # ML-predicted PK concentrations (mean)
        'ml_pk_upper': {},   # +1σ band
        'ml_pk_lower': {},   # –1σ band
        'ml_active': False,
    }

    config = st.session_state.current_config
    simulator = st.session_state.simulator
    bio_params = config['bio_params']

    # ------------------------------------------------------------------
    # ML model (optional)
    # ------------------------------------------------------------------
    use_ml = st.session_state.get('use_ml_model', False)
    ml_model = load_ml_model() if use_ml else None
    if use_ml and ml_model is None:
        st.warning("⚠️ ML checkpoint not found or incompatible (retrain required). "
                   "Falling back to analytical PK.")
        use_ml = False
    st.session_state.simulation_data['ml_active'] = use_ml

    # Initialise per-drug lists
    for drug_name in config['selected_drugs']:
        st.session_state.simulation_data['drug_concentrations'][drug_name] = []
        st.session_state.simulation_data['ml_pk_mean'][drug_name] = []
        st.session_state.simulation_data['ml_pk_upper'][drug_name] = []
        st.session_state.simulation_data['ml_pk_lower'][drug_name] = []

    for disease_name in config['selected_diseases']:
        st.session_state.simulation_data['disease_severity'][disease_name] = []

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    duration = config['simulation_duration']
    time_step = 1.0  # 1 hour steps

    for t in np.arange(0, duration, time_step):
        st.session_state.simulation_data['time_points'].append(t)

        drug_effects = {}
        for drug_name in config['selected_drugs']:
            drug_props = simulator.drug_database[drug_name]
            if drug_name in config['drug_configs']:
                drug_props.dose_mg = config['drug_configs'][drug_name]['dose']
                drug_props.dosing_frequency = config['drug_configs'][drug_name]['frequency']

            concentration = simulator.calculate_drug_concentration(drug_props, t, bio_params)
            st.session_state.simulation_data['drug_concentrations'][drug_name].append(concentration)
            drug_effects[drug_name] = concentration / drug_props.therapeutic_window[1]

            # ML prediction + MC Dropout uncertainty
            if use_ml and ml_model is not None:
                feat = build_ml_feature(drug_props, bio_params, t)
                pk_mean, pk_std, _pd_mean, _pd_std = mc_predict(ml_model, feat, n_samples=50)
                # Scale back: model outputs are normalised by /1000
                pk_mu = pk_mean * 1000.0
                pk_sigma = pk_std * 1000.0
                st.session_state.simulation_data['ml_pk_mean'][drug_name].append(pk_mu)
                st.session_state.simulation_data['ml_pk_upper'][drug_name].append(pk_mu + pk_sigma)
                st.session_state.simulation_data['ml_pk_lower'][drug_name].append(pk_mu - pk_sigma)
            else:
                st.session_state.simulation_data['ml_pk_mean'][drug_name].append(None)
                st.session_state.simulation_data['ml_pk_upper'][drug_name].append(None)
                st.session_state.simulation_data['ml_pk_lower'][drug_name].append(None)

        # Disease progression
        current_diseases = {}
        for disease_name in config['selected_diseases']:
            disease_state = simulator.disease_database[disease_name]
            if disease_name in config['disease_configs']:
                disease_state.severity = config['disease_configs'][disease_name]['severity']
            updated_disease = simulator.simulate_disease_progression(disease_state, t, drug_effects)
            current_diseases[disease_name] = updated_disease
            st.session_state.simulation_data['disease_severity'][disease_name].append(updated_disease.severity)

        # Side effects — pass current analytical concentration
        current_side_effects = []
        for drug_name in config['selected_drugs']:
            conc = st.session_state.simulation_data['drug_concentrations'][drug_name][-1] \
                   if st.session_state.simulation_data['drug_concentrations'][drug_name] else 0.0
            effects = simulator.check_side_effects(drug_name, t, drug_concentration=conc)
            current_side_effects.extend(effects)
        
        st.session_state.simulation_data['side_effects'].append(current_side_effects)
        
        # Calculate system status
        status = simulator.get_system_status(bio_params, 
                                          {name: st.session_state.simulation_data['drug_concentrations'][name][-1] 
                                           if st.session_state.simulation_data['drug_concentrations'][name] else 0
                                           for name in config['selected_drugs']},
                                          current_diseases,
                                          current_side_effects)
        st.session_state.simulation_data['system_status'].append(status)
    
    st.session_state.simulation_running = False
    st.success("✅ Simulation completed successfully!")

def reset_simulation():
    """Reset the simulation"""
    st.session_state.simulation_running = False
    st.session_state.simulation_data = {
        'time_points': [],
        'drug_concentrations': {},
        'disease_severity': {},
        'side_effects': [],
        'system_status': [],
        'ml_pk_mean': {},
        'ml_pk_upper': {},
        'ml_pk_lower': {},
        'ml_active': False,
    }
    st.rerun()

def render_main_dashboard():
    """Render main simulation dashboard"""
    if not st.session_state.simulation_data['time_points']:
        st.markdown("""
        <div class="main-header">
            <h1>🧬 PK-PD Biological System Simulator</h1>
            <p>Configure your simulation parameters and click "Start Simulation" to begin real-time biological system modeling</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🧬 Real-Time Biological System Simulation")
    st.markdown(f"**Simulation Duration:** {st.session_state.current_config['simulation_duration']} hours")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.simulation_data['system_status']:
            latest_status = st.session_state.simulation_data['system_status'][-1]
            overall = latest_status.get('overall_health', 1.0)
            st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{overall:.0%}</div>', unsafe_allow_html=True)
            st.markdown("**Overall Health**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        active_drugs = len([d for d in st.session_state.current_config['selected_drugs']])
        st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{active_drugs}</div>', unsafe_allow_html=True)
        st.markdown("**Active Drugs**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        active_diseases = len([d for d in st.session_state.current_config['selected_diseases']])
        st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{active_diseases}</div>', unsafe_allow_html=True)
        st.markdown("**Diseases**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        current_side_effects = len(st.session_state.simulation_data['side_effects'][-1]) if st.session_state.simulation_data['side_effects'] else 0
        st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{current_side_effects}</div>', unsafe_allow_html=True)
        st.markdown("**Active Side Effects**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Simulation Timeline
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
        st.subheader("💊 Drug Concentration Timeline")

        sim_data = st.session_state.simulation_data
        times = sim_data['time_points']
        fig = go.Figure()

        for drug_name, concentrations in sim_data['drug_concentrations'].items():
            fig.add_trace(go.Scatter(
                x=times, y=concentrations,
                mode='lines', name=f"{drug_name} (analytical)",
                line=dict(width=2)
            ))

            # ML predictions + ±1σ confidence band
            if sim_data.get('ml_active') and sim_data['ml_pk_mean'].get(drug_name):
                ml_mean = sim_data['ml_pk_mean'][drug_name]
                ml_upper = sim_data['ml_pk_upper'][drug_name]
                ml_lower = sim_data['ml_pk_lower'][drug_name]
                if all(v is not None for v in ml_mean):
                    # Upper bound (invisible, used for fill)
                    fig.add_trace(go.Scatter(
                        x=times, y=ml_upper,
                        mode='lines', line=dict(width=0),
                        showlegend=False, name=f"{drug_name} ML upper"
                    ))
                    # Lower bound with fill to upper
                    fig.add_trace(go.Scatter(
                        x=times, y=ml_lower,
                        mode='lines', line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(100,149,237,0.2)',
                        showlegend=False, name=f"{drug_name} ML lower"
                    ))
                    # Mean prediction line
                    fig.add_trace(go.Scatter(
                        x=times, y=ml_mean,
                        mode='lines', name=f"{drug_name} (ML ±1σ)",
                        line=dict(width=2, dash='dash')
                    ))

        fig.update_layout(
            title="Drug Concentrations Over Time",
            xaxis_title="Time (hours)",
            yaxis_title="Concentration (mg/L)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
        st.subheader("🏥 Disease Progression")
        
        fig = go.Figure()
        for disease_name, severities in st.session_state.simulation_data['disease_severity'].items():
            fig.add_trace(go.Scatter(
                x=st.session_state.simulation_data['time_points'],
                y=severities,
                mode='lines',
                name=disease_name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Disease Severity Over Time",
            xaxis_title="Time (hours)",
            yaxis_title="Severity (0-1)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Side Effects Timeline
    st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
    st.subheader("⚠️ Side Effects Timeline")
    
    # Create side effects timeline
    side_effects_data = []
    for i, time_point in enumerate(st.session_state.simulation_data['time_points']):
        if i < len(st.session_state.simulation_data['side_effects']):
            for effect in st.session_state.simulation_data['side_effects'][i]:
                side_effects_data.append({
                    'time': time_point,
                    'effect': effect.name,
                    'severity': effect.severity,
                    'active': effect.active
                })
    
    if side_effects_data:
        side_effects_df = pd.DataFrame(side_effects_data)
        
        # Create timeline chart
        fig = px.scatter(
            side_effects_df,
            x='time',
            y='effect',
            color='severity',
            symbol='active',
            title="Side Effects Occurrence Timeline",
            height=400,
            template="plotly_white"
        )
        fig.update_layout(yaxis_title="Side Effect")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No side effects detected in simulation period")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Organ Health Radar + Drug Effects
    st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
    st.subheader("🫀 Organ Health & Drug Effects")

    if st.session_state.simulation_data['system_status']:
        latest_status = st.session_state.simulation_data['system_status'][-1]
        organ_status  = latest_status['organ_status']
        drug_effects  = latest_status.get('organ_drug_effects', {})

        ALL_ORGANS = list(organ_status.keys())
        organ_labels = [o.replace('_', ' ').title() for o in ALL_ORGANS]
        health_vals  = [organ_status[o] for o in ALL_ORGANS]

        col_radar, col_effects = st.columns([1, 1])

        with col_radar:
            # Radar / spider chart of all organs
            fig_radar = go.Figure(go.Scatterpolar(
                r=health_vals + [health_vals[0]],
                theta=organ_labels + [organ_labels[0]],
                fill='toself',
                fillcolor='rgba(0, 150, 255, 0.15)',
                line=dict(color='rgba(0, 150, 255, 0.8)', width=2),
                name="Current Health"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
                title="Organ Health Radar",
                height=380,
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_effects:
            # Table: organ, health %, colour, which drug is helping/hurting
            st.markdown("**Per-Organ Status with Drug Contributions**")
            for organ, health in organ_status.items():
                icon = "🟢" if health >= 0.85 else ("🟡" if health >= 0.65 else "🔴")
                label = organ.replace('_', ' ').title()
                st.markdown(f"{icon} **{label}**: {health:.0%}")
                # Show drug contributions
                contributions = drug_effects.get(organ, {})
                if contributions:
                    parts = []
                    for d, delta in contributions.items():
                        arrow = "↑" if delta > 0 else "↓"
                        color = "green" if delta > 0 else "red"
                        parts.append(f'<span style="color:{color}">{d} {arrow}{abs(delta):.0%}</span>')
                    st.markdown("&nbsp;&nbsp;&nbsp; " + " | ".join(parts), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Organ Health Timeline
    if len(st.session_state.simulation_data['system_status']) > 1:
        st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
        st.subheader("📈 Organ Health Over Time")

        times = st.session_state.simulation_data['time_points']
        statuses = st.session_state.simulation_data['system_status']

        # Collect organ timeline data
        all_organs_in_data = list(statuses[0]['organ_status'].keys()) if statuses else []
        selected_organs_to_plot = st.multiselect(
            "Select organs to display",
            [o.replace('_', ' ').title() for o in all_organs_in_data],
            default=[o.replace('_', ' ').title() for o in all_organs_in_data[:6]],
            key="organ_timeline_select"
        )
        selected_organ_keys = [o.lower().replace(' ', '_') for o in selected_organs_to_plot]

        fig_timeline = go.Figure()
        color_seq = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b',
                     '#e377c2','#7f7f7f','#bcbd22','#17becf','#aec7e8','#ffbb78']
        for idx, organ in enumerate(selected_organ_keys):
            values = [s['organ_status'].get(organ, 1.0) for s in statuses]
            fig_timeline.add_trace(go.Scatter(
                x=times[:len(values)], y=values,
                mode='lines', name=organ.replace('_', ' ').title(),
                line=dict(width=2, color=color_seq[idx % len(color_seq)])
            ))

        fig_timeline.update_layout(
            title="Organ Health Trajectory",
            xaxis_title="Time (hours)",
            yaxis_title="Organ Health (0–1.2)",
            yaxis=dict(range=[0, 1.25]),
            height=380,
            template="plotly_white",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # System Status and Recommendations
    st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
    st.subheader("📊 Clinical Summary & Recommendations")

    if st.session_state.simulation_data['system_status']:
        latest_status = st.session_state.simulation_data['system_status'][-1]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Overall System Health**")
            oh = latest_status['overall_health']
            health_color = "#2ecc71" if oh >= 0.8 else ("#f39c12" if oh >= 0.6 else "#e74c3c")
            st.markdown(
                f'<div style="font-size:2rem;font-weight:bold;color:{health_color}">{oh:.0%}</div>',
                unsafe_allow_html=True
            )
            if latest_status['recommendations']:
                st.markdown("**Recommendations:**")
                for rec in latest_status['recommendations']:
                    st.markdown(f"• {rec}")
            else:
                st.markdown("✅ No immediate clinical concerns detected.")

        with col2:
            if latest_status['warnings']:
                st.markdown("**⚠️ Active Warnings**")
                for warning in latest_status['warnings']:
                    lvl = warning.split(":")[0].strip().upper() if ":" in warning else ""
                    if lvl == "HIGH" or lvl == "INTERACTION":
                        st.markdown(f'<div class="alert-high">{warning}</div>', unsafe_allow_html=True)
                    elif lvl == "MEDIUM":
                        st.markdown(f'<div class="alert-medium">{warning}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-low">{warning}</div>', unsafe_allow_html=True)
            else:
                st.success("No active warnings.")

    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        render_configuration_panel()
        render_simulation_control()
        
        # Quick Stats
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("📈 Quick Stats")
        
        if st.session_state.simulation_data['time_points']:
            total_time = st.session_state.simulation_data['time_points'][-1]
            st.markdown(f"**Simulation Time:** {total_time:.1f} hours")
            
            total_side_effects = sum(len(effects) for effects in st.session_state.simulation_data['side_effects'])
            st.markdown(f"**Total Side Effects:** {total_side_effects}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    render_main_dashboard()

if __name__ == "__main__":
    main()
