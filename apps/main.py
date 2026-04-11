"""
PK-PD Interactive Dashboard — FastAPI Backend
==============================================
Serves the HTML dashboard and exposes simulation REST API endpoints.
Run with:
    uvicorn apps.fastapi_app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)

# ===========================================================================
# ── Simulation Domain Classes (extracted from advanced_dashboard.py) ─────
# ===========================================================================

@dataclass
class BiologicalParameters:
    age: float = 65.0
    weight: float = 70.0
    sex: int = 1            # 0 = female, 1 = male
    kidney_function: float = 0.8
    liver_function: float = 0.9
    heart_rate: float = 72.0
    blood_pressure: Tuple[float, float] = (120.0, 80.0)
    metabolic_rate: float = 1.0


@dataclass
class DrugProperties:
    name: str = ""
    molecular_weight: float = 300.0
    logp: float = 1.5
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
    name: str = ""
    severity: float = 0.5
    progression_rate: float = 0.01
    affected_organs: List[str] = field(default_factory=list)
    biomarkers: Dict[str, float] = field(default_factory=dict)


class SideEffect:
    def __init__(self, name: str, severity: str, onset_time: float,
                 duration: float, probability: float, description: str):
        self.name = name
        self.severity = severity
        self.onset_time = onset_time
        self.duration = duration
        self.probability = probability
        self.description = description
        self.active = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None


# ---------------------------------------------------------------------------
# Drug → receptor binding table
# ---------------------------------------------------------------------------
DRUG_RECEPTORS: Dict[str, List[Dict[str, Any]]] = {
    "Lisinopril":    [{"receptor": "ACE",                "effect": "inhibition",  "organ_affinity": ["heart","kidneys","lungs"]}],
    "Metoprolol":    [{"receptor": "β1-Adrenergic",      "effect": "blockade",    "organ_affinity": ["heart","brain"]}],
    "Ibuprofen":     [{"receptor": "COX-1",              "effect": "inhibition",  "organ_affinity": ["gut","kidneys"]},
                      {"receptor": "COX-2",              "effect": "inhibition",  "organ_affinity": ["joints","immune_system"]}],
    "Naproxen":      [{"receptor": "COX-1",              "effect": "inhibition",  "organ_affinity": ["gut"]},
                      {"receptor": "COX-2",              "effect": "inhibition",  "organ_affinity": ["joints"]}],
    "Atorvastatin":  [{"receptor": "HMG-CoA Reductase",  "effect": "inhibition",  "organ_affinity": ["liver"]}],
    "Amlodipine":    [{"receptor": "L-Ca²⁺ Channel",     "effect": "blockade",    "organ_affinity": ["heart","brain"]}],
    "Metformin":     [{"receptor": "AMPK",               "effect": "activation",  "organ_affinity": ["liver","muscle"]}],
    "Glipizide":     [{"receptor": "KATP (SUR1)",        "effect": "closure",     "organ_affinity": ["pancreas"]}],
    "Amoxicillin":   [{"receptor": "PBP",                "effect": "inhibition",  "organ_affinity": ["immune_system","gut"]}],
    "Ciprofloxacin": [{"receptor": "DNA Gyrase",         "effect": "inhibition",  "organ_affinity": ["immune_system"]}],
    "Sertraline":    [{"receptor": "SERT",               "effect": "inhibition",  "organ_affinity": ["brain"]}],
    "Lorazepam":     [{"receptor": "GABA-A",             "effect": "potentiation","organ_affinity": ["brain"]}],
    "Levothyroxine": [{"receptor": "TR-α/β",             "effect": "agonism",     "organ_affinity": ["heart","brain","muscle"]}],
    "Prednisone":    [{"receptor": "GR (Glucocorticoid)","effect": "agonism",     "organ_affinity": ["immune_system","lungs"]}],
    "Acetaminophen": [{"receptor": "COX-3 / TRPV1",      "effect": "inhibition",  "organ_affinity": ["brain","liver"]}],
}


class BiologicalSystemSimulator:
    """Pure-Python simulation engine — no Streamlit dependencies."""

    def __init__(self):
        self.drug_database       = self._build_drug_database()
        self.disease_database    = self._build_disease_database()
        self.side_effects_db     = self._build_side_effects_database()
        self.drug_disease_map    = self._build_drug_disease_map()
        self.drug_incompatibilities = self._build_drug_incompatibilities()
        self.organ_drug_effects  = self._build_organ_drug_effects()

    # ------------------------------------------------------------------ dbs
    def _build_drug_database(self) -> Dict[str, DrugProperties]:
        return {
            "Acetaminophen": DrugProperties("Acetaminophen", 151.16,  1.0, 0.88,  2.5, 21.0,  42.0, 500.0,  "oral","qid",  0.10, (10.0,  50.0)),
            "Ibuprofen":     DrugProperties("Ibuprofen",     206.29,  3.5, 0.80,  2.0,  6.3,   9.8, 400.0,  "oral","tid",  0.99, (10.0,  40.0)),
            "Naproxen":      DrugProperties("Naproxen",      230.26,  3.2, 0.95, 17.0,  0.42,  9.0, 500.0,  "oral","bid",  0.99, (20.0,  80.0)),
            "Lisinopril":    DrugProperties("Lisinopril",    405.49, -0.5, 0.25, 12.0,  2.1, 120.0,  10.0,  "oral","daily",0.00, ( 5.0,  40.0)),
            "Metoprolol":    DrugProperties("Metoprolol",    267.37,  1.9, 0.50,  3.5, 14.7, 280.0,  50.0,  "oral","bid",  0.12, (30.0, 200.0)),
            "Atorvastatin":  DrugProperties("Atorvastatin",  558.64,  6.4, 0.14, 14.0, 12.6, 381.0,  20.0,  "oral","daily",0.98, (10.0,  80.0)),
            "Amlodipine":    DrugProperties("Amlodipine",    408.88,  3.0, 0.64, 35.0,  4.2, 840.0,   5.0,  "oral","daily",0.97, ( 5.0,  15.0)),
            "Metformin":     DrugProperties("Metformin",     129.16, -6.1, 0.55,  6.0, 35.0, 126.0, 850.0,  "oral","bid",  0.00, ( 0.5,   2.5)),
            "Glipizide":     DrugProperties("Glipizide",     445.54,  1.9, 0.90,  7.0,  1.4,  14.0,   5.0,  "oral","daily",0.98, (50.0, 150.0)),
            "Amoxicillin":   DrugProperties("Amoxicillin",   365.40, -2.5, 0.75,  1.5, 14.0,  21.0, 500.0,  "oral","tid",  0.18, ( 1.0,   8.0)),
            "Ciprofloxacin": DrugProperties("Ciprofloxacin", 331.35,  0.3, 0.70,  4.0, 37.8, 210.0, 500.0,  "oral","bid",  0.30, ( 0.5,   4.0)),
            "Sertraline":    DrugProperties("Sertraline",    306.23,  5.1, 0.44, 26.0,  1.4, 273.0,  50.0,  "oral","daily",0.98, (10.0, 150.0)),
            "Lorazepam":     DrugProperties("Lorazepam",     321.16,  2.4, 0.85, 14.0,  1.26,121.0,   1.0,  "oral","bid",  0.91, (10.0,  60.0)),
            "Levothyroxine": DrugProperties("Levothyroxine", 776.87,  3.5, 0.70,168.0,  0.105,10.5,  0.1,  "oral","daily",0.999,( 0.5,   2.0)),
            "Prednisone":    DrugProperties("Prednisone",    358.43,  1.5, 0.70,  3.5,  3.57, 35.0,  20.0,  "oral","daily",0.70, (10.0, 100.0)),
        }

    def _build_disease_database(self) -> Dict[str, DiseaseState]:
        return {
            "Hypertension":                DiseaseState("Hypertension",          0.55,0.004,["heart","kidneys","brain"],          {"systolic_bp":150.,"diastolic_bp":95.,"heart_rate":82.}),
            "Heart Failure (Mild)":        DiseaseState("Heart Failure (Mild)",  0.40,0.006,["heart","lungs","kidneys"],           {"ejection_fraction":50.,"bnp":250.,"heart_rate":90.}),
            "Heart Failure (Moderate)":    DiseaseState("Heart Failure (Moderate)",0.65,0.009,["heart","lungs","kidneys","liver"],  {"ejection_fraction":35.,"bnp":600.,"heart_rate":100.}),
            "Heart Failure (Severe)":      DiseaseState("Heart Failure (Severe)",0.85,0.015,["heart","lungs","kidneys","liver","brain"],{"ejection_fraction":20.,"bnp":1200.,"heart_rate":115.}),
            "Type 2 Diabetes (Mild)":      DiseaseState("Type 2 Diabetes (Mild)",0.35,0.005,["pancreas","liver"],                  {"fasting_glucose":130.,"hba1c":7.2,"insulin_resistance":1.5}),
            "Type 2 Diabetes (Moderate)":  DiseaseState("Type 2 Diabetes (Mod)",0.55,0.008,["pancreas","liver","kidneys"],         {"fasting_glucose":160.,"hba1c":8.5,"insulin_resistance":2.5}),
            "Type 2 Diabetes (Severe)":    DiseaseState("Type 2 Diabetes (Sev)",0.75,0.012,["pancreas","liver","kidneys","heart"], {"fasting_glucose":200.,"hba1c":10.5,"insulin_resistance":4.0}),
            "COPD (Moderate)":             DiseaseState("COPD (Moderate)",       0.55,0.007,["lungs","heart"],                     {"fev1_pct":58.,"o2_saturation":92.,"fvc":70.}),
            "COPD (Severe)":               DiseaseState("COPD (Severe)",         0.80,0.012,["lungs","heart","muscle"],            {"fev1_pct":33.,"o2_saturation":87.,"fvc":55.}),
            "Chronic Kidney Disease":      DiseaseState("Chronic Kidney Disease",0.50,0.006,["kidneys","heart","brain"],           {"gfr":40.,"creatinine":2.2,"urea":25.}),
            "Arthritis":                   DiseaseState("Arthritis",             0.45,0.003,["joints","immune_system"],            {"crp":8.,"esr":45.,"pain_score":6.}),
            "Depression":                  DiseaseState("Depression",            0.50,0.004,["brain"],                             {"cortisol":25.,"serotonin_index":0.5,"sleep_quality":0.4}),
            "Anxiety Disorder":            DiseaseState("Anxiety Disorder",      0.45,0.004,["brain","heart"],                    {"cortisol":28.,"heart_rate":90.,"gaba_index":0.6}),
            "Hypothyroidism":              DiseaseState("Hypothyroidism",        0.40,0.003,["heart","brain","muscle"],            {"tsh":8.,"t4_free":0.6,"metabolic_rate":0.75}),
            "Bacterial Infection":         DiseaseState("Bacterial Infection",   0.60,0.020,["immune_system","gut","liver"],       {"wbc":14.,"crp":50.,"temperature":38.8}),
        }

    def _build_side_effects_database(self) -> Dict[str, List[SideEffect]]:
        return {
            "Acetaminophen": [
                SideEffect("Liver Toxicity",     "high",  24.0, 48.0, 0.05, "Elevated liver enzymes; hepatotoxicity at supratherapeutic doses"),
                SideEffect("Nausea",             "low",    1.0,  2.0, 0.12, "Mild gastrointestinal discomfort"),
                SideEffect("Allergic Reaction",  "medium", 0.5,  4.0, 0.02, "Skin rash, urticaria, potential anaphylaxis"),
            ],
            "Ibuprofen": [
                SideEffect("GI Bleeding",        "high",  12.0, 24.0, 0.07, "Gastric ulcers and gastrointestinal bleeding"),
                SideEffect("Kidney Damage",      "medium",48.0, 72.0, 0.05, "Reduced renal perfusion; acute kidney injury risk"),
                SideEffect("Headache",           "low",    2.0,  4.0, 0.18, "Mild to moderate headache"),
            ],
            "Naproxen": [
                SideEffect("GI Irritation",      "medium", 4.0, 24.0, 0.12, "Abdominal pain, dyspepsia, peptic ulceration"),
                SideEffect("Fluid Retention",    "medium",24.0, 48.0, 0.08, "Oedema due to renal prostaglandin inhibition"),
                SideEffect("Kidney Damage",      "medium",72.0, 96.0, 0.04, "Impaired renal function with prolonged use"),
            ],
            "Lisinopril": [
                SideEffect("Angioedema",         "high",   2.0, 24.0, 0.01, "Swelling of face/lips/tongue — medical emergency"),
                SideEffect("Dry Cough",          "low",   24.0,168.0, 0.20, "Persistent bradykinin-mediated dry cough"),
                SideEffect("Hyperkalemia",       "medium",48.0, 72.0, 0.05, "Elevated potassium; risk in renal impairment"),
                SideEffect("Hypotension",        "medium", 1.0,  4.0, 0.08, "First-dose hypotension"),
            ],
            "Metoprolol": [
                SideEffect("Bradycardia",        "medium", 2.0,  8.0, 0.10, "Heart rate below 50 bpm"),
                SideEffect("Fatigue",            "low",   24.0, 72.0, 0.25, "Reduced exercise tolerance, lethargy"),
                SideEffect("Bronchospasm",       "high",   1.0,  6.0, 0.04, "Beta-2 mediated bronchoconstriction — caution in asthma/COPD"),
                SideEffect("Cold Extremities",   "low",   12.0, 48.0, 0.15, "Peripheral vasoconstriction"),
            ],
            "Atorvastatin": [
                SideEffect("Myopathy",           "medium",168.0,336.0,0.05, "Muscle pain, weakness; check CK if symptomatic"),
                SideEffect("Liver Enzyme Elev.", "medium", 48.0, 96.0, 0.03, "Elevated ALT/AST; monitor LFTs"),
                SideEffect("Rhabdomyolysis",     "high",  336.0,168.0,0.005,"Severe muscle breakdown; risk increases with interactions"),
            ],
            "Amlodipine": [
                SideEffect("Peripheral Oedema",  "low",   24.0, 72.0, 0.20, "Ankle swelling due to vasodilation"),
                SideEffect("Flushing",           "low",    1.0,  4.0, 0.12, "Facial flushing and warmth"),
                SideEffect("Reflex Tachycardia", "medium", 2.0,  6.0, 0.08, "Compensatory heart rate increase"),
            ],
            "Metformin": [
                SideEffect("Lactic Acidosis",    "high",  72.0, 48.0, 0.01, "Rare but serious; risk increased in renal impairment"),
                SideEffect("GI Distress",        "medium", 1.0, 24.0, 0.30, "Diarrhoea, nausea, abdominal discomfort — take with food"),
                SideEffect("Vitamin B12 Defic.", "low",   720.0,168.0, 0.10, "Long-term B12 malabsorption; monitor annually"),
            ],
            "Glipizide": [
                SideEffect("Hypoglycaemia",      "high",   2.0,  6.0, 0.12, "Blood glucose <3.5 mmol/L; dizziness, sweating, confusion"),
                SideEffect("Weight Gain",        "low",   168.0,336.0,0.20, "Insulin secretagogue effect promotes weight gain"),
                SideEffect("Nausea",             "low",    1.0,  4.0, 0.08, "Mild GI discomfort"),
            ],
            "Amoxicillin": [
                SideEffect("Allergic Reaction",  "high",   0.5,  6.0, 0.05, "Penicillin hypersensitivity; anaphylaxis risk"),
                SideEffect("Diarrhoea",          "low",    4.0, 24.0, 0.20, "Antibiotic-associated GI disruption"),
                SideEffect("C.diff Infection",   "high",  72.0,168.0, 0.03, "Dysbiosis leading to C. diff colitis"),
            ],
            "Ciprofloxacin": [
                SideEffect("Tendinopathy",       "medium", 48.0, 96.0, 0.04, "Tendon inflammation/rupture, especially Achilles"),
                SideEffect("CNS Effects",        "medium", 4.0, 12.0, 0.06, "Headache, dizziness, seizure risk in susceptible patients"),
                SideEffect("QT Prolongation",    "high",   4.0, 12.0, 0.03, "Cardiac arrhythmia risk"),
            ],
            "Sertraline": [
                SideEffect("Nausea",             "low",    4.0, 24.0, 0.25, "Common early-treatment GI symptom — usually resolves"),
                SideEffect("Insomnia",           "low",   12.0, 72.0, 0.15, "Sleep disturbance, vivid dreams"),
                SideEffect("Serotonin Syndrome", "high",   2.0, 12.0, 0.01, "Hyperthermia, agitation, clonus — risk with MAOIs"),
            ],
            "Lorazepam": [
                SideEffect("Sedation",           "medium", 1.0,  6.0, 0.40, "CNS depression; impairs driving/coordination"),
                SideEffect("Respiratory Depr.",  "high",   1.0,  6.0, 0.03, "Reduced respiratory drive; additive with opioids/alcohol"),
                SideEffect("Dependence",         "high",  336.0,672.0,0.15, "Physical and psychological dependence with prolonged use"),
            ],
            "Levothyroxine": [
                SideEffect("Palpitations",       "medium",24.0, 48.0, 0.08, "Supraphysiological dosing causes tachyarrhythmia"),
                SideEffect("Insomnia",           "low",   24.0, 72.0, 0.10, "CNS stimulation at high doses"),
                SideEffect("Chest Pain",         "high",  12.0, 24.0, 0.02, "Angina risk in ischaemic heart disease patients"),
            ],
            "Prednisone": [
                SideEffect("Hyperglycaemia",     "medium", 2.0, 24.0, 0.30, "Steroid-induced glucose elevation"),
                SideEffect("Immunosuppression",  "high",  24.0, 72.0, 0.50, "Increased infection risk"),
                SideEffect("GI Ulceration",      "medium",12.0, 48.0, 0.10, "Gastric irritation; take with food or PPI cover"),
                SideEffect("Osteoporosis",       "medium",720.0,336.0,0.20, "Long-term bone density loss"),
            ],
        }

    def _build_drug_disease_map(self) -> Dict[str, List[str]]:
        return {
            "Hypertension":              ["Lisinopril","Metoprolol","Amlodipine","Atorvastatin"],
            "Heart Failure (Mild)":      ["Lisinopril","Metoprolol","Atorvastatin"],
            "Heart Failure (Moderate)":  ["Lisinopril","Metoprolol","Atorvastatin"],
            "Heart Failure (Severe)":    ["Lisinopril","Metoprolol"],
            "Type 2 Diabetes (Mild)":    ["Metformin","Glipizide"],
            "Type 2 Diabetes (Moderate)":["Metformin","Glipizide"],
            "Type 2 Diabetes (Severe)":  ["Metformin","Glipizide"],
            "COPD (Moderate)":           ["Prednisone"],
            "COPD (Severe)":             ["Prednisone"],
            "Chronic Kidney Disease":    ["Lisinopril","Amlodipine"],
            "Arthritis":                 ["Ibuprofen","Naproxen","Prednisone","Acetaminophen"],
            "Depression":                ["Sertraline"],
            "Anxiety Disorder":          ["Sertraline","Lorazepam"],
            "Hypothyroidism":            ["Levothyroxine"],
            "Bacterial Infection":       ["Amoxicillin","Ciprofloxacin"],
        }

    def _build_drug_incompatibilities(self) -> Dict[tuple, str]:
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

    def _build_organ_drug_effects(self) -> Dict[str, Dict[str, float]]:
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

    # ----------------------------------------------------------------- math
    def calculate_concentration(self, drug: DrugProperties, t: float,
                                 bio: BiologicalParameters) -> float:
        cl = drug.clearance * bio.kidney_function * bio.liver_function * bio.metabolic_rate
        vd = drug.volume_distribution * (bio.weight / 70.0)
        tau = {"daily": 24.0, "bid": 12.0, "tid": 8.0, "qid": 6.0}.get(drug.dosing_frequency, 24.0)
        k_el = cl / max(vd, 1e-6)
        doses = int(t / tau) + 1
        conc = 0.0
        for i in range(doses):
            t_since = t - i * tau
            if t_since >= 0:
                c0 = (drug.dose_mg * drug.bioavailability) / max(vd, 1e-6)
                conc += c0 * math.exp(-k_el * t_since)
        return float(max(0.0, conc))

    def get_pk_metrics(self, drug: DrugProperties, bio: BiologicalParameters,
                       duration: float) -> Dict[str, float]:
        cl = drug.clearance * bio.kidney_function * bio.liver_function * bio.metabolic_rate
        vd = drug.volume_distribution * (bio.weight / 70.0)
        k_el = cl / max(vd, 1e-6)
        t_half = 0.693 / max(k_el, 1e-9)
        # Evaluate on fine grid to find Cmax / Tmax
        ts = np.arange(0, duration, 0.25)
        concs = np.array([self.calculate_concentration(drug, float(t_), bio) for t_ in ts])
        idx_max = int(np.argmax(concs))
        cmax = float(concs[idx_max])
        tmax = float(ts[idx_max])
        auc  = float(np.trapezoid(concs, ts) if hasattr(np, 'trapezoid') else np.trapz(concs, ts))
        return {"cmax": round(cmax, 3), "tmax": round(tmax, 2),
                "auc": round(auc, 1), "half_life": round(t_half, 2)}

    def simulate_disease_progression(self, disease: DiseaseState, t: float,
                                      drug_effects: Dict[str, float]) -> DiseaseState:
        sev = disease.severity + disease.progression_rate * t
        for effect in drug_effects.values():
            if effect > 0:
                sev *= (1 - effect * 0.3)
        sev = float(np.clip(sev, 0.0, 1.0))
        bm = {}
        for key, base in disease.biomarkers.items():
            bm[key] = round(base * (1 + sev * 0.2), 2)
        from dataclasses import replace
        return DiseaseState(disease.name, sev, disease.progression_rate,
                            disease.affected_organs[:], bm)

    def check_side_effects(self, drug_name: str, t: float,
                            concentration: float = 0.0) -> List[Dict]:
        effects = self.side_effects_db.get(drug_name, [])
        drug = self.drug_database.get(drug_name)
        tox_threshold = (drug.therapeutic_window[1] * 1.5) if drug else 100.0
        if tox_threshold > 0 and concentration > 0:
            z = 5.0 * (concentration - tox_threshold) / tox_threshold
            cmult = 1.0 / (1.0 + math.exp(-z))
        else:
            cmult = 0.5
        result = []
        for e in effects:
            if t >= e.onset_time:
                eff_prob = e.probability * max(cmult, 0.01)
                active = random.random() < eff_prob
                if active:
                    result.append({
                        "name": e.name, "severity": e.severity,
                        "onset_time": e.onset_time, "duration": e.duration,
                        "probability": round(eff_prob, 3), "description": e.description,
                        "active": True, "start_time": t, "end_time": t + e.duration,
                    })
        return result

    def get_system_status(self, bio: BiologicalParameters,
                          drug_concentrations: Dict[str, float],
                          disease_states: List[DiseaseState],
                          side_effects: List[Dict]) -> Dict:
        ALL_ORGANS = ["brain","heart","liver","kidneys","lungs","muscle",
                      "adipose","gut","skin","pancreas","immune_system","joints"]
        organ_health: Dict[str, float] = {}
        organ_drug_effects: Dict[str, Dict[str, float]] = {o: {} for o in ALL_ORGANS}

        for org in ALL_ORGANS:
            h = 1.0
            if org == "kidneys": h *= bio.kidney_function
            elif org == "liver":  h *= bio.liver_function
            for ds in disease_states:
                if org in ds.affected_organs:
                    h -= ds.severity * 0.30
            for dname, conc in drug_concentrations.items():
                if dname in self.organ_drug_effects and org in self.organ_drug_effects[dname]:
                    tw_hi = self.drug_database[dname].therapeutic_window[1] if dname in self.drug_database else 50.0
                    delta = self.organ_drug_effects[dname][org] * (conc / (tw_hi + 1e-10))
                    h += delta
                    organ_drug_effects[org][dname] = round(delta, 4)
            organ_health[org] = round(float(np.clip(h, 0.0, 1.2)), 3)

        warnings: List[str] = []
        recommendations: List[str] = []
        drugs = list(drug_concentrations.keys())
        for i, d1 in enumerate(drugs):
            for d2 in drugs[i+1:]:
                key = (d1, d2) if (d1, d2) in self.drug_incompatibilities else (d2, d1)
                if key in self.drug_incompatibilities:
                    warnings.append(f"INTERACTION: {d1} + {d2} — {self.drug_incompatibilities[key]}")
        for eff in side_effects:
            if eff["severity"] == "high":
                warnings.append(f"ALERT: {eff['name']} — {eff['description']}")
        for org, h in organ_health.items():
            label = org.replace("_", " ").title()
            if h < 0.60:
                recommendations.append(f"Monitor {label} closely (health: {h:.0%})")
            elif h < 0.75:
                recommendations.append(f"Watch {label} function (health: {h:.0%})")
        if bio.kidney_function < 0.5:
            for d in ["Metformin","Ciprofloxacin","Amoxicillin"]:
                if d in drug_concentrations:
                    recommendations.append(f"Reduce {d} dose — GFR severely reduced")
        if bio.liver_function < 0.5:
            for d in ["Acetaminophen","Atorvastatin","Sertraline"]:
                if d in drug_concentrations:
                    recommendations.append(f"Caution with {d} — hepatic impairment")

        return {
            "overall_health": round(float(np.mean(list(organ_health.values()))), 3),
            "organ_health": organ_health,
            "organ_drug_effects": organ_drug_effects,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    def compute_cell_data(self, organ: str, drug_concentrations: Dict[str, float],
                          organ_health: float, disease: Optional[str]) -> Dict:
        """Compute cell-level physiological data for the cell viewer."""
        base_health = float(np.clip(organ_health, 0.0, 1.0))
        # Membrane integrity: degraded by harmful drugs, supported by health
        harmful_delta = sum(
            self.organ_drug_effects.get(d, {}).get(organ, 0.0)
            for d in drug_concentrations if self.organ_drug_effects.get(d, {}).get(organ, 0.0) < 0
        )
        membrane_integrity = float(np.clip(base_health - abs(harmful_delta) * 0.5, 0.1, 1.0))
        atp_level = float(np.clip(base_health * 0.95 + (1 - base_health) * 0.3 + random.gauss(0, 0.02), 0.1, 1.0))
        mitochondrial_activity = float(np.clip(atp_level * 0.9 + random.gauss(0, 0.02), 0.05, 1.0))
        inflammatory_markers = float(np.clip(1.0 - base_health + random.gauss(0, 0.03), 0.0, 1.0))
        apoptosis_risk = float(np.clip((1.0 - base_health) ** 1.5 + abs(harmful_delta) * 0.3, 0.0, 1.0))
        recovery_rate = float(np.clip(base_health * 0.02 - (1.0 - base_health) * 0.015, -0.03, 0.03))
        # Receptor occupancy per drug
        drug_binding_events = []
        total_occupancy = 0.0
        for drug_name, conc in drug_concentrations.items():
            receptors = DRUG_RECEPTORS.get(drug_name, [])
            for rec in receptors:
                if organ in rec["organ_affinity"] or not rec["organ_affinity"]:
                    tw_hi = self.drug_database[drug_name].therapeutic_window[1] if drug_name in self.drug_database else 50.0
                    occ = float(np.clip(conc / (tw_hi + 1e-10), 0.0, 1.0))
                    total_occupancy += occ
                    drug_binding_events.append({
                        "receptor": rec["receptor"],
                        "drug": drug_name,
                        "occupancy": round(occ, 3),
                        "effect": rec["effect"],
                        "drug_conc": round(conc, 3),
                    })
        receptor_occupancy = float(np.clip(total_occupancy / max(len(drug_binding_events), 1), 0.0, 1.0))
        cellular_drug_conc = sum(drug_concentrations.values()) * (1.0 - 0.4 * (1.0 - membrane_integrity))
        return {
            "organ": organ,
            "health": round(base_health, 3),
            "cell_data": {
                "membrane_integrity":        round(membrane_integrity, 3),
                "atp_level":                 round(atp_level, 3),
                "receptor_occupancy":        round(receptor_occupancy, 3),
                "drug_concentration_cellular": round(cellular_drug_conc, 3),
                "mitochondrial_activity":    round(mitochondrial_activity, 3),
                "inflammatory_markers":      round(inflammatory_markers, 3),
                "apoptosis_risk":            round(apoptosis_risk, 3),
                "recovery_rate":             round(recovery_rate, 4),
            },
            "drug_effects": {d: round(self.organ_drug_effects.get(d, {}).get(organ, 0.0), 4)
                             for d in drug_concentrations},
            "drug_binding_events": drug_binding_events,
        }


# ===========================================================================
# ── FastAPI Application ───────────────────────────────────────────────────
# ===========================================================================
app = FastAPI(title="PK-PD Interactive Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR    = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR  = os.path.join(BASE_DIR, "templates")
os.makedirs(STATIC_DIR,   exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Shared simulator instance
_simulator = BiologicalSystemSimulator()


# ──────────────────────────────────────────────── Pydantic request models ──
class SimulateRequest(BaseModel):
    drug: str = "Lisinopril"
    dose: float = 10.0
    drug2: Optional[str] = None
    dose2: Optional[float] = None
    disease: str = "Hypertension"
    severity: float = 0.55
    age: float = 65.0
    weight: float = 70.0
    sex: int = 1
    kidney_function: float = 0.8
    liver_function: float = 0.9
    heart_rate: float = 72.0
    metabolic_rate: float = 1.0
    duration_hours: float = 72.0

class CellRequest(BaseModel):
    organ: str
    drug: str
    dose: float = 10.0
    drug2: Optional[str] = None
    dose2: Optional[float] = None
    organ_health: float = 0.85
    disease: Optional[str] = None
    age: float = 65.0
    weight: float = 70.0
    kidney_function: float = 0.8
    liver_function: float = 0.9


# ───────────────────────────────────────────────────────────── routes ──────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    html_path = os.path.join(TEMPLATE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/drugs")
async def get_drugs():
    out = []
    for name, d in _simulator.drug_database.items():
        out.append({
            "name": name,
            "molecular_weight": d.molecular_weight,
            "bioavailability": d.bioavailability,
            "half_life": d.half_life,
            "protein_binding": d.protein_binding,
            "administration_route": d.administration_route,
            "dosing_frequency": d.dosing_frequency,
            "default_dose": d.dose_mg,
            "therapeutic_window": list(d.therapeutic_window),
            "drug_class": _infer_class(name),
            "indicated_for": [dis for dis, drugs in _simulator.drug_disease_map.items() if name in drugs],
        })
    return out


@app.get("/api/diseases")
async def get_diseases():
    out = []
    for name, ds in _simulator.disease_database.items():
        out.append({
            "name": name,
            "default_severity": ds.severity,
            "progression_rate": ds.progression_rate,
            "affected_organs": ds.affected_organs,
            "biomarkers": ds.biomarkers,
            "indicated_drugs": _simulator.drug_disease_map.get(name, []),
        })
    return out


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    sim = _simulator
    bio = BiologicalParameters(
        age=req.age, weight=req.weight, sex=req.sex,
        kidney_function=req.kidney_function, liver_function=req.liver_function,
        heart_rate=req.heart_rate, metabolic_rate=req.metabolic_rate,
    )
    # Build drug list
    drugs: List[Tuple[str, float]] = [(req.drug, req.dose)]
    if req.drug2 and req.drug2 != "None" and req.drug2 != "":
        drugs.append((req.drug2, req.dose2 or sim.drug_database.get(req.drug2, DrugProperties()).dose_mg))

    drug_props: Dict[str, DrugProperties] = {}
    for dname, dose in drugs:
        if dname in sim.drug_database:
            import copy
            dp = copy.deepcopy(sim.drug_database[dname])
            dp.dose_mg = dose
            drug_props[dname] = dp

    disease_name = req.disease
    disease_base = sim.disease_database.get(disease_name)
    if disease_base:
        import copy
        disease_obj = copy.deepcopy(disease_base)
        disease_obj.severity = req.severity
    else:
        disease_obj = DiseaseState(name=disease_name, severity=req.severity)

    # Time step: 1h for ≤96h, 2h for longer
    step = 1.0 if req.duration_hours <= 96 else 2.0
    time_points = list(np.arange(0, req.duration_hours + step / 2, step))

    concentrations: Dict[str, List[float]] = {d: [] for d in drug_props}
    disease_severity: List[float] = []
    organ_health_series: List[Dict] = []
    all_side_effects: List[Dict] = []
    biomarker_series: Dict[str, List[float]] = {k: [] for k in disease_obj.biomarkers}

    for t in time_points:
        drug_concs_t: Dict[str, float] = {}
        drug_effects_t: Dict[str, float] = {}
        for dname, dp in drug_props.items():
            c = sim.calculate_concentration(dp, t, bio)
            concentrations[dname].append(round(c, 4))
            drug_concs_t[dname] = c
            drug_effects_t[dname] = c / (dp.therapeutic_window[1] + 1e-10)

        # Disease progression
        prog = sim.simulate_disease_progression(disease_obj, t, drug_effects_t)
        disease_severity.append(round(prog.severity, 4))
        for bk, bv in prog.biomarkers.items():
            if bk in biomarker_series:
                biomarker_series[bk].append(round(bv, 3))

        # Side effects (sampled once at each time point with fixed seed for reproducibility)
        for dname in drug_props:
            conc = drug_concs_t[dname]
            effs = sim.check_side_effects(dname, t, conc)
            all_side_effects.extend(effs)

        # Organ health
        status = sim.get_system_status(bio, drug_concs_t, [prog], [])
        organ_health_series.append(status["organ_health"])

    # Final status with full side effects
    final_concs = {d: concentrations[d][-1] for d in drug_props}
    final_disease = sim.simulate_disease_progression(disease_obj, time_points[-1], {d: final_concs[d] / (drug_props[d].therapeutic_window[1] + 1e-10) for d in drug_props})
    final_status = sim.get_system_status(bio, final_concs, [final_disease], all_side_effects)

    # PK metrics
    pk_metrics = {}
    for dname, dp in drug_props.items():
        pk_metrics[dname] = sim.get_pk_metrics(dp, bio, req.duration_hours)
        pk_metrics[dname]["therapeutic_window"] = list(dp.therapeutic_window)

    # Deduplicate side effects (keep unique names per drug)
    seen = set()
    unique_effects = []
    for e in all_side_effects:
        key = e["name"]
        if key not in seen:
            seen.add(key)
            unique_effects.append(e)

    # Therapeutic status
    primary_drug = list(drug_props.values())[0]
    last_conc = concentrations[req.drug][-1]
    tw_lo, tw_hi = primary_drug.therapeutic_window
    if last_conc < tw_lo:
        therapeutic_status = "sub-therapeutic"
    elif last_conc > tw_hi * 1.5:
        therapeutic_status = "toxic"
    elif last_conc > tw_hi:
        therapeutic_status = "supra-therapeutic"
    else:
        therapeutic_status = "therapeutic"

    return {
        "time_points": [round(t, 2) for t in time_points],
        "concentrations": concentrations,
        "therapeutic_windows": {d: list(drug_props[d].therapeutic_window) for d in drug_props},
        "disease_severity": disease_severity,
        "disease_name": disease_name,
        "organ_health_series": organ_health_series,
        "organ_health_final": final_status["organ_health"],
        "organ_drug_effects": final_status["organ_drug_effects"],
        "side_effects": unique_effects,
        "warnings": final_status["warnings"],
        "recommendations": final_status["recommendations"],
        "pk_metrics": pk_metrics,
        "biomarkers": biomarker_series,
        "therapeutic_status": therapeutic_status,
        "overall_health": final_status["overall_health"],
        "drugs": list(drug_props.keys()),
        "heart_rate": bio.heart_rate,
    }


@app.post("/api/cell")
async def cell_detail(req: CellRequest):
    sim = _simulator
    bio = BiologicalParameters(
        age=req.age, weight=req.weight,
        kidney_function=req.kidney_function, liver_function=req.liver_function,
    )
    drug_concs: Dict[str, float] = {}
    for dname, dose in [(req.drug, req.dose), (req.drug2, req.dose2)]:
        if dname and dname not in ("None", ""):
            dp_base = sim.drug_database.get(dname)
            if dp_base:
                import copy
                dp = copy.deepcopy(dp_base)
                dp.dose_mg = dose or dp_base.dose_mg
                c = sim.calculate_concentration(dp, dp.half_life * 2, bio)
                drug_concs[dname] = round(c, 4)

    cell_data = sim.compute_cell_data(req.organ, drug_concs, req.organ_health, req.disease)
    return cell_data


@app.post("/api/interactions")
async def check_interactions(drugs: List[str]):
    results = []
    sim = _simulator
    for i, d1 in enumerate(drugs):
        for d2 in drugs[i+1:]:
            key = (d1, d2) if (d1, d2) in sim.drug_incompatibilities else (d2, d1)
            if key in sim.drug_incompatibilities:
                results.append({"drug1": d1, "drug2": d2,
                                 "message": sim.drug_incompatibilities[key], "severity": "high"})
    return results


# ─────────────────────────────────────────────────────── helpers ───────────
def _infer_class(name: str) -> str:
    mapping = {
        "Acetaminophen": "Analgesic", "Ibuprofen": "NSAID", "Naproxen": "NSAID",
        "Lisinopril": "ACE Inhibitor", "Metoprolol": "Beta-Blocker",
        "Atorvastatin": "Statin", "Amlodipine": "CCB",
        "Metformin": "Biguanide", "Glipizide": "Sulfonylurea",
        "Amoxicillin": "Penicillin Antibiotic", "Ciprofloxacin": "Fluoroquinolone",
        "Sertraline": "SSRI", "Lorazepam": "Benzodiazepine",
        "Levothyroxine": "Thyroid Hormone", "Prednisone": "Corticosteroid",
    }
    return mapping.get(name, "Unknown")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
