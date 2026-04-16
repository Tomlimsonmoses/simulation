"""
PK-PD Interactive Dashboard — FastAPI Backend
==============================================
Serves the HTML dashboard and exposes simulation REST API endpoints.
Run with:
    uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import copy
import logging
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
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

# DataLoader — all seed data lives in data/raw/; files are created on first run
from data.data_loader import get_data_loader  # noqa: E402

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


# Drug → receptor binding table is loaded from data/raw/drug_receptors.csv
# (accessed via BiologicalSystemSimulator._drug_receptors)


class BiologicalSystemSimulator:
    """Pure-Python simulation engine — no Streamlit dependencies."""

    def __init__(self):
        _dl = get_data_loader()
        self.drug_database          = self._build_drug_database(_dl.load_drugs())
        self.disease_database       = self._build_disease_database(_dl.load_diseases())
        self.side_effects_db        = self._build_side_effects_database(_dl.load_side_effects())
        self.drug_disease_map       = self._build_drug_disease_map(_dl.load_drug_disease_map())
        self.drug_incompatibilities = self._build_drug_incompatibilities(_dl.load_drug_interactions())
        self.organ_drug_effects     = self._build_organ_drug_effects(_dl.load_drug_organ_effects())
        self._drug_receptors        = self._build_drug_receptors(_dl.load_drug_receptors())

    # ------------------------------------------------------------------ dbs
    def _build_drug_database(self, rows: List[Dict]) -> Dict[str, DrugProperties]:
        result: Dict[str, DrugProperties] = {}
        for r in rows:
            result[r["drug_name"]] = DrugProperties(
                name=r["drug_name"],
                molecular_weight=float(r["molecular_weight"]),
                logp=float(r["logp"]),
                bioavailability=float(r["bioavailability"]),
                half_life=float(r["half_life"]),
                clearance=float(r["clearance"]),
                volume_distribution=float(r["volume_distribution"]),
                dose_mg=float(r["dose_mg"]),
                administration_route=r["administration_route"],
                dosing_frequency=r["dosing_frequency"],
                protein_binding=float(r["protein_binding"]),
                therapeutic_window=(
                    float(r["therapeutic_window_lo"]),
                    float(r["therapeutic_window_hi"]),
                ),
            )
        return result

    def _build_disease_database(self, disease_dict: Dict) -> Dict[str, DiseaseState]:
        result: Dict[str, DiseaseState] = {}
        for name, d in disease_dict.items():
            result[name] = DiseaseState(
                name=name,
                severity=float(d["severity"]),
                progression_rate=float(d["progression_rate"]),
                affected_organs=list(d["affected_organs"]),
                biomarkers={k: float(v) for k, v in d["biomarkers"].items()},
            )
        return result

    def _build_side_effects_database(self, rows: List[Dict]) -> Dict[str, List[SideEffect]]:
        result: Dict[str, List[SideEffect]] = {}
        for r in rows:
            se = SideEffect(
                r["effect_name"], r["severity"],
                float(r["onset_hours"]), float(r["duration_hours"]),
                float(r["probability"]), r["description"],
            )
            result.setdefault(r["drug_name"], []).append(se)
        return result

    def _build_drug_disease_map(self, rows: List[Dict]) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for r in rows:
            result.setdefault(r["disease_name"], []).append(r["drug_name"])
        return result

    def _build_drug_incompatibilities(self, rows: List[Dict]) -> Dict[tuple, str]:
        return {(r["drug1"], r["drug2"]): r["description"] for r in rows}

    def _build_organ_drug_effects(self, rows: List[Dict]) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for r in rows:
            result.setdefault(r["drug_name"], {})[r["organ"]] = float(r["effect_magnitude"])
        return result

    def _build_drug_receptors(self, rows: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            entry = {
                "receptor": r["receptor"],
                "effect": r["effect_type"],
                "organ_affinity": [o.strip() for o in r["organ_affinity"].split("|")],
            }
            result.setdefault(r["drug_name"], []).append(entry)
        return result

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
            receptors = self._drug_receptors.get(drug_name, [])
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

# Mechanistic engine — initialised at startup; falls back to analytical PK if unavailable
_engine = None
_logger = logging.getLogger(__name__)


@app.on_event("startup")
async def _startup():
    global _engine
    try:
        from simulation.mechanistic_engine import MechanisticPKPDEngine
        _engine = MechanisticPKPDEngine()
        _logger.info("MechanisticPKPDEngine loaded successfully")
    except Exception as exc:
        _logger.warning(
            "MechanisticPKPDEngine not available — using analytical fallback. Reason: %s", exc
        )


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

class LongTermSimulateRequest(BaseModel):
    drug: str = "Atorvastatin"
    dose: float = 20.0
    drug2: Optional[str] = None
    dose2: Optional[float] = None
    disease: str = "Hypertension"
    severity: float = 0.55
    age: float = 65.0
    weight: float = 70.0
    sex: int = 1
    kidney_function: float = 0.8
    liver_function: float = 0.9
    duration_days: float = 180.0


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

    # ── Simulation —————————————————————————————————————————————————————
    time_arr = np.array(time_points, dtype=float)
    pk_metrics: Dict[str, Dict] = {}
    final_prog = disease_obj

    # Engine-name → viz-name mapping for organ concentration series
    _ENG_TO_VIZ = {
        "kidney": "kidneys", "lung": "lungs", "fat": "adipose",
        "brain": "brain", "heart": "heart", "liver": "liver",
        "muscle": "muscle", "skin": "skin", "gut": "gut",
        "spleen": "immune_system",
    }

    if _engine is not None:
        # ── Mechanistic PBPK/PD path — full biological intelligence ──────
        _eng_results: Dict[str, Any] = {}

        # Phase 4 — CYP competitive inhibition: adjust clearance before engine
        _CYP_INHIBITION = {
            ("Erythromycin",  "Atorvastatin"): 0.45,
            ("Erythromycin",  "Amlodipine"):   0.35,
            ("Erythromycin",  "Lorazepam"):    0.30,
            ("Ciprofloxacin", "Metformin"):    0.40,
            ("Ciprofloxacin", "Glipizide"):    0.20,
            ("Prednisone",    "Metformin"):    0.15,
            ("Ibuprofen",     "Metformin"):    0.25,
            ("Naproxen",      "Metformin"):    0.20,
            ("Sertraline",    "Metoprolol"):   0.60,
            ("Amoxicillin",   "Glipizide"):    0.15,
        }
        _cyp_warnings: List[str] = []
        drug_name_list = list(drug_props.keys())
        for i_d, d1 in enumerate(drug_name_list):
            for d2 in drug_name_list[i_d+1:]:
                for (inh, sub), frac in _CYP_INHIBITION.items():
                    if d1 == inh and d2 == sub:
                        drug_props[d2] = copy.deepcopy(drug_props[d2])
                        drug_props[d2].clearance *= (1.0 - frac)
                        _cyp_warnings.append(
                            f"CYP interaction: {d1} inhibits metabolism of {d2} — "
                            f"clearance reduced ~{int(frac*100)}%, expect elevated exposure & prolonged t\u00bd"
                        )
                    elif d2 == inh and d1 == sub:
                        drug_props[d1] = copy.deepcopy(drug_props[d1])
                        drug_props[d1].clearance *= (1.0 - frac)
                        _cyp_warnings.append(
                            f"CYP interaction: {d2} inhibits metabolism of {d1} — "
                            f"clearance reduced ~{int(frac*100)}%, expect elevated exposure & prolonged t\u00bd"
                        )

        for dname, dp in drug_props.items():
            result = _engine.simulate_drug(dp, bio, [disease_obj], time_arr)
            _eng_results[dname] = result
            concentrations[dname] = [round(float(c), 4) for c in result.plasma_conc]
            if not disease_severity:
                disease_severity = [round(float(s), 4) for s in result.disease_severity]
            pk_metrics[dname] = {
                "cmax":      result.pk_stats.get("Cmax (mg/L)", 0.0),
                "tmax":      result.pk_stats.get("Tmax (h)", 0.0),
                "auc":       result.pk_stats.get("AUClast (mg\u00b7h/L)", 0.0),
                "half_life": result.pk_stats.get("t\u00bd (h)", dp.half_life),
                "css":       result.pk_stats.get("Css_avg (mg/L)", 0.0),
                "cl_eff":    result.pk_stats.get("CL_eff (L/h)", dp.clearance),
                "vd_eff":    result.pk_stats.get("Vd_eff (L)", dp.volume_distribution),
                "therapeutic_window": list(dp.therapeutic_window),
                "mechanistic": True,
            }

            # Phase 1 — Emergent side effects from concentration dynamics ──
            tw_hi = dp.therapeutic_window[1]
            conc_arr = result.plasma_conc

            # Band 1: scan plasma for threshold crossings
            _BANDS = [(3.0, "high"), (2.0, "high"), (1.5, "medium")]
            _emitted_plasma = False
            for band_mult, band_sev in _BANDS:
                threshold = tw_hi * band_mult
                above = conc_arr > threshold
                if not np.any(above):
                    continue
                crossings = np.where(above)[0]
                onset_idx   = int(crossings[0])
                offset_idx  = int(crossings[-1])
                onset_h     = float(time_arr[onset_idx])
                dt          = float(time_arr[1] - time_arr[0]) if len(time_arr) > 1 else 1.0
                duration_h  = float(time_arr[offset_idx]) - onset_h + dt
                peak_ratio  = float(np.max(conc_arr[crossings]) / (tw_hi + 1e-10))

                # Identify most-exposed organ at crossing
                _org_tox_names = {
                    "liver": "Hepatotoxicity", "kidney": "Nephrotoxicity",
                    "brain": "Neurotoxicity",  "heart": "Cardiotoxicity",
                    "gut":   "GI Toxicity",    "muscle": "Myotoxicity",
                    "lung":  "Pulmonary Toxicity",
                }
                toxic_organ_name = "Supratherapeutic Exposure"
                if result.organ_conc:
                    max_ratio = 0.0
                    for eng_org, tox_nm in _org_tox_names.items():
                        if eng_org not in result.organ_conc:
                            continue
                        org_arr = result.organ_conc[eng_org]
                        ref = float(org_arr[onset_idx])
                        plasma_ref = float(conc_arr[onset_idx]) + 1e-10
                        r = (ref / plasma_ref) * band_mult
                        if r > max_ratio:
                            max_ratio = r
                            toxic_organ_name = tox_nm

                all_side_effects.append({
                    "name":       toxic_organ_name,
                    "severity":   band_sev,
                    "onset_time": round(onset_h, 2),
                    "duration":   round(duration_h, 2),
                    "probability": 1.0,
                    "description": (
                        f"Plasma concentration reached {peak_ratio:.1f}\u00d7 the therapeutic maximum "
                        f"at t={onset_h:.1f}h. Duration above threshold: {duration_h:.1f}h."
                    ),
                    "active":     True,
                    "start_time": round(onset_h, 2),
                    "end_time":   round(onset_h + duration_h, 2),
                    "emergent":   True,
                })
                _emitted_plasma = True
                break  # only emit the most severe band

            # Band 2: organ-level concentration toxicity
            if result.organ_conc:
                _org_thresholds = {
                    "liver":  tw_hi * 4.0,
                    "kidney": tw_hi * 3.5,
                    "brain":  tw_hi * 2.5,
                    "heart":  tw_hi * 3.0,
                    "gut":    tw_hi * 5.0,
                }
                _org_tox_labels = {
                    "liver": "Hepatotoxicity", "kidney": "Nephrotoxicity",
                    "brain": "Neurotoxicity",  "heart": "Cardiotoxicity",
                    "gut": "GI Mucosal Injury",
                }
                for eng_org, org_thresh in _org_thresholds.items():
                    if eng_org not in result.organ_conc:
                        continue
                    org_arr2 = result.organ_conc[eng_org]
                    above2 = org_arr2 > org_thresh
                    if not np.any(above2):
                        continue
                    c2 = np.where(above2)[0]
                    onset_h2 = float(time_arr[int(c2[0])])
                    dt2 = float(time_arr[1] - time_arr[0]) if len(time_arr) > 1 else 1.0
                    dur_h2 = float(time_arr[int(c2[-1])]) - onset_h2 + dt2
                    peak2   = float(np.max(org_arr2[c2]))
                    tox_lbl = _org_tox_labels[eng_org]
                    already = any(e["name"] == tox_lbl and e.get("emergent") for e in all_side_effects)
                    if not already:
                        all_side_effects.append({
                            "name":       tox_lbl,
                            "severity":   "high" if peak2 > org_thresh * 1.5 else "medium",
                            "onset_time": round(onset_h2, 2),
                            "duration":   round(dur_h2, 2),
                            "probability": 1.0,
                            "description": (
                                f"{eng_org.title()} concentration ({peak2:.2f} mg/L) exceeded "
                                f"safety threshold at t={onset_h2:.1f}h."
                            ),
                            "active":     True,
                            "start_time": round(onset_h2, 2),
                            "end_time":   round(onset_h2 + dur_h2, 2),
                            "emergent":   True,
                        })

            # Band 3: static-DB probabilistic effects not covered by emergent detection
            static_effs = sim.check_side_effects(dname, req.duration_hours, pk_metrics[dname]["cmax"])
            emergent_names = {e["name"] for e in all_side_effects if e.get("emergent")}
            for eff in static_effs:
                if eff["name"] not in emergent_names:
                    all_side_effects.append(eff)

        # Phase 2 — Organ health time-series (thinned to ≤73 points) ──────
        _primary_key = req.drug if req.drug in _eng_results else next(iter(_eng_results))
        _primary_result = _eng_results[_primary_key]
        _thin = max(1, len(time_arr) // 73)
        for t_idx in range(0, len(time_arr), _thin):
            drug_concs_t: Dict[str, float] = {
                dname: float(concentrations[dname][t_idx]) for dname in drug_props
            }
            sev_t = float(disease_severity[t_idx]) if t_idx < len(disease_severity) else req.severity
            dis_t = copy.deepcopy(disease_obj)
            dis_t.severity = sev_t
            status_t = sim.get_system_status(bio, drug_concs_t, [dis_t], [])
            organ_health_series.append(status_t["organ_health"])

        # Receptor occupancy + PD response (thinned, primary drug)
        _receptor_occupancy = [
            round(float(_primary_result.receptor_occupancy[i]), 4)
            for i in range(0, len(time_arr), _thin)
        ]
        _pd_response = [
            round(float(_primary_result.pd_response[i]), 4)
            for i in range(0, len(time_arr), _thin)
        ]
        _series_time = [
            round(float(time_arr[i]), 2)
            for i in range(0, len(time_arr), _thin)
        ]

        # Biomarker series
        sev_final = disease_severity[-1] if disease_severity else req.severity
        for bk, bv in disease_obj.biomarkers.items():
            if bk in biomarker_series:
                bv_final = bv * (1.0 + sev_final * 0.2)
                biomarker_series[bk] = [
                    round(bv + (bv_final - bv) * i / max(len(time_points) - 1, 1), 3)
                    for i in range(len(time_points))
                ]
        disease_severity = disease_severity or [req.severity] * len(time_points)

        final_concs = {d: concentrations[d][-1] for d in drug_props}
        final_eff = {d: final_concs[d] / (drug_props[d].therapeutic_window[1] + 1e-10)
                     for d in drug_props}
        final_prog = sim.simulate_disease_progression(disease_obj, req.duration_hours, final_eff)
    else:
        # ── Fallback: analytical single-compartment PK ────────────────────
        for t in time_points:
            drug_concs_t: Dict[str, float] = {}
            drug_effects_t: Dict[str, float] = {}
            for dname, dp in drug_props.items():
                c = sim.calculate_concentration(dp, t, bio)
                concentrations[dname].append(round(c, 4))
                drug_concs_t[dname] = c
                drug_effects_t[dname] = c / (dp.therapeutic_window[1] + 1e-10)
            prog = sim.simulate_disease_progression(disease_obj, t, drug_effects_t)
            disease_severity.append(round(prog.severity, 4))
            for bk, bv in prog.biomarkers.items():
                if bk in biomarker_series:
                    biomarker_series[bk].append(round(bv, 3))
            for dname in drug_props:
                all_side_effects.extend(sim.check_side_effects(dname, t, drug_concs_t[dname]))
            organ_health_series.append(
                sim.get_system_status(bio, drug_concs_t, [prog], [])["organ_health"]
            )
        final_concs = {d: concentrations[d][-1] for d in drug_props}
        final_prog = sim.simulate_disease_progression(
            disease_obj, time_points[-1],
            {d: final_concs[d] / (drug_props[d].therapeutic_window[1] + 1e-10) for d in drug_props},
        )
        for dname, dp in drug_props.items():
            pk_metrics[dname] = sim.get_pk_metrics(dp, bio, req.duration_hours)
            pk_metrics[dname]["therapeutic_window"] = list(dp.therapeutic_window)

    # Final system status (common to both paths)
    final_concs = {d: concentrations[d][-1] for d in drug_props}
    final_status = sim.get_system_status(bio, final_concs, [final_prog], all_side_effects)

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
    _primary_drug_key = req.drug if req.drug in concentrations else list(concentrations.keys())[0]
    last_conc = concentrations[_primary_drug_key][-1]
    tw_lo, tw_hi = primary_drug.therapeutic_window
    if last_conc < tw_lo:
        therapeutic_status = "sub-therapeutic"
    elif last_conc > tw_hi * 1.5:
        therapeutic_status = "toxic"
    elif last_conc > tw_hi:
        therapeutic_status = "supra-therapeutic"
    else:
        therapeutic_status = "therapeutic"

    # Merge CYP warnings into final status warnings
    all_warnings = list(final_status["warnings"])
    if _engine is not None and "_cyp_warnings" in dir():
        all_warnings = _cyp_warnings + all_warnings

    # Build series_time_points for thinned series (same as organ_health_series)
    _thin_final = max(1, len(time_points) // 73)
    series_time_points = [round(float(time_points[i]), 2)
                          for i in range(0, len(time_points), _thin_final)]

    # For fallback path, generate placeholder receptor occupancy
    if _engine is None:
        _receptor_occupancy = [0.0] * len(series_time_points)
        _pd_response        = [0.0] * len(series_time_points)
        _series_time        = series_time_points

    return {
        "time_points": [round(t, 2) for t in time_points],
        "concentrations": concentrations,
        "therapeutic_windows": {d: list(drug_props[d].therapeutic_window) for d in drug_props},
        "disease_severity": disease_severity,
        "disease_name": disease_name,
        # Time-series organ health (len ≤ 73)
        "organ_health_series": organ_health_series,
        "series_time_points": _series_time if _engine is not None else series_time_points,
        # Receptor occupancy & PD response time-series (primary drug)
        "receptor_occupancy": _receptor_occupancy,
        "pd_response": _pd_response,
        "organ_health_final": final_status["organ_health"],
        "organ_drug_effects": final_status["organ_drug_effects"],
        "side_effects": unique_effects,
        "warnings": all_warnings,
        "recommendations": final_status["recommendations"],
        "pk_metrics": pk_metrics,
        "biomarkers": biomarker_series,
        "therapeutic_status": therapeutic_status,
        "overall_health": final_status["overall_health"],
        "drugs": list(drug_props.keys()),
        "heart_rate": bio.heart_rate,
        "engine": "mechanistic" if _engine is not None else "analytical",
    }


# ───────────────────────────────────────────── long-term realistic sim ─────
@app.post("/api/simulate/longterm")
async def simulate_longterm(req: LongTermSimulateRequest):
    """
    Run a realistic multi-scale biological simulation spanning hours → years.

    Unlike /api/simulate (which uses a 1-compartment model), this endpoint
    uses ODE-based immune cascades, organ-damage biomarker kinetics, and
    long-term adaptation models (CYP enzyme induction, receptor downregulation,
    bone remodeling, microbiome disruption, and more).

    Request body parameters:
    - drug / dose: primary drug and dose (mg)
    - drug2 / dose2: optional second drug (combination)
    - disease / severity: disease name (0-1 severity)
    - age, weight, sex: patient demographics
    - kidney_function, liver_function: organ function (0-1)
    - duration_days: simulation length from 0.5 to 1825 (5 years)

    Returns: comprehensive JSON with plasma concentrations, immune markers,
    23 biomarker trajectories, adaptations, clinical events, and a summary.
    """
    from simulation.realistic_simulator import get_realistic_simulator

    drug_names = [req.drug]
    doses: Dict[str, float] = {req.drug: req.dose}

    if req.drug2 and req.drug2 not in ("None", "", "null"):
        drug_names.append(req.drug2)
        # Use provided dose2, or fall back to database default
        dp2 = _simulator.drug_database.get(req.drug2)
        doses[req.drug2] = req.dose2 if req.dose2 is not None else (dp2.dose_mg if dp2 else 10.0)

    sim = get_realistic_simulator()
    result = sim.simulate(
        drug_names=drug_names,
        doses_mg=doses,
        disease_name=req.disease,
        disease_severity=req.severity,
        patient_age=req.age,
        patient_weight_kg=req.weight,
        patient_sex=req.sex,
        kidney_function=req.kidney_function,
        liver_function=req.liver_function,
        dosing_frequency="daily",
        duration_days=req.duration_days,
    )
    return result.to_api_dict()


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


# ─────────────────────────────────────────────────────── population ────────
class PopulationRequest(BaseModel):
    drug: str = "Lisinopril"
    dose: float = 10.0
    disease: str = "Hypertension"
    severity: float = 0.55
    age: float = 65.0
    weight: float = 70.0
    sex: int = 1
    kidney_function: float = 0.8
    liver_function: float = 0.9
    n_patients: int = 50
    duration_hours: float = 72.0


@app.post("/api/population")
async def population_analysis(req: PopulationRequest):
    """Run a population cohort simulation and return percentile plasma envelopes."""
    from fastapi import HTTPException
    if _engine is None:
        raise HTTPException(status_code=503, detail="Mechanistic engine not available")
    from simulation.population_cohort import PopulationCohortRunner
    sim = _simulator
    dp = copy.deepcopy(sim.drug_database.get(req.drug, DrugProperties(name=req.drug)))
    dp.dose_mg = req.dose
    bio = BiologicalParameters(
        age=req.age, weight=req.weight, sex=req.sex,
        kidney_function=req.kidney_function, liver_function=req.liver_function,
    )
    disease_base = sim.disease_database.get(req.disease)
    disease_obj = copy.deepcopy(disease_base) if disease_base else DiseaseState(
        name=req.disease, severity=req.severity
    )
    disease_obj.severity = req.severity
    step = 1.0 if req.duration_hours <= 96 else 2.0
    time_arr = np.arange(0, req.duration_hours + step / 2, step)
    runner = PopulationCohortRunner()
    cohort = runner.run(_engine, dp, bio, [disease_obj], n=req.n_patients, time_points=time_arr)
    tw = cohort.therapeutic_window
    return {
        "time_points": [round(float(t), 2) for t in cohort.time],
        "p05":  [round(float(v), 4) for v in cohort.plasma_p05],
        "p25":  [round(float(v), 4) for v in cohort.plasma_p25],
        "p50":  [round(float(v), 4) for v in cohort.plasma_p50],
        "p75":  [round(float(v), 4) for v in cohort.plasma_p75],
        "p95":  [round(float(v), 4) for v in cohort.plasma_p95],
        "therapeutic_window": [float(tw[0]), float(tw[1])],
        "frac_within_window": round(float(cohort.frac_within_window), 4),
        "n_patients": cohort.n_patients,
        "n_succeeded": cohort.n_succeeded,
        "drug": req.drug,
        "disease": req.disease,
        "cmax_p50": round(float(np.median(cohort.cmax_distribution)), 4),
        "auc_p50":  round(float(np.median(cohort.auc_distribution)), 4),
    }


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
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
