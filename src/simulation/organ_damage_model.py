"""
Organ Damage Model
==================
Drug-specific cellular damage mechanisms and biomarker release kinetics.

Every drug can damage or protect organs via specific molecular pathways.
Damage accumulates on a real timescale (hours to months), and clinical
biomarkers rise/fall accordingly.

Implemented damage pathways
---------------------------
Acetaminophen (APAP)
  CYP2E1/CYP3A4 → NAPQI (N-acetyl-p-benzoquinone imine)
  NAPQI + GSH → conjugate (safe)
  NAPQI excess → mitochondrial adducts → hepatocyte necrosis
  ALT/AST rise 24-72 h after overdose; can peak > 500× ULN

NSAIDs (Ibuprofen, Naproxen)
  COX-1 inhibition → ↓prostaglandin E2 in gastric mucosa → ulceration
  COX-2 inhibition → ↓prostaglandin I2 in kidney → afferent vasoconstriction
  Renal consequences: GFR ↓, creatinine ↑ (days-weeks with chronic use)
  GI consequences: mucosal erosion, bleeding risk

Statins (Atorvastatin)
  HMG-CoA reductase inhibition → ↓mevalonate → ↓CoQ10 (ubiquinone)
  CoQ10 ↓ in skeletal muscle mitochondria → electron-transport-chain dysfunction
  ETC dysfunction → ATP ↓, ROS ↑, lipid peroxidation
  Myocyte damage → CK (creatine kinase) ↑, myalgia (weeks-months)
  Rhabdomyolysis (0.1% incidence) → myoglobin ↑ → acute kidney injury risk

ACE Inhibitors (Lisinopril)
  ACE inhibition → bradykinin NOT cleaved → bradykinin accumulates in lung
  Bradykinin → B2 receptor → sensitises cough-stretch receptors (slowly)
  Cough onset: 1-4 weeks post-initiation; prevalence 15-20% (40% Asian)
  Protective: ↓angiotensin II → ↓efferent arteriolar constriction → ↓GFR pressure
  → Slows CKD progression (net renoprotective despite acute small GFR ↓)

Corticosteroids (Prednisone)
  GR (glucocorticoid receptor) activation:
    → ↓ RANKL → ↓ osteoblast survival → ↓ bone formation
    → ↑ osteoclast activity → ↑ bone resorption
    Net: bone mineral density (BMD) ↓ 2-5% in first year (fracture risk +75%)
    → Hyperglycaemia: ↑ hepatic gluconeogenesis, ↓ peripheral glucose uptake
    → Immunosuppression: ↓ neutrophil adhesion, ↓ lymphocyte proliferation
    → ↓ collagen synthesis: skin atrophy/bruising
    → Adrenal suppression: HPA axis → adrenal atrophy after months

Benzodiazepines (Lorazepam)
  GABA-A receptor potentiation → acute sedation/anxiolysis
  Prolonged use → receptor internalization (down-regulation)
  Tolerance: anxiety-anxiolysis mismatch (weeks-months)
  Dependence: GABA-A subunit composition changes (months)
  Withdrawal syndrome: rebound anxiety, insomnia, seizure risk

SSRIs (Sertraline)
  SERT (serotonin transporter) block → synaptic 5-HT ↑ (immediate)
  5-HT1A autoreceptor: initially causes tachyphylaxis (blunts effect)
  Autoreceptor desensitisation (2-4 weeks) → therapeutic effect emerges
  Hippocampal neurogenesis (4-12 weeks) → mood stabilisation
  Early: nausea (5-HT3), insomnia, possibly sexual dysfunction

Antibiotics (Amoxicillin, Ciprofloxacin)
  Bactericidal → pathogen elimination (hours-days)
  Gut microbiome: Lactobacillus and Bifidobacterium ↓ (within 24-48h)
  C. diff risk (Clostridioides difficile): if spores present + disruption
  Microbiome partial recovery: 2-4 weeks post-treatment
  Fluoroquinolone (Ciprofloxacin): tendon collagen disruption (weeks)
    CYP inhibition → elevated co-administered drug levels

Metformin
  AMPK activation → ↓ hepatic gluconeogenesis → ↓ fasting glucose
  GI side effects: mitochondrial effect on intestinal cells → lactate ↑ locally
  B12 malabsorption: competition at ileal IF-B12 intrinsic factor receptor
    → B12 ↓ over months-years → peripheral neuropathy, megaloblastic anaemia
  Lactic acidosis: rare (1/10,000); risk ↑ if GFR < 30

Glipizide
  Sulphonylurea → SUR1 receptor → β-cell KATP channel block → insulin secretion
  Hypoglycaemia: main risk; 2-6h post-dose
  Weight gain: ~2-4 kg over months (insulin secretagogue)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

import json as _json
from pathlib import Path as _Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simulation config loader — read from file, seed on first run
# ---------------------------------------------------------------------------
_SIM_CFG_DIR = _Path(__file__).parent.parent.parent / "data" / "raw" / "simulation"


def _load_or_seed_sim_json(name: str, seed_data: dict) -> dict:
    """Return parsed JSON config; write seed_data to disk if file is missing."""
    path = _SIM_CFG_DIR / name
    if path.exists():
        try:
            with open(path, encoding="utf-8") as _f:
                return _json.load(_f)
        except Exception as _e:
            logger.warning("Could not read %s: %s — using defaults", name, _e)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as _f:
        _json.dump(seed_data, _f, indent=2, ensure_ascii=False)
    logger.info("Seeded simulation config: %s", name)
    return seed_data


# ---------------------------------------------------------------------------
# Biomarker ODE state indices (22 variables)
# ---------------------------------------------------------------------------
class _BM:
    # ── Liver biomarkers ─────────────────────────────────────────────────────
    ALT          = 0   # alanine aminotransferase (U/L); normal 7–56
    AST          = 1   # aspartate aminotransferase (U/L); normal 10–40
    ALP          = 2   # alkaline phosphatase (U/L); normal 44–147
    BILIRUBIN    = 3   # total bilirubin (µmol/L); normal < 20
    ALBUMIN      = 4   # albumin (g/L); normal 35–50
    INR          = 5   # International Normalised Ratio; normal ~1.0; liver synthetic function

    # ── Kidney biomarkers ────────────────────────────────────────────────────
    CREATININE   = 6   # creatinine (µmol/L); normal M 62–115, F 53–97
    BUN          = 7   # blood urea nitrogen (mmol/L); normal 2.5–7.1
    GFR          = 8   # estimated GFR (mL/min/1.73m²); normal > 90
    POTASSIUM    = 9   # serum K⁺ (mmol/L); normal 3.5–5.0

    # ── Muscle / cardiac biomarkers ──────────────────────────────────────────
    CK           = 10  # creatine kinase (U/L); normal M 38–174, F 26–140
    CK_MB        = 11  # CK-MB isoenzyme (U/L); normal < 5 (cardiac-specific)
    TROPONIN_I   = 12  # troponin I (ng/L); normal < 30
    MYOGLOBIN    = 13  # myoglobin (µg/L); normal < 90

    # ── Bone markers ────────────────────────────────────────────────────────
    BMD          = 14  # bone mineral density (g/cm²); lumbar spine normal ~1.0
    CALCIUM      = 15  # serum Ca²⁺ (mmol/L); normal 2.2–2.6
    PTH          = 16  # parathyroid hormone (pmol/L); normal 1.6–6.9

    # ── Haematological ──────────────────────────────────────────────────────
    NEUTROPHIL_B = 17  # duplicate of immune cascade (mirrored for drug effects)
    B12_LEVEL    = 18  # vitamin B12 (pmol/L); normal 150–700
    FOLATE       = 19  # serum folate (nmol/L); normal > 7

    # ── Drug-specific intermediates ──────────────────────────────────────────
    GSH          = 20  # hepatic glutathione (normalised, 1 = replete, 0 = depleted)
    BRADYKININ   = 21  # bradykinin level (arbitrary units, 0–1 normalised)

N_BM = 22


# ---------------------------------------------------------------------------
# Normal laboratory ranges and baseline values
# ---------------------------------------------------------------------------
BIOMARKER_BASELINE: Dict[str, Tuple[float, float, float]] = {
    # name: (value, lower_normal, upper_normal)
    "ALT":       (25.0,   7.0,  56.0),
    "AST":       (22.0,  10.0,  40.0),
    "ALP":       (80.0,  44.0, 147.0),
    "BILIRUBIN": (12.0,   3.0,  20.0),
    "ALBUMIN":   (42.0,  35.0,  50.0),
    "INR":       ( 1.0,   0.8,   1.2),
    "CREATININE":(85.0,  62.0, 115.0),
    "BUN":       ( 4.5,   2.5,   7.1),
    "GFR":       (95.0,  90.0, 120.0),
    "POTASSIUM": ( 4.1,   3.5,   5.0),
    "CK":        (90.0,  38.0, 174.0),
    "CK_MB":     ( 3.0,   0.0,   5.0),
    "TROPONIN_I":( 5.0,   0.0,  30.0),
    "MYOGLOBIN": (40.0,   0.0,  90.0),
    "BMD":       ( 1.00,  0.85,  1.20),
    "CALCIUM":   ( 2.35,  2.20,  2.60),
    "PTH":       ( 3.50,  1.60,  6.90),
    "NEUTROPHIL_B":(4.0,  1.8,   7.5),
    "B12_LEVEL": (350.0, 150.0, 700.0),
    "FOLATE":    (20.0,   7.0,  40.0),
    "GSH":       ( 1.00,  0.50,  1.00),   # fraction (1 = replete)
    "BRADYKININ":( 0.05,  0.00,  0.30),   # normalised
}

BM_NAMES = [
    "ALT","AST","ALP","BILIRUBIN","ALBUMIN","INR",
    "CREATININE","BUN","GFR","POTASSIUM",
    "CK","CK_MB","TROPONIN_I","MYOGLOBIN",
    "BMD","CALCIUM","PTH",
    "NEUTROPHIL_B","B12_LEVEL","FOLATE",
    "GSH","BRADYKININ",
]


def make_biomarker_baseline(disease_name: str = "", disease_severity: float = 0.0) -> np.ndarray:
    """Build initial biomarker state vector from healthy baseline ± disease adjustments."""
    y0 = np.array([BIOMARKER_BASELINE[n][0] for n in BM_NAMES], dtype=float)

    # Disease-specific biomarker adjustments
    _adjustments: Dict[str, Dict[str, float]] = {
        "Chronic Kidney Disease": {
            "CREATININE": 85 + 80 * disease_severity,
            "BUN":         4.5 + 10 * disease_severity,
            "GFR":         95 - 60 * disease_severity,
            "POTASSIUM":   4.1 + 1.0 * disease_severity,
        },
        "Heart Failure (Mild)":    {"ALT": 30.0, "AST": 28.0, "CREATININE": 95.0, "GFR": 80.0},
        "Heart Failure (Moderate)":{"ALT": 45.0, "AST": 40.0, "CREATININE": 110.0, "GFR": 65.0},
        "Heart Failure (Severe)":  {"ALT": 70.0, "AST": 60.0, "CREATININE": 140.0, "GFR": 45.0},
        "Type 2 Diabetes (Mild)":  {"CREATININE": 95.0, "GFR": 82.0},
        "Type 2 Diabetes (Moderate)": {"CREATININE": 105.0, "GFR": 72.0, "ALT": 35.0},
        "Type 2 Diabetes (Severe)":   {"CREATININE": 130.0, "GFR": 55.0, "ALT": 50.0, "B12_LEVEL": 280.0},
        "Arthritis":               {"CK": 120.0},
        "Hypothyroidism":          {"ALP": 60.0, "CK": 200.0},
        "Bacterial Infection":     {"CK": 100.0, "NEUTROPHIL_B": 14.0, "CREATININE": 100.0},
        "COPD (Moderate)":         {"CK": 110.0},
        "COPD (Severe)":           {"CK": 130.0, "CREATININE": 100.0, "GFR": 75.0},
    }
    adj = _adjustments.get(disease_name, {})
    for bm_name, val in adj.items():
        if bm_name in BM_NAMES:
            idx = BM_NAMES.index(bm_name)
            y0[idx] = val
    return y0


# ---------------------------------------------------------------------------
# Drug-specific damage parameters
# ---------------------------------------------------------------------------

@dataclass
class DrugDamageProfile:
    """Defines the toxicity mechanisms and target organs for a drug."""
    # Hepatotoxicity parameters
    hepatotox_rate: float = 0.0       # rate of hepatocyte damage per mg/L above threshold
    hepatox_threshold: float = 1000.0 # plasma concentration threshold (mg/L) for liver damage
    napqi_production: float = 0.0     # NAPQI production fraction (acetaminophen specific)
    # Nephrotoxicity parameters
    nephrotox_rate: float = 0.0       # rate of GFR reduction per mg/L
    nephrotox_threshold: float = 1000.0
    renal_vasoconstriction: float = 0.0  # COX-2 inhibition → renal vasoconstriction
    # Myotoxicity parameters (statins)
    coq10_depletion_rate: float = 0.0 # CoQ10 depletion rate (fraction of max)
    coq10_threshold: float = 0.0      # minimum therapeutic exposure for CoQ10 effect
    # GI damage
    gi_damage_rate: float = 0.0       # GI mucosal damage rate
    # Cardiac
    qt_prolongation: float = 0.0      # QT prolongation in ms per mg/L
    # Bone loss (prolonged corticosteroids)
    bone_loss_rate: float = 0.0       # BMD % loss per hour of therapeutic exposure
    # B12 malabsorption (metformin)
    b12_absorption_inhibit: float = 0.0  # fraction reduction in B12 absorption
    # Bradykinin accumulation (ACE inhibitors)
    bradykinin_accumulation: float = 0.0
    # Protective effects (positive = organ protection)
    cardiac_protection: float = 0.0
    renal_protection: float = 0.0


# Drug damage profiles — grounded in published mechanistic data
DRUG_DAMAGE_PROFILES: Dict[str, DrugDamageProfile] = {
    "Acetaminophen": DrugDamageProfile(
        hepatotox_rate     = 0.0005,
        hepatox_threshold  = 60.0,   # > 60 mg/L is overtly hepatotoxic (overdose ~200 mg/L)
        napqi_production   = 0.08,   # ~8% at therapeutic dose, ↑ with depletion of GSH
    ),
    "Ibuprofen": DrugDamageProfile(
        gi_damage_rate         = 0.0003,
        renal_vasoconstriction = 0.004,   # COX-2 → afferent arteriole constriction
        nephrotox_rate         = 0.0002,
        nephrotox_threshold    = 30.0,
    ),
    "Naproxen": DrugDamageProfile(
        gi_damage_rate         = 0.0004,
        renal_vasoconstriction = 0.003,
        nephrotox_rate         = 0.00015,
        nephrotox_threshold    = 40.0,
    ),
    "Lisinopril": DrugDamageProfile(
        bradykinin_accumulation = 0.95,    # max ACE inhibition fraction (0–1) — near-complete at therapeutic dose
        renal_protection        = 0.0002,  # efferent dilation → ↓ intraglomerular pressure
        cardiac_protection      = 0.0003,
    ),
    "Metoprolol": DrugDamageProfile(
        cardiac_protection = 0.0004,
        qt_prolongation    = 0.001,   # mild QT effect at high doses
    ),
    "Atorvastatin": DrugDamageProfile(
        coq10_depletion_rate = 0.0005,  # CoQ10 starts depleting within weeks
        coq10_threshold      = 3.0,    # therapeutic concs (mg/L ≈ 5–20 ng/mL)
        hepatotox_rate       = 0.00002, # transaminase elevation is uncommon
        hepatox_threshold    = 100.0,
    ),
    "Amlodipine": DrugDamageProfile(),  # generally safe profile
    "Metformin": DrugDamageProfile(
        b12_absorption_inhibit = 0.40,     # max fraction of ileal B12-IF absorption inhibited
        gi_damage_rate         = 0.00005,  # GI irritation
    ),
    "Glipizide": DrugDamageProfile(),
    "Amoxicillin": DrugDamageProfile(
        gi_damage_rate = 0.0001,
    ),
    "Ciprofloxacin": DrugDamageProfile(
        gi_damage_rate  = 0.00015,
        nephrotox_rate  = 0.00005,
        qt_prolongation = 0.002,
    ),
    "Sertraline": DrugDamageProfile(
        gi_damage_rate  = 0.00005,  # GI 5-HT3 activation → initial nausea
    ),
    "Lorazepam": DrugDamageProfile(),
    "Levothyroxine": DrugDamageProfile(
        cardiac_protection = -0.00005,  # at high doses → tachyarrhythmia risk
    ),
    "Prednisone": DrugDamageProfile(
        bone_loss_rate     = 3.5e-6,     # max net BMD depletion rate [g/cm²/h] at full GR occupancy (~3%/yr)
        hepatotox_rate     = 0.00001,    # mild hepatic lipid accumulation
        gi_damage_rate     = 0.0002,     # GI mucosal damage
    ),
}

# CoQ10 baseline (normalised 1 = replete, 0 = depleted)
_COQ10_BASELINE = 1.0
_COQ10_RECOVERY_RATE = 0.0003   # ~ 1% recovery per hour without statin
_COQ10_SYNTHESIS_RATE = 0.0008  # baseline synthesis

# Bone remodeling baseline osteoblast/osteoclast balance
_OSTEOBLAST_BASELINE = 1.0
_OSTEOCLAST_BASELINE = 1.0
_BMD_FORMATION_RATE  = 0.0000015
_BMD_RESORPTION_RATE = 0.0000015

# GFR homeostasis rate
_GFR_RECOVERY_RATE = 0.001   # GFR self-repairs if insult removed

# CK half-life in plasma: ~12h
_CK_HALFLIFE_H = 12.0
_TROPONIN_HALFLIFE_H = 6.0
_CK_CLR_RATE = 0.693 / _CK_HALFLIFE_H

# ---------------------------------------------------------------------------
# Override constants from data file (writes file on first run)
# ---------------------------------------------------------------------------
def _drug_damage_profiles_as_dict() -> dict:
    """Serialise DRUG_DAMAGE_PROFILES dataclass instances to plain dicts."""
    import dataclasses
    return {k: dataclasses.asdict(v) for k, v in DRUG_DAMAGE_PROFILES.items()}


_organ_damage_cfg = _load_or_seed_sim_json("organ_damage_config.json", {
    "biomarker_baseline": {
        k: list(v) for k, v in BIOMARKER_BASELINE.items()
    },
    "drug_damage_profiles": _drug_damage_profiles_as_dict(),
    "rate_constants": {
        "coq10_baseline":      _COQ10_BASELINE,
        "coq10_recovery_rate": _COQ10_RECOVERY_RATE,
        "coq10_synthesis_rate":_COQ10_SYNTHESIS_RATE,
        "osteoblast_baseline": _OSTEOBLAST_BASELINE,
        "osteoclast_baseline": _OSTEOCLAST_BASELINE,
        "bmd_formation_rate":  _BMD_FORMATION_RATE,
        "bmd_resorption_rate": _BMD_RESORPTION_RATE,
        "gfr_recovery_rate":   _GFR_RECOVERY_RATE,
        "ck_halflife_h":       _CK_HALFLIFE_H,
        "troponin_halflife_h": _TROPONIN_HALFLIFE_H,
    },
})

# Reload BIOMARKER_BASELINE as dict of tuples
BIOMARKER_BASELINE = {
    k: tuple(v) for k, v in _organ_damage_cfg["biomarker_baseline"].items()
}
# Rebuild DRUG_DAMAGE_PROFILES from JSON
DRUG_DAMAGE_PROFILES = {
    k: DrugDamageProfile(**v)
    for k, v in _organ_damage_cfg["drug_damage_profiles"].items()
}
# Reload rate constants
_rc = _organ_damage_cfg["rate_constants"]
_COQ10_BASELINE        = float(_rc["coq10_baseline"])
_COQ10_RECOVERY_RATE   = float(_rc["coq10_recovery_rate"])
_COQ10_SYNTHESIS_RATE  = float(_rc["coq10_synthesis_rate"])
_OSTEOBLAST_BASELINE   = float(_rc["osteoblast_baseline"])
_OSTEOCLAST_BASELINE   = float(_rc["osteoclast_baseline"])
_BMD_FORMATION_RATE    = float(_rc["bmd_formation_rate"])
_BMD_RESORPTION_RATE   = float(_rc["bmd_resorption_rate"])
_GFR_RECOVERY_RATE     = float(_rc["gfr_recovery_rate"])
_CK_HALFLIFE_H         = float(_rc["ck_halflife_h"])
_TROPONIN_HALFLIFE_H   = float(_rc["troponin_halflife_h"])
_CK_CLR_RATE           = 0.693 / max(_CK_HALFLIFE_H, 0.1)


@dataclass
class DamageState:
    """Running state of cumulative organ damage (not part of ODE vector directly)."""
    coq10_level: float = 1.0        # statin-depleted CoQ10 (normalised)
    hepatocyte_health: float = 1.0  # fraction of functional hepatocytes
    myocyte_health: float = 1.0     # fraction of functional myocytes
    gfr_multiplier: float = 1.0     # ratio to baseline (1 = normal)
    gfr_protection: float = 0.0     # cumulative renoprotective benefit
    gi_mucosal_health: float = 1.0  # GI mucosa integrity
    bmd_current: float = 1.0        # current BMD (g/cm²)
    bradykinin_sens: float = 0.0    # cough receptor sensitisation (0–1)
    b12_absorption: float = 1.0     # fraction of normal B12 absorption
    osteoblast_activity: float = 1.0
    osteoclast_activity: float = 1.0


class OrganDamageEngine:
    """
    Tracks drug-specific organ damage accumulation and biomarker dynamics.

    For each simulation step (given drug concentration and elapsed time), updates:
    - Internal damage state (coq10, hepatocyte health, GFR, bone, etc.)
    - Biomarker ODE output (ALT, AST, CK, creatinine, bradykinin, etc.)

    Returns time-series arrays of all biomarkers at clinical lab units.
    """

    def __init__(self) -> None:
        self._damage: DamageState = DamageState()
        self._biomarker_state: Optional[np.ndarray] = None

    def simulate(
        self,
        drug_names: List[str],
        plasma_conc_series: Dict[str, np.ndarray],  # drug → concentration array (mg/L)
        time_h: np.ndarray,
        disease_name: str = "",
        disease_severity: float = 0.0,
        patient_weight_kg: float = 70.0,
        patient_age: float = 65.0,
        kidney_function: float = 1.0,
        liver_function: float = 1.0,
        existing_damage: Optional[DamageState] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run the biomarker ODE system for the given time array.

        Parameters
        ----------
        drug_names : list of drug names being simulated
        plasma_conc_series : dict of drug → plasma concentration array (same time grid)
        time_h : time array (hours)
        ...

        Returns
        -------
        dict of biomarker_name → np.ndarray of values over time
        """
        if existing_damage is not None:
            self._damage = existing_damage

        y0 = make_biomarker_baseline(disease_name, disease_severity)

        # Adjust for baseline organ function
        y0[_BM.GFR] *= kidney_function
        y0[_BM.CREATININE] *= (1.0 / max(kidney_function, 0.1))
        y0[_BM.ALT] *= (1.0 + (1.0 - liver_function) * 0.5)
        y0[_BM.AST] *= (1.0 + (1.0 - liver_function) * 0.5)

        # Collect drug profiles
        profiles = {d: DRUG_DAMAGE_PROFILES.get(d, DrugDamageProfile()) for d in drug_names}

        # Pre-compute therapeutic windows (for threshold checking)
        from app.server import _simulator  # imported lazily to avoid circular import issues
        _tw: Dict[str, float] = {}
        for d in drug_names:
            try:
                dp = _simulator.drug_database.get(d)
                _tw[d] = dp.therapeutic_window[1] if dp else 50.0
            except Exception:
                _tw[d] = 50.0

        # The damage state object tracks slow-timescale changes
        damage = existing_damage or DamageState()

        def ode_rhs(t_val: float, y: np.ndarray) -> np.ndarray:
            return self._biomarker_odes(
                t_val, y, drug_names, plasma_conc_series, profiles,
                time_h, damage, disease_severity, kidney_function, liver_function,
            )

        try:
            sol = solve_ivp(
                ode_rhs,
                (float(time_h[0]), float(time_h[-1])),
                y0.copy(),
                method="LSODA",
                t_eval=time_h,
                rtol=1e-4,
                atol=1e-6,
                max_step=4.0,
            )
            traj = np.clip(sol.y, 0.0, None)
        except Exception as exc:
            logger.warning("OrganDamage ODE failed: %s — returning flat baseline", exc)
            traj = np.tile(y0[:, None], (1, len(time_h)))

        # Package into named dict
        result: Dict[str, np.ndarray] = {}
        for i, name in enumerate(BM_NAMES):
            result[name] = traj[i, :len(time_h)]

        # Update persistent damage state from final ODE state
        self._update_damage_state(damage, traj[:, -1], drug_names, profiles, plasma_conc_series, time_h[-1])
        self._damage = damage

        return result

    def get_damage_state(self) -> DamageState:
        return self._damage

    # ── Biomarker ODE system ─────────────────────────────────────────────────

    def _biomarker_odes(
        self,
        t: float,
        y: np.ndarray,
        drug_names: List[str],
        plasma_series: Dict[str, np.ndarray],
        profiles: Dict[str, DrugDamageProfile],
        time_h: np.ndarray,
        damage: DamageState,
        disease_severity: float,
        kf: float,   # kidney function
        lf: float,   # liver function
    ) -> np.ndarray:
        dy = np.zeros(N_BM)

        # Interpolate drug concentrations at time t
        drug_concs: Dict[str, float] = {}
        for d in drug_names:
            arr = plasma_series.get(d, np.zeros(len(time_h)))
            drug_concs[d] = float(np.interp(t, time_h, arr))

        # ── Unpack state ─────────────────────────────────────────────────────
        alt    = max(y[_BM.ALT], 0.0)
        ast    = max(y[_BM.AST], 0.0)
        alp    = max(y[_BM.ALP], 0.0)
        bilir  = max(y[_BM.BILIRUBIN], 0.0)
        alb    = max(y[_BM.ALBUMIN], 0.0)
        inr    = max(y[_BM.INR], 0.0)
        creat  = max(y[_BM.CREATININE], 0.0)
        bun    = max(y[_BM.BUN], 0.0)
        gfr    = max(y[_BM.GFR], 1.0)
        potass = max(y[_BM.POTASSIUM], 0.0)
        ck     = max(y[_BM.CK], 0.0)
        ckmb   = max(y[_BM.CK_MB], 0.0)
        trop   = max(y[_BM.TROPONIN_I], 0.0)
        myog   = max(y[_BM.MYOGLOBIN], 0.0)
        bmd    = max(y[_BM.BMD], 0.0)
        ca     = max(y[_BM.CALCIUM], 0.0)
        pth    = max(y[_BM.PTH], 0.0)
        b12    = max(y[_BM.B12_LEVEL], 0.0)
        folate = max(y[_BM.FOLATE], 0.0)
        gsh    = max(y[_BM.GSH], 0.0)
        bk     = max(y[_BM.BRADYKININ], 0.0)

        # ── Compute aggregate damage drivers from all drugs ──────────────────
        total_napqi_rate    = 0.0
        total_gsh_depletion = 0.0
        total_gi_damage     = 0.0
        total_renal_vaso    = 0.0
        total_nephrotox     = 0.0
        total_coq10_deplete = 0.0
        total_bone_loss     = 0.0
        total_b12_inhibit   = 0.0
        total_bk_accum      = 0.0
        total_renal_protect = 0.0
        total_cardiac_prot  = 0.0
        hepatotox_damage    = 0.0

        for d, conc in drug_concs.items():
            p = profiles.get(d, DrugDamageProfile())

            # NAPQI (acetaminophen metabolite)
            if p.napqi_production > 0:
                napqi_rate = p.napqi_production * conc * (1.0 - gsh * 0.8)
                total_napqi_rate += napqi_rate
                total_gsh_depletion += napqi_rate * 0.6  # GSH consumed per NAPQI unit

            # Hepatotoxicity (direct)
            if conc > p.hepatox_threshold and p.hepatotox_rate > 0:
                excess = conc - p.hepatox_threshold
                hepatotox_damage += p.hepatotox_rate * excess * (1.0 - gsh * 0.5)

            # GI damage (COX-1 inhibition, corticosteroids, antibiotics)
            if p.gi_damage_rate > 0:
                total_gi_damage += p.gi_damage_rate * conc

            # Renal vasoconstriction (COX-2 inhibition)
            if p.renal_vasoconstriction > 0:
                total_renal_vaso += p.renal_vasoconstriction * conc

            # Direct nephrotoxicity
            if conc > p.nephrotox_threshold and p.nephrotox_rate > 0:
                total_nephrotox += p.nephrotox_rate * (conc - p.nephrotox_threshold)

            # Statin CoQ10 depletion (starts at any therapeutic concentration)
            if p.coq10_depletion_rate > 0:
                occ = conc / (p.coq10_threshold + conc + 1e-6)
                total_coq10_deplete += p.coq10_depletion_rate * occ

            # Bone loss (corticosteroids)
            # bone_loss_rate = max net BMD depletion rate [1/h] at saturated GR.
            # Hill equation EC50 for glucocorticoid receptor ≈ 0.01 mg/L (prednisolone equiv.)
            if p.bone_loss_rate > 0:
                ec50_bone = 0.01
                occ_bone = conc / (conc + ec50_bone)
                total_bone_loss += p.bone_loss_rate * occ_bone

            # B12 malabsorption (metformin)
            # b12_absorption_inhibit = max fraction of ileal absorption blocked (0–1).
            # EC50 for metformin TCII/IF-receptor competition ≈ 0.5 mg/L.
            if p.b12_absorption_inhibit > 0:
                ec50_b12 = 0.5
                occ_b12 = conc / (conc + ec50_b12)
                total_b12_inhibit = min(0.95, total_b12_inhibit + p.b12_absorption_inhibit * occ_b12)

            # Bradykinin (ACE inhibitors)
            # bradykinin_accumulation = max ACE inhibition fraction (0–1).
            # EC50 for ACE inhibition by Lisinopril ≈ 0.001 mg/L.
            if p.bradykinin_accumulation > 0:
                ec50_ace = 0.001
                occ_ace = conc / (conc + ec50_ace)
                total_bk_accum = min(1.0, total_bk_accum + p.bradykinin_accumulation * occ_ace)

            # Renal protection (ACE inhibitors, ARBs)
            if p.renal_protection > 0:
                total_renal_protect += p.renal_protection * conc

            # Cardiac protection
            if p.cardiac_protection > 0:
                total_cardiac_prot += p.cardiac_protection * conc

        # ── Glutathione (GSH) — liver defence against NAPQI ─────────────────
        # GSH synthesised at rate ~2 mM/h in liver; consumed by NAPQI
        gsh_synthesis = 0.005 * (1.0 - gsh)   # replenishment (faster when depleted)
        dy[_BM.GSH] = gsh_synthesis - total_gsh_depletion * 0.01

        # ── Liver biomarkers ─────────────────────────────────────────────────
        # ALT: released from necrotic hepatocytes; t½ ~18-24h in plasma
        # Production ∝ hepatocyte necrosis rate; baseline production maintains normal level
        alt_baseline = BIOMARKER_BASELINE["ALT"][0] * (1.0 + (1.0 - lf) * 0.5)
        napqi_necrosis = total_napqi_rate * (1.0 - gsh * 0.9) * 50.0
        alt_prod = float(np.clip(alt_baseline * 0.038 + napqi_necrosis + hepatotox_damage * 500.0, 0.0, 5000.0))
        dy[_BM.ALT] = alt_prod - (0.693 / 20.0) * alt    # t½ ~20h
        dy[_BM.AST] = (
            alt_prod * 0.7 + total_coq10_deplete * 100.0  # also from muscle (CK pathway)
            - (0.693 / 12.0) * ast                         # t½ ~12h
        )
        dy[_BM.ALP] = (
            alp * 0.001 * (disease_severity * 0.5 - 1.0 + lf * 0.5)  # slow turnover
            + BIOMARKER_BASELINE["ALP"][0] * 0.001
        )
        # Bilirubin: rises with hepatocyte loss or bile duct obstruction
        dy[_BM.BILIRUBIN] = (
            0.05 * (1.0 - lf) * disease_severity
            - 0.03 * bilir
            + BIOMARKER_BASELINE["BILIRUBIN"][0] * 0.03
        )
        # Albumin: synthesised by liver; t½ ~20 days — very slow to fall
        dy[_BM.ALBUMIN] = 0.0001 * (BIOMARKER_BASELINE["ALBUMIN"][0] * lf - alb)
        # INR: coagulation function; rises with liver failure
        dy[_BM.INR] = 0.0002 * (
            (1.0 + (1.0 - lf) * 0.5 + max(0.0, alt / 200.0 - 1.0) * 0.1) - inr
        )

        # ── Kidney biomarkers ─────────────────────────────────────────────────
        # GFR: acute reduction from NSAID-induced vasoconstriction; slower chronic damage
        gfr_baseline = BIOMARKER_BASELINE["GFR"][0] * kf
        gfr_nsaid_effect = total_renal_vaso * gfr_baseline * 0.12
        gfr_chronic_damage = total_nephrotox * gfr_baseline
        gfr_protection = total_renal_protect * gfr_baseline
        dy[_BM.GFR] = (
            _GFR_RECOVERY_RATE * (gfr_baseline - gfr)     # homeostasis
            - gfr_nsaid_effect                              # acute NSAID effect
            - gfr_chronic_damage                            # chronic toxicity
            + gfr_protection                                # ACE-I renoprotection
        )

        # Creatinine: produced by muscle at constant rate; cleared by GFR
        # Mass balance: dCreat/dt = production - renal_clearance
        # At steady-state: Creat_ss = (production_rate / GFR) × constant
        creat_prod = 0.014 * 70.0  # ~10 mg/kg/day muscle production
        creat_clearance = (gfr / 1000.0) * creat   # GFR × concentration (simplified)
        # Clearance coefficient calibrated so that at GFR=95 and creat=85 µmol/L,
        # production exactly equals clearance (net rate = 0 at baseline).
        # k_clear = creat_prod / (GFR_baseline × creat_baseline) = 0.98/(95×85) = 0.1213e-3
        dy[_BM.CREATININE] = creat_prod - creat_clearance * 0.1213

        dy[_BM.BUN] = 0.002 * (
            (BIOMARKER_BASELINE["BUN"][0] / max(kf, 0.1)) - bun
        )

        # Potassium: rises with ACE inhibitor use (↓ aldosterone → K+ retention)
        # bradykinin_accumulation is now the max ACE inhibition fraction (0-1).
        # Use same Hill-equation approach for K+ target (max ≈0.35 mmol/L rise).
        ace_inhib_total = 0.0
        for d_k, c_k in drug_concs.items():
            p_k = DRUG_DAMAGE_PROFILES.get(d_k, DrugDamageProfile())
            if p_k.bradykinin_accumulation > 0:
                occ_k = c_k / (c_k + 0.001)   # same EC50 as bradykinin pathway
                ace_inhib_total = min(1.0, ace_inhib_total + p_k.bradykinin_accumulation * occ_k)
        k_target = 4.1 + ace_inhib_total * 0.35 + (1.0 - kf) * 0.6
        dy[_BM.POTASSIUM] = 0.002 * (k_target - potass)

        # ── Muscle / cardiac biomarkers ──────────────────────────────────────
        # CoQ10-mediated myopathy (statins): weeks-months onset
        coq10 = max(damage.coq10_level, 0.01)
        mito_dysfunction = max(0.0, 1.0 - coq10)   # 0 = no dysfunction, 1 = fully impaired

        # CK: released from damaged myocytes; t½ ~12h; rises in statin myopathy
        ck_baseline = BIOMARKER_BASELINE["CK"][0]
        ck_myo_release = mito_dysfunction * 5.0 * ck_baseline   # CK release from myopathy
        ck_disease = disease_severity * 30.0                      # CK elevated in COPD/heart failure
        dy[_BM.CK] = (
            ck_baseline * _CK_CLR_RATE   # production maintains baseline
            + ck_myo_release              # statin myopathy
            + ck_disease
            - _CK_CLR_RATE * ck          # clearance
        )

        # CK-MB: cardiac-specific; elevated in coronary events or severe damage
        dy[_BM.CK_MB] = (
            BIOMARKER_BASELINE["CK_MB"][0] * 0.04
            + max(0.0, mito_dysfunction - 0.5) * 2.0  # cardiac-specific with severe CoQ10 depletion
            - (0.693 / 8.0) * ckmb
        )

        # Troponin I: ultra-sensitive cardiac marker; t½ ~6h
        dy[_BM.TROPONIN_I] = (
            BIOMARKER_BASELINE["TROPONIN_I"][0] * (0.693 / _TROPONIN_HALFLIFE_H)
            + max(0.0, total_cardiac_prot) * (-0.1)  # protective drugs reduce troponin
            - (0.693 / _TROPONIN_HALFLIFE_H) * trop
        )

        # Myoglobin: rises with acute muscle damage; clears rapidly (t½ ~2-3h)
        dy[_BM.MYOGLOBIN] = (
            BIOMARKER_BASELINE["MYOGLOBIN"][0] * (0.693 / 2.5)
            + mito_dysfunction * 30.0  # acute myopathy
            - (0.693 / 2.5) * myog
        )

        # ── Bone markers ─────────────────────────────────────────────────────
        # BMD: very slow net change driven by glucocorticoid occupancy.
        # total_bone_loss is now the net depletion rate [g/cm²/h] from Hill eq.
        # At no drug → total_bone_loss=0 → net BMD change=0 (homeostasis).
        # Prednisone 10mg at Cavg 0.057 mg/L → ~2.6%/yr loss (literature: 2-5%/yr).
        dy[_BM.BMD] = -total_bone_loss

        # Calcium: mild secondary hyperparathyroidism as BMD falls
        dy[_BM.CALCIUM] = (
            0.001 * (BIOMARKER_BASELINE["CALCIUM"][0] - ca)
            + total_bone_loss * 5000.0  # small Ca² release with bone resorption
        )

        # PTH: rises when calcium falls (secondary hyperparathyroidism)
        ca_target = BIOMARKER_BASELINE["CALCIUM"][0]
        dy[_BM.PTH] = 0.005 * ((ca_target - ca) * 5.0 - (pth - BIOMARKER_BASELINE["PTH"][0]))

        # ── Haematological ───────────────────────────────────────────────────
        dy[_BM.NEUTROPHIL_B] = 0.02 * (BIOMARKER_BASELINE["NEUTROPHIL_B"][0] - y[_BM.NEUTROPHIL_B])

        # B12: dietary intake – absorption – utilisation
        # Metformin inhibits ileal absorption of IF-B12 complex
        b12_intake = 0.0002   # ~2.4 µg/day → normalised
        # total_b12_inhibit is now the direct inhibition fraction (0–1) from Hill eq.
        b12_absorption = b12_intake * (1.0 - total_b12_inhibit)   # fraction absorbed
        b12_absorption = max(0.0, b12_absorption)
        dy[_BM.B12_LEVEL] = b12_absorption * 200.0 - 0.0001 * b12  # slow turnover

        dy[_BM.FOLATE] = 0.0001 * (BIOMARKER_BASELINE["FOLATE"][0] - folate)

        # ── Bradykinin (ACE inhibitor cough sensitisation) ───────────────────
        # total_bk_accum is now the ACE inhibition fraction (0–1).
        # 75% of bradykinin is normally cleared by ACE; the rest by other peptidases.
        # Blocking ACE dramatically reduces clearance → BK accumulates.
        # Baseline: bk_ss = 0.005/0.100 = 0.05  (normalised)
        # Full ACE inhibition: bk_ss = 0.005/0.025 = 0.20  (4× baseline → cough)
        _ace_frac_of_bk_clearance = 0.75
        bk_clearance_rate = 0.10 * (1.0 - _ace_frac_of_bk_clearance * total_bk_accum)
        dy[_BM.BRADYKININ] = (
            0.005                      # basal kinin generation
            - bk_clearance_rate * bk   # reduced clearance when ACE is blocked
        )

        return dy

    def _update_damage_state(
        self,
        damage: DamageState,
        final_y: np.ndarray,
        drug_names: List[str],
        profiles: Dict[str, DrugDamageProfile],
        plasma_series: Dict[str, np.ndarray],
        total_time_h: float,
    ) -> None:
        """Update cumulative slow-timescale damage state from final ODE values."""
        # Average drug concentrations over the simulation
        for d in drug_names:
            arr = plasma_series.get(d, np.zeros(1))
            avg_conc = float(np.mean(arr))
            p = profiles.get(d, DrugDamageProfile())

            # CoQ10 depletion (statins) — accumulates over weeks of therapy
            if p.coq10_depletion_rate > 0:
                occ = avg_conc / (p.coq10_threshold + avg_conc + 1e-6)
                depletion = p.coq10_depletion_rate * occ * total_time_h
                damage.coq10_level = max(0.05, damage.coq10_level - depletion)

            # Bone density (corticosteroids) — very slow, use BMD ODE result
        damage.bmd_current = float(final_y[_BM.BMD])

        # Bradykinin sensitisation
        damage.bradykinin_sens = min(1.0, float(final_y[_BM.BRADYKININ]) * 2.0)

        # B12 absorption (metformin long-term)
        b12_depletion_rate_per_h = sum(
            p.b12_absorption_inhibit * float(np.mean(plasma_series.get(d, np.zeros(1))))
            for d, p in profiles.items()
        )
        damage.b12_absorption = max(0.1, damage.b12_absorption - b12_depletion_rate_per_h * total_time_h * 0.1)

        # GFR multiplier from creatinine trajectory
        creat_final = float(final_y[_BM.CREATININE])
        creat_baseline = BIOMARKER_BASELINE["CREATININE"][0]
        damage.gfr_multiplier = min(1.2, max(0.1, creat_baseline / max(creat_final, 1.0)))


def interpret_biomarker(name: str, value: float) -> Tuple[str, str]:
    """
    Return (status, interpretation) for a biomarker value.
    status: 'normal', 'mildly_elevated', 'moderately_elevated', 'critically_elevated',
            'low', or 'critically_low'
    """
    info = BIOMARKER_BASELINE.get(name)
    if info is None:
        return ("unknown", "No reference range available")

    _, lo, hi = info

    if value < lo * 0.5:
        return ("critically_low", f"{name} = {value:.1f}; critically below normal ({lo}–{hi})")
    if value < lo:
        return ("low", f"{name} = {value:.1f}; below normal range ({lo}–{hi})")
    if value <= hi:
        return ("normal", f"{name} = {value:.1f}; within normal range ({lo}–{hi})")
    if value <= hi * 2.0:
        return ("mildly_elevated", f"{name} = {value:.1f}; mildly above normal ({lo}–{hi})")
    if value <= hi * 5.0:
        return ("moderately_elevated", f"{name} = {value:.1f}; moderately elevated (>2× ULN)")
    return ("critically_elevated", f"{name} = {value:.1f}; critically elevated (>5× ULN)")
