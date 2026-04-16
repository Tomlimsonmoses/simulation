"""
Biological Cascade Engine
=========================
ODE-based models of the human innate/adaptive immune system, cytokine networks,
acute-phase response, endocrine feedback axes, and blood-cell dynamics.

All parameters are grounded in published pharmacology/physiology literature.
Units
-----
  cells     : cells/µL (standard clinical blood count units)
  cytokines : pg/mL
  hormones  : standard clinical units documented per variable
  time      : hours  (internal); converted from days/months at ingestion

References (simplified, open-literature)
-----------------------------------------
  Daun et al. (2008) J Theor Biol — innate immune model framework
  Reynolds et al. (2006) J Theor Biol — inflammation/pathogen ODE system
  Marathe et al. (2014) — cytokine dynamics
  Pharmacological Basis of Therapeutics (Brunton et al., 13th ed.)
  Goodman & Gilman pharmacokinetic tables
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

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
# State vector indices — these define the layout of y[] in all ODE calls
# ---------------------------------------------------------------------------
class _IDX:
    """Named indices for the biological cascade ODE state vector (35 variables)."""
    # ── Innate immune cells (cells/µL) ─────────────────────────────────────
    NEUTROPHIL   = 0   # PMN; first responder to infection/damage
    MACROPHAGE   = 1   # tissue macrophage (activated state fraction)
    NK_CELL      = 2   # natural killer cell
    MONOCYTE     = 3   # circulating monocyte
    # ── Adaptive immune cells (cells/µL) ────────────────────────────────────
    TH_CELL      = 4   # CD4+ T-helper lymphocyte
    TC_CELL      = 5   # CD8+ cytotoxic T-lymphocyte
    B_CELL       = 6   # B lymphocyte (naïve + memory)
    ANTIBODY     = 7   # circulating IgG (arbitrary units, normalised to 1 = baseline)
    # ── Pro-inflammatory cytokines (pg/mL) ─────────────────────────────────
    IL1B         = 8   # interleukin-1β
    IL6          = 9   # interleukin-6 (drives CRP, fever)
    TNFa         = 10  # tumour necrosis factor-α
    IL17         = 11  # interleukin-17 (chronic inflammation, arthritis)
    IFNG         = 12  # interferon-γ (viral/adaptive immunity)
    # ── Anti-inflammatory mediators ─────────────────────────────────────────
    IL10         = 13  # interleukin-10 (anti-inflammatory brake)
    IL4          = 14  # interleukin-4 (Th2/anti-inflammatory)
    # ── Acute-phase proteins (clinical lab units) ───────────────────────────
    CRP          = 15  # C-reactive protein (mg/L); normal < 5
    FIBRINOGEN   = 16  # fibrinogen (g/L); normal 2–4
    # ── HPA axis hormones ───────────────────────────────────────────────────
    CRH          = 17  # corticotropin-releasing hormone (pg/mL)
    ACTH         = 18  # ACTH (pg/mL); normal 10–50
    CORTISOL     = 19  # cortisol (nmol/L); normal morning ~500, nadir ~100
    # ── Thyroid axis ────────────────────────────────────────────────────────
    TSH          = 20  # thyroid-stimulating hormone (mIU/L); normal 0.4–4.0
    T4_FREE      = 21  # free T4 (pmol/L); normal 9–24
    T3_FREE      = 22  # free T3 (pmol/L); normal 2.5–6.5
    # ── Glucose–insulin axis ────────────────────────────────────────────────
    GLUCOSE      = 23  # blood glucose (mmol/L); fasting normal 4.0–5.6
    INSULIN      = 24  # insulin (pmol/L); fasting normal 20–90
    GLUCAGON     = 25  # glucagon (pmol/L); normal 30–130
    # ── Erythroid / platelet ────────────────────────────────────────────────
    RBC          = 26  # red blood cells (×10¹²/L); normal M 4.5–5.9, F 4.0–5.2
    HEMOGLOBIN   = 27  # haemoglobin (g/dL); normal M 13.5–17.5
    PLATELET     = 28  # platelets (×10⁹/L); normal 150–400
    # ── Total white blood cell count ────────────────────────────────────────
    WBC          = 29  # WBC (×10⁹/L); normal 4.5–11.0
    # ── Complement / coagulation proxies ────────────────────────────────────
    COMPLEMENT   = 30  # complement activation (normalised 0–1)
    D_DIMER      = 31  # D-dimer (µg/mL FEU); normal < 0.5
    # ── Circadian driver (internal) ─────────────────────────────────────────
    CIRC_PHASE   = 32  # circadian clock phase (radians, 0–2π over 24 h)
    # ── Pathogen load (for infectious disease simulations) ──────────────────
    PATHOGEN     = 33  # relative pathogen load (log₁₀ CFU/mL equivalent, 0–10)
    # ── Inflammation index (composite disease activity score) ───────────────
    INFLAM_INDEX = 34  # 0 = none, 1 = severe

N_STATE = 35   # total size of state vector


# ---------------------------------------------------------------------------
# Baseline healthy-adult state
# ---------------------------------------------------------------------------
HEALTHY_BASELINE: Dict[str, float] = {
    "NEUTROPHIL":   4.0,      # cells/µL ×10³ equivalent
    "MACROPHAGE":   0.05,     # activated fraction (0–1)
    "NK_CELL":      0.20,
    "MONOCYTE":     0.50,
    "TH_CELL":      0.80,
    "TC_CELL":      0.40,
    "B_CELL":       0.25,
    "ANTIBODY":     1.00,    # normalised baseline
    "IL1B":         0.50,    # pg/mL (nearly undetectable at rest)
    "IL6":          1.50,    # pg/mL normal < 3 pg/mL
    "TNFa":         5.00,    # pg/mL normal 4–8
    "IL17":         2.00,
    "IFNG":         0.50,
    "IL10":         3.00,    # pg/mL
    "IL4":          1.50,
    "CRP":          1.00,    # mg/L (low-grade baseline)
    "FIBRINOGEN":   2.50,    # g/L
    "CRH":          10.0,    # pg/mL
    "ACTH":         20.0,    # pg/mL
    "CORTISOL":     400.0,   # nmol/L (morning peak; will oscillate)
    "TSH":          2.0,     # mIU/L
    "T4_FREE":      15.0,    # pmol/L
    "T3_FREE":      4.5,     # pmol/L
    "GLUCOSE":      5.0,     # mmol/L (fasting)
    "INSULIN":      50.0,    # pmol/L
    "GLUCAGON":     60.0,    # pmol/L
    "RBC":          5.0,     # ×10¹²/L
    "HEMOGLOBIN":   14.5,    # g/dL
    "PLATELET":     250.0,   # ×10⁹/L
    "WBC":          6.5,     # ×10⁹/L
    "COMPLEMENT":   0.10,    # activation fraction (low at rest)
    "D_DIMER":      0.25,    # µg/mL
    "CIRC_PHASE":   0.0,     # radians; start at midnight (trough cortisol)
    "PATHOGEN":     0.0,     # no infection
    "INFLAM_INDEX": 0.05,    # tiny background inflammation
}


def make_baseline_state(disease_adjustments: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Build the initial state vector from healthy baseline, optionally
    modified by disease-specific adjustments.
    """
    names = [
        "NEUTROPHIL","MACROPHAGE","NK_CELL","MONOCYTE",
        "TH_CELL","TC_CELL","B_CELL","ANTIBODY",
        "IL1B","IL6","TNFa","IL17","IFNG",
        "IL10","IL4",
        "CRP","FIBRINOGEN",
        "CRH","ACTH","CORTISOL",
        "TSH","T4_FREE","T3_FREE",
        "GLUCOSE","INSULIN","GLUCAGON",
        "RBC","HEMOGLOBIN","PLATELET","WBC",
        "COMPLEMENT","D_DIMER",
        "CIRC_PHASE","PATHOGEN","INFLAM_INDEX",
    ]
    y0 = np.array([HEALTHY_BASELINE[n] for n in names], dtype=float)
    if disease_adjustments:
        for name, val in disease_adjustments.items():
            if name in names:
                y0[names.index(name)] = val
    return y0


# ---------------------------------------------------------------------------
# Disease-specific initial state adjustments
# ---------------------------------------------------------------------------
DISEASE_INITIAL_STATES: Dict[str, Dict[str, float]] = {
    "Hypertension": {
        "IL6": 3.5, "CRP": 4.0, "INFLAM_INDEX": 0.15,
        "FIBRINOGEN": 3.2, "PLATELET": 280.0,
    },
    "Heart Failure (Mild)": {
        "IL6": 8.0, "TNFa": 12.0, "CRP": 8.0, "BNP": 250.0,
        "INFLAM_INDEX": 0.25, "NEUTROPHIL": 7.0, "WBC": 8.5,
    },
    "Heart Failure (Moderate)": {
        "IL6": 15.0, "TNFa": 22.0, "CRP": 15.0,
        "INFLAM_INDEX": 0.40, "NEUTROPHIL": 9.0, "WBC": 10.5,
        "FIBRINOGEN": 4.2,
    },
    "Heart Failure (Severe)": {
        "IL6": 35.0, "TNFa": 40.0, "CRP": 30.0,
        "INFLAM_INDEX": 0.65, "NEUTROPHIL": 12.0, "WBC": 13.0,
    },
    "Type 2 Diabetes (Mild)": {
        "GLUCOSE": 8.0, "INSULIN": 150.0, "IL6": 4.0, "CRP": 3.0,
        "INFLAM_INDEX": 0.15, "GLUCAGON": 90.0,
    },
    "Type 2 Diabetes (Moderate)": {
        "GLUCOSE": 10.5, "INSULIN": 220.0, "IL6": 7.5, "CRP": 6.5,
        "INFLAM_INDEX": 0.25, "GLUCAGON": 120.0, "FIBRINOGEN": 3.8,
    },
    "Type 2 Diabetes (Severe)": {
        "GLUCOSE": 14.0, "INSULIN": 300.0, "IL6": 12.0, "CRP": 12.0,
        "INFLAM_INDEX": 0.40, "GLUCAGON": 160.0,
    },
    "COPD (Moderate)": {
        "NEUTROPHIL": 8.0, "IL6": 10.0, "IL17": 8.0, "CRP": 10.0,
        "WBC": 9.0, "INFLAM_INDEX": 0.30, "FIBRINOGEN": 4.0,
    },
    "COPD (Severe)": {
        "NEUTROPHIL": 12.0, "IL6": 20.0, "IL17": 15.0, "CRP": 20.0,
        "WBC": 12.0, "INFLAM_INDEX": 0.55,
    },
    "Chronic Kidney Disease": {
        "IL6": 6.0, "CRP": 7.0, "INFLAM_INDEX": 0.20,
        "NEUTROPHIL": 7.5, "FIBRINOGEN": 3.5, "PLATELET": 230.0,
    },
    "Arthritis": {
        "IL6": 25.0, "TNFa": 30.0, "IL1B": 8.0, "IL17": 20.0,
        "CRP": 25.0, "INFLAM_INDEX": 0.50, "WBC": 9.0,
        "FIBRINOGEN": 4.5,
    },
    "Depression": {
        "IL6": 4.5, "IL1B": 2.0, "CRP": 3.5, "CORTISOL": 550.0,
        "INFLAM_INDEX": 0.15, "ACTH": 35.0,
    },
    "Anxiety Disorder": {
        "CORTISOL": 580.0, "ACTH": 40.0, "CRH": 18.0,
        "IL6": 3.5, "INFLAM_INDEX": 0.10,
    },
    "Hypothyroidism": {
        "TSH": 8.0, "T4_FREE": 8.0, "T3_FREE": 2.5,
        "GLUCOSE": 5.5, "CHOLESTEROL": 6.8, "INFLAM_INDEX": 0.12,
    },
    "Bacterial Infection": {
        "PATHOGEN": 6.0,   # log₁₀ ~10⁶ CFU/mL
        "NEUTROPHIL": 14.0, "WBC": 15.0, "IL6": 40.0,
        "TNFa": 60.0, "IL1B": 15.0, "CRP": 80.0,
        "INFLAM_INDEX": 0.80, "FIBRINOGEN": 5.0,
    },
}


# ---------------------------------------------------------------------------
# BiologicalCascadeEngine
# ---------------------------------------------------------------------------

@dataclass
class CascadeInput:
    """Drug effects passed into the biological cascade per simulation step."""
    # Antimicrobial effect on pathogen (0 = none, 1 = complete bactericidal)
    antimicrobial_effect: float = 0.0
    # Anti-inflammatory effect (0–1): e.g. corticosteroids, NSAIDs fraction
    anti_inflammatory_effect: float = 0.0
    # Immunosuppressive effect (0–1): e.g. corticosteroids, methotrexate
    immunosuppressive_effect: float = 0.0
    # Specific receptor-mediated effects (fractional change from 0)
    cortisol_supplement: float = 0.0    # exogenous glucocorticoid dose
    thyroid_supplement: float = 0.0     # T4/T3 supplement
    insulin_sensitizer: float = 0.0     # metformin-like AMPK activation
    insulin_secretagogue: float = 0.0   # sulphonylurea-like effect
    beta_blocker_intensity: float = 0.0 # β1 blockade (0–1)
    ace_inhibition: float = 0.0         # ACE inhibition (0–1)
    serotonin_reuptake_block: float = 0.0  # SSRI effect
    # Direct cytokine modulation (negative = suppression)
    il6_modulation: float = 0.0
    tnfa_modulation: float = 0.0


@dataclass
class CascadeResult:
    """Output produced by BiologicalCascadeEngine.simulate()."""
    time_h: np.ndarray
    state_trajectories: np.ndarray   # shape (N_STATE, T)
    # Derived clinical labels
    wbc_series: np.ndarray
    crp_series: np.ndarray
    il6_series: np.ndarray
    tnfa_series: np.ndarray
    cortisol_series: np.ndarray
    glucose_series: np.ndarray
    tsh_series: np.ndarray
    platelets_series: np.ndarray
    pathogen_series: np.ndarray
    inflammation_index_series: np.ndarray
    # Peak / final summary
    peak_crp: float
    peak_il6: float
    peak_wbc: float
    final_pathogen: float
    disease_controlled: bool


class BiologicalCascadeEngine:
    """
    Simulates the systemic human biological response to drug exposure and
    disease states using a coupled ODE system.

    The 35-variable ODE system describes:
    1. Innate immune response (neutrophils, macrophages, NK cells, monocytes)
    2. Adaptive immune response (T-helper, cytotoxic T, B cells, IgG)
    3. Pro- and anti-inflammatory cytokine network (IL-1β, IL-6, TNF-α, IL-10, IL-4, IL-17, IFN-γ)
    4. Acute-phase proteins (CRP, fibrinogen)
    5. HPA axis (CRH → ACTH → cortisol) with circadian rhythm
    6. Thyroid axis (TSH → T4 → T3) with feedback
    7. Glucose-insulin-glucagon dynamics
    8. Blood counts (RBC, WBC, platelets, haemoglobin)
    9. Pathogen dynamics (for infectious disease)
    10. Composite inflammation index
    """

    # ── Kinetic constants (literature-derived approximations) ───────────────

    # Innate immune kinetics
    _K_N_PROD    = 0.10   # basal neutrophil production rate (cells/µL/h)
    _K_N_DEATH   = 0.025  # natural apoptosis rate (1/h, t½ ~28h)
    _K_N_RECRUIT = 2.50   # max IL-6-driven neutrophil mobilisation
    _K_N_ACT_M   = 0.30   # macrophage activation by TNFa+pathogens

    # Cytokine kinetics
    _K_IL6_PROD  = 0.80   # IL-6 production by activated macrophages (pg/mL/h)
    _K_IL6_CLR   = 0.15   # IL-6 clearance (1/h, t½ ~2-4h in vivo → ~4.6h)
    _K_TNF_PROD  = 1.20
    _K_TNF_CLR   = 0.20   # TNFa t½ ~1.5h in vivo
    _K_IL1B_PROD = 0.50
    _K_IL1B_CLR  = 0.18
    _K_IL10_PROD = 0.40   # anti-inflammatory IL-10 produced by M2 macrophages
    _K_IL10_CLR  = 0.12
    _K_IL17_PROD = 0.20   # IL-17 from activated Th17 cells
    _K_IL17_CLR  = 0.10

    # CRP kinetics (produced by hepatocytes stimulated by IL-6)
    _K_CRP_PROD  = 0.60   # mg/L per pg/mL IL-6 per hour (lag included via IL-6 dynamics)
    _K_CRP_CLR   = 0.036  # CRP t½ ~19h (0.693/19 = 0.036)
    _K_FIB_PROD  = 0.015
    _K_FIB_CLR   = 0.010

    # HPA axis kinetics
    _K_CRH_PROD  = 2.00   # basal CRH secretion
    _K_CRH_CLR   = 0.50   # CRH t½ ~1.4h
    _K_ACTH_PROD = 4.00
    _K_ACTH_CLR  = 0.25   # ACTH t½ ~10-15 min → simplify to 0.25 (1-compartment approx)
    _K_CORT_PROD = 20.0
    _K_CORT_CLR  = 0.10   # cortisol t½ ~70-90 min → ~0.08 /h

    # Thyroid axis (very slow)
    _K_TSH_PROD  = 0.020
    _K_TSH_CLR   = 0.008  # TSH t½ ~50-70 min, but biological action is hours-days
    _K_T4_PROD   = 0.030
    _K_T4_CLR    = 0.003  # T4 t½ ~7 days
    _K_T3_PROD   = 0.050
    _K_T3_CLR    = 0.020  # T3 t½ ~1 day

    # Glucose/insulin kinetics (simplified Bergman minimal model)
    _K_GLUC_BASAL = 5.0   # mmol/L target (fasting)
    _K_INS_CLR   = 0.10   # insulin clearance
    _K_GLC_INS   = 0.08   # glucose uptake per unit insulin
    _K_GCG_PROD  = 0.30   # glucagon production
    _K_GCG_CLR   = 0.20

    # Pathogen dynamics (infection)
    _K_PATH_GROWTH = 1.30  # pathogen doubling ~0.5h → growth rate ~1.4/h
    _K_PATH_KILL   = 0.40  # neutrophil killing rate per unit
    _K_PATH_CAP    = 10.0  # carrying capacity (log scale)

    # Adaptive immune (slower) – activation peaks day 7-14
    _K_TCELL_ACT  = 0.005  # T-cell clonal expansion rate (1/h)
    _K_BCELL_ACT  = 0.003
    _K_AB_PROD    = 0.010  # antibody from B cells
    _K_AB_CLR     = 0.001

    def __init__(self) -> None:
        pass

    # ── Public API ──────────────────────────────────────────────────────────

    def simulate(
        self,
        duration_hours: float,
        disease_name: str,
        disease_severity: float,
        cascade_input: CascadeInput,
        patient_age: float = 65.0,
        extra_baseline: Optional[Dict[str, float]] = None,
        time_resolution_h: float = 1.0,
    ) -> CascadeResult:
        """
        Run the biological cascade ODE system.

        Parameters
        ----------
        duration_hours  : total simulation time in hours
        disease_name    : disease name (used to select initial state)
        disease_severity: 0–1 scaling of disease initial state deviations
        cascade_input   : drug effect inputs
        patient_age     : patient age modifies immune response
        extra_baseline  : additional state overrides
        time_resolution_h : output resolution

        Returns
        -------
        CascadeResult with full state trajectories
        """
        # Build initial state
        disease_adj = {
            k: HEALTHY_BASELINE.get(k, 0.0)
               + (v - HEALTHY_BASELINE.get(k, 0.0)) * disease_severity
            for k, v in DISEASE_INITIAL_STATES.get(disease_name, {}).items()
        }
        if extra_baseline:
            disease_adj.update(extra_baseline)

        y0 = make_baseline_state(disease_adj)

        # Age scaling: immune senescence > 60y → reduce neutrophil reserve
        age_immune_scale = 1.0 - max(0.0, (patient_age - 60.0) / 100.0) * 0.25

        t_span = (0.0, duration_hours)
        t_eval = np.arange(0.0, duration_hours + time_resolution_h / 2, time_resolution_h)

        # Closure over fixed parameters
        ci = cascade_input

        def ode_system(t: float, y: np.ndarray) -> np.ndarray:
            return self._odes(t, y, ci, age_immune_scale)

        try:
            sol = solve_ivp(
                ode_system,
                t_span,
                y0,
                method="LSODA",
                t_eval=t_eval,
                rtol=1e-4,
                atol=1e-6,
                max_step=min(4.0, time_resolution_h * 2.0),
            )
            traj = np.clip(sol.y, 0.0, None)   # no negative concentrations
        except Exception as exc:
            logger.warning("BiologicalCascade ODE failed: %s — returning flat baseline", exc)
            n_t = len(t_eval)
            traj = np.tile(y0[:, None], (1, n_t))

        t_out = t_eval[:traj.shape[1]]

        # Extract key series
        crp_series   = traj[_IDX.CRP]
        il6_series   = traj[_IDX.IL6]
        tnfa_series  = traj[_IDX.TNFa]
        wbc_series   = traj[_IDX.WBC]
        cort_series  = traj[_IDX.CORTISOL]
        gluc_series  = traj[_IDX.GLUCOSE]
        tsh_series   = traj[_IDX.TSH]
        plt_series   = traj[_IDX.PLATELET]
        path_series  = traj[_IDX.PATHOGEN]
        idx_series   = traj[_IDX.INFLAM_INDEX]

        final_pathogen  = float(path_series[-1])
        disease_ctrl    = bool(final_pathogen < 0.5) if cascade_input.antimicrobial_effect > 0 else True

        return CascadeResult(
            time_h=t_out,
            state_trajectories=traj,
            wbc_series=wbc_series,
            crp_series=crp_series,
            il6_series=il6_series,
            tnfa_series=tnfa_series,
            cortisol_series=cort_series,
            glucose_series=gluc_series,
            tsh_series=tsh_series,
            platelets_series=plt_series,
            pathogen_series=path_series,
            inflammation_index_series=idx_series,
            peak_crp=float(np.max(crp_series)),
            peak_il6=float(np.max(il6_series)),
            peak_wbc=float(np.max(wbc_series)),
            final_pathogen=final_pathogen,
            disease_controlled=disease_ctrl,
        )

    # ── ODE system ──────────────────────────────────────────────────────────

    def _odes(
        self,
        t: float,
        y: np.ndarray,
        ci: CascadeInput,
        age_scale: float,
    ) -> np.ndarray:
        """
        Right-hand side of the full 35-variable ODE system.
        Evaluated at time t (hours) given state vector y.
        """
        dy = np.zeros(N_STATE)

        # ── Unpack state ────────────────────────────────────────────────────
        neut    = max(y[_IDX.NEUTROPHIL], 0.0)
        macro   = max(y[_IDX.MACROPHAGE], 0.0)
        nk      = max(y[_IDX.NK_CELL], 0.0)
        mono    = max(y[_IDX.MONOCYTE], 0.0)
        th      = max(y[_IDX.TH_CELL], 0.0)
        tc      = max(y[_IDX.TC_CELL], 0.0)
        bcell   = max(y[_IDX.B_CELL], 0.0)
        ab      = max(y[_IDX.ANTIBODY], 0.0)
        il1b    = max(y[_IDX.IL1B], 0.0)
        il6     = max(y[_IDX.IL6], 0.0)
        tnfa    = max(y[_IDX.TNFa], 0.0)
        il17    = max(y[_IDX.IL17], 0.0)
        ifng    = max(y[_IDX.IFNG], 0.0)
        il10    = max(y[_IDX.IL10], 0.0)
        il4     = max(y[_IDX.IL4], 0.0)
        crp     = max(y[_IDX.CRP], 0.0)
        fib     = max(y[_IDX.FIBRINOGEN], 0.0)
        crh     = max(y[_IDX.CRH], 0.0)
        acth    = max(y[_IDX.ACTH], 0.0)
        cort    = max(y[_IDX.CORTISOL], 0.0)
        tsh     = max(y[_IDX.TSH], 0.0)
        t4      = max(y[_IDX.T4_FREE], 0.0)
        t3      = max(y[_IDX.T3_FREE], 0.0)
        gluc    = max(y[_IDX.GLUCOSE], 0.0)
        ins     = max(y[_IDX.INSULIN], 0.0)
        gcg     = max(y[_IDX.GLUCAGON], 0.0)
        rbc     = max(y[_IDX.RBC], 0.0)
        hgb     = max(y[_IDX.HEMOGLOBIN], 0.0)
        plt     = max(y[_IDX.PLATELET], 0.0)
        wbc     = max(y[_IDX.WBC], 0.0)
        comp    = max(y[_IDX.COMPLEMENT], 0.0)
        ddimer  = max(y[_IDX.D_DIMER], 0.0)
        phase   = y[_IDX.CIRC_PHASE]
        pathogen= max(y[_IDX.PATHOGEN], 0.0)
        inflam  = max(y[_IDX.INFLAM_INDEX], 0.0)

        # ── Drug effect shorthands ──────────────────────────────────────────
        ab_drug    = float(np.clip(ci.anti_inflammatory_effect, 0.0, 1.0))
        immuno_sup = float(np.clip(ci.immunosuppressive_effect + ci.cortisol_supplement * 0.4, 0.0, 1.0))
        antimic    = float(np.clip(ci.antimicrobial_effect, 0.0, 1.0))
        ins_sens   = float(np.clip(ci.insulin_sensitizer, 0.0, 1.0))
        ins_secr   = float(np.clip(ci.insulin_secretagogue, 0.0, 1.0))
        cort_exog  = float(np.clip(ci.cortisol_supplement * 300.0, 0.0, 400.0))
        t4_exog    = float(np.clip(ci.thyroid_supplement * 10.0, 0.0, 15.0))

        # ── Anti-inflammatory brake (IL-10 + IL-4 + drug) ──────────────────
        anti_brake = il10 / (il10 + 8.0) + il4 / (il4 + 5.0) + ab_drug
        anti_brake = float(np.clip(anti_brake, 0.0, 1.0))

        # ── Circadian driver: CRH and cortisol follow 24-h rhythm ──────────
        # Cortisol peaks ~0800h (phase=π). Start phase at midnight (0).
        circ_freq = 2.0 * math.pi / 24.0               # radians/hour
        dy[_IDX.CIRC_PHASE] = circ_freq                # pure phase advance
        circ = math.cos(phase - math.pi) * 0.5 + 0.5  # 0 at nadir, 1 at peak

        # ── 1. Pathogen dynamics ─────────────────────────────────────────────
        if pathogen > 0:
            path_growth = self._K_PATH_GROWTH * pathogen * (1.0 - pathogen / self._K_PATH_CAP)
            path_kill_innate = self._K_PATH_KILL * (neut + nk * 0.5) * pathogen / (pathogen + 2.0)
            path_kill_adaptive = 0.02 * tc * pathogen * (1 if t > 168 else 0)  # adaptive kicks in after ~7 days
            path_kill_ab = 0.01 * ab * pathogen if t > 336 else 0.0           # antibodies from ~day 14
            path_kill_drug = antimic * 2.5 * pathogen                          # drug bactericidal
            dy[_IDX.PATHOGEN] = path_growth - path_kill_innate - path_kill_adaptive - path_kill_ab - path_kill_drug
        else:
            dy[_IDX.PATHOGEN] = 0.0

        # ── 2. Innate immune cells ───────────────────────────────────────────
        # Neutrophils: basally produced by bone marrow, recruited by IL-6/G-CSF during infection
        n_recruit_signal = (il6 / (il6 + 15.0) + pathogen / (pathogen + 3.0)) * 5.0
        n_immuno_inhibit = immuno_sup * 0.60  # corticosteroids suppress recruitment
        dy[_IDX.NEUTROPHIL] = (
            self._K_N_PROD * age_scale * (1.0 - n_immuno_inhibit) * (1.0 + n_recruit_signal)
            - self._K_N_DEATH * neut
        )

        # Macrophage activation: activated by TNFa, pathogen; suppressed by IL-10 + drug
        macro_act_drive = (tnfa / (tnfa + 20.0)) * (1.0 + pathogen / (pathogen + 2.0))
        dy[_IDX.MACROPHAGE] = (
            self._K_N_ACT_M * macro_act_drive * (1.0 - anti_brake * 0.7) * (1.0 + inflam)
            - 0.05 * macro
        )

        # NK cells: activated by IL-12 (approximated by IL-6 proxy) and IL-15
        dy[_IDX.NK_CELL] = (
            0.010 * (il6 / (il6 + 20.0)) * (1.0 - immuno_sup * 0.5)
            - 0.012 * nk
            + 0.002
        )

        # Monocytes
        dy[_IDX.MONOCYTE] = 0.05 * (0.50 - mono) - 0.02 * mono * (inflam)

        # ── 3. Cytokine network ──────────────────────────────────────────────
        # IL-1β — produced by activated macrophages and neutrophils during infection/inflammation
        il1b_drive = macro * (1.0 + pathogen / (pathogen + 1.0)) * (1.0 + inflam)
        dy[_IDX.IL1B] = (
            self._K_IL1B_PROD * il1b_drive / (1.0 + anti_brake * 2.0)
            - self._K_IL1B_CLR * il1b
            + HEALTHY_BASELINE["IL1B"] * self._K_IL1B_CLR  # maintain baseline
            + ci.il6_modulation * 0.3  # repurpose il6_modulation as general cytokine
        )

        # IL-6 — key pleotropic cytokine driving acute phase response
        il6_drive = (macro + neut * 0.3) * (1.0 + pathogen / (pathogen + 1.0)) * (1.0 + inflam * 2.0)
        dy[_IDX.IL6] = (
            self._K_IL6_PROD * il6_drive / (1.0 + anti_brake * 2.5)
            - self._K_IL6_CLR * il6
            + HEALTHY_BASELINE["IL6"] * self._K_IL6_CLR
            + ci.il6_modulation * 2.0
        )

        # TNF-α — predominantly from macrophages, early pro-inflammatory signal
        tnfa_drive = macro * (1.0 + pathogen / (pathogen + 0.5)) * (1.0 + inflam * 1.5)
        dy[_IDX.TNFa] = (
            self._K_TNF_PROD * tnfa_drive / (1.0 + anti_brake * 3.0)
            - self._K_TNF_CLR * tnfa
            + HEALTHY_BASELINE["TNFa"] * self._K_TNF_CLR
            + ci.tnfa_modulation * 2.0
        )

        # IL-17 — from Th17 cells; chronically elevated in autoimmune/COPD
        il17_drive = th * inflam * (1.0 - immuno_sup * 0.7)
        dy[_IDX.IL17] = (
            self._K_IL17_PROD * il17_drive
            - self._K_IL17_CLR * il17
            + HEALTHY_BASELINE["IL17"] * self._K_IL17_CLR
        )

        # IFN-γ — from NK cells and cytotoxic T; primarily anti-viral
        dy[_IDX.IFNG] = (
            0.10 * nk * (pathogen / (pathogen + 2.0))
            - 0.15 * ifng
            + HEALTHY_BASELINE["IFNG"] * 0.15
        )

        # IL-10 — anti-inflammatory; produced by monocytes/macrophages as counter-regulation
        il10_drive = (macro * 0.5 + il6 / (il6 + 30.0)) * (1.0 + inflam * 0.5)
        dy[_IDX.IL10] = (
            self._K_IL10_PROD * il10_drive
            - self._K_IL10_CLR * il10
            + HEALTHY_BASELINE["IL10"] * self._K_IL10_CLR
            + ab_drug * 0.5  # anti-inflammatory drugs boost IL-10 pathway
        )

        # IL-4 — Th2 cytokine, rises with corticosteroids
        dy[_IDX.IL4] = (
            0.05 * (1.0 + immuno_sup * 0.5)
            - 0.08 * il4
            + HEALTHY_BASELINE["IL4"] * 0.08
        )

        # ── 4. Acute-phase proteins (hepatocyte-derived, driven by IL-6) ────
        # CRP: synthesised by liver hepatocytes; peaks 24-48h after infection
        # IL-6 is the primary driver; there's a 6-8h lag (captured by IL-6 kinetics)
        crp_baseline = HEALTHY_BASELINE["CRP"]
        dy[_IDX.CRP] = (
            self._K_CRP_PROD * (il6 / (il6 + 5.0)) * (1.0 + inflam)
            - self._K_CRP_CLR * crp
            + crp_baseline * self._K_CRP_CLR  # baseline production
        )

        # Fibrinogen: acute-phase reactant; slower kinetics than CRP
        dy[_IDX.FIBRINOGEN] = (
            self._K_FIB_PROD * (il6 / (il6 + 10.0)) * (1.0 + inflam * 0.5)
            - self._K_FIB_CLR * fib
            + HEALTHY_BASELINE["FIBRINOGEN"] * self._K_FIB_CLR
        )

        # ── 5. HPA axis (Hypothalamic-Pituitary-Adrenal) ────────────────────
        # CRH secreted by hypothalamus; driven by stress + IL-1β/IL-6; inhibited by cortisol
        crh_baseline = HEALTHY_BASELINE["CRH"]
        crh_drive = (1.0 + inflam * 0.5 + il1b / (il1b + 5.0)) * circ
        cortisol_inhibit = cort / (cort + 500.0)   # negative feedback
        dy[_IDX.CRH] = (
            self._K_CRH_PROD * crh_drive * (1.0 - cortisol_inhibit * 0.6)
            - self._K_CRH_CLR * crh
            + crh_baseline * self._K_CRH_CLR * 0.5
        )

        # ACTH: from pituitary; stimulated by CRH; inhibited by cortisol
        dy[_IDX.ACTH] = (
            self._K_ACTH_PROD * crh / (crh + 8.0) * (1.0 - cortisol_inhibit * 0.7)
            - self._K_ACTH_CLR * acth
            + HEALTHY_BASELINE["ACTH"] * self._K_ACTH_CLR * 0.3
        )

        # Cortisol: from adrenal cortex; stimulated by ACTH; exogenous supplement possible
        # Half-life ~70-90 minutes; pulsatile but model as continuous
        dy[_IDX.CORTISOL] = (
            self._K_CORT_PROD * acth / (acth + 15.0) * circ
            - self._K_CORT_CLR * cort
            + HEALTHY_BASELINE["CORTISOL"] * self._K_CORT_CLR * 0.20
            + cort_exog  # exogenous glucocorticoid (e.g. prednisone)
        )

        # ── 6. Thyroid axis (very slow; timescale days-weeks) ───────────────
        # TSH: from pituitary; inhibited by T4/T3 (negative feedback)
        t4t3_feedback = (t4 / HEALTHY_BASELINE["T4_FREE"] + t3 / HEALTHY_BASELINE["T3_FREE"]) / 2.0
        dy[_IDX.TSH] = (
            self._K_TSH_PROD * (1.0 / (t4t3_feedback + 0.1)) * (1.0 - immuno_sup * 0.2)
            - self._K_TSH_CLR * tsh
            + HEALTHY_BASELINE["TSH"] * self._K_TSH_CLR * 0.5
        )

        # T4: produced by thyroid gland (stimulated by TSH); t½ ~7 days
        dy[_IDX.T4_FREE] = (
            self._K_T4_PROD * tsh / (tsh + 1.0)
            - self._K_T4_CLR * t4
            + HEALTHY_BASELINE["T4_FREE"] * self._K_T4_CLR * 0.5
            + t4_exog  # levothyroxine supplement
        )

        # T3: converted from T4 (mainly) and directly secreted (small fraction)
        t4_to_t3_conversion = 0.035 * t4  # ~30% of circulating T3 from conversion
        dy[_IDX.T3_FREE] = (
            self._K_T3_PROD * (tsh / (tsh + 1.0)) * 0.3  # direct secretion
            + t4_to_t3_conversion
            - self._K_T3_CLR * t3
            + HEALTHY_BASELINE["T3_FREE"] * self._K_T3_CLR * 0.3
        )

        # ── 7. Glucose–insulin dynamics (simplified Bergman minimal model) ──
        # Glucose: rises from glucagon, GI absorption; falls via insulin action
        # Chronic inflammation → insulin resistance (IL-6, TNFa → IRS-1 inhibition)
        insulin_resistance = 1.0 + (inflam * 0.5 + tnfa / (tnfa + 50.0) * 0.3) + immuno_sup * 0.8
        basal_hep_glucose = 0.05   # hepatic gluconeogenesis base rate

        dy[_IDX.GLUCOSE] = (
            gcg / (gcg + 40.0) * 0.08 + basal_hep_glucose   # glucagon + gluconeogenesis
            - self._K_GLC_INS * (ins / 80.0) / insulin_resistance * gluc  # insulin-mediated uptake
            + (gluc - self._K_GLUC_BASAL) * (-0.04)           # passive homeostasis
        )

        # Insulin: secreted by pancreatic β cells in response to glucose
        gluc_stim = max(0.0, gluc - 4.0) / 3.0   # secretion starts when glucose > 4 mmol/L
        dy[_IDX.INSULIN] = (
            5.0 * gluc_stim * (1.0 + ins_secr * 3.0)   # glucose-stimulated + sulphonylurea
            - self._K_INS_CLR * ins
            - ins * immuno_sup * 0.10                    # steroids suppress insulin secretion
        )

        # Glucagon: secreted when glucose low; suppressed by insulin and glucose
        dy[_IDX.GLUCAGON] = (
            self._K_GCG_PROD * (1.0 / (gluc / 4.5 + 0.5))   # hypoglycemia → glucagon
            - self._K_GCG_CLR * gcg
            + HEALTHY_BASELINE["GLUCAGON"] * self._K_GCG_CLR * 0.3
        )

        # ── 8. Blood counts ──────────────────────────────────────────────────
        # RBC: produced by EPO (not modelled explicitly); slow turnover (t½ 120 days = 2880h)
        rbc_baseline = HEALTHY_BASELINE["RBC"]
        dy[_IDX.RBC] = (
            0.00024 * (rbc_baseline - rbc)  # slow homeostasis
            - 0.00003 * inflam * rbc        # anaemia of chronic disease
        )
        dy[_IDX.HEMOGLOBIN] = (
            0.00024 * (HEALTHY_BASELINE["HEMOGLOBIN"] - hgb)
            - 0.00003 * inflam * hgb
        )

        # Platelets: produced by megakaryocytes (t½ ~10 days = 240h)
        dy[_IDX.PLATELET] = (
            0.003 * (HEALTHY_BASELINE["PLATELET"] - plt)
            + 0.001 * il6 / (il6 + 20.0) * 50  # thrombocytosis in inflammation
            - 0.003 * plt * (immuno_sup * 0.3)  # steroid-mediated demargination
        )

        # WBC: tracks neutrophil + lymphocyte dynamics
        wbc_target = neut + (th + tc + bcell) * 10.0 + mono * 5.0 + 1.0
        dy[_IDX.WBC] = 0.15 * (wbc_target - wbc)   # WBC tracks composite of cell counts

        # ── 9. Complement and D-dimer ────────────────────────────────────────
        dy[_IDX.COMPLEMENT] = (
            0.10 * (pathogen / (pathogen + 1.0) + inflam * 0.3)
            - 0.20 * comp
            + HEALTHY_BASELINE["COMPLEMENT"] * 0.20
        )
        dy[_IDX.D_DIMER] = (
            0.005 * (inflam + pathogen / (pathogen + 2.0))
            - 0.04 * ddimer
            + HEALTHY_BASELINE["D_DIMER"] * 0.04
        )

        # ── 10. Adaptive immune ──────────────────────────────────────────────
        # T-cell expansion: peaks day 7-14 post-antigen exposure
        ag_lag = max(0.0, 1.0 - math.exp(-max(0.0, t - 72.0) / 96.0))  # activates after ~3 days
        dy[_IDX.TH_CELL] = (
            self._K_TCELL_ACT * pathogen * (1.0 - immuno_sup * 0.8) * ag_lag * th
            - 0.002 * th
            + 0.001  # naïve T-cell replenishment
        )
        dy[_IDX.TC_CELL] = (
            self._K_TCELL_ACT * pathogen * (1.0 - immuno_sup * 0.8) * ag_lag * tc
            - 0.002 * tc + 0.001
        )
        dy[_IDX.B_CELL] = (
            self._K_BCELL_ACT * pathogen * ag_lag * (1.0 - immuno_sup * 0.7) * bcell
            - 0.003 * bcell + 0.001
        )
        dy[_IDX.ANTIBODY] = (
            self._K_AB_PROD * bcell * ag_lag
            - self._K_AB_CLR * ab
            + HEALTHY_BASELINE["ANTIBODY"] * self._K_AB_CLR  # baseline IgG maintenance
        )

        # ── 11. Composite inflammation index ────────────────────────────────
        # Weighted composite of CRP, IL-6, TNFa, and pathogen load
        inflam_target = float(np.clip(
            (crp / 100.0) * 0.3
            + (il6 / 100.0) * 0.3
            + (tnfa / 100.0) * 0.2
            + (pathogen / 10.0) * 0.2,
            0.0, 1.0
        ))
        dy[_IDX.INFLAM_INDEX] = 0.10 * (inflam_target - inflam)

        return dy


# ---------------------------------------------------------------------------
# Helper: build drug-specific CascadeInput from drug name and concentration
# ---------------------------------------------------------------------------

# Map drug names to cascade effects — grounded in mechanism of action
_DRUG_CASCADE_PARAMS: Dict[str, Dict[str, float]] = {
    "Acetaminophen":   {"anti_inflammatory_effect": 0.15},
    "Ibuprofen":       {"anti_inflammatory_effect": 0.55, "tnfa_modulation": -1.0},
    "Naproxen":        {"anti_inflammatory_effect": 0.50, "tnfa_modulation": -0.8},
    "Lisinopril":      {"ace_inhibition": 0.85, "il6_modulation": -0.5},
    "Metoprolol":      {"beta_blocker_intensity": 0.75, "anti_inflammatory_effect": 0.05},
    "Atorvastatin":    {"anti_inflammatory_effect": 0.20, "il6_modulation": -1.5},
    "Amlodipine":      {"anti_inflammatory_effect": 0.05},
    "Metformin":       {"insulin_sensitizer": 0.65, "anti_inflammatory_effect": 0.10},
    "Glipizide":       {"insulin_secretagogue": 0.70},
    "Amoxicillin":     {"antimicrobial_effect": 0.85},
    "Ciprofloxacin":   {"antimicrobial_effect": 0.92},
    "Sertraline":      {"serotonin_reuptake_block": 0.90, "anti_inflammatory_effect": 0.08},
    "Lorazepam":       {"anti_inflammatory_effect": 0.03},
    "Levothyroxine":   {"thyroid_supplement": 0.90},
    "Prednisone":      {
        "anti_inflammatory_effect": 0.80,
        "immunosuppressive_effect": 0.72,
        "cortisol_supplement": 0.85,
        "il6_modulation": -5.0,
        "tnfa_modulation": -4.0,
    },
}

# Override module-level constants from data file (writes file on first run)
_cascade_cfg = _load_or_seed_sim_json("biological_cascade_config.json", {
    "healthy_baseline": HEALTHY_BASELINE,
    "disease_initial_states": DISEASE_INITIAL_STATES,
    "drug_cascade_params": _DRUG_CASCADE_PARAMS,
})
HEALTHY_BASELINE       = _cascade_cfg["healthy_baseline"]
DISEASE_INITIAL_STATES = _cascade_cfg["disease_initial_states"]
_DRUG_CASCADE_PARAMS   = _cascade_cfg["drug_cascade_params"]


def build_cascade_input_for_drug(
    drug_name: str,
    plasma_conc: float,
    therapeutic_window_hi: float,
) -> CascadeInput:
    """
    Build a CascadeInput for a drug given its current plasma concentration.
    Scales effect by receptor-occupancy Hill equation: E = Emax * C / (EC50 + C)
    where EC50 is approximated as half the upper therapeutic window.
    """
    params = _DRUG_CASCADE_PARAMS.get(drug_name, {})
    if not params:
        return CascadeInput()

    # Receptor occupancy (0–1) based on concentration relative to EC50
    ec50 = max(therapeutic_window_hi * 0.5, 1e-6)
    occupancy = float(np.clip(plasma_conc / (ec50 + plasma_conc + 1e-12), 0.0, 1.0))

    kwargs = {k: v * occupancy for k, v in params.items() if not k.endswith("_modulation")}
    for k, v in params.items():
        if k.endswith("_modulation"):
            kwargs[k] = v * occupancy  # can be negative

    return CascadeInput(**{k: v for k, v in kwargs.items() if hasattr(CascadeInput, k)
                           or k in CascadeInput.__dataclass_fields__})
