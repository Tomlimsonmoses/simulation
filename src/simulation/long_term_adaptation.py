"""
Long-Term Adaptation Engine
============================
Models pharmacological adaptations that develop over days, weeks, and months
of continuous drug therapy.

Mechanisms implemented (all ODE/kinetic-based, no random number generation)
---------------------------------------------------------------------------

1. CYP Enzyme Induction / Inhibition
   CYP3A4, CYP2D6, CYP2C9, CYP2C19 activity changes drug clearance.
   • Induction: pregnane X receptor (PXR) and constitutive androstane receptor (CAR)
     activation → mRNA ↑ → protein synthesis ↑ (onset: days; peak: 1-2 weeks)
   • Inhibition: direct competitive or mechanism-based (immediate or irreversible)
   • Returns effective clearance multiplier over time

2. Receptor Downregulation / Upregulation
   Prolonged agonist / antagonist exposure changes receptor density and sensitivity.
   Drug classes affected:
   • β-adrenergic (beta-blockers): up-regulation on withdrawal
   • GABA-A (benzodiazepines): subunit composition changes → tolerance (weeks)
   • SERT (SSRIs): gradual downregulation of 5-HT1A autoreceptors (weeks)
   • Glucocorticoid receptor (corticosteroids): downregulation of GR (weeks-months)
   • Adrenal suppression (corticosteroids): ACTH ↓ → adrenal atrophy (months)

3. HPA Axis Suppression (Corticosteroids)
   Exogenous glucocorticoid → negative feedback on CRH/ACTH → adrenal atrophy
   • After stopping corticosteroids: adrenal insufficiency can persist for months
   • Model: adrenal gland volume → cortisol secretory reserve

4. Microbiome Disruption (Antibiotics)
   • Bactericidal effect on gut flora (broad vs. narrow spectrum)
   • Gut diversity index declines within 24-48h
   • Clostridium difficile bloom risk if treated with broad-spectrum antibiotics
   • Recovery: Lactobacillus and Bifidobacterium repopulate over 2-4 weeks
   • Some species altered for months post-treatment

5. Thyroid Hormone Adaptation (Levothyroxine)
   • Dose adjustment tracked: TSH takes 6-8 weeks to stabilise after dose change
   • Cardiovascular effects (HR, BP) adapt with T3 levels

6. Serotonergic Neuroplasticity (SSRIs)
   • 5-HT1A autoreceptor desensitisation: 2-4 weeks
   • Hippocampal neurogenesis (BDNF ↑): 4-12 weeks
   • Full antidepressant effect realised at 4-8 weeks
   • Discontinuation syndrome: fast offset with short t½ SSRIs

7. Statin Long-Term Cardiovascular Effects
   • LDL reduction: gradual (weeks) via HMG-CoA inhibition → hepatic LDL-R upregulation
   • Plaque stabilisation: months
   • Myopathy risk: accumulates with CoQ10 depletion (see organ_damage_model.py)

8. Insulin Resistance Adaptation (Corticosteroids, Antipsychotics)
   • Glucocorticoids → hepatic gluconeogenesis ↑ + peripheral insulin resistance
   • metabolic syndrome risk builds over months

9. Tolerance and Dependence Tracking
   • Benzodiazepines: physical dependence after 2-4 weeks of regular use
   • Opioids: tolerance after days (model included for generality)

All time constants are grounded in published pharmacology literature.
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
# State vector indices (28 adaptation variables)
# ---------------------------------------------------------------------------
class _ADP:
    # ── CYP enzyme activity (normalised; 1 = baseline) ───────────────────
    CYP3A4       = 0   # major drug-metabolising enzyme (substrate ~50% of drugs)
    CYP2D6       = 1   # ~25% of drugs; genetically polymorphic (not modelled here)
    CYP2C9       = 2   # NSAIDs, warfarin, glipizide
    CYP2C19      = 3   # PPIs, clopidogrel, some SSRIs
    PGP          = 4   # P-glycoprotein efflux transporter

    # ── Receptor density / sensitivity ────────────────────────────────────
    BETA1_R      = 5   # β1-adrenergic receptor (normalised; 1 = baseline density)
    GABA_A_R     = 6   # GABA-A receptor sensitivity (downregulated by chronic BZD)
    SERT         = 7   # serotonin transporter (downregulated by chronic SSRI)
    HT1A_AUTO    = 8   # 5-HT1A autoreceptor density (desensitises with SSRI)
    GR           = 9   # glucocorticoid receptor (downregulates with chronic steroids)

    # ── HPA axis / adrenal function ───────────────────────────────────────
    ADRENAL_VOL  = 10  # adrenal gland volume (normalised; 1 = normal, atrophies with steroids)
    CORTISOL_RES = 11  # adrenal cortisol secretory reserve (normalised)

    # ── Microbiome diversity ──────────────────────────────────────────────
    MICRO_DIV    = 12  # gut microbiome diversity index (Shannon diversity, normalised 0–1)
    LACTOBACILLUS= 13  # Lactobacillus spp. abundance (normalised)
    BIFIDOBACT   = 14  # Bifidobacterium abundance (normalised)
    CDIFF_RISK   = 15  # C. difficile bloom risk (0–1; rises with diversity collapse)

    # ── Neuroplasticity (SSRIs) ───────────────────────────────────────────
    BDNF         = 16  # BDNF mRNA level (normalised; ↑ with SSRI, weeks-months)
    HIPPO_NEURO  = 17  # hippocampal neurogenesis index (0–1; clinical antidepressant proxy)
    SEROTONIN_SYN= 18  # synaptic 5-HT availability (relative, proxy for antidepressant effect)

    # ── Lipid metabolism ─────────────────────────────────────────────────
    LDL          = 19  # LDL-cholesterol (mmol/L); normal < 3.0
    HDL          = 20  # HDL-cholesterol (mmol/L); normal > 1.0
    LDL_R_LIVER  = 21  # hepatic LDL receptor density (normalised; ↑ with statins)

    # ── Metabolic / glucose adaptation ───────────────────────────────────
    INSULIN_RESIST = 22  # insulin resistance index (1 = normal, ↑ with steroids)
    HEPATIC_GLUC   = 23  # hepatic gluconeogenesis rate multiplier

    # ── Benzodiazepine dependence ─────────────────────────────────────────
    BZD_TOLERANCE  = 24  # tolerance (0 = naive, 1 = fully tolerant; ↑ weeks)
    BZD_DEPEND     = 25  # physical dependence index (0–1; withdrawal risk)

    # ── Thyroid adaptation ────────────────────────────────────────────────
    THYROID_RESERVE= 26  # endogenous thyroid reserve (1 = full, ↓ with prolonged T4 supplement)

    # ── Bone remodeling balance ───────────────────────────────────────────
    RANK_RANKL     = 27  # RANK/RANKL balance (1 = normal; ↓ with steroids → more resorption)

N_ADP = 28


ADP_NAMES = [
    "CYP3A4","CYP2D6","CYP2C9","CYP2C19","PGP",
    "BETA1_R","GABA_A_R","SERT","HT1A_AUTO","GR",
    "ADRENAL_VOL","CORTISOL_RES",
    "MICRO_DIV","LACTOBACILLUS","BIFIDOBACT","CDIFF_RISK",
    "BDNF","HIPPO_NEURO","SEROTONIN_SYN",
    "LDL","HDL","LDL_R_LIVER",
    "INSULIN_RESIST","HEPATIC_GLUC",
    "BZD_TOLERANCE","BZD_DEPEND",
    "THYROID_RESERVE","RANK_RANKL",
]


# ---------------------------------------------------------------------------
# Baseline adaptation state (all normalised where possible)
# ---------------------------------------------------------------------------
ADAPTATION_BASELINE: Dict[str, float] = {
    "CYP3A4":        1.0,
    "CYP2D6":        1.0,
    "CYP2C9":        1.0,
    "CYP2C19":       1.0,
    "PGP":           1.0,
    "BETA1_R":       1.0,
    "GABA_A_R":      1.0,
    "SERT":          1.0,
    "HT1A_AUTO":     1.0,
    "GR":            1.0,
    "ADRENAL_VOL":   1.0,
    "CORTISOL_RES":  1.0,
    "MICRO_DIV":     1.0,    # Shannon diversity ~4 in healthy adult
    "LACTOBACILLUS": 1.0,
    "BIFIDOBACT":    1.0,
    "CDIFF_RISK":    0.05,   # low baseline C. diff risk
    "BDNF":          1.0,
    "HIPPO_NEURO":   0.5,   # half-maximal at baseline
    "SEROTONIN_SYN": 1.0,
    "LDL":           3.5,   # mmol/L (borderline elevated typical Western diet)
    "HDL":           1.3,
    "LDL_R_LIVER":   1.0,
    "INSULIN_RESIST":1.0,
    "HEPATIC_GLUC":  1.0,
    "BZD_TOLERANCE": 0.0,
    "BZD_DEPEND":    0.0,
    "THYROID_RESERVE":1.0,
    "RANK_RANKL":    1.0,
}


def make_adaptation_baseline(
    disease_name: str = "",
    disease_severity: float = 0.0,
    patient_age: float = 65.0,
) -> np.ndarray:
    y0 = np.array([ADAPTATION_BASELINE[n] for n in ADP_NAMES], dtype=float)

    # Age effects: CYP3A4 activity decreases with age (≥65 → ~20% reduction)
    age_cyp_scale = max(0.6, 1.0 - max(0.0, patient_age - 60.0) * 0.008)
    y0[_ADP.CYP3A4]  *= age_cyp_scale
    y0[_ADP.CYP2D6]  *= age_cyp_scale
    y0[_ADP.MICRO_DIV] *= max(0.7, 1.0 - max(0.0, patient_age - 60.0) * 0.005)

    # Disease adjustments
    if disease_name == "Type 2 Diabetes (Moderate)" or disease_name == "Type 2 Diabetes (Severe)":
        scale = disease_severity
        y0[_ADP.INSULIN_RESIST] = 1.0 + 2.0 * scale
        y0[_ADP.HEPATIC_GLUC]   = 1.0 + 1.5 * scale
        y0[_ADP.LDL]            = 3.5 + 1.0 * scale
        y0[_ADP.HDL]            = 1.3 - 0.3 * scale

    if "Hypothyroidism" in disease_name:
        y0[_ADP.THYROID_RESERVE] = max(0.2, 1.0 - disease_severity * 0.7)

    return y0


# ---------------------------------------------------------------------------
# Drug effect profiles on adaptation systems
# ---------------------------------------------------------------------------

@dataclass
class AdaptationDrugEffect:
    """Drug-specific effects on adaptation variables (rates per mg/L · h)."""
    # CYP enzyme effects (positive = induction; negative = inhibition)
    cyp3a4_effect: float = 0.0   # rate constant for CYP3A4 change per mg/L
    cyp2d6_effect: float = 0.0
    cyp2c9_effect: float = 0.0
    cyp2c19_effect: float = 0.0
    pgp_effect: float = 0.0

    # Receptor effects (negative = downregulation; positive = upregulation of alternative)
    beta1_r_effect: float = 0.0
    gaba_a_effect: float = 0.0
    sert_effect: float = 0.0
    ht1a_auto_effect: float = 0.0
    gr_effect: float = 0.0

    # Adrenal suppression (per mg/L · h; corticosteroids only)
    adrenal_suppression: float = 0.0

    # Microbiome disruption (per mg/L · h)
    micro_disruption: float = 0.0
    broad_spectrum: bool = False  # True → higher C.diff risk

    # Neuroplasticity (SSRIs; slow positive effect)
    bdnf_induction: float = 0.0
    serotonin_syn_effect: float = 0.0

    # Metabolic effects
    ldl_reduction: float = 0.0   # statin → ↓ LDL via LDL-R upregulation
    hdl_increase: float = 0.0
    insulin_resist_change: float = 0.0  # steroids → insulin resistance ↑
    hepatic_gluc_change: float = 0.0

    # BZD tolerance
    bzd_tolerance_rate: float = 0.0
    bzd_dependence_rate: float = 0.0

    # Thyroid adaptation
    thyroid_suppress: float = 0.0

    # Bone remodeling
    rankl_suppress: float = 0.0  # steroids → suppress RANKL balance


# Drug adaptation profiles
ADAPTATION_DRUG_EFFECTS: Dict[str, AdaptationDrugEffect] = {
    "Acetaminophen": AdaptationDrugEffect(),

    "Ibuprofen": AdaptationDrugEffect(
        cyp2c9_effect = -0.0003,  # mild CYP2C9 inhibition at high doses
    ),
    "Naproxen": AdaptationDrugEffect(
        cyp2c9_effect = -0.0002,
    ),
    "Lisinopril": AdaptationDrugEffect(
        # Gradual cardiovascular adaptation
        insulin_resist_change = -0.0001,  # slight insulin sensitisation over time
    ),
    "Metoprolol": AdaptationDrugEffect(
        beta1_r_effect = 0.00015,  # β1 upregulation with chronic blockade (withdrawal risk)
    ),
    "Atorvastatin": AdaptationDrugEffect(
        ldl_reduction  = 0.90,     # max LDL-R upregulation (Hill occ-based); 40mg → ~40% LDL cut
        hdl_increase   = 0.05,     # ~5% HDL increase at therapeutic dose
        cyp3a4_effect  = -0.0001,  # weak CYP3A4 inhibition (substrate, not major inhibitor)
    ),
    "Amlodipine": AdaptationDrugEffect(
        cyp3a4_effect = -0.0002,   # CYP3A4 substrate; weak competitive inhibition
    ),
    "Metformin": AdaptationDrugEffect(
        insulin_resist_change = -0.0004,  # gradually improves insulin sensitivity
        hepatic_gluc_change   = -0.0003,  # ↓ hepatic gluconeogenesis
        ldl_reduction         = 0.0001,
    ),
    "Glipizide": AdaptationDrugEffect(
        insulin_resist_change = 0.00002,  # slight weight gain → insulin resistance ↑ long-term
    ),
    "Amoxicillin": AdaptationDrugEffect(
        micro_disruption = 0.010,   # significant but recoverable
        broad_spectrum   = False,   # beta-lactam → mainly gram-positive
    ),
    "Ciprofloxacin": AdaptationDrugEffect(
        micro_disruption = 0.015,   # fluoroquinolone → broader disruption
        broad_spectrum   = True,
        cyp3a4_effect    = -0.0005, # weak CYP3A4 inhibition
    ),
    "Sertraline": AdaptationDrugEffect(
        sert_effect       = -0.0002,  # gradual SERT downregulation
        ht1a_auto_effect  = -0.0003,  # 5-HT1A autoreceptor desensitises
        serotonin_syn_effect = 0.0004,
        bdnf_induction     = 0.0002,
    ),
    "Lorazepam": AdaptationDrugEffect(
        gaba_a_effect       = -0.0005,  # GABA-A receptor downregulation (tolerance)
        bzd_tolerance_rate  = 0.00010,
        bzd_dependence_rate = 0.00005,
    ),
    "Levothyroxine": AdaptationDrugEffect(
        thyroid_suppress = 0.0002,  # exogenous T4 suppresses endogenous thyroid
    ),
    "Prednisone": AdaptationDrugEffect(
        gr_effect            = -0.0002,  # GR downregulation with chronic use
        adrenal_suppression  = 0.0003,
        insulin_resist_change = 0.0006,
        hepatic_gluc_change   = 0.0004,
        rankl_suppress        = 0.0004,
        cyp3a4_effect         = 0.0001,  # mild CYP3A4 induction
    ),
}


# Override module-level constants from data file (writes file on first run)
def _adaptation_drug_effects_as_dict() -> dict:
    import dataclasses
    return {k: dataclasses.asdict(v) for k, v in ADAPTATION_DRUG_EFFECTS.items()}


_adaptation_cfg = _load_or_seed_sim_json("long_term_adaptation_config.json", {
    "adaptation_baseline": ADAPTATION_BASELINE,
    "drug_effects": _adaptation_drug_effects_as_dict(),
})
ADAPTATION_BASELINE = _adaptation_cfg["adaptation_baseline"]
ADAPTATION_DRUG_EFFECTS = {
    k: AdaptationDrugEffect(**v)
    for k, v in _adaptation_cfg["drug_effects"].items()
}


# ---------------------------------------------------------------------------
# AdaptationEngine
# ---------------------------------------------------------------------------

class LongTermAdaptationEngine:
    """
    Simulates long-term pharmacological adaptations using a 28-variable ODE system.

    Returns time-series of adaptation state variables, which can be used to:
    - Adjust effective drug clearance (CYP enzyme activity changes)
    - Determine therapeutic onset timescales (receptor adaptation)
    - Track microbiome disruption and recovery
    - Show metabolic consequences (LDL, insulin resistance)
    - Flag tolerance and dependence development
    - Compute cough/withdrawal/bone-loss risks

    Key timescales:
    - CYP induction: days-weeks (half-time ~3-7 days for rifampicin-like induction)
    - Receptor downregulation: days-weeks
    - Microbiome recovery: 2-4 weeks
    - Neuroplasticity (BDNF/neurogenesis): 4-12 weeks
    - LDL reduction (statins): weeks (LDL-R upregulation)
    - Adrenal atrophy: months
    - Bone metabolism: months-years (BMD changes)
    """

    def simulate(
        self,
        duration_hours: float,
        drug_names: List[str],
        plasma_conc_series: Dict[str, np.ndarray],
        time_h: np.ndarray,
        disease_name: str = "",
        disease_severity: float = 0.0,
        patient_age: float = 65.0,
        existing_state: Optional[np.ndarray] = None,
        time_resolution_h: float = 4.0,
    ) -> Dict[str, np.ndarray]:
        """
        Run the adaptation ODE system.

        Returns dict of adaptation_name → np.ndarray over `time_h`.
        """
        y0 = existing_state if existing_state is not None else make_adaptation_baseline(
            disease_name, disease_severity, patient_age
        )

        # Build drug effect lookups
        drug_effects = {d: ADAPTATION_DRUG_EFFECTS.get(d, AdaptationDrugEffect()) for d in drug_names}

        def ode_rhs(t_val: float, y: np.ndarray) -> np.ndarray:
            return self._odes(t_val, y, drug_names, plasma_conc_series, drug_effects, time_h)

        try:
            sol = solve_ivp(
                ode_rhs,
                (float(time_h[0]), float(time_h[-1])),
                y0.copy(),
                method="LSODA",
                t_eval=time_h,
                rtol=1e-4,
                atol=1e-6,
                max_step=min(24.0, time_resolution_h * 4.0),
            )
            traj = np.clip(sol.y, 0.0, None)
        except Exception as exc:
            logger.warning("LongTermAdaptation ODE failed: %s — returning flat baseline", exc)
            traj = np.tile(y0[:, None], (1, len(time_h)))

        n_t = min(traj.shape[1], len(time_h))
        result: Dict[str, np.ndarray] = {}
        for i, name in enumerate(ADP_NAMES):
            result[name] = traj[i, :n_t]

        return result

    def _odes(
        self,
        t: float,
        y: np.ndarray,
        drug_names: List[str],
        plasma_series: Dict[str, np.ndarray],
        effects: Dict[str, AdaptationDrugEffect],
        time_h: np.ndarray,
    ) -> np.ndarray:
        dy = np.zeros(N_ADP)

        # Interpolate drug concentrations at time t
        drug_concs: Dict[str, float] = {}
        for d in drug_names:
            arr = plasma_series.get(d, np.zeros(len(time_h)))
            drug_concs[d] = max(0.0, float(np.interp(t, time_h, arr)))

        # Unpack current state
        cyp3a4   = max(y[_ADP.CYP3A4], 0.0)
        cyp2d6   = max(y[_ADP.CYP2D6], 0.0)
        cyp2c9   = max(y[_ADP.CYP2C9], 0.0)
        cyp2c19  = max(y[_ADP.CYP2C19], 0.0)
        pgp      = max(y[_ADP.PGP], 0.0)
        beta1r   = max(y[_ADP.BETA1_R], 0.0)
        gabaa    = max(y[_ADP.GABA_A_R], 0.0)
        sert     = max(y[_ADP.SERT], 0.0)
        ht1a     = max(y[_ADP.HT1A_AUTO], 0.0)
        gr       = max(y[_ADP.GR], 0.0)
        adrenal  = max(y[_ADP.ADRENAL_VOL], 0.0)
        cort_res = max(y[_ADP.CORTISOL_RES], 0.0)
        micro    = max(y[_ADP.MICRO_DIV], 0.0)
        lacto    = max(y[_ADP.LACTOBACILLUS], 0.0)
        bifido   = max(y[_ADP.BIFIDOBACT], 0.0)
        cdiff    = max(y[_ADP.CDIFF_RISK], 0.0)
        bdnf     = max(y[_ADP.BDNF], 0.0)
        hippo    = max(y[_ADP.HIPPO_NEURO], 0.0)
        serotonin= max(y[_ADP.SEROTONIN_SYN], 0.0)
        ldl      = max(y[_ADP.LDL], 0.0)
        hdl      = max(y[_ADP.HDL], 0.0)
        ldlr     = max(y[_ADP.LDL_R_LIVER], 0.0)
        ir       = max(y[_ADP.INSULIN_RESIST], 0.0)
        hepgluc  = max(y[_ADP.HEPATIC_GLUC], 0.0)
        bzd_tol  = max(y[_ADP.BZD_TOLERANCE], 0.0)
        bzd_dep  = max(y[_ADP.BZD_DEPEND], 0.0)
        thy_res  = max(y[_ADP.THYROID_RESERVE], 0.0)
        rank_bal = max(y[_ADP.RANK_RANKL], 0.0)

        # ── Aggregate drug effects across all active drugs ──────────────────
        tot_cyp3a4_eff   = 0.0
        tot_cyp2d6_eff   = 0.0
        tot_cyp2c9_eff   = 0.0
        tot_cyp2c19_eff  = 0.0
        tot_pgp_eff      = 0.0
        tot_beta1r_eff   = 0.0
        tot_gaba_eff     = 0.0
        tot_sert_eff     = 0.0
        tot_ht1a_eff     = 0.0
        tot_gr_eff       = 0.0
        tot_adrenal_sup  = 0.0
        tot_micro_disrupt= 0.0
        is_broad_spectrum= False
        tot_bdnf         = 0.0
        tot_5ht_syn      = 0.0
        tot_ldl_red      = 0.0
        tot_hdl_inc      = 0.0
        tot_ir_change    = 0.0
        tot_hepgluc_chg  = 0.0
        tot_bzd_tol      = 0.0
        tot_bzd_dep      = 0.0
        tot_thy_sup      = 0.0
        tot_rankl_sup    = 0.0

        for d, conc in drug_concs.items():
            e = effects.get(d, AdaptationDrugEffect())
            tot_cyp3a4_eff   += e.cyp3a4_effect   * conc
            tot_cyp2d6_eff   += e.cyp2d6_effect   * conc
            tot_cyp2c9_eff   += e.cyp2c9_effect   * conc
            tot_cyp2c19_eff  += e.cyp2c19_effect  * conc
            tot_pgp_eff      += e.pgp_effect       * conc
            tot_beta1r_eff   += e.beta1_r_effect   * conc
            tot_gaba_eff     += e.gaba_a_effect    * conc
            tot_sert_eff     += e.sert_effect      * conc
            tot_ht1a_eff     += e.ht1a_auto_effect * conc
            tot_gr_eff       += e.gr_effect        * conc
            tot_adrenal_sup  += e.adrenal_suppression * conc
            tot_micro_disrupt+= e.micro_disruption * conc
            if e.broad_spectrum:
                is_broad_spectrum = True
            tot_bdnf         += e.bdnf_induction   * conc
            tot_5ht_syn      += e.serotonin_syn_effect * conc
            # LDL: statins use Hill-equation occupancy (very potent — low EC50)
            # Other drugs (metformin): linear conc scaling
            if e.ldl_reduction > 0.01:  # statin-class (max factor > 0.01)
                _ec50_ldl = 0.003   # mg/L for HMG-CoA reductase inhibition
                tot_ldl_red += e.ldl_reduction * conc / (conc + _ec50_ldl)
            else:
                tot_ldl_red += e.ldl_reduction * conc   # weak / non-statin effect
            tot_hdl_inc      += e.hdl_increase     * conc
            tot_ir_change    += e.insulin_resist_change * conc
            tot_hepgluc_chg  += e.hepatic_gluc_change * conc
            tot_bzd_tol      += e.bzd_tolerance_rate  * conc
            tot_bzd_dep      += e.bzd_dependence_rate * conc
            tot_thy_sup      += e.thyroid_suppress    * conc
            tot_rankl_sup    += e.rankl_suppress      * conc

        # ── 1. CYP enzyme adaptation ────────────────────────────────────────
        # Half-time for CYP3A4 induction: ~3-7 days (k_adapt = 0.693/120h = 0.006 /h)
        # Inhibition is direct (faster) → k_inhib depends on drug binding kinetics
        k_cyp_return = 0.005   # return to baseline rate when drug removed (t½ ~6 days)

        # CYP3A4: can be induced (↑) or inhibited (↓) depending on drug
        if tot_cyp3a4_eff > 0:  # induction
            cyp3a4_target = min(3.0, 1.0 + tot_cyp3a4_eff * 200.0)
        else:  # inhibition
            cyp3a4_target = max(0.05, 1.0 + tot_cyp3a4_eff * 200.0)
        dy[_ADP.CYP3A4] = 0.003 * (cyp3a4_target - cyp3a4)

        cyp2d6_target = max(0.05, 1.0 + tot_cyp2d6_eff * 200.0)
        dy[_ADP.CYP2D6] = 0.003 * (cyp2d6_target - cyp2d6)

        cyp2c9_target = max(0.05, 1.0 + tot_cyp2c9_eff * 200.0)
        dy[_ADP.CYP2C9] = 0.003 * (cyp2c9_target - cyp2c9)

        cyp2c19_target = max(0.05, 1.0 + tot_cyp2c19_eff * 200.0)
        dy[_ADP.CYP2C19] = 0.003 * (cyp2c19_target - cyp2c19)

        pgp_target = max(0.1, 1.0 + tot_pgp_eff * 100.0)
        dy[_ADP.PGP] = 0.002 * (pgp_target - pgp)

        # ── 2. β1-adrenergic receptor (beta-blockers) ────────────────────────
        # Chronic blockade → compensatory upregulation (Bmax ↑)
        # This is clinically relevant: abrupt withdrawal → rebound tachycardia
        beta1r_target = max(0.5, 1.0 + tot_beta1r_eff * 500.0)  # upregulation
        dy[_ADP.BETA1_R] = 0.001 * (beta1r_target - beta1r)

        # ── 3. GABA-A receptor (benzodiazepines — tolerance) ─────────────────
        # Down-regulation of GABA-A receptors: clinical tolerance (onset weeks)
        # Rate: t½ for tolerance ~3 weeks = 504h
        gaba_target = max(0.20, 1.0 + tot_gaba_eff * 1000.0)
        dy[_ADP.GABA_A_R] = 0.00080 * (gaba_target - gabaa)

        # ── 4. SERT and 5-HT1A autoreceptor (SSRIs) ──────────────────────────
        # SERT downregulation: gradual (weeks)
        sert_target = max(0.30, 1.0 + tot_sert_eff * 800.0)
        dy[_ADP.SERT] = 0.00050 * (sert_target - sert)

        # 5-HT1A autoreceptor desensitisation: starts 1 week, peaks 2-4 weeks
        # This is key to SSRI therapeutic onset — autoreceptors blunt early 5-HT ↑
        ht1a_target = max(0.20, 1.0 + tot_ht1a_eff * 1500.0)
        dy[_ADP.HT1A_AUTO] = 0.00060 * (ht1a_target - ht1a)    # t½ ~2.5 weeks

        # ── 5. Glucocorticoid receptor (chronic corticosteroids) ─────────────
        gr_target = max(0.30, 1.0 + tot_gr_eff * 2000.0)
        dy[_ADP.GR] = 0.00050 * (gr_target - gr)

        # ── 6. Adrenal suppression (HPA axis – chronic corticosteroids) ──────
        # Exogenous glucocorticoid → ACTH ↓ → adrenal atrophy
        # Time constant: months (adrenal atrophy t½ ~3-6 months = 2160-4320h)
        adrenal_target = max(0.05, 1.0 - tot_adrenal_sup * 500.0)
        dy[_ADP.ADRENAL_VOL]  = 0.000150 * (adrenal_target - adrenal)
        dy[_ADP.CORTISOL_RES] = 0.000200 * (adrenal_target * 0.9 - cort_res)

        # ── 7. Gut microbiome dynamics ───────────────────────────────────────
        # Antibiotics → rapid depletion (48-72h); slow recovery (2-4 weeks)
        # Recovery rate: ~0.005/h ≈ 1% per hour = ~4 weeks to full recovery
        _MICRO_RECOVERY = 0.003
        _MICRO_DEPLETION_HALF = 0.020   # rate per mg/L·h

        dy[_ADP.MICRO_DIV]  = _MICRO_RECOVERY * (1.0 - micro) - tot_micro_disrupt * micro * 0.5
        dy[_ADP.LACTOBACILLUS] = _MICRO_RECOVERY * 1.2 * (1.0 - lacto) - tot_micro_disrupt * lacto
        dy[_ADP.BIFIDOBACT]    = _MICRO_RECOVERY * 1.0 * (1.0 - bifido) - tot_micro_disrupt * bifido * 1.2

        # C. diff risk: inversely proportional to diversity; boosted by broad-spectrum
        cdiff_susceptibility = 1.0 - micro
        cdiff_boost = 1.5 if is_broad_spectrum else 1.0
        dy[_ADP.CDIFF_RISK] = (
            0.002 * cdiff_susceptibility * cdiff_boost
            - 0.005 * cdiff          # natural clearance
            + 0.002 * cdiff_susceptibility  # background colonisation
        )

        # ── 8. BDNF and hippocampal neurogenesis (SSRIs) ─────────────────────
        # BDNF induction: 4-12 weeks with chronic SSRI → antidepressant effect
        bdnf_target = 1.0 + tot_bdnf * 2000.0
        dy[_ADP.BDNF] = 0.00030 * (bdnf_target - bdnf)    # t½ ~4 weeks

        # Hippocampal neurogenesis: driven by BDNF; very slow (months)
        hippo_target = min(1.0, 0.5 + (bdnf - 1.0) * 1.0 + tot_bdnf * 1000.0)
        dy[_ADP.HIPPO_NEURO] = 0.00020 * (hippo_target - hippo)  # t½ ~2 months

        # Synaptic 5-HT availability
        # Immediate: blocked by SERT block; modulated by autoreceptor desensitisation
        # Full antidepressant effect = SERT block + autoreceptor desensitisation
        serotonin_target = 1.0 + tot_5ht_syn * 1000.0 * (1.0 + (1.0 - ht1a) * 0.5)
        dy[_ADP.SEROTONIN_SYN] = 0.020 * (serotonin_target - serotonin)

        # ── 9. LDL/HDL cholesterol (statins) ─────────────────────────────────
        # tot_ldl_red is now an occupancy-weighted sum (0–1 range for statins),
        # so we no longer need the *3000 multiplier.
        ldlr_target = 1.0 + tot_ldl_red    # e.g. 1.64 for Atorvastatin 40mg → ~38% LDL drop
        dy[_ADP.LDL_R_LIVER] = 0.00080 * (ldlr_target - ldlr)  # t½ ~1 week

        # LDL falls with LDL-R upregulation
        ldl_target = ADAPTATION_BASELINE["LDL"] / max(ldlr, 0.5)
        dy[_ADP.LDL] = 0.00050 * (ldl_target - ldl)

        hdl_target = ADAPTATION_BASELINE["HDL"] + tot_hdl_inc * 500.0
        dy[_ADP.HDL] = 0.00030 * (hdl_target - hdl)

        # ── 10. Insulin resistance (corticosteroids → diabetogenic) ──────────
        ir_target = max(0.5, 1.0 + tot_ir_change * 1000.0)
        dy[_ADP.INSULIN_RESIST] = 0.00050 * (ir_target - ir)

        hepgluc_target = max(0.5, 1.0 + tot_hepgluc_chg * 1000.0)
        dy[_ADP.HEPATIC_GLUC] = 0.00050 * (hepgluc_target - hepgluc)

        # ── 11. BZD tolerance and dependence ─────────────────────────────────
        # Tolerance: driven by GABA-A downregulation (rate × concentration)
        dy[_ADP.BZD_TOLERANCE] = (
            tot_bzd_tol * 5.0 * (1.0 - bzd_tol)     # accumulates with use
            - 0.0001 * bzd_tol                        # very slow spontaneous recovery
        )
        # Dependence: lags tolerance; requires even slower reversal
        dy[_ADP.BZD_DEPEND] = (
            tot_bzd_dep * 5.0 * (bzd_tol * 2.0) * (1.0 - bzd_dep)  # driven by tolerance
            - 0.00003 * bzd_dep
        )

        # ── 12. Thyroid reserve ───────────────────────────────────────────────
        # Exogenous T4 suppresses endogenous TSH → thyroid atrophies slowly
        thy_res_target = max(0.1, 1.0 - tot_thy_sup * 500.0)
        dy[_ADP.THYROID_RESERVE] = 0.00020 * (thy_res_target - thy_res)

        # ── 13. RANK/RANKL bone remodeling balance ────────────────────────────
        # Corticosteroids suppress RANKL → more osteoclast survival, less osteoblast
        rank_target = max(0.1, 1.0 - tot_rankl_sup * 800.0)
        dy[_ADP.RANK_RANKL] = 0.000030 * (rank_target - rank_bal)  # very slow bone remodeling

        return dy


# ---------------------------------------------------------------------------
# Derived clinical metrics from adaptation state
# ---------------------------------------------------------------------------

def compute_clearance_multiplier(adaptation_state: Dict[str, np.ndarray], drug_name: str) -> np.ndarray:
    """
    Compute the effective clearance multiplier for a drug based on CYP enzyme changes.
    A statin patient on rifampicin: CYP3A4 ↑ 3× → atorvastatin clearance ↑ 3×
    A CYP2D6 inhibitor patient: CYP2D6 ↓ → metoprolol clearance ↓ → accumulation.
    """
    # Map drugs to their primary CYP enzyme
    _CYP_MAP = {
        "Atorvastatin":  "CYP3A4",
        "Amlodipine":    "CYP3A4",
        "Lorazepam":     "CYP3A4",
        "Sertraline":    "CYP2D6",
        "Metoprolol":    "CYP2D6",
        "Ibuprofen":     "CYP2C9",
        "Naproxen":      "CYP2C9",
        "Glipizide":     "CYP2C9",
        "Levothyroxine": None,
        "Metformin":     None,  # renal clearance, not CYP
        "Lisinopril":    None,
        "Amoxicillin":   None,
        "Ciprofloxacin": None,
        "Prednisone":    "CYP3A4",
        "Acetaminophen": "CYP3A4",
    }
    cyp = _CYP_MAP.get(drug_name)
    if cyp is None:
        return np.ones(len(next(iter(adaptation_state.values()))))
    cyp_activity = adaptation_state.get(cyp, np.ones(1))
    return np.clip(cyp_activity, 0.1, 5.0)


def compute_antidepressant_response(adaptation_state: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute the expected antidepressant response trajectory for SSRIs.
    Captures the clinically documented 4-8 week lag before full effect.
    """
    # Antidepressant effect = combination of:
    # 1. Synaptic 5-HT availability (immediate but incomplete without autoreceptor desenitisation)
    # 2. 5-HT1A autoreceptor desensitisation (2-4 weeks)
    # 3. Hippocampal neurogenesis (4-12 weeks)
    serotonin = adaptation_state.get("SEROTONIN_SYN", np.ones(1))
    ht1a      = adaptation_state.get("HT1A_AUTO", np.ones(1))
    hippo     = adaptation_state.get("HIPPO_NEURO", np.ones(1) * 0.5)
    bdnf      = adaptation_state.get("BDNF", np.ones(1))

    # Autoreceptor desensitised → tachyphylaxis resolved → full serotonin effect
    sert_block_efficiency = (serotonin - 1.0) * (2.0 - ht1a)  # more effect when ht1a down
    neurogenic_boost = (hippo - 0.5) * 2.0

    response = np.clip(
        0.3 * np.clip(sert_block_efficiency, 0.0, 1.0)  # acute serotonin effect
        + 0.4 * np.clip(2.0 - ht1a, 0.0, 1.0)          # autoreceptor desensitisation
        + 0.3 * np.clip(neurogenic_boost, 0.0, 1.0),    # neuroplasticity
        0.0, 1.0
    )
    return response


def compute_withdrawal_risk(adaptation_state: Dict[str, np.ndarray], drug_name: str) -> np.ndarray:
    """
    Compute withdrawal risk if drug is suddenly stopped.
    High for benzodiazepines (GABA rebound), beta-blockers (adrenergic rebound),
    corticosteroids (adrenal insufficiency), SSRIs (discontinuation syndrome).
    """
    n = len(next(iter(adaptation_state.values())))
    risk = np.zeros(n)

    if drug_name in ("Lorazepam",):
        bzd_dep = adaptation_state.get("BZD_DEPEND", np.zeros(n))
        risk = np.clip(bzd_dep * 2.0, 0.0, 1.0)

    elif drug_name in ("Metoprolol",):
        beta1_up = adaptation_state.get("BETA1_R", np.ones(n))
        risk = np.clip((beta1_up - 1.0) * 2.0, 0.0, 1.0)

    elif drug_name in ("Prednisone",):
        adrenal = adaptation_state.get("ADRENAL_VOL", np.ones(n))
        risk = np.clip((1.0 - adrenal) * 1.5, 0.0, 1.0)

    elif drug_name in ("Sertraline",):
        sert = adaptation_state.get("SERT", np.ones(n))
        risk = np.clip((1.0 - sert) * 0.6, 0.0, 1.0)

    return risk
