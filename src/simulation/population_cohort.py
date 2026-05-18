"""
Population PK/PD Cohort Runner
================================

Generates a virtual patient population and runs mechanistic PK/PD simulations
for each patient, returning percentile statistics suitable for dashboard plotting.

Key design decisions:
- Uses the same MechanisticPKPDEngine instance passed in (no re-instantiation).
- Patient variability is driven by allometric scaling + log-normal IIV on CL/Vd.
- Special sub-groups: pregnant, elderly, paediatric, renal-impaired,
  immunocompromised — each with physiologically informed parameter shifts.
- Paediatric weights follow WHO/CDC growth chart centiles (age-specific).
- Adult-only diseases (hypertension, T2D, CAD, COPD, etc.) are not assigned
  to paediatric patients — they are reassigned as young adults instead.
- Sex-specific adult weight distributions are applied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Per-patient variability ─────────────────────────────────────────────────
_IIV_CL_CV = 0.30
_IIV_VD_CV = 0.25
_IIV_BA_CV = 0.15
_IIV_KF_SD = 0.12
_IIV_LF_SD = 0.10

# ── WHO/CDC weight-for-age lookup (mean kg, SD kg) — combined sex ───────────
# Source: CDC Clinical Growth Charts 2000 + WHO Child Growth Standards
_PAED_WT_BY_AGE: Dict[int, tuple] = {
    1:  (10.2, 1.3),  2:  (12.5, 1.5),  3:  (14.6, 1.8),
    4:  (16.8, 2.0),  5:  (18.8, 2.5),  6:  (21.2, 3.0),
    7:  (23.6, 3.6),  8:  (26.4, 4.5),  9:  (29.6, 6.0),
    10: (33.8, 7.2), 11: (38.2, 8.8), 12: (43.1, 10.2),
    13: (48.8, 11.4), 14: (54.5, 11.8), 15: (59.2, 11.5),
    16: (63.8, 10.8), 17: (67.2, 10.5),
}

# ── Diseases that are essentially adult-onset only (age < 16 is unrealistic) ─
_ADULT_ONLY_DISEASES = {
    "hypertension", "essential hypertension",
    "type 2 diabetes", "type 2 diabetes mellitus", "type 2 diabetes (viral)",
    "viral + diabetes",
    "coronary artery disease", "ischaemic heart disease",
    "heart failure", "cardiac failure",
    "atrial fibrillation",
    "copd", "chronic obstructive pulmonary disease",
    "osteoporosis", "osteoarthritis",
    "alzheimer", "alzheimer's disease", "dementia",
    "peripheral artery disease", "peripheral vascular disease",
    "non-alcoholic fatty liver", "nafld",
    "metabolic syndrome", "dyslipidaemia",
    "gout", "chronic kidney disease",
}

# ── Population density / setting prevalences ────────────────────────────────
_SETTING_PROFILES: Dict[str, Dict] = {
    "urban":        {"elderly_p": 0.18, "pregnant_p": 0.05, "paediatric_p": 0.12,
                     "renal_p": 0.10, "immuno_p": 0.06, "density_km2": 5000},
    "suburban":     {"elderly_p": 0.22, "pregnant_p": 0.06, "paediatric_p": 0.15,
                     "renal_p": 0.11, "immuno_p": 0.05, "density_km2": 1200},
    "rural":        {"elderly_p": 0.28, "pregnant_p": 0.07, "paediatric_p": 0.18,
                     "renal_p": 0.13, "immuno_p": 0.04, "density_km2": 120},
    "hospital":     {"elderly_p": 0.40, "pregnant_p": 0.08, "paediatric_p": 0.10,
                     "renal_p": 0.30, "immuno_p": 0.20, "density_km2": 15000},
    "school":       {"elderly_p": 0.02, "pregnant_p": 0.01, "paediatric_p": 0.70,
                     "renal_p": 0.02, "immuno_p": 0.03, "density_km2": 3000},
    "nursing_home": {"elderly_p": 0.90, "pregnant_p": 0.00, "paediatric_p": 0.00,
                     "renal_p": 0.45, "immuno_p": 0.30, "density_km2": 2000},
}
_DEFAULT_SETTING = "urban"

SUBGROUP_LABELS = [
    "general_adult", "elderly", "pregnant", "paediatric",
    "renal_impaired", "immunocompromised",
]


# ── Realistic weight helpers ─────────────────────────────────────────────────

def _paed_weight(age: float, sex: int, rng) -> float:
    """Return a WHO/CDC-anchored weight (kg) for a child aged 1–17."""
    age_int = max(1, min(17, int(round(age))))
    mean_wt, sd_wt = _PAED_WT_BY_AGE[age_int]
    # Males heavier post-puberty; females slightly heavier 11–13
    if sex == 1 and age >= 13:
        mean_wt += 2.5
    elif sex == 0 and 11 <= age <= 13:
        mean_wt += 1.2
    lo = mean_wt * 0.72   # roughly 3rd centile
    hi = mean_wt * 1.52   # roughly 97th centile (accounts for overweight)
    return float(np.clip(rng.normal(mean_wt, sd_wt), lo, hi))


def _adult_weight(age: float, sex: int, rng, mean_override: float = None) -> float:
    """
    Sex- and age-appropriate adult weight (kg).
    Males: mean ~83 kg; Females: mean ~70 kg.
    Slight decline after age 75 due to sarcopenia.
    """
    if sex == 1:  # male
        base = mean_override if mean_override else 83.0
        sd   = 16.0
        lo, hi = 55.0, 145.0
    else:          # female
        base = mean_override if mean_override else 70.0
        sd   = 14.0
        lo, hi = 44.0, 130.0
    # Sarcopenia — gradual weight loss after 75
    if age > 75:
        base -= (age - 75) * 0.4
    return float(np.clip(rng.normal(base, sd), lo, hi))


def _elderly_kf(age: float, rng) -> float:
    """eGFR-based kidney function fraction. Gradual decline ~1%/year after 40."""
    base_kf = max(0.30, 1.0 - 0.010 * max(0, age - 40))
    return float(np.clip(rng.normal(base_kf, 0.09), 0.25, 0.92))


# ── Main runner ──────────────────────────────────────────────────────────────

@dataclass
class CohortResult:
    time: np.ndarray
    plasma_p05: np.ndarray
    plasma_p25: np.ndarray
    plasma_p50: np.ndarray
    plasma_p75: np.ndarray
    plasma_p95: np.ndarray
    cmax_distribution: List[float]
    auc_distribution: List[float]
    tmax_distribution: List[float]
    t_half_distribution: List[float]
    therapeutic_window: tuple
    frac_within_window: float
    n_patients: int
    n_succeeded: int
    patient_demographics: List[Dict]
    subgroup_summary: Dict = field(default_factory=dict)
    overall_response_rate: float = 0.0
    overall_adverse_rate: float = 0.0
    setting: str = "urban"
    density_km2: int = 5000


class PopulationCohortRunner:

    def run(
        self,
        engine,
        base_drug,
        base_bio,
        diseases: list,
        n: int = 50,
        time_points: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        setting: str = "urban",
        disease_name: str = "",
    ) -> CohortResult:
        if time_points is None:
            t_end = float(getattr(base_drug, "half_life", 8.0)) * 8.0
            t_end = float(np.clip(t_end, 24.0, 168.0))
            time_points = np.linspace(0.0, t_end, max(int(t_end) + 1, 97))

        rng = np.random.default_rng(seed)
        t_window = getattr(base_drug, "therapeutic_window", (1.0, 100.0))
        t_lo, t_hi = float(t_window[0]), float(t_window[1])

        profile = _SETTING_PROFILES.get(setting, _SETTING_PROFILES[_DEFAULT_SETTING])

        # Is this disease adult-only? (paediatric patients will be reclassified)
        disease_lower = disease_name.lower().strip()
        is_adult_only_disease = any(d in disease_lower for d in _ADULT_ONLY_DISEASES)

        sg_traces: Dict[str, List] = {k: [] for k in SUBGROUP_LABELS}
        sg_cmax:   Dict[str, List] = {k: [] for k in SUBGROUP_LABELS}
        sg_auc:    Dict[str, List] = {k: [] for k in SUBGROUP_LABELS}
        sg_resp:   Dict[str, List] = {k: [] for k in SUBGROUP_LABELS}
        sg_adv:    Dict[str, List] = {k: [] for k in SUBGROUP_LABELS}

        plasma_traces, cmax_vals, auc_vals, tmax_vals, thalf_vals = [], [], [], [], []
        demographics: List[Dict] = []
        n_succeeded = 0

        base_cl  = float(base_drug.clearance)
        base_vd  = float(base_drug.volume_distribution)
        base_ba  = float(base_drug.bioavailability)
        base_age = float(getattr(base_bio, "age", 45.0))
        base_kf  = float(getattr(base_bio, "kidney_function", 1.0))
        base_lf  = float(getattr(base_bio, "liver_function", 1.0))

        for i in range(n):
            try:
                # ── Assign subgroup ───────────────────────────────────────
                roll = rng.random()
                cumulative = 0.0

                def _exceeds(p):
                    nonlocal cumulative
                    cumulative += p
                    return roll < cumulative

                is_elderly  = _exceeds(profile["elderly_p"])
                is_pregnant = not is_elderly  and _exceeds(profile["pregnant_p"])
                is_paed     = not is_elderly  and not is_pregnant and _exceeds(profile["paediatric_p"])
                is_renal    = not is_elderly  and not is_pregnant and not is_paed and _exceeds(profile["renal_p"])
                is_immuno   = not is_elderly  and not is_pregnant and not is_paed and not is_renal and _exceeds(profile["immuno_p"])

                # Reclassify paediatric → young adult if disease is adult-only
                if is_paed and is_adult_only_disease:
                    is_paed = False  # becomes general_adult with young-adult age

                if is_elderly:       sg = "elderly"
                elif is_pregnant:    sg = "pregnant"
                elif is_paed:        sg = "paediatric"
                elif is_renal:       sg = "renal_impaired"
                elif is_immuno:      sg = "immunocompromised"
                else:                sg = "general_adult"

                # ── Demographics ──────────────────────────────────────────
                if is_elderly:
                    sex    = int(rng.integers(0, 2))
                    age    = float(np.clip(rng.normal(74, 7), 65, 95))
                    weight = _adult_weight(age, sex, rng, mean_override=68.0 if sex == 1 else 62.0)
                    kf     = _elderly_kf(age, rng)
                    lf     = float(np.clip(rng.normal(0.78, 0.09), 0.52, 0.94))

                elif is_pregnant:
                    sex    = 0  # female
                    age    = float(np.clip(rng.normal(28, 5), 18, 42))
                    # Pregnancy weight gain: pre-pregnancy base + gestational gain
                    pre_preg_wt = float(np.clip(rng.normal(65, 10), 48, 95))
                    gestation_wk = float(rng.uniform(8, 38))
                    gain = 0.45 * gestation_wk if gestation_wk <= 20 else (9.0 + 0.50 * (gestation_wk - 20))
                    weight = float(np.clip(pre_preg_wt + gain, 52, 115))
                    kf     = float(np.clip(rng.normal(0.96, 0.05), 0.80, 1.05))  # GFR ↑ in pregnancy
                    lf     = float(np.clip(rng.normal(0.88, 0.07), 0.68, 1.0))

                elif is_paed:
                    sex  = int(rng.integers(0, 2))
                    age  = float(np.clip(rng.normal(9, 4), 1, 17))
                    weight = _paed_weight(age, sex, rng)
                    # Children generally have good organ function unless neonatal
                    kf   = float(np.clip(rng.normal(0.97, 0.04) if age > 2 else rng.normal(0.80, 0.08), 0.65, 1.05))
                    lf   = float(np.clip(rng.normal(0.97, 0.04) if age > 1 else rng.normal(0.85, 0.07), 0.70, 1.05))

                elif is_renal:
                    sex    = int(rng.integers(0, 2))
                    age    = float(np.clip(rng.normal(max(base_age, 45), 12), 18, 85))
                    weight = _adult_weight(age, sex, rng)
                    kf     = float(np.clip(rng.normal(0.38, 0.11), 0.12, 0.62))
                    lf     = float(np.clip(rng.normal(base_lf, _IIV_LF_SD), 0.55, 1.0))

                elif is_immuno:
                    sex    = int(rng.integers(0, 2))
                    age    = float(np.clip(rng.normal(base_age, 14), 18, 80))
                    weight = _adult_weight(age, sex, rng,
                                           mean_override=(78.0 if sex == 1 else 65.0))  # slight cachexia
                    kf     = float(np.clip(rng.normal(base_kf, _IIV_KF_SD), 0.35, 1.0))
                    lf     = float(np.clip(rng.normal(0.78, 0.11), 0.50, 1.0))

                else:  # general_adult (includes reclassified paediatric)
                    sex    = int(rng.integers(0, 2))
                    # Reclassified paediatric patients become young adults (18–25)
                    if is_adult_only_disease and roll < (profile["elderly_p"] + profile["pregnant_p"] + profile["paediatric_p"]):
                        age = float(np.clip(rng.normal(21, 2.5), 18, 26))
                    else:
                        age = float(np.clip(rng.normal(base_age, 14), 18, 64))
                    weight = _adult_weight(age, sex, rng)
                    kf     = float(np.clip(rng.normal(base_kf, _IIV_KF_SD), 0.55, 1.0))
                    lf     = float(np.clip(rng.normal(base_lf, _IIV_LF_SD), 0.60, 1.0))

                # ── Allometric PK scaling ─────────────────────────────────
                allom_wt  = weight / 70.0
                cl_allom  = base_cl * (allom_wt ** 0.75)
                vd_allom  = base_vd * allom_wt

                cl_patient = cl_allom * float(np.exp(rng.normal(0.0, _IIV_CL_CV)))
                vd_patient = vd_allom * float(np.exp(rng.normal(0.0, _IIV_VD_CV)))
                ba_patient = float(np.clip(base_ba * np.exp(rng.normal(0.0, _IIV_BA_CV)), 0.05, 1.0))

                # ── Subgroup-specific PK adjustments ─────────────────────
                if is_elderly:
                    # CL declines ~1.2%/year from age 65 (hepatic + renal)
                    age_factor = max(0.42, 1.0 - 0.012 * (age - 65))
                    cl_patient *= age_factor
                elif is_pregnant:
                    cl_patient *= 1.25   # CYP3A4 induction
                    vd_patient *= 1.18   # ↑ plasma volume
                elif is_paed:
                    # Per-kg CL higher in children; absolute Vd smaller
                    if age < 2:
                        ped_cl_factor = 1.8
                    elif age < 5:
                        ped_cl_factor = 1.5
                    elif age < 12:
                        ped_cl_factor = 1.2
                    else:
                        ped_cl_factor = 1.0
                    cl_patient *= ped_cl_factor
                elif is_renal:
                    cl_patient *= (0.38 + 0.62 * max(kf, 0.12))
                elif is_immuno:
                    cl_patient *= 0.84  # reduced hepatic phase-I

                # Sex effect on Vd (females ~8% lower Vd for many drugs)
                if sex == 0:
                    vd_patient *= 1.07

                # ── Build patient objects ─────────────────────────────────
                patient_bio = _clone_namespace(base_bio)
                patient_bio.weight          = weight
                patient_bio.age             = age
                patient_bio.sex             = sex
                patient_bio.kidney_function = min(kf, 1.0)
                patient_bio.liver_function  = min(lf, 1.0)

                patient_drug = _clone_namespace(base_drug)
                patient_drug.clearance           = float(np.clip(cl_patient, 0.1, 500.0))
                patient_drug.volume_distribution = float(np.clip(vd_patient, 1.0, 5000.0))
                patient_drug.bioavailability     = ba_patient

                # ── Simulate ──────────────────────────────────────────────
                result = engine.simulate_drug(
                    patient_drug, patient_bio, diseases,
                    time_points, apply_ml_correction=False,
                )
                plasma = result.plasma_conc
                cmax  = result.pk_stats.get("Cmax (mg/L)", float(np.max(plasma)))
                auc   = result.pk_stats.get("AUClast (mg·h/L)", 0.0)
                tmax  = result.pk_stats.get("Tmax (h)", 0.0)
                thalf = result.pk_stats.get("t½ (h)", 0.0)

                plasma_traces.append(plasma)
                cmax_vals.append(cmax)
                auc_vals.append(auc)
                tmax_vals.append(tmax)
                thalf_vals.append(thalf)

                in_window = int(t_lo <= cmax <= t_hi)
                above_hi  = int(cmax > t_hi * 1.5)

                sg_traces[sg].append(plasma)
                sg_cmax[sg].append(cmax)
                sg_auc[sg].append(auc)
                sg_resp[sg].append(in_window)
                sg_adv[sg].append(above_hi)

                # BMI for display
                # Use rough height estimate from weight/age
                bmi_note = ""
                if age >= 18:
                    # rough height (cm): males ~176, females ~163 with age adjustments
                    ht = (176 if sex == 1 else 163) - max(0, (age - 70) * 0.3)
                    bmi = weight / ((ht / 100) ** 2)
                    bmi_note = f"{bmi:.1f}"

                demographics.append({
                    "patient":         i + 1,
                    "subgroup":        sg,
                    "weight_kg":       round(weight, 1),
                    "age":             round(age, 0),
                    "sex":             "F" if sex == 0 else "M",
                    "kidney_function": round(min(kf, 1.0), 2),
                    "liver_function":  round(min(lf, 1.0), 2),
                    "CL (L/h)":        round(patient_drug.clearance, 2),
                    "Vd (L)":          round(patient_drug.volume_distribution, 1),
                    "Cmax (mg/L)":     round(cmax, 4),
                    "AUC (mg·h/L)":    round(auc, 2),
                    "in_window":       bool(in_window),
                    "adverse":         bool(above_hi),
                })
                n_succeeded += 1

            except Exception as exc:
                logger.debug("Patient %d simulation failed: %s", i, exc)

        if not plasma_traces:
            raise RuntimeError("All patient simulations failed in population cohort")

        traces   = np.array(plasma_traces)
        cmax_arr = np.array(cmax_vals)
        frac_win = float(np.sum((cmax_arr >= t_lo) & (cmax_arr <= t_hi)) / len(cmax_arr))
        adv_rate = float(np.sum(cmax_arr > t_hi * 1.5) / len(cmax_arr))

        # ── Subgroup summary ──────────────────────────────────────────────
        subgroup_summary: Dict = {}
        for sg_key in SUBGROUP_LABELS:
            n_sg = len(sg_cmax[sg_key])
            if n_sg == 0:
                continue
            sg_c = np.array(sg_cmax[sg_key])
            sg_a = np.array(sg_auc[sg_key])
            subgroup_summary[sg_key] = {
                "n":             n_sg,
                "cmax_median":   round(float(np.median(sg_c)), 4),
                "auc_median":    round(float(np.median(sg_a)), 2),
                "response_rate": round(float(np.mean(sg_resp[sg_key])), 4),
                "adverse_rate":  round(float(np.mean(sg_adv[sg_key])), 4),
                "plasma_p50":    [round(float(v), 4) for v in np.percentile(np.array(sg_traces[sg_key]), 50, axis=0)],
                "plasma_p05":    [round(float(v), 4) for v in np.percentile(np.array(sg_traces[sg_key]),  5, axis=0)],
                "plasma_p95":    [round(float(v), 4) for v in np.percentile(np.array(sg_traces[sg_key]), 95, axis=0)],
            }

        return CohortResult(
            time                  = time_points,
            plasma_p05            = np.percentile(traces,  5, axis=0),
            plasma_p25            = np.percentile(traces, 25, axis=0),
            plasma_p50            = np.percentile(traces, 50, axis=0),
            plasma_p75            = np.percentile(traces, 75, axis=0),
            plasma_p95            = np.percentile(traces, 95, axis=0),
            cmax_distribution     = cmax_vals,
            auc_distribution      = auc_vals,
            tmax_distribution     = tmax_vals,
            t_half_distribution   = thalf_vals,
            therapeutic_window    = (t_lo, t_hi),
            frac_within_window    = frac_win,
            n_patients            = n,
            n_succeeded           = n_succeeded,
            patient_demographics  = demographics,
            subgroup_summary      = subgroup_summary,
            overall_response_rate = frac_win,
            overall_adverse_rate  = adv_rate,
            setting               = setting,
            density_km2           = profile["density_km2"],
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clone_namespace(obj) -> SimpleNamespace:
    ns = SimpleNamespace()
    for attr in dir(obj):
        if not attr.startswith("_"):
            try:
                setattr(ns, attr, getattr(obj, attr))
            except Exception:
                pass
    return ns
