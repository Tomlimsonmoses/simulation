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
- Population density settings affect disease-transmission context metadata
  (urban / suburban / rural / hospital / school).
- All n patients are simulated synchronously; each call to simulate_drug is
  < 5 ms (analytical Bateman), so n=200 takes ~1 s.

Usage::

    from simulation.population_cohort import PopulationCohortRunner, CohortResult

    runner = PopulationCohortRunner()
    result = runner.run(engine, drug, base_bio, diseases, n=100,
                        time_points=None, setting="urban")
    # result.plasma_p50  — median plasma profile
    # result.subgroup_summary — dict keyed by subgroup label
    # result.demographics   — full per-patient list
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Per-patient variability parameters ─────────────────────────────────────
_IIV_CL_CV   = 0.30   # 30% coefficient of variation on CL (log-normal)
_IIV_VD_CV   = 0.25   # 25% CV on Vd (log-normal)
_IIV_BA_CV   = 0.15   # 15% CV on bioavailability (log-normal, clamped 0.05-1)
_IIV_KF_SD   = 0.15   # additive SD on kidney_function (normal, clipped 0.3-1.0)
_IIV_LF_SD   = 0.12   # additive SD on liver_function  (normal, clipped 0.5-1.0)
_IIV_WT_SD   = 12.0   # kg SD on body weight (normal, clipped 40-140)

# ── Population density / setting prevalence adjustments ────────────────────
# Each setting shifts the probability that a patient is in a high-risk group.
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

# ── Subgroup labels ─────────────────────────────────────────────────────────
SUBGROUP_LABELS = [
    "general_adult", "elderly", "pregnant", "paediatric",
    "renal_impaired", "immunocompromised",
]


@dataclass
class CohortResult:
    """Statistics from a population cohort simulation."""

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
    """
    Runs n virtual patient simulations and returns percentile statistics
    plus per-subgroup breakdowns (elderly, pregnant, paediatric, etc.).
    """

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
    ) -> CohortResult:
        """
        Run a population cohort simulation.

        Parameters
        ----------
        engine     : MechanisticPKPDEngine instance
        base_drug  : DrugProperties reference (mean)
        base_bio   : BiologicalParameters reference (mean)
        diseases   : list of DiseaseState — shared
        n          : number of virtual patients
        time_points: optional time grid; defaults to 0–96 h at 1 h resolution
        seed       : optional random seed
        setting    : population density/setting key (urban/rural/hospital/etc.)

        Returns
        -------
        CohortResult
        """
        if time_points is None:
            t_end = float(getattr(base_drug, "half_life", 8.0)) * 8.0
            t_end = float(np.clip(t_end, 24.0, 168.0))
            time_points = np.linspace(0.0, t_end, max(int(t_end) + 1, 97))

        rng = np.random.default_rng(seed)
        t_window = getattr(base_drug, "therapeutic_window", (1.0, 100.0))
        t_lo, t_hi = float(t_window[0]), float(t_window[1])

        profile = _SETTING_PROFILES.get(setting, _SETTING_PROFILES[_DEFAULT_SETTING])

        # Per-subgroup accumulators
        sg_traces: Dict[str, List] = {k: [] for k in SUBGROUP_LABELS}
        sg_cmax:   Dict[str, List] = {k: [] for k in SUBGROUP_LABELS}
        sg_auc:    Dict[str, List] = {k: [] for k in SUBGROUP_LABELS}
        sg_resp:   Dict[str, List] = {k: [] for k in SUBGROUP_LABELS}  # 1=responded
        sg_adv:    Dict[str, List] = {k: [] for k in SUBGROUP_LABELS}  # 1=adverse

        plasma_traces: List[np.ndarray] = []
        cmax_vals, auc_vals, tmax_vals, thalf_vals = [], [], [], []
        demographics: List[Dict] = []
        n_succeeded = 0

        base_wt  = float(getattr(base_bio, "weight", 70.0))
        base_kf  = float(getattr(base_bio, "kidney_function", 1.0))
        base_lf  = float(getattr(base_bio, "liver_function", 1.0))
        base_age = float(getattr(base_bio, "age", 45.0))
        base_cl  = float(base_drug.clearance)
        base_vd  = float(base_drug.volume_distribution)
        base_ba  = float(base_drug.bioavailability)

        for i in range(n):
            try:
                # ── Assign subgroup by setting prevalences ───────────────
                roll = rng.random()
                is_elderly  = roll < profile["elderly_p"]
                is_pregnant = (not is_elderly) and roll < (profile["elderly_p"] + profile["pregnant_p"])
                is_paed     = (not is_elderly) and (not is_pregnant) and roll < (
                    profile["elderly_p"] + profile["pregnant_p"] + profile["paediatric_p"])
                is_renal    = (not is_elderly) and (not is_pregnant) and (not is_paed) and roll < (
                    profile["elderly_p"] + profile["pregnant_p"] + profile["paediatric_p"] + profile["renal_p"])
                is_immuno   = (not is_elderly) and (not is_pregnant) and (not is_paed) and (not is_renal) and roll < (
                    profile["elderly_p"] + profile["pregnant_p"] + profile["paediatric_p"] + profile["renal_p"] + profile["immuno_p"])

                # Determine subgroup label
                if is_elderly:       sg = "elderly"
                elif is_pregnant:    sg = "pregnant"
                elif is_paed:        sg = "paediatric"
                elif is_renal:       sg = "renal_impaired"
                elif is_immuno:      sg = "immunocompromised"
                else:                sg = "general_adult"

                # ── Demographics per subgroup ─────────────────────────────
                if is_elderly:
                    age    = float(np.clip(rng.normal(74, 7), 65, 95))
                    weight = float(np.clip(rng.normal(68, 12), 45, 110))
                    kf     = float(np.clip(rng.normal(0.62, 0.12), 0.30, 0.85))
                    lf     = float(np.clip(rng.normal(0.80, 0.10), 0.55, 0.95))
                    sex    = int(rng.integers(0, 2))
                elif is_pregnant:
                    age    = float(np.clip(rng.normal(28, 5), 18, 42))
                    weight = float(np.clip(rng.normal(75, 10), 52, 110))
                    kf     = float(np.clip(rng.normal(0.95, 0.06), 0.75, 1.0))
                    lf     = float(np.clip(rng.normal(0.88, 0.08), 0.65, 1.0))
                    sex    = 0  # female
                elif is_paed:
                    age    = float(np.clip(rng.normal(9, 4), 1, 17))
                    weight = float(np.clip(rng.normal(32, 14), 8, 65))
                    kf     = float(np.clip(rng.normal(0.95, 0.05), 0.75, 1.0))
                    lf     = float(np.clip(rng.normal(0.95, 0.05), 0.75, 1.0))
                    sex    = int(rng.integers(0, 2))
                elif is_renal:
                    age    = float(np.clip(rng.normal(base_age, 12), 18, 80))
                    weight = float(np.clip(rng.normal(base_wt, _IIV_WT_SD), 45, 130))
                    kf     = float(np.clip(rng.normal(0.40, 0.10), 0.15, 0.60))
                    lf     = float(np.clip(rng.normal(base_lf, _IIV_LF_SD), 0.55, 1.0))
                    sex    = int(rng.integers(0, 2))
                elif is_immuno:
                    age    = float(np.clip(rng.normal(base_age, 14), 18, 80))
                    weight = float(np.clip(rng.normal(base_wt - 5, _IIV_WT_SD), 42, 120))
                    kf     = float(np.clip(rng.normal(base_kf, _IIV_KF_SD), 0.35, 1.0))
                    lf     = float(np.clip(rng.normal(0.78, 0.12), 0.50, 1.0))
                    sex    = int(rng.integers(0, 2))
                else:  # general_adult
                    age    = float(np.clip(rng.normal(base_age, 14), 18, 64))
                    weight = float(np.clip(rng.normal(base_wt, _IIV_WT_SD), 45, 130))
                    kf     = float(np.clip(rng.normal(base_kf, _IIV_KF_SD), 0.55, 1.0))
                    lf     = float(np.clip(rng.normal(base_lf, _IIV_LF_SD), 0.60, 1.0))
                    sex    = int(rng.integers(0, 2))

                # ── Allometric PK scaling ────────────────────────────────
                allom_wt = weight / 70.0
                cl_allom = base_cl * (allom_wt ** 0.75)
                vd_allom = base_vd * allom_wt

                # ── IIV ─────────────────────────────────────────────────
                cl_patient = cl_allom * float(np.exp(rng.normal(0.0, _IIV_CL_CV)))
                vd_patient = vd_allom * float(np.exp(rng.normal(0.0, _IIV_VD_CV)))
                ba_patient = float(np.clip(
                    base_ba * np.exp(rng.normal(0.0, _IIV_BA_CV)), 0.05, 1.0
                ))

                # ── Subgroup-specific PK adjustments ────────────────────
                if is_elderly:
                    age_factor = max(1.0 - 0.012 * (age - 65), 0.45)
                    cl_patient *= age_factor
                elif is_pregnant:
                    # Pregnancy ↑ CL ~25% (CYP3A4 induction), ↑ Vd ~15–20%
                    cl_patient *= 1.25
                    vd_patient *= 1.18
                elif is_paed:
                    # Children: CL/kg higher but abs Vd lower
                    ped_factor = 1.5 if age < 5 else (1.2 if age < 12 else 1.0)
                    cl_patient *= ped_factor
                elif is_renal:
                    kf_factor = max(kf, 0.15)
                    cl_patient *= (0.4 + 0.6 * kf_factor)
                elif is_immuno:
                    cl_patient *= 0.85   # reduced hepatic phase-I activity

                # ── Sex effect on Vd ────────────────────────────────────
                if sex == 0:
                    vd_patient *= 1.08

                # ── Build per-patient objects ────────────────────────────
                patient_bio = _clone_namespace(base_bio)
                patient_bio.weight          = weight
                patient_bio.age             = age
                patient_bio.sex             = sex
                patient_bio.kidney_function = kf
                patient_bio.liver_function  = lf

                patient_drug = _clone_namespace(base_drug)
                patient_drug.clearance           = float(np.clip(cl_patient, 0.1, 500.0))
                patient_drug.volume_distribution = float(np.clip(vd_patient, 1.0, 5000.0))
                patient_drug.bioavailability     = ba_patient

                # ── Simulate ─────────────────────────────────────────────
                result = engine.simulate_drug(
                    patient_drug, patient_bio, diseases,
                    time_points, apply_ml_correction=False,
                )
                plasma = result.plasma_conc
                cmax   = result.pk_stats.get("Cmax (mg/L)", float(np.max(plasma)))
                auc    = result.pk_stats.get("AUClast (mg·h/L)", 0.0)
                tmax   = result.pk_stats.get("Tmax (h)", 0.0)
                thalf  = result.pk_stats.get("t½ (h)", 0.0)

                plasma_traces.append(plasma)
                cmax_vals.append(cmax)
                auc_vals.append(auc)
                tmax_vals.append(tmax)
                thalf_vals.append(thalf)

                # ── Therapeutic response & adverse flag ──────────────────
                in_window  = int(t_lo <= cmax <= t_hi)
                above_hi   = int(cmax > t_hi * 1.5)   # rough "adverse" threshold

                sg_traces[sg].append(plasma)
                sg_cmax[sg].append(cmax)
                sg_auc[sg].append(auc)
                sg_resp[sg].append(in_window)
                sg_adv[sg].append(above_hi)

                demographics.append({
                    "patient":          i + 1,
                    "subgroup":         sg,
                    "weight_kg":        round(weight, 1),
                    "age":              round(age, 0),
                    "sex":              "F" if sex == 0 else "M",
                    "kidney_function":  round(kf, 2),
                    "liver_function":   round(lf, 2),
                    "CL (L/h)":         round(patient_drug.clearance, 2),
                    "Vd (L)":           round(patient_drug.volume_distribution, 1),
                    "Cmax (mg/L)":      round(cmax, 4),
                    "AUC (mg·h/L)":     round(auc, 2),
                    "in_window":        bool(in_window),
                    "adverse":          bool(above_hi),
                })
                n_succeeded += 1

            except Exception as exc:
                logger.debug("Patient %d simulation failed: %s", i, exc)

        if not plasma_traces:
            raise RuntimeError("All patient simulations failed in population cohort")

        traces    = np.array(plasma_traces)
        cmax_arr  = np.array(cmax_vals)
        frac_win  = float(np.sum((cmax_arr >= t_lo) & (cmax_arr <= t_hi)) / len(cmax_arr))
        adv_rate  = float(np.sum(cmax_arr > t_hi * 1.5) / len(cmax_arr))

        # ── Subgroup summary ─────────────────────────────────────────────────
        subgroup_summary: Dict = {}
        for sg_key in SUBGROUP_LABELS:
            n_sg = len(sg_cmax[sg_key])
            if n_sg == 0:
                continue
            sg_c = np.array(sg_cmax[sg_key])
            sg_a = np.array(sg_auc[sg_key])
            subgroup_summary[sg_key] = {
                "n":                  n_sg,
                "cmax_median":        round(float(np.median(sg_c)), 4),
                "auc_median":         round(float(np.median(sg_a)), 2),
                "response_rate":      round(float(np.mean(sg_resp[sg_key])), 4),
                "adverse_rate":       round(float(np.mean(sg_adv[sg_key])), 4),
                "plasma_p50":         [round(float(v), 4) for v in np.percentile(np.array(sg_traces[sg_key]), 50, axis=0)],
                "plasma_p05":         [round(float(v), 4) for v in np.percentile(np.array(sg_traces[sg_key]), 5,  axis=0)],
                "plasma_p95":         [round(float(v), 4) for v in np.percentile(np.array(sg_traces[sg_key]), 95, axis=0)],
            }

        return CohortResult(
            time                 = time_points,
            plasma_p05           = np.percentile(traces, 5,  axis=0),
            plasma_p25           = np.percentile(traces, 25, axis=0),
            plasma_p50           = np.percentile(traces, 50, axis=0),
            plasma_p75           = np.percentile(traces, 75, axis=0),
            plasma_p95           = np.percentile(traces, 95, axis=0),
            cmax_distribution    = cmax_vals,
            auc_distribution     = auc_vals,
            tmax_distribution    = tmax_vals,
            t_half_distribution  = thalf_vals,
            therapeutic_window   = (t_lo, t_hi),
            frac_within_window   = frac_win,
            n_patients           = n,
            n_succeeded          = n_succeeded,
            patient_demographics = demographics,
            subgroup_summary     = subgroup_summary,
            overall_response_rate = frac_win,
            overall_adverse_rate  = adv_rate,
            setting              = setting,
            density_km2          = profile["density_km2"],
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

# ── Helpers ──────────────────────────────────────────────────────────────────

def _clone_namespace(obj) -> SimpleNamespace:
    """Shallow-copy a dataclass or namespace into a SimpleNamespace so fields
    can be mutated without affecting the original.
    """
    ns = SimpleNamespace()
    for attr in dir(obj):
        if not attr.startswith("_"):
            try:
                setattr(ns, attr, getattr(obj, attr))
            except Exception:
                pass
    return ns
