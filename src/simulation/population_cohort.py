"""
Population PK/PD Cohort Runner
================================

Generates a virtual patient population and runs mechanistic PK/PD simulations
for each patient, returning percentile statistics suitable for dashboard plotting.

Key design decisions:
- Uses the same MechanisticPKPDEngine instance passed in (no re-instantiation).
- Patient variability is driven by allometric scaling + log-normal IIV on CL/Vd.
- BiologicalParameters / DrugProperties are cloned via SimpleNamespace so the
  original objects are never mutated.
- All n patients are simulated synchronously; each call to simulate_drug is
  < 5 ms (analytical Bateman), so n=100 takes ~0.5 s.

Usage::

    from simulation.population_cohort import PopulationCohortRunner, CohortResult

    runner = PopulationCohortRunner()
    result = runner.run(engine, drug, base_bio, diseases, n=100, time_points=None)
    # result.plasma_p50  — median plasma profile
    # result.plasma_p05  — 5th percentile
    # result.plasma_p95  — 95th percentile
    # result.cmax_distribution  — list of Cmax values across patients
    # result.frac_within_window — fraction of population within therapeutic window
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
_IIV_VD_CV   = 0.25   # 25% coefficient of variation on Vd (log-normal)
_IIV_BA_CV   = 0.15   # 15% CV on bioavailability (log-normal, clamped 0.05-1)
_IIV_KF_SD   = 0.15   # additive SD on kidney_function (normal, clipped 0.3-1.0)
_IIV_LF_SD   = 0.12   # additive SD on liver_function  (normal, clipped 0.5-1.0)
_IIV_WT_SD   = 12.0   # kg SD on body weight (normal, clipped 40-140)


@dataclass
class CohortResult:
    """Statistics from a population cohort simulation."""

    time: np.ndarray                         # shared time axis (h)
    plasma_p05: np.ndarray                   # 5th  percentile envelope
    plasma_p25: np.ndarray                   # 25th percentile envelope
    plasma_p50: np.ndarray                   # median
    plasma_p75: np.ndarray                   # 75th percentile envelope
    plasma_p95: np.ndarray                   # 95th percentile envelope
    cmax_distribution: List[float]           # Cmax per patient (mg/L)
    auc_distribution: List[float]            # AUClast per patient (mg·h/L)
    tmax_distribution: List[float]           # Tmax per patient (h)
    t_half_distribution: List[float]         # t½ per patient (h)
    therapeutic_window: tuple                # (lo, hi) mg/L
    frac_within_window: float                # fraction of patients with Cmax in window
    n_patients: int                          # number of simulated patients
    n_succeeded: int                         # patients that completed without error
    patient_demographics: List[Dict]         # weight, age, kf, lf, sex per patient


class PopulationCohortRunner:
    """
    Runs n virtual patient simulations and returns percentile statistics.

    Thread-safety: not guaranteed (reuses the passed engine instance).
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
    ) -> CohortResult:
        """
        Run a population cohort simulation.

        Parameters
        ----------
        engine     : MechanisticPKPDEngine instance
        base_drug  : DrugProperties — reference (mean) drug object
        base_bio   : BiologicalParameters — reference (mean) patient
        diseases   : list of DiseaseState — shared across population
        n          : number of virtual patients (default 50)
        time_points: optional time grid; defaults to 0–96 h at 1 h resolution
        seed       : optional random seed for reproducibility

        Returns
        -------
        CohortResult
        """
        if time_points is None:
            t_end = float(getattr(base_drug, "half_life", 8.0)) * 8.0
            t_end = float(np.clip(t_end, 24.0, 168.0))  # 1 day – 1 week
            time_points = np.linspace(0.0, t_end, max(int(t_end) + 1, 97))

        rng = np.random.default_rng(seed)
        t_window = getattr(base_drug, "therapeutic_window", (1.0, 100.0))
        t_lo, t_hi = float(t_window[0]), float(t_window[1])

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
                # ── Patient demographics ─────────────────────────────────
                weight = float(np.clip(rng.normal(base_wt, _IIV_WT_SD), 40.0, 140.0))
                age    = float(np.clip(rng.normal(base_age, 15.0), 18.0, 85.0))
                sex    = int(rng.integers(0, 2))
                kf     = float(np.clip(rng.normal(base_kf, _IIV_KF_SD), 0.30, 1.0))
                lf     = float(np.clip(rng.normal(base_lf, _IIV_LF_SD), 0.50, 1.0))

                # ── Allometric PK scaling ────────────────────────────────
                #   CL ∝ (weight/70)^0.75  — standard allometric exponent
                #   Vd ∝  weight/70        — linear volume scaling
                allom_wt = weight / 70.0
                cl_allom = base_cl * (allom_wt ** 0.75)
                vd_allom = base_vd * allom_wt

                # ── Inter-individual variability (log-normal IIV) ────────
                cl_patient = cl_allom * float(
                    np.exp(rng.normal(0.0, _IIV_CL_CV))
                )
                vd_patient = vd_allom * float(
                    np.exp(rng.normal(0.0, _IIV_VD_CV))
                )
                ba_patient = float(np.clip(
                    base_ba * np.exp(rng.normal(0.0, _IIV_BA_CV)), 0.05, 1.0
                ))

                # ── Age effect on CL: > 65 → -1% per year ───────────────
                if age > 65:
                    age_factor = max(1.0 - 0.01 * (age - 65), 0.5)
                    cl_patient *= age_factor

                # ── Sex effect on Vd: female → ~+15% adipose ────────────
                if sex == 0:   # female
                    vd_patient *= 1.08  # mild Vd increase (lipophilic distribution)

                # ── Build per-patient objects ────────────────────────────
                patient_bio = _clone_namespace(base_bio)
                patient_bio.weight          = weight
                patient_bio.age             = age
                patient_bio.sex             = sex
                patient_bio.kidney_function = kf
                patient_bio.liver_function  = lf

                patient_drug = _clone_namespace(base_drug)
                patient_drug.clearance             = float(np.clip(cl_patient, 0.1, 500.0))
                patient_drug.volume_distribution   = float(np.clip(vd_patient, 1.0, 5000.0))
                patient_drug.bioavailability       = ba_patient

                # ── Simulate (no ML correction for speed) ────────────────
                result = engine.simulate_drug(
                    patient_drug, patient_bio, diseases,
                    time_points, apply_ml_correction=False,
                )
                plasma_traces.append(result.plasma_conc)
                cmax_vals.append(result.pk_stats.get("Cmax (mg/L)", float(np.max(result.plasma_conc))))
                auc_vals.append(result.pk_stats.get("AUClast (mg·h/L)", 0.0))
                tmax_vals.append(result.pk_stats.get("Tmax (h)", 0.0))
                thalf_vals.append(result.pk_stats.get("t½ (h)", 0.0))
                demographics.append({
                    "patient": i + 1,
                    "weight_kg": round(weight, 1),
                    "age": round(age, 0),
                    "sex": "F" if sex == 0 else "M",
                    "kidney_function": round(kf, 2),
                    "liver_function": round(lf, 2),
                    "CL (L/h)": round(patient_drug.clearance, 2),
                    "Vd (L)": round(patient_drug.volume_distribution, 1),
                })
                n_succeeded += 1

            except Exception as exc:
                logger.debug("Patient %d simulation failed: %s", i, exc)

        if not plasma_traces:
            raise RuntimeError("All patient simulations failed in population cohort")

        traces = np.array(plasma_traces)   # (n_succeeded, T)
        cmax_arr = np.array(cmax_vals)
        frac_window = float(np.sum((cmax_arr >= t_lo) & (cmax_arr <= t_hi)) / len(cmax_arr))

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
            frac_within_window   = frac_window,
            n_patients           = n,
            n_succeeded          = n_succeeded,
            patient_demographics = demographics,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clone_namespace(obj) -> SimpleNamespace:
    """
    Shallow-copy a dataclass or namespace into a SimpleNamespace so fields
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
