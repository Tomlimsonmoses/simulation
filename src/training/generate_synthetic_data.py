"""
Synthetic PK-PD Data Generator
================================
Generates physiologically realistic pharmacokinetic / pharmacodynamic training
samples using established mathematical models:

  PK  —  Bateman equation (oral, 1-compartment) with log-normal IIV
  PD  —  Sigmoid Emax (Hill equation) with inter-individual EC50 variability

Saves to  data/raw/synthetic_clinical_data.csv  (same schema as clinical_trials.csv).

Usage:
    python -m src.training.generate_synthetic_data          # 50 000 samples
    python src/training/generate_synthetic_data.py --n 100000
"""

from __future__ import annotations
import argparse
import math
import os
import random
import sys
import csv

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Reference PK parameters (literature-based) ────────────────────────────────
# Each drug: {bioavailability, ka (1/h), ke (1/h), Vd_ref (L/kg),
#              Emax, EC50 (mg/L), Hill_n, drug_class, route}
_DRUG_PARAMS = {
    "Lisinopril":      dict(F=0.25, ka=0.8,  ke=0.08, Vd=0.53, Emax=1.0, EC50=0.06, n=1.5, cls="small_molecule", route="oral"),
    "Atorvastatin":    dict(F=0.14, ka=1.5,  ke=0.18, Vd=5.14, Emax=0.9, EC50=0.10, n=1.2, cls="small_molecule", route="oral"),
    "Metformin":       dict(F=0.55, ka=0.6,  ke=0.11, Vd=1.00, Emax=0.8, EC50=0.20, n=1.0, cls="small_molecule", route="oral"),
    "Amlodipine":      dict(F=0.64, ka=0.5,  ke=0.025,Vd=20.0, Emax=1.0, EC50=0.005,n=1.8, cls="small_molecule", route="oral"),
    "Sertraline":      dict(F=0.44, ka=0.7,  ke=0.026,Vd=20.0, Emax=0.85,EC50=0.005,n=1.5, cls="small_molecule", route="oral"),
    "Metoprolol":      dict(F=0.40, ka=1.2,  ke=0.11, Vd=5.60, Emax=0.9, EC50=0.05, n=1.3, cls="small_molecule", route="oral"),
    "Prednisone":      dict(F=0.80, ka=2.0,  ke=0.25, Vd=0.97, Emax=1.0, EC50=0.08, n=2.0, cls="small_molecule", route="oral"),
    "Ciprofloxacin":   dict(F=0.70, ka=1.5,  ke=0.23, Vd=2.50, Emax=0.95,EC50=0.12, n=1.4, cls="small_molecule", route="oral"),
    "Amoxicillin":     dict(F=0.93, ka=2.5,  ke=0.70, Vd=0.30, Emax=0.85,EC50=0.30, n=1.0, cls="small_molecule", route="oral"),
    "Acetaminophen":   dict(F=0.88, ka=3.0,  ke=0.35, Vd=0.85, Emax=0.80,EC50=4.0,  n=1.2, cls="small_molecule", route="oral"),
    "Ibuprofen":       dict(F=0.87, ka=2.5,  ke=0.40, Vd=0.15, Emax=0.90,EC50=3.0,  n=1.5, cls="small_molecule", route="oral"),
    "Naproxen":        dict(F=0.99, ka=1.8,  ke=0.065,Vd=0.16, Emax=0.88,EC50=5.0,  n=1.3, cls="small_molecule", route="oral"),
    "Erythromycin":    dict(F=0.35, ka=1.0,  ke=0.32, Vd=0.74, Emax=0.85,EC50=0.50, n=1.1, cls="small_molecule", route="oral"),
    "Glipizide":       dict(F=0.97, ka=1.5,  ke=0.20, Vd=0.30, Emax=0.75,EC50=0.20, n=1.4, cls="small_molecule", route="oral"),
    "Levothyroxine":   dict(F=0.70, ka=0.3,  ke=0.004,Vd=10.0, Emax=0.70,EC50=0.04, n=1.0, cls="small_molecule", route="oral"),
    "Lorazepam":       dict(F=0.92, ka=1.8,  ke=0.043,Vd=1.30, Emax=0.95,EC50=0.02, n=2.5, cls="small_molecule", route="oral"),
    "Atorvastatin_IV": dict(F=1.00, ka=99.0, ke=0.18, Vd=5.14, Emax=0.9, EC50=0.10, n=1.2, cls="small_molecule", route="intravenous"),
    "Ciprofloxacin_IV":dict(F=1.00, ka=99.0, ke=0.23, Vd=2.50, Emax=0.95,EC50=0.12, n=1.4, cls="small_molecule", route="intravenous"),
}

_DRUG_NAMES  = list(_DRUG_PARAMS.keys())
_ETHNICITIES = ["Caucasian", "Asian", "African", "Hispanic", "Other"]
_SEXES       = ["M", "F"]
_DISEASES    = [
    "Hypertension", "Diabetes", "Hyperlipidemia", "Infection",
    "Anxiety", "Asthma", "Heart Failure", "Arthritis", "Hypothyroidism",
]


def _bateman(t: float, dose: float, F: float, ka: float, ke: float,
             Vd_L: float) -> float:
    """Plasma concentration (mg/L) at time t (h) — Bateman equation."""
    if ka <= ke:                         # degenerate: use approximation
        ka = ke * 1.001
    c = (F * dose * ka) / (Vd_L * (ka - ke)) * (
        math.exp(-ke * t) - math.exp(-ka * t)
    )
    return max(0.0, c)


def _emax(conc: float, Emax: float, EC50: float, n: float) -> float:
    """Sigmoid Emax (Hill) model — normalised response [0, 1]."""
    cn = conc ** n
    ec50n = EC50 ** n
    return Emax * cn / (ec50n + cn + 1e-12)


def _allometric_vd(Vd_ref_per_kg: float, weight_kg: float) -> float:
    """Allometric scaling of volume of distribution."""
    return Vd_ref_per_kg * (weight_kg ** 0.75)


def _renal_cl_adjust(ke: float, kidney_fn: float) -> float:
    """Scale elimination for renal impairment (linear approximation)."""
    return ke * (0.3 + 0.7 * kidney_fn)


def _hepatic_cl_adjust(ka: float, liver_fn: float) -> float:
    """Reduce first-pass absorption for hepatic impairment."""
    return ka * (0.4 + 0.6 * liver_fn)


def generate_samples(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    trial_id = 1

    for _ in range(n):
        drug_name = rng.choice(_DRUG_NAMES)
        p = _DRUG_PARAMS[drug_name]

        # Patient demographics
        age      = float(rng.integers(18, 90))
        weight   = float(np.clip(rng.normal(75, 15), 40, 180))
        sex      = rng.choice(_SEXES)
        ethnicity = rng.choice(_ETHNICITIES)
        disease  = rng.choice(_DISEASES)
        severity = float(rng.beta(2, 3))          # skewed toward mild

        # Organ function (inter-individual variability)
        kidney_fn = float(np.clip(rng.beta(5, 2), 0.1, 1.0))   # most patients >0.5
        liver_fn  = float(np.clip(rng.beta(6, 2), 0.1, 1.0))

        # Dose with ±20 % variability around a sensible range per drug
        base_dose = p["dose"] if "dose" in p else 100.0

        # Add log-normal IIV to PK parameters (CV ≈ 30%)
        omega = 0.30
        F_i   = float(np.clip(p["F"]  * rng.lognormal(0, omega), 0.05, 1.0))
        ka_i  = float(np.clip(_hepatic_cl_adjust(p["ka"], liver_fn) * rng.lognormal(0, omega), 0.05, 20.0))
        ke_i  = float(np.clip(_renal_cl_adjust(p["ke"], kidney_fn)  * rng.lognormal(0, omega), 0.005, 5.0))
        Vd_L  = _allometric_vd(p["Vd"], weight)
        Vd_i  = float(np.clip(Vd_L * rng.lognormal(0, omega * 0.5), 1.0, 500.0))

        # EC50 IIV (pharmacodynamic variability)
        ec50_i = float(np.clip(p["EC50"] * rng.lognormal(0, omega), p["EC50"] * 0.1, p["EC50"] * 10))
        emax_i = float(np.clip(p["Emax"] + rng.normal(0, 0.05), 0.1, 1.2))
        n_hill = p["n"]

        # Dose (mg) — choose realistic values with variability
        dose_options = [base_dose * f for f in [0.25, 0.5, 1.0, 1.5, 2.0]]
        dose = float(rng.choice(dose_options)) * float(rng.lognormal(0, 0.10))
        dose = float(np.clip(dose, base_dose * 0.1, base_dose * 4.0))

        # Multiple time points per patient (4–8 samples)
        n_times = int(rng.integers(4, 9))
        t_half  = math.log(2) / max(ke_i, 1e-6)
        t_max   = max(24.0, t_half * 5)
        time_pts = sorted(rng.uniform(0.25, t_max, n_times).tolist())

        for t in time_pts:
            conc = _bateman(t, dose, F_i, ka_i, ke_i, Vd_i)
            # Proportional residual error (σ = 15 %)
            eps_pk = float(rng.normal(0, conc * 0.15 + 0.001))
            conc_obs = float(max(0.0, conc + eps_pk))

            resp = _emax(conc_obs, emax_i, ec50_i, n_hill) * 100.0   # scale to 0-100
            eps_pd = float(rng.normal(0, resp * 0.12 + 0.1))
            resp_obs = float(np.clip(resp + eps_pd, 0.0, 100.0))

            adverse = bool(conc_obs > (ec50_i * 5))   # proxy toxicity flag

            records.append({
                "trial_id":      trial_id,
                "drug_name":     drug_name,
                "dose":          round(dose, 3),
                "time_point":    round(t, 3),
                "age":           int(age),
                "weight":        round(weight, 1),
                "sex":           sex,
                "ethnicity":     ethnicity,
                "disease":       disease,
                "severity":      round(severity, 4),
                "kidney_fn":     round(kidney_fn, 4),
                "liver_fn":      round(liver_fn, 4),
                "concentration": round(conc_obs, 6),
                "response_value":round(resp_obs, 4),
                "adverse_event": adverse,
                # True PK parameters (useful for physics-informed training)
                "true_ke":       round(ke_i, 6),
                "true_ka":       round(ka_i, 6),
                "true_Vd":       round(Vd_i, 3),
                "true_F":        round(F_i, 4),
            })
            trial_id += 1

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PK-PD training data")
    parser.add_argument("--n",    type=int, default=50_000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str, default=os.path.join(_ROOT, "data", "raw", "synthetic_clinical_data.csv"))
    args = parser.parse_args()

    print(f"Generating {args.n:,} synthetic PK-PD samples (seed={args.seed})…")
    df = generate_samples(n=args.n, seed=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows -> {args.out}")
    print(f"  Drugs:        {df['drug_name'].nunique()} unique")
    print(f"  Ethnicities:  {df['ethnicity'].nunique()} groups")
    print(f"  Concentration range: [{df['concentration'].min():.4f}, {df['concentration'].max():.4f}] mg/L")
    print(f"  Response range:      [{df['response_value'].min():.2f}, {df['response_value'].max():.2f}]")
    print(f"  Adverse events:      {df['adverse_event'].sum()} ({df['adverse_event'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
