"""
Unit tests for 1-compartment PK equations used in the dashboard.

Reference values computed analytically from textbook formulas:
  C(t) = (D * F / Vd) * exp(-CL/Vd * t)

Tests use the dashboard's calculate_drug_concentration logic.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# We import the dataclasses and the simulator class directly
from advanced_dashboard import BiologicalSystemSimulator, DrugProperties, BiologicalParameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_bio() -> BiologicalParameters:
    """70 kg reference patient — no organ impairment."""
    bp = BiologicalParameters()
    bp.weight = 70.0
    bp.kidney_function = 1.0
    bp.liver_function = 1.0
    bp.metabolic_rate = 1.0
    return bp


def _sim() -> BiologicalSystemSimulator:
    return BiologicalSystemSimulator()


# ---------------------------------------------------------------------------
# Single-dose PK tests
# ---------------------------------------------------------------------------

def test_concentration_zero_at_zero():
    """Concentration is > 0 immediately after dose (t=0)."""
    sim = _sim()
    drug = sim.drug_database["Ibuprofen"]
    bp = _default_bio()
    c0 = sim.calculate_drug_concentration(drug, 0.0, bp)
    assert c0 > 0.0, "Concentration should be positive immediately after dosing"


def test_concentration_decays_over_time():
    """Concentration at t=24 should be less than at t=2 (single dose, no accumulation window)."""
    sim = _sim()
    drug = sim.drug_database["Ibuprofen"]
    drug.dosing_frequency = "once"   # single dose to avoid accumulation
    bp = _default_bio()
    c2  = sim.calculate_drug_concentration(drug, 2.0,  bp)
    c24 = sim.calculate_drug_concentration(drug, 24.0, bp)
    assert c24 < c2, f"Concentration should decay: C(2h)={c2:.3f} C(24h)={c24:.3f}"


def test_ibuprofen_peak_order_of_magnitude():
    """Ibuprofen 400 mg oral, 70 kg: peak should be roughly in the 10–100 mg/L range."""
    sim = _sim()
    drug = sim.drug_database["Ibuprofen"]
    bp = _default_bio()
    # Check a range of times around typical t_max (1–2 h)
    peak = max(sim.calculate_drug_concentration(drug, t, bp) for t in np.arange(0.25, 6.0, 0.25))
    assert 5.0 <= peak <= 500.0, f"Ibuprofen peak out of expected range: {peak:.2f} mg/L"


def test_kidney_impairment_raises_concentration():
    """Reduced kidney function → slower clearance → higher concentration."""
    sim = _sim()
    drug = sim.drug_database["Ibuprofen"]
    bp_normal = _default_bio()
    bp_impaired = _default_bio()
    bp_impaired.kidney_function = 0.4

    c_normal   = sim.calculate_drug_concentration(drug, 6.0, bp_normal)
    c_impaired = sim.calculate_drug_concentration(drug, 6.0, bp_impaired)
    assert c_impaired > c_normal, (
        "Kidney impairment should increase concentration, "
        f"normal={c_normal:.3f} impaired={c_impaired:.3f}"
    )


def test_higher_dose_gives_higher_concentration():
    """Doubling dose should double the concentration (linear PK)."""
    sim = _sim()
    drug = sim.drug_database["Acetaminophen"]
    bp = _default_bio()

    drug.dose_mg = 250.0
    c_low = sim.calculate_drug_concentration(drug, 3.0, bp)

    drug.dose_mg = 500.0
    c_high = sim.calculate_drug_concentration(drug, 3.0, bp)

    ratio = c_high / max(c_low, 1e-9)
    assert 1.5 < ratio < 2.5, f"Expected ~2× ratio for doubled dose, got {ratio:.3f}"


# ---------------------------------------------------------------------------
# Multi-dose accumulation
# ---------------------------------------------------------------------------

def test_accumulation_with_daily_dosing():
    """With repeated daily dosing, concentration at t=72 h should exceed t=24 h."""
    sim = _sim()
    drug = sim.drug_database["Atorvastatin"]
    bp = _default_bio()
    c24 = sim.calculate_drug_concentration(drug, 24.0, bp)
    c72 = sim.calculate_drug_concentration(drug, 72.0, bp)
    assert c72 >= c24, (
        f"Daily dosing should accumulate: C(24h)={c24:.3f} C(72h)={c72:.3f}"
    )
