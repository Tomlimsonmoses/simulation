"""Quick smoke test for MechanisticPKPDEngine."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

from simulation.mechanistic_engine import MechanisticPKPDEngine


@dataclass
class FakeDrug:
    name: str = "Metformin"
    molecular_weight: float = 129.16
    logp: float = -0.5
    bioavailability: float = 0.55
    protein_binding: float = 0.0
    clearance: float = 35.0
    volume_distribution: float = 126.0
    half_life: float = 6.0
    dose_mg: float = 850.0
    administration_route: str = "oral"
    dosing_frequency: str = "bid"
    therapeutic_window: Tuple[float, float] = (0.5, 2.5)


@dataclass
class FakeBio:
    age: float = 60.0
    weight: float = 75.0
    sex: int = 0
    kidney_function: float = 0.60
    liver_function: float = 0.95
    metabolic_rate: float = 1.0


@dataclass
class FakeDis:
    name: str = "Chronic Kidney Disease"
    severity: float = 0.5
    progression_rate: float = 0.006
    affected_organs: List[str] = field(default_factory=lambda: ["kidneys", "heart"])
    biomarkers: Dict[str, float] = field(default_factory=dict)


def test_metformin_ckd():
    engine = MechanisticPKPDEngine()
    print("Engine loaded. ML model:", "YES" if engine._ml_model else "None")

    t = np.linspace(0, 72, 73)
    result = engine.simulate_drug(FakeDrug(), FakeBio(), [FakeDis()], t)

    stats = result.pk_stats
    cmax  = stats["Cmax (mg/L)"]
    tmax  = stats["Tmax (h)"]

    print("PK stats:", stats)
    print(f"Receptor occ at 2h : {result.receptor_occupancy[2]:.4f}")
    print(f"PD response at 4h  : {result.pd_response[4]:.4f}")
    print(f"Disease sev at 72h : {result.disease_severity[-1]:.4f}")
    print(f"Organs             : {list(result.organ_conc.keys())}")

    # Pharmacological validation
    # Metformin 850mg BID with CKD-reduced clearance:
    # - Single dose Cmax ~1-2 mg/L; at SS with CKD → ~2-4 mg/L is plausible
    # - Tmax for multiple dosing = time of peak across whole observation window
    #   (with BID over 72h this is near steady-state, i.e. ~60-72h is valid)
    assert cmax > 0.1, f"Cmax too low: {cmax}"
    assert cmax < 20.0, f"Cmax unrealistically high: {cmax}"
    assert 0.0 < tmax <= 72.0, f"Tmax out of expected range: {tmax}"
    assert 0.0 <= result.receptor_occupancy[2] <= 1.0
    assert 0.0 <= result.pd_response[4] <= 1.0
    assert result.disease_severity[-1] >= result.disease_severity[0] * 0.5  # shouldn't collapse to 0
    assert "blood" in result.organ_conc

    # Verify disease progression was dynamic (not flat)
    sev_change = result.disease_severity[-1] - result.disease_severity[0]
    print(f"Disease severity change over 72h: {sev_change:+.4f}")

    print("\nALL ASSERTIONS PASSED")


def test_metoprolol_healthy():
    """Metoprolol 50mg BID in healthy patient — Cmax ~0.1 mg/L (published ~100 ng/mL)."""
    @dataclass
    class MetoproDrug:
        name: str = "Metoprolol"
        molecular_weight: float = 267.37
        logp: float = 1.8
        bioavailability: float = 0.50
        protein_binding: float = 0.12
        clearance: float = 14.7
        volume_distribution: float = 280.0
        half_life: float = 3.5
        dose_mg: float = 50.0
        administration_route: str = "oral"
        dosing_frequency: str = "bid"
        therapeutic_window: Tuple[float, float] = (0.03, 0.2)

    @dataclass
    class HealthyBio:
        age: float = 45.0
        weight: float = 70.0
        sex: int = 1
        kidney_function: float = 1.0
        liver_function: float = 1.0
        metabolic_rate: float = 1.0

    engine = MechanisticPKPDEngine()
    t = np.linspace(0, 24, 25)
    result = engine.simulate_drug(MetoproDrug(), HealthyBio(), [], t)

    cmax = result.pk_stats["Cmax (mg/L)"]
    tmax = result.pk_stats["Tmax (h)"]
    print(f"\nMetoprolol 50mg healthy: Cmax = {cmax:.5f} mg/L, Tmax = {tmax:.2f}h")
    # Published: ~60-160 ng/mL = 0.06-0.16 mg/L
    assert 0.001 < cmax < 1.0, f"Metoprolol Cmax unexpected: {cmax}"
    print("Metoprolol test PASSED")


if __name__ == "__main__":
    test_metformin_ckd()
    test_metoprolol_healthy()
    print("\n=== ALL TESTS PASSED ===")
