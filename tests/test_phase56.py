"""
Smoke tests for Phase 5 (GNN Integration) and Phase 6 (Population Cohort).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Phase 5 ────────────────────────────────────────────────────────────────
from src.simulation.organ_graph_builder import OrganGraphBuilder, gnn_infer_organ_pd
from src.simulation.mechanistic_engine import MechanisticPKPDEngine, EngineResult
from dataclasses import dataclass, field
from typing import Tuple, Dict, List


# Minimal dataclasses matching dashboard DrugProperties / BiologicalParameters
@dataclass
class Drug:
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
class Bio:
    age: float = 60.0
    weight: float = 75.0
    sex: int = 0
    kidney_function: float = 0.60
    liver_function: float = 0.95
    metabolic_rate: float = 1.0


@dataclass
class Disease:
    name: str = "Chronic Kidney Disease"
    severity: float = 0.5
    progression_rate: float = 0.006
    affected_organs: List[str] = field(default_factory=lambda: ["kidneys", "heart"])
    biomarkers: Dict[str, float] = field(default_factory=dict)


def test_organ_graph_builder():
    builder = OrganGraphBuilder()
    organ_conc = {
        "brain": 12.1, "heart": 5.3, "liver": 4.8, "kidney": 6.1,
        "lung": 3.2, "muscle": 2.1, "fat": 0.9, "skin": 1.8,
        "gut": 3.5, "spleen": 2.4, "blood": 5.0,
    }
    data = builder.build_graph(
        organ_conc,
        drug_mw=267.0, drug_logp=1.8, drug_ba=0.50, drug_pb=0.12,
        drug_cl=14.7, drug_vd=280.0, drug_hl=3.5, drug_dose=50.0,
        disease_severity=0.3, patient_weight=75.0,
    )
    assert data.x.shape == (13, 16), f"x.shape={data.x.shape}"
    assert data.edge_index.shape[0] == 2
    assert data.drug_features.shape == (1, 8)
    assert data.disease_features.shape == (1, 4)
    print(f"  Edges: {data.edge_index.shape[1]}")
    print("test_organ_graph_builder PASSED")


def test_engine_gnn_fields():
    engine = MechanisticPKPDEngine()
    t = np.linspace(0, 48, 97)
    result = engine.simulate_drug(Drug(), Bio(), [Disease()], t)

    # GNN checkpoint doesn't exist, so gnn_available should be False
    assert isinstance(result.gnn_available, bool)
    print(f"  gnn_available = {result.gnn_available}")
    print(f"  gnn_organ_pd  = {result.gnn_organ_pd}")

    # Existing PK should still work
    pk = result.pk_stats
    assert pk["Cmax (mg/L)"] > 0.1, f"Cmax too low: {pk['Cmax (mg/L)']}"
    print(f"  Cmax = {pk['Cmax (mg/L)']:.4f} mg/L")
    print("test_engine_gnn_fields PASSED")


def test_population_cohort():
    from src.simulation.population_cohort import PopulationCohortRunner

    engine = MechanisticPKPDEngine()
    runner = PopulationCohortRunner()
    t = np.linspace(0, 48, 49)
    cohort = runner.run(engine, Drug(), Bio(), [Disease()], n=20, time_points=t, seed=99)

    assert cohort.n_succeeded > 0
    assert len(cohort.cmax_distribution) == cohort.n_succeeded
    assert cohort.plasma_p50.shape == t.shape
    assert 0.0 <= cohort.frac_within_window <= 1.0

    cmax_arr = np.array(cohort.cmax_distribution)
    cv = 100 * cmax_arr.std() / max(cmax_arr.mean(), 1e-9)
    print(f"  n_succeeded = {cohort.n_succeeded}/{cohort.n_patients}")
    print(f"  Cmax median = {np.median(cmax_arr):.4f} mg/L")
    print(f"  Cmax CV%    = {cv:.1f}%")
    print(f"  frac_within_window = {cohort.frac_within_window:.1%}")
    assert 5.0 < cv < 80.0, f"CV% {cv:.1f}% out of expected range"
    print("test_population_cohort PASSED")


if __name__ == "__main__":
    print("=== Phase 5-6 Smoke Tests ===")
    test_organ_graph_builder()
    print()
    test_engine_gnn_fields()
    print()
    test_population_cohort()
    print()
    print("ALL TESTS PASSED")
