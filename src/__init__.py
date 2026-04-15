"""
PK-PD Biological System Simulator — Python package.

Live simulation stack
---------------------
MechanisticPKPDEngine  — primary ODE/Bateman PBPK/PD engine (app/server.py)
PopulationCohortRunner — virtual patient cohort (/api/cohort endpoint)
OrganGraphBuilder      — builds PyG organ graph for optional GNN inference

Training utilities (CLI, not used by live server)
--------------------------------------------------
  python -m src.training.train          # train SimpleMLP / DeepMLP / ResidualMLP
  python -m src.training.train_staged   # train StagedPKPDModel
  python -m src.training.evaluate       # evaluate all checkpoints
"""

__version__ = "1.0.0"
__author__ = "PK-PD Research Team"

# Live simulation exports
from .simulation.mechanistic_engine import MechanisticPKPDEngine, EngineResult
from .simulation.population_cohort import PopulationCohortRunner
from .simulation.organ_graph_builder import OrganGraphBuilder
from .models.mlp_architectures import SimpleMLP, DeepMLP, ResidualMLP
from .models.pkpd_staged import StagedPKPDModel

__all__ = [
    "MechanisticPKPDEngine",
    "EngineResult",
    "PopulationCohortRunner",
    "OrganGraphBuilder",
    "SimpleMLP",
    "DeepMLP",
    "ResidualMLP",
    "StagedPKPDModel",
]
