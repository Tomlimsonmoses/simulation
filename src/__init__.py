"""
GNN-Based PK-PD Model for Organ Drug Response

This package implements a comprehensive Graph Neural Network approach for
Pharmacokinetic-Pharmacodynamic (PK-PD) modeling to simulate organ responses
to drugs in virtual human biological systems.

Main Components:
- Models: GNN architecture, PK simulator, PD simulator
- Biological System: Virtual human, disease models
- Data: Dataset specifications, processing pipeline
- Training: Training framework, evaluation metrics
- Simulation: Main simulation runner with customizable parameters
"""

__version__ = "1.0.0"
__author__ = "PK-PD Research Team"

# Core imports
from .models.gnn_pkpd import PKPDGNN, PKPDDataProcessor, PKPDTrainer
from .models.pk_simulator import MultiCompartmentPK, PhysiologicallyBasedPK, DrugProperties
from .models.pd_simulator import SystemicPDModel, OrganSpecificPD
from .biological_system.organ_network import VirtualHuman, PopulationGenerator
from .biological_system.disease_models import DiseaseSimulator, DiseaseDrugInteractions
from .simulation.runner import PKPDSimulationRunner, SimulationConfig
from .data.dataset_specs import DatasetRequirements, DataPipeline
from .training.trainer import PKPDTrainer, PKPDLossFunction

__all__ = [
    # Core classes
    "PKPDGNN",
    "PKPDSimulationRunner",
    "VirtualHuman",
    "DiseaseSimulator",
    
    # Model components
    "MultiCompartmentPK",
    "PhysiologicallyBasedPK",
    "SystemicPDModel",
    "OrganSpecificPD",
    
    # Configuration
    "SimulationConfig",
    "DrugProperties",
    
    # Data and training
    "DatasetRequirements",
    "DataPipeline",
    "PKPDTrainer",
    "PKPDLossFunction",
    
    # Utilities
    "PopulationGenerator",
    "DiseaseDrugInteractions",
    "PKPDDataProcessor"
]
