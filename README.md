# GNN-Based PK-PD Model for Organ Drug Response

This project implements a Graph Neural Network (GNN) and Deep Multilayer Perceptron (MLP) approach for Pharmacokinetic-Pharmacodynamic (PK-PD) modeling to simulate organ responses to drugs in a virtual human biological system. 

It handles:
- **Pharmacokinetics (PK)**: How drugs move through the body (absorption, distribution, metabolism, excretion)
- **Pharmacodynamics (PD)**: How organs respond to drugs (receptor binding, signal transduction, physiological effects)
- **Simulations**: Interactive or script-based predictions for various patient profiles

## Project Structure

```
PK-PD/
├── apps/                          # Streamlit web interfaces
│   ├── advanced_dashboard.py      # Primary simulation dashboard (run this)
│   └── clinical_dashboard.py      # Secondary EHR-style clinical interface
├── scripts/                       # CLI entry points
│   ├── train.py                   # Train all neural network models
│   ├── evaluate.py                # Evaluate trained models
│   └── simulate.py                # Run drug response simulations
├── src/                           # Core library
│   ├── models/                    # Neural network architectures
│   │   ├── mlp_architectures.py   # Shared MLP classes (SimpleMLP, DeepMLP, ResidualMLP)
│   │   ├── gnn_pkpd.py            # Graph Neural Network PK-PD model
│   │   ├── pk_simulator.py        # Pharmacokinetics simulation
│   │   └── pd_simulator.py        # Pharmacodynamics simulation
│   ├── biological_system/         # Virtual human physiology
│   │   ├── organ_network.py       # 12-organ model with blood flow topology
│   │   └── disease_models.py      # Disease progression and drug-disease interactions
│   ├── domain/                    # Domain knowledge modules
│   │   ├── drug_database.py       # Comprehensive drug properties database
│   │   ├── disease_modifiers.py   # Disease-specific PK-PD parameter adjustments
│   │   ├── clinical_validation.py # Clinical trial validation framework
│   │   └── pk_pd_interface.py     # High-level PK-PD model interface
│   ├── simulation/
│   │   └── runner.py              # Primary simulation engine
│   ├── training/
│   │   └── trainer.py             # Training loop, loss functions, cross-validation
│   └── data/                      # Data ingestion and preparation pipeline
│       ├── generators/            # Synthetic data generation
│       ├── ingestion/             # CSV ingestor and consolidation engine
│       ├── processors/            # Data transformation
│       ├── validators/            # Data validation
│       └── exporters/             # Export utilities
├── data/                          # All project data
│   ├── raw/                       # Source CSV datasets (drugs, organs, clinical trials)
│   └── knowledge/                 # Knowledge base JSON files
├── checkpoints/                   # Trained model weights (.pth files)
├── results/                       # All outputs
│   ├── models/                    # Model performance reports and training data
│   ├── simulations/               # Simulation result PNGs and JSON outputs
│   └── ingestion/                 # Data ingestion result files
├── notebooks/
│   ├── demos/                     # Interactive demos and working examples
│   └── production/                # Step-by-step production pipeline notebooks
├── examples/                      # Runnable code examples
├── tests/                         # Test suite
├── archive/                       # Preserved historical files (deprecated)
├── requirements.txt
├── README.md
└── SETUP.md
```

## Installation and Setup

1. **Create and activate a virtual environment:**
```bash
# Windows
python -m venv .env
.env\Scripts\activate

# Linux/Mac
python3 -m venv .env
source .env/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Training Models
Train neural networks from scratch using the datasets in `data/raw/`:
```bash
python scripts/train.py
```
This script will:
- Load all required clinical and biological data
- Setup training for Simple MLP, Deep MLP, and Residual MLP
- Train models for 50 epochs each
- Save model checkpoints to `checkpoints/`
- Output performance comparisons (R², RMSE, MAE) and training curves to `results/simulations/`

### 2. Evaluating Models
Evaluate trained MLP models:
```bash
python scripts/evaluate.py
```
This script will:
- Run forward passes over a test hold-out set
- Evaluate temporal and component-wise accuracy (PK vs PD separate scores)
- Emit final error metrics to the console and diagnostic plots to `results/simulations/`

### 3. Running Simulations
Simulate drug effects across different virtual populations and dosages:
```bash
python scripts/simulate.py
```
This module:
- Loads the best checkpoint from `checkpoints/`
- Runs simulated scenarios (young adult, elderly, pediatric, obese) for multiple drugs
- Generates dose-response curves, time-course predictions, and population-level response estimates
- Saves visualizations to `results/simulations/`

### 4. Launching the Dashboard
To interact with the full simulation dashboard:
```bash
.\env\Scripts\streamlit run apps/advanced_dashboard.py
```
Or for the clinical EHR-style interface:
```bash
.\env\Scripts\streamlit run apps/clinical_dashboard.py
```

## Output Locations
- **Model Checkpoints:** `checkpoints/` as `.pth` files
- **Evaluation Plots:** `results/simulations/` as `.png` images
- **Model Reports:** `results/models/performance_report.txt`
- **Ingestion Results:** `results/ingestion/`

## Contact & License

Please refer to documentation in `src/` for extending dataset specifications or simulation modes.
`MIT License`
