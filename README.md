# GNN-Based PK-PD Model for Organ Drug Response

This project implements a Graph Neural Network (GNN) and Deep Multilayer Perceptron (MLP) approach for Pharmacokinetic-Pharmacodynamic (PK-PD) modeling to simulate organ responses to drugs in a virtual human biological system. 

It handles:
- **Pharmacokinetics (PK)**: How drugs move through the body (absorption, distribution, metabolism, excretion)
- **Pharmacodynamics (PD)**: How organs respond to drugs (receptor binding, signal transduction, physiological effects)
- **Simulations**: Interactive or script-based predictions for various patient profiles

## Project Structure

The project has been organized for maximum clarity and ease of use:

```
├── train.py               # Main script to train all ML models
├── evaluate.py            # Comprehensive script to evaluate trained models
├── simulate.py            # Script to run patient simulations under various scenarios
├── clinical_app.py        # Clinical decision support interface
├── checkpoints/           # Trained model weights (.pth files)
├── results/               # Output plots, evaluation metrics, and curves
├── notebooks/             # Jupyter notebooks for data exploration and validation
├── archive/               # Older diagnostic or duplicate evaluation scripts
├── real_data/             # Core real datasets (drugs, organs, clinical trials)
├── data/                  # Synthetic or intermediate generated data
├── src/                   # Core biological, modeling, and simulation logic
│   ├── models/            # Neural network architectures
│   ├── biological_system/ # Virtual human components
│   ├── training/          # Training loop and loss functions
│   ├── simulation/        # Core simulation engine logic
│   └── data/              # Data parsing and specification logic
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
To train the neural networks from scratch using the datasets in `real_data/`, run:
```bash
python train.py
```
This script will:
- Load all required clinical and biological data
- Setup training for Simple MLP, Deep MLP, and Residual MLP
- Train models for 50 epochs each
- Save model checkpoints to `checkpoints/`
- Output performance comparisons (R², RMSE, MAE) and training curves to `results/`

### 2. Evaluating Models
To comprehensively evaluate the complex GNN models, run:
```bash
python evaluate.py
```
This script will:
- Run forward passes over a test hold-out set
- Evaluate temporal and component-wise accuracy (PK vs PD separate scores)
- Emit final error metrics to the console and diagnostic plots to `results/`

### 3. Running Simulations
To simulate drug effects across different virtual populations and dosages, run:
```bash
python simulate.py
```
This module:
- Loads the best checkpoint from `checkpoints/`
- Runs simulated scenarios (young adult, elderly, pediatric, obese) for multiple drugs
- Generates dose-response curves, time-course predictions, and population-level response estimates
- Visualizes everything and saves to `results/`

### 4. Exploring Clinical App
To interact with a textual simulation interface meant for health practitioners, run:
```bash
python clinical_app.py
```

## Output Locations
- **Model Checkpoints:** Found in `checkpoints/` as `.pth` files.
- **Evaluation Plots:** Found in `results/` as `.png` images.

## Contact & License

Please refer to documentation in `src/` for extending dataset specifications or simulation modes.
`MIT License`
