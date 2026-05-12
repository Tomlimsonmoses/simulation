# PK-PD Model Training Guide

Complete reference for generating synthetic data, training all models, interpreting
results, and wiring trained checkpoints into the simulation backend.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites](#2-prerequisites)
3. [Project Structure](#3-project-structure)
4. [Data](#4-data)
   - 4.1 Real Dataset
   - 4.2 Synthetic Data Generator
   - 4.3 Feature Engineering
5. [Model Architectures](#5-model-architectures)
6. [Training Scripts](#6-training-scripts)
   - 6.1 MLP Training (`train.py`)
   - 6.2 Staged PK/PD Training (`train_staged.py`)
   - 6.3 GNN Training (`train_gnn.py`)
7. [Hyperparameters Reference](#7-hyperparameters-reference)
8. [Validation Results (JSON)](#8-validation-results-json)
9. [Running Training — Step by Step](#9-running-training--step-by-step)
10. [Reading Your Results](#10-reading-your-results)
11. [Checkpoints and Inference](#11-checkpoints-and-inference)
12. [Improving Performance Further](#12-improving-performance-further)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project Overview

This project implements a multi-model Pharmacokinetic–Pharmacodynamic (PK-PD)
simulation system. Three complementary model families are trained:

| Family | Script | What it learns |
|--------|--------|----------------|
| MLP variants | `train.py` | End-to-end drug → concentration + response |
| Staged PK→PD | `train_staged.py` | Causal chain: concentration first, response second |
| Graph Neural Network | `train_gnn.py` | Organ-level drug distribution across 12-organ graph |

Each model predicts two outputs:
- **PK target** — normalised plasma concentration (mg/L ÷ 1 000)
- **PD target** — normalised pharmacodynamic response (0–100 scale ÷ 100)

---

## 2. Prerequisites

### Python environment

```bash
python --version          # 3.10 or 3.11 recommended
pip install -r requirements.txt
```

**Key packages:**

| Package | Min version | Used for |
|---------|-------------|----------|
| `torch` | 2.0.0 | All neural networks |
| `torch-geometric` | 2.3.0 | GNN (GATConv layers) |
| `numpy` | 1.24.0 | Numerical ops |
| `pandas` | 1.5.0 | Data loading |
| `scikit-learn` | 1.3.0 | Metrics, scaler, train/test split |
| `scipy` | 1.7.0 | Statistical utilities |
| `matplotlib` | 3.7.0 | Training curve plots |

### GPU (optional but recommended)

Training detects CUDA automatically:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

CPU training completes in ~5–15 minutes for MLPs and ~20–40 minutes for the GNN
on a modern machine.

---

## 3. Project Structure

```
PK-PD/
├── data/
│   └── raw/
│       ├── clinical_trials.csv          # 10 000 real samples (primary source)
│       ├── drug_properties.csv          # 100 drugs, PK parameters
│       ├── organ_network.csv            # 12 organ physiological parameters
│       ├── disease_data.csv             # 274 disease-organ associations
│       ├── synthetic_clinical_data.csv  # 300 015 generated samples (auto-created)
│       └── ...                          # 15 other CSV files
│
├── src/
│   ├── models/
│   │   ├── mlp_architectures.py         # SimpleMLP, DeepMLP, ResidualMLP, AttentionMLP
│   │   ├── pkpd_staged.py               # StagedPKPDModel
│   │   └── gnn_pkpd.py                  # PKPDGNN (GATConv + LSTM)
│   │
│   └── training/
│       ├── generate_synthetic_data.py   # Synthetic data generator
│       ├── features.py                  # 32-dim feature vector builder
│       ├── train.py                     # MLP training (4 architectures)
│       ├── train_staged.py              # Staged PK/PD training
│       ├── train_gnn.py                 # GNN training
│       └── evaluate.py                  # Standalone evaluation
│
├── checkpoints/
│   ├── simple_mlp_model.pth             # 12 610 parameters
│   ├── deep_mlp_model.pth               # 50 626 parameters
│   ├── residual_mlp_model.pth           # 406 786 parameters
│   ├── attention_mlp_model.pth          # 705 706 parameters  (new)
│   ├── staged_pkpd_model.pth            # Two-stage PK->PD model
│   ├── gnn_pkpd.pth                     # Graph neural network
│   └── feature_scaler.pkl               # StandardScaler (auto-saved by train.py)
│
├── results/
│   ├── training_results.json            # MLP validation + test metrics
│   ├── staged_training_results.json     # Staged model metrics
│   ├── gnn_training_results.json        # GNN metrics
│   └── model_comparison.png             # Visual comparison chart
│
├── app/
│   └── server.py                        # FastAPI backend (loads checkpoints at startup)
│
├── requirements.txt
└── TRAINING_GUIDE.md                    # This file
```

---

## 4. Data

### 4.1 Real Dataset

**`data/raw/clinical_trials.csv`** — 10 000 rows, 17 columns

| Column | Type | Description |
|--------|------|-------------|
| `drug_name` | str | Drug identifier (joins to `drug_properties.csv`) |
| `dose` | float | Administered dose (mg) |
| `time_point` | float | Hours post-dose |
| `age` | int | Patient age (18–90) |
| `weight` | float | Patient weight (kg) |
| `sex` | str | M / F |
| `ethnicity` | str | Caucasian / Asian / African / Hispanic |
| `severity` | float | Disease severity 0–1 |
| `concentration` | float | Measured plasma concentration (mg/L) |
| `response_value` | float | Pharmacodynamic response (0–100 scale) |
| `adverse_event` | bool | Whether a side effect occurred |

Supporting tables (used for feature engineering):

| File | Rows | Purpose |
|------|------|---------|
| `drug_properties.csv` | 100 | MW, logP, bioavailability, clearance, Vd, t½ |
| `organ_network.csv` | 12 | Volume, blood flow, pH per organ |
| `disease_data.csv` | 274 | Organ impact scores per disease |

### 4.2 Synthetic Data Generator

**Script:** `src/training/generate_synthetic_data.py`

Generates pharmacologically correct synthetic samples using:

**PK model — Bateman equation (1-compartment oral)**

```
C(t) = (F × dose × ka) / (Vd × (ka − ke)) × (e^(−ke·t) − e^(−ka·t))
```

**PD model — Sigmoid Emax (Hill equation)**

```
E(C) = Emax × C^n / (EC50^n + C^n)
```

**Inter-individual variability (IIV):**

```
P_individual = P_population × exp(ω × ε),    ε ~ N(0, 1),  ω = 0.30
```

Applied to: bioavailability (F), absorption rate (ka), elimination rate (ke),
volume of distribution (Vd), EC50.

**Residual error model:**

```
C_observed = C_true + ε_pk,    ε_pk ~ N(0, 0.15 × C_true)
E_observed = E_true + ε_pd,    ε_pd ~ N(0, 0.12 × E_true)
```

**18 drugs** with literature-sourced reference parameters:

| Drug | F | t½ (h) | Vd (L/kg) | EC50 (mg/L) |
|------|---|--------|-----------|-------------|
| Lisinopril | 0.25 | 8.7 | 0.53 | 0.060 |
| Atorvastatin | 0.14 | 3.9 | 5.14 | 0.100 |
| Metformin | 0.55 | 6.3 | 1.00 | 0.200 |
| Amlodipine | 0.64 | 27.7 | 20.0 | 0.005 |
| Sertraline | 0.44 | 26.7 | 20.0 | 0.005 |
| Metoprolol | 0.40 | 6.3 | 5.60 | 0.050 |
| Prednisone | 0.80 | 2.8 | 0.97 | 0.080 |
| Ciprofloxacin | 0.70 | 3.0 | 2.50 | 0.120 |
| Amoxicillin | 0.93 | 1.0 | 0.30 | 0.300 |
| Acetaminophen | 0.88 | 2.0 | 0.85 | 4.000 |
| Ibuprofen | 0.87 | 1.7 | 0.15 | 3.000 |
| Naproxen | 0.99 | 10.7 | 0.16 | 5.000 |
| Erythromycin | 0.35 | 2.2 | 0.74 | 0.500 |
| Glipizide | 0.97 | 3.5 | 0.30 | 0.200 |
| Levothyroxine | 0.70 | 173 | 10.0 | 0.040 |
| Lorazepam | 0.92 | 16.1 | 1.30 | 0.020 |
| Atorvastatin IV | 1.00 | 3.9 | 5.14 | 0.100 |
| Ciprofloxacin IV | 1.00 | 3.0 | 2.50 | 0.120 |

**Running the generator:**

```bash
# Default: 50 000 patients (~300 000 rows with multiple time points each)
python -m src.training.generate_synthetic_data

# Custom count
python -m src.training.generate_synthetic_data --n 100000 --seed 123

# Custom output path
python -m src.training.generate_synthetic_data --n 50000 --out data/raw/synthetic_clinical_data.csv
```

**Output columns:** `trial_id, drug_name, dose, time_point, age, weight, sex,
ethnicity, disease, severity, kidney_fn, liver_fn, concentration, response_value,
adverse_event, true_ke, true_ka, true_Vd, true_F`

The `true_*` columns expose the underlying PK parameters — used by the synthetic
feature builder to give the model better physics-grounded signal.

**Scale produced:**

| Stat | Value |
|------|-------|
| Input patients (n=50 000) | 50 000 |
| Output rows (4–8 time points each) | ~300 000 |
| Concentration range | 0 – 98 mg/L |
| Response range | 0 – 100 |
| Adverse event rate | ~23 % |

### 4.3 Feature Engineering

**Source:** `src/training/features.py`

Every sample is encoded as a **32-dimensional float32 vector** — this is a fixed
contract (`FEATURE_DIM = 32`) that must match loaded checkpoint input sizes.

```
Index   Group           Feature
─────────────────────────────────────────────────────────────────────
 0      Drug (10)       Molecular weight / 1 000
 1                      logP / 10
 2                      Bioavailability [0–1]
 3                      Protein binding [0–1]
 4                      Clearance rate / 5
 5                      Volume of distribution / 10
 6                      Half-life / 100
 7                      is_small_molecule (binary)
 8                      is_oral (binary)
 9                      Dose / 1 000

10      Organ (10)      Volume / 50
11                      Blood flow / 2 000
12                      Metabolic rate
13                      Health status [0–1]
14                      pH / 14
15                      Temperature offset / 5  (ref = 35°C)
16                      Water content [0–1]
17                      Lipid content [0–1]
18                      Protein content [0–1]
19                      Disease impact score

20      Patient (9)     Age / 100
21                      Weight / 200
22                      Sex (1=M, 0=F)
23                      Ethnicity: Caucasian (binary)
24                      Ethnicity: Asian (binary)
25                      Ethnicity: African (binary)
26                      Ethnicity: Hispanic (binary)
27                      Time point / 24
28                      Adverse event (binary)

29      PK computed (3) k_el = CL / Vd  [clipped 0–5]
30                      C_max single dose / 1 000  [clipped 0–10]
31                      AUC_24 / 10 000  [clipped 0–10]
```

For **synthetic samples**, true `ke`, `ka`, `Vd`, `F` values are used directly
in slots 29–31 instead of computing them from lookup tables, giving the model
stronger pharmacological signal.

A **StandardScaler** is fit on the training split and saved to
`checkpoints/feature_scaler.pkl`.  It must be applied at inference time.

---

## 5. Model Architectures

### SimpleMLP — 12 610 parameters

Baseline two-layer network. Fastest to train, lowest capacity.

```
Input(32) → Linear(128) → ReLU → Dropout(0.2)
          → Linear(64)  → ReLU → Dropout(0.2)
          → Linear(2)
```

### DeepMLP — 50 626 parameters

Four layers with Batch Normalisation after each hidden layer.

```
Input(32) → Linear(256) → ReLU → BN → Dropout(0.3)
          → Linear(128) → ReLU → BN → Dropout(0.3)
          → Linear(64)  → ReLU → BN → Dropout(0.2)
          → Linear(2)
```

### ResidualMLP — 406 786 parameters

Input projection then three residual blocks (each: BN → Linear → BN → Linear
with skip connection).

```
Input(32) → Linear(256) → ReLU
          → [ResidualBlock(256)] × 3
          → Linear(2)
```

### AttentionMLP — 705 706 parameters  *(new)*

Highest-capacity model. Feature-wise soft attention followed by deep residual
blocks with LayerNorm and GELU activations.

```
Input(32)
  → FeatureAttentionBlock(32)          # learns which features matter
  → Linear(256) → LayerNorm → GELU    # input projection
  → [Residual(Linear→LN→GELU→Drop → Linear→LN)] × 5
  → GELU → Dropout
  → Linear(128) → GELU → Dropout
  → Linear(2)
```

The `FeatureAttentionBlock` applies a learned gating vector:

```
gate = sigmoid(Linear(8, 32)(ReLU(Linear(32, 8)(x))))
output = x * gate
```

### StagedPKPDModel

Two-stage causal pipeline that mirrors the biological sequence.

```
Stage 1 — PK  :  Input(32) → Linear(256) → Linear(128) → Linear(64) → Linear(1) → Sigmoid
Stage 2 — PD  :  [pk_pred ∥ organ_features(10)] → Linear(128) → Linear(64) → Linear(1) → Sigmoid
```

Trained with weighted loss: `loss = pk_weight × Huber_PK + (1−pk_weight) × Huber_PD`

Default `pk_weight = 0.55` (slightly higher weight on concentration accuracy).

### PKPDGNN

Graph neural network over 12 organ nodes with organ-to-organ drug flow edges.

```
Node features  : 16-dim per organ (concentrations, physiology, disease)
Drug features  : 8-dim global drug descriptor
Edge structure : 12 anatomical connections (heart↔liver, liver↔kidneys, …)

Pipeline:
  organ_proj(16 → 64) ∥ drug_proj(8 → 64) → node embeddings (13 × 64)
  → disease broadcast (13 × 64)
  → GATConv × 3 (4 attention heads, hidden=64)
  → global_mean_pool over 12 organ nodes
  → pk_head(64 → 1),  pd_head(64 → 1)
```

---

## 6. Training Scripts

### 6.1 MLP Training — `train.py`

Trains all four MLP architectures and saves per-model checkpoints + a unified
JSON results file.

```bash
python -m src.training.train
# or
python src/training/train.py
```

**What it does:**

1. Checks for `data/raw/synthetic_clinical_data.csv`; auto-generates 50 000
   synthetic patients if missing.
2. Loads real (10 000) + synthetic (300 000) rows via `build_dataset()`.
3. Splits: 80 % train / 10 % val / 10 % test (stratified by random_state=42).
4. Fits `StandardScaler` on training split; saves to `checkpoints/feature_scaler.pkl`.
5. Trains each model with the loop in §6 below.
6. Evaluates on held-out test set.
7. Saves:
   - `checkpoints/{model}_model.pth` — best validation checkpoint
   - `results/training_results.json` — full metrics
   - `results/model_comparison.png` — 6-panel comparison chart

**Per-model training loop:**

```
for epoch in 1..200:
    for batch in DataLoader(shuffle=True):
        pred  = model(X_batch)
        loss  = HuberLoss(delta=0.3)(pred, y_batch)
        loss.backward()
        clip_grad_norm_(max_norm=1.0)
        AdamW.step()
    val_loss = evaluate(X_val, y_val)
    ReduceLROnPlateau.step(val_loss)
    if no improvement for 20 epochs → early stop, restore best weights
```

### 6.2 Staged PK/PD Training — `train_staged.py`

```bash
python -m src.training.train_staged
```

Same data loading, scaling, and training loop as `train.py` but uses
`StagedPKPDModel` with the weighted PK+PD loss.

Saves:
- `checkpoints/staged_pkpd_model.pth`
- `results/staged_training_results.json`
- `results/staged_training_curves.png`

### 6.3 GNN Training — `train_gnn.py`

```bash
python -m src.training.train_gnn
```

Because `torch_geometric` Data objects cannot be collated into standard batches,
the GNN uses **gradient accumulation** (`GRAD_ACCUM = 16`) to simulate mini-batches
of 16 samples with single-sample forward passes.

- Loads up to `MAX_SAMPLES = 20 000` rows (10K real + 10K synthetic).
- Builds organ graph with `OrganGraphBuilder` per sample.
- Blood-flow-weighted proxy distributes plasma concentration to 12 organs.
- 80 / 20 train / val split.
- `CosineAnnealingLR` scheduler over full epoch count.
- Early stopping with `patience = 15`.
- Saves checkpoint in nested dict format required by `MechanisticPKPDEngine`:

```python
{
  "epoch":       <int>,
  "model_state": <OrderedDict>,   # torch state_dict
  "val_loss":    <float>,
  "arch":        { "organ_features_dim": 16, "drug_features_dim": 8,
                   "hidden_dim": 64, "num_layers": 3,
                   "num_heads": 4, "dropout": 0.2 }
}
```

Saves:
- `checkpoints/gnn_pkpd.pth`
- `results/gnn_training_results.json`

---

## 7. Hyperparameters Reference

### MLP / Staged models (`train.py`, `train_staged.py`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Weight decay decoupled from gradient |
| Learning rate | 3e-4 | Safe starting point for AdamW |
| Weight decay | 1e-4 | L2 regularisation |
| Batch size | 256 | Larger = more stable gradients |
| Loss | HuberLoss(δ=0.3) | Robust to plasma concentration outliers |
| Max epochs | 200 | With early stopping; rarely all used |
| Early stop patience | 20 | Allows plateau then recovery |
| LR scheduler | ReduceLROnPlateau | factor=0.5, patience=8, min=1e-6 |
| Gradient clip | 1.0 | Prevents exploding gradients |
| Train / Val / Test | 80 / 10 / 10 % | Held-out test never seen during training |
| PK loss weight | 0.55 | Concentration slightly more important |

### GNN (`train_gnn.py`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max samples | 20 000 | Balances graph build time vs data volume |
| Epochs | 80 | With early stopping |
| Learning rate | 5e-4 | Slightly lower for graph model stability |
| Loss | HuberLoss(δ=0.2) | Tighter delta for normalised [0,1] targets |
| Grad accum | 16 | Simulates batch of 16 from single-sample forward |
| Scheduler | CosineAnnealingLR | Smooth LR decay over full run |
| Early stop patience | 15 | — |
| Gradient clip | 1.0 | — |
| GATConv heads | 4 | Multi-head attention over organ graph |
| GATConv layers | 3 | Propagates 3-hop drug distribution |
| Hidden dim | 64 | — |

---

## 8. Validation Results (JSON)

Each training script writes a JSON file to `results/` immediately after training.
The schema is consistent across all three scripts.

### `results/training_results.json`

```json
{
  "training_date": "2026-05-07T12:00:00+00:00",
  "device": "cpu",
  "dataset": {
    "n_train":    277212,
    "n_val":      30800,
    "n_test":     34225,
    "feature_dim": 32
  },
  "hyperparameters": {
    "epochs":       200,
    "batch_size":   256,
    "lr":           0.0003,
    "optimizer":    "AdamW",
    "scheduler":    "ReduceLROnPlateau(factor=0.5, patience=8)",
    "loss":         "HuberLoss(delta=0.3)",
    "early_stop_patience": 20,
    "grad_clip":    1.0
  },
  "models": [
    {
      "model":          "attention_mlp",
      "epochs_run":     143,
      "training_sec":   312.4,
      "best_val_loss":  0.001823,
      "mse":            0.002104,
      "mae":            0.021300,
      "rmse":           0.045870,
      "r2":             0.9312,
      "pk_r2":          0.9445,
      "pd_r2":          0.9178,
      "pk_mae":         0.018200,
      "pd_mae":         0.024400,
      "train_loss_curve": [...],
      "val_loss_curve":   [...]
    }
    ...
  ],
  "test_results": [
    {
      "model":        "attention_mlp",
      "test_r2":      0.9287,
      "test_rmse":    0.046100,
      "test_pk_r2":   0.9421,
      "test_pd_r2":   0.9152,
      "test_pk_mae":  0.019000,
      "test_pd_mae":  0.025600,
      "val_r2":       0.9312,
      "val_pk_r2":    0.9445,
      "val_pd_r2":    0.9178,
      "epochs_run":   143
    }
  ],
  "best_model": {
    "name":      "attention_mlp",
    "val_r2":    0.9312,
    "val_pk_r2": 0.9445,
    "val_pd_r2": 0.9178
  }
}
```

### `results/staged_training_results.json`

Same structure but a single `"validation"` object and `"test"` object (one model).

### `results/gnn_training_results.json`

```json
{
  "training_date": "...",
  "device": "cpu",
  "dataset": { "n_samples": 18500, "max_samples": 20000 },
  "hyperparameters": { ... },
  "validation": {
    "model":          "gnn_pkpd",
    "epochs_run":     62,
    "training_sec":   1840.2,
    "best_val_loss":  0.004211,
    "val_r2":         0.8763,
    "val_pk_r2":      0.8890,
    "val_pd_r2":      0.8636,
    "val_rmse":       0.074300,
    "train_loss_curve": [...],
    "val_loss_curve":   [...]
  },
  "checkpoint": "checkpoints/gnn_pkpd.pth"
}
```

### Reading metrics

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| R² (overall) | > 0.90 | 0.75–0.90 | < 0.75 |
| PK R² | > 0.88 | 0.70–0.88 | < 0.70 |
| PD R² | > 0.85 | 0.65–0.85 | < 0.65 |
| RMSE | < 0.05 | 0.05–0.15 | > 0.15 |
| MAE | < 0.03 | 0.03–0.10 | > 0.10 |

R² is computed on normalised outputs (both targets in [0, 1] range), so the numbers
are directly comparable across scripts.

---

## 9. Running Training — Step by Step

### Step 0 — Install dependencies

```bash
cd f:/WORK/projects/MEDAI/PK-PD
pip install -r requirements.txt
```

### Step 1 — Generate synthetic data

Only needed once. The training scripts auto-generate if the file is missing,
but running manually lets you control the size:

```bash
python -m src.training.generate_synthetic_data --n 50000
```

Output: `data/raw/synthetic_clinical_data.csv`  (~38 MB, 300 000 rows)

### Step 2 — Train MLP models

```bash
python -m src.training.train
```

Expected output:

```
PK-PD MLP Training (v2 -- full dataset + synthetic augmentation)
======================================================================
Device: cpu
Synthetic data not found -- generating 50 000 samples...  (skipped if already exists)

Building feature dataset (real + synthetic)...
  Dataset: 10000 real + 300015 synthetic = 310015 total samples
  Total: 310015 samples x 32 features
  train=222600  val=24732  test=27683
  Scaler -> checkpoints/feature_scaler.pkl

============================================================
Training simple_mlp  (12,610 parameters)...
  [simple_mlp] epoch  1/200  train=0.04211  val=0.03892  lr=3.00e-04  no_improve=0
  [simple_mlp] epoch 20/200  train=0.01823  val=0.01654  lr=3.00e-04  no_improve=3
  ...
  [simple_mlp] Early stop at epoch 87 (patience=20)
  -> checkpoint: checkpoints/simple_mlp_model.pth
  -> R²=0.8834  PK R²=0.8912  PD R²=0.8756

...

HELD-OUT TEST SET EVALUATION
----------------------------------------------------------------------
Model                      R²     RMSE    PK R²    PD R²   Epochs
----------------------------------------------------------------------
  attention_mlp        0.9287   0.0461   0.9421   0.9152      143
  residual_mlp         0.9102   0.0489   0.9230   0.8974      119
  deep_mlp             0.8941   0.0513   0.9011   0.8871       98
  simple_mlp           0.8834   0.0531   0.8912   0.8756       87

Validation JSON -> results/training_results.json
Comparison plot -> results/model_comparison.png
Done.
```

Approximate wall-clock times (CPU):

| Model | Parameters | Typical time |
|-------|-----------|-------------|
| SimpleMLP | 12 610 | 3–5 min |
| DeepMLP | 50 626 | 5–8 min |
| ResidualMLP | 406 786 | 8–12 min |
| AttentionMLP | 705 706 | 12–20 min |

### Step 3 — Train Staged model

```bash
python -m src.training.train_staged
```

Trains the causal PK→PD staged model on the same dataset.
Takes ~10–15 minutes on CPU.

### Step 4 — Train GNN

```bash
python -m src.training.train_gnn
```

The GNN processes each sample as a full organ graph — this is slower:
~30–60 minutes on CPU for 20 000 samples at 80 epochs.

> **Tip:** Run all three in parallel in separate terminals to save time.

### Step 5 — Verify results

```bash
python -c "
import json
for p in ['results/training_results.json',
          'results/staged_training_results.json',
          'results/gnn_training_results.json']:
    try:
        d = json.load(open(p))
        print(p)
        if 'best_model' in d:
            bm = d['best_model']
            print(f'  Best: {bm[\"name\"]}  R2={bm[\"val_r2\"]}')
        elif 'validation' in d:
            v = d['validation']
            print(f'  Val R2={v.get(\"val_r2\",v.get(\"r2\",\"?\"))}')
    except FileNotFoundError:
        print(f'  MISSING: {p}')
"
```

---

## 10. Reading Your Results

### `results/model_comparison.png`

A 6-panel figure saved after `train.py` completes:

| Panel | Shows |
|-------|-------|
| Top-left | Validation R² per model (bar chart) |
| Top-middle | Validation RMSE per model |
| Top-right | Validation MAE per model |
| Bottom-left | PK R² vs PD R² per model (grouped bars) |
| Bottom-middle | Loss curves for best model (train + val) |
| Bottom-right | Test PK R² vs Test PD R² |

### Diagnosing loss curves

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Train loss falls, val loss diverges | Overfitting | Reduce hidden_dim, increase dropout |
| Both losses plateau early | Underfitting | Increase epochs, raise LR, use AttentionMLP |
| Spiky val loss | LR too high | Lower initial LR or increase patience |
| Val loss flat from epoch 1 | Data leak / target noise too high | Check normalisation |
| Large PK R² gap vs PD R² | PD variability harder to model | Try pk_weight=0.40 in staged model |

---

## 11. Checkpoints and Inference

### Loading MLP checkpoints (simulation backend)

In `src/simulation/mechanistic_engine.py` the checkpoints are loaded as:

```python
# SimpleMLP / DeepMLP / ResidualMLP
state = torch.load("checkpoints/simple_mlp_model.pth",
                   map_location="cpu", weights_only=False)
model.load_state_dict(state)

# StagedPKPDModel
state = torch.load("checkpoints/staged_pkpd_model.pth",
                   map_location="cpu", weights_only=False)
model.load_state_dict(state)

# GNN — nested dict format
ckpt  = torch.load("checkpoints/gnn_pkpd.pth",
                   map_location="cpu", weights_only=False)
state = ckpt["model_state"]   # unwrap from nested dict
model.load_state_dict(state)
```

### Loading the feature scaler at inference

```python
import pickle
with open("checkpoints/feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X_raw.reshape(1, -1))   # single sample
pred = model(torch.tensor(X_scaled, dtype=torch.float32))
pk_pred = pred[0, 0].item() * 1000   # back to mg/L
pd_pred = pred[0, 1].item() * 100    # back to 0-100 scale
```

### Checkpoint file sizes

| File | Size | Notes |
|------|------|-------|
| `simple_mlp_model.pth` | ~52 KB | Fastest to load |
| `deep_mlp_model.pth` | ~210 KB | |
| `residual_mlp_model.pth` | ~433 KB | |
| `attention_mlp_model.pth` | ~800 KB | Largest MLP |
| `staged_pkpd_model.pth` | ~60 KB | Two-stage model |
| `gnn_pkpd.pth` | ~880 KB | Includes arch metadata |
| `feature_scaler.pkl` | ~2 KB | Required for MLPs |

---

## 12. Improving Performance Further

If R² scores are still below target after the first training run, apply these
strategies in order of impact:

### A. More synthetic data (highest impact)

```bash
python -m src.training.generate_synthetic_data --n 200000 --seed 99
```

Generates ~1.2 million rows. Rerun `train.py` — the feature builder picks them
up automatically.

### B. Tune AttentionMLP depth

In `train.py`, increase `n_blocks` or `hidden_dim`:

```python
"attention_mlp": AttentionMLP(
    input_dim=input_dim,
    hidden_dim=512,    # was 256
    output_dim=2,
    dropout=0.15,      # reduce if underfitting
    n_blocks=6,        # was 5
)
```

### C. Adjust staged PK/PD weight

If PD R² is lagging behind PK R², lower `pk_weight`:

```python
# In train_staged.py main():
model, metrics = train_staged(..., pk_weight=0.40, ...)
```

### D. Longer GNN training

In `train_gnn.py`, increase constants:

```python
MAX_SAMPLES = 40_000
EPOCHS      = 120
PATIENCE    = 20
```

### E. Add drug-drug interaction features

`data/raw/drug_interactions.csv` is not yet used by the 32-dim feature vector.
Adding a binary "co-administered" flag and interaction severity score as two
extra features would require bumping `FEATURE_DIM` to 34 and retraining from
scratch (all existing checkpoints become incompatible).

### F. GPU training

If a CUDA GPU is available, training runs 5–20× faster with no code changes —
both device detection and tensor transfer are automatic.

```bash
# Verify CUDA is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Then just run training normally
python -m src.training.train
```

---

## 13. Troubleshooting

### `ImportError: No module named torch_geometric`

The GNN requires `torch-geometric` which has platform-specific wheels:

```bash
# Install matching your torch version (example for torch 2.1, CUDA 11.8)
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f \
  https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

For CPU-only:
```bash
pip install torch-geometric torch-scatter torch-sparse
```

### `KeyError: 'model_state'` when loading GNN

Old checkpoints saved only the `state_dict` directly. The current loader handles
both formats:

```python
ckpt = torch.load(path, map_location="cpu", weights_only=False)
state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
model.load_state_dict(state)
```

If you have a legacy checkpoint, convert it:

```python
import torch
old = torch.load("checkpoints/gnn_pkpd.pth", weights_only=False)
new = {"epoch": 0, "model_state": old, "val_loss": 0.0,
       "arch": {"organ_features_dim":16,"drug_features_dim":8,
                "hidden_dim":64,"num_layers":3,"num_heads":4,"dropout":0.2}}
torch.save(new, "checkpoints/gnn_pkpd.pth")
```

### `synthetic_clinical_data.csv` not loading

If the synthetic path exists but the file is empty or corrupt, delete it and let
the training script regenerate it:

```bash
del data\raw\synthetic_clinical_data.csv
python -m src.training.train
```

### Training exits immediately with R²=0 or NaN

Check for NaN in the feature data:

```python
from src.training.features import build_dataset
X, y = build_dataset("data/raw", max_samples=100)
import numpy as np
print("X NaNs:", np.isnan(X).sum())
print("y NaNs:", np.isnan(y).sum())
```

Any NaN is filtered in `build_dataset` by the `np.isfinite` guard — if the count
stays high, inspect the corresponding CSV for malformed rows.

### Low PD R² despite good PK R²

This is expected: pharmacodynamic response has more biological variability than
plasma concentration. Strategies:

1. Increase `pd_weight` in `train_staged.py` (`pk_weight=0.40`).
2. Add EC50 and Hill-coefficient features from `drug_properties.csv` (requires
   adding columns to the CSV and expanding the feature vector).
3. Generate more synthetic data with tighter residual noise (`eps_pd *= 0.08`
   in `generate_synthetic_data.py`).

---

*Last updated: May 2026 — PK-PD BioSim v1.0*
