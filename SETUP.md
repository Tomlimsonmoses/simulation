# Setup Guide for GNN-Based PK-PD Model

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .env

# Activate environment
# Windows:
.env\Scripts\activate
# Linux/Mac:
source .env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```python
# Test basic imports
from src.models.gnn_pkpd import PKPDGNN
from src.simulation.runner import PKPDSimulationRunner
from src.biological_system.organ_network import VirtualHuman

print("Installation successful!")
```

### 3. Run Your First Simulation

```python
from src.simulation.runner import run_quick_simulation

# Quick simulation with default parameters
results = run_quick_simulation(
    drug_name="example_drug",
    dose=100.0,
    time_horizon=48.0
)

print(f"Simulation completed! Generated {len(results.output_times)} time points")
```

## Detailed Setup

### System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended
- 2GB+ disk space

### Dependencies

The project requires the following main packages:

#### Core ML Libraries
- `torch>=2.0.0` - Deep learning framework
- `torch-geometric>=2.3.0` - Graph neural networks
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation

#### Scientific Computing
- `scipy>=1.7.0` - Scientific computing
- `scikit-learn>=1.0.0` - Machine learning utilities
- `networkx>=2.6.0` - Graph algorithms

#### Visualization
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization
- `plotly>=5.0.0` - Interactive plots

#### Development Tools
- `jupyter>=1.0.0` - Jupyter notebooks
- `ipywidgets>=7.6.0` - Interactive widgets
- `tqdm>=4.62.0` - Progress bars
- `pydantic>=1.8.0` - Data validation

### Installation Steps

#### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd windsurf-project

# Or download and extract the project files
```

#### 2. Create Virtual Environment

```bash
# Create environment
python -m venv .env

# Activate (Windows)
.env\Scripts\activate

# Activate (Linux/Mac)
source .env/bin/activate
```

#### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# For GPU support (optional)
# Check CUDA version first: nvidia-smi
# Install appropriate PyTorch version from https://pytorch.org/
```

#### 4. Verify Installation

Run the test script:

```python
python examples/basic_simulation.py
```

Or test individual components:

```python
# Test GNN model
from src.models.gnn_pkpd import PKPDGNN
model = PKPDGNN()
print(f"GNN model created with {sum(p.numel() for p in model.parameters())} parameters")

# Test virtual human
from src.biological_system.organ_network import VirtualHuman
vh = VirtualHuman()
print(f"Virtual human created with {len(vh.organs)} organs")

# Test simulation runner
from src.simulation.runner import PKPDSimulationRunner, SimulationConfig
config = SimulationConfig()
runner = PKPDSimulationRunner(config)
print("Simulation runner initialized successfully")
```

## Project Structure

```
windsurf-project/
├── .env/                  # Virtual environment
├── src/                   # Source code
│   ├── models/           # GNN, PK, PD models
│   ├── biological_system/ # Virtual human, diseases
│   ├── data/            # Dataset specifications
│   ├── training/        # Training framework
│   └── simulation/      # Main simulation runner
├── examples/             # Usage examples
├── notebooks/           # Jupyter notebooks
├── data/               # Data directory
├── tests/              # Unit tests
├── requirements.txt     # Dependencies
└── README.md          # Project documentation
```

## Dataset Requirements

The system requires specific datasets for training. See `src/data/dataset_specs.py` for detailed requirements:

### Essential Datasets (Required for Training)
1. **Drug Properties Database** - Physicochemical and pharmacological properties
2. **Human Organ Network Topology** - Organ connections and parameters
3. **Clinical Trial Outcomes** - PK/PD measurements from human studies
4. **Disease Characterization Data** - Disease effects on physiology

### Optional Datasets (Enhanced Performance)
5. **Population Demographics** - Inter-individual variability
6. **Biomarker Reference Data** - Normal ranges and disease associations
7. **PK/PD Parameters** - Drug-specific parameters
8. **Pharmacogenomics Data** - Genetic variants affecting drug response

### Data Sources
- DrugBank, ChEMBL, PubChem (drug properties)
- ClinicalTrials.gov, FDA databases (clinical data)
- Human Physiome Project (physiological parameters)
- NHANES, UK Biobank (population data)
- PharmGKB, ClinVar (genomic data)

## Getting Started with Examples

### 1. Basic Simulation

```bash
python examples/basic_simulation.py
```

### 2. Disease Simulation

```bash
python examples/disease_simulation.py
```

### 3. Population Simulation

```bash
python examples/population_simulation.py
```

### 4. Model Training

```bash
python examples/training_example.py
```

### 5. Interactive Demo

```bash
jupyter notebook notebooks/demo.ipynb
```

## Configuration

### Simulation Parameters

Key parameters you can customize:

```python
config = SimulationConfig(
    drug_name="your_drug",
    dose=10.0,                    # mg
    administration_route="oral",   # oral, iv, im, sc
    time_horizon=48.0,           # hours
    patient_age=45,              # years
    patient_weight=70.0,         # kg
    patient_sex="male",           # male, female
    disease_type="cardiovascular_hypertension",  # optional
    pk_model_type="pbpk",         # compartmental, pbpk
    pd_model_type="organ_specific", # organ_specific, systemic
    enable_population_variability=False,  # for population simulation
    population_size=1             # number of virtual subjects
)
```

### Model Parameters

```python
model = PKPDGNN(
    organ_features_dim=16,       # organ feature dimension
    drug_features_dim=8,         # drug feature dimension
    hidden_dim=64,               # hidden layer size
    num_layers=3,                # number of GNN layers
    num_heads=4,                 # attention heads
    dropout=0.2                  # dropout rate
)
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure you're in the project directory
cd windsurf-project

# Activate virtual environment
# Windows:
.env\Scripts\activate
# Linux/Mac:
source .env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. CUDA Issues
```bash
# Check CUDA installation
nvidia-smi

# Install CPU-only PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or install appropriate CUDA version from pytorch.org
```

#### 3. Memory Issues
```python
# Reduce batch size or population size
config.population_size = 10  # Instead of 100
config.time_step = 0.5      # Instead of 0.1
```

#### 4. Missing Datasets
```python
# Use synthetic data for testing
from src.data.dataset_specs import DataPipeline
pipeline = DataPipeline(DatasetRequirements())
synthetic_data = pipeline.generate_synthetic_data({
    DatasetType.DRUG_PROPERTIES: 1000,
    DatasetType.ORGAN_NETWORK: 12
})
```

### Performance Optimization

#### GPU Acceleration
```python
# Check if CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PKPDGNN().to(device)
```

#### Memory Management
```python
# Use smaller batches
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Enable gradient checkpointing for large models
model = PKPDGNN(use_checkpointing=True)
```

## Next Steps

1. **Collect Datasets**: Gather the required datasets specified in `src/data/dataset_specs.py`
2. **Train Model**: Use the training framework with your data
3. **Run Simulations**: Test with your specific drugs and conditions
4. **Validate Results**: Compare with experimental data
5. **Customize**: Adapt models for your specific use case

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the examples in the `examples/` directory
3. Examine the Jupyter notebook demo
4. Check dataset specifications in `src/data/dataset_specs.py`

## Citation

If you use this system in your research, please cite:

```
GNN-Based PK-PD Model for Organ Drug Response Simulation
Version 1.0.0
https://github.com/your-repo/windsurf-project
```
