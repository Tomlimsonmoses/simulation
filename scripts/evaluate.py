"""
Model Evaluation Script for PK-PD Modeling

This script evaluates trained MLP models on the clinical dataset
and outputs clear evaluation metrics and visualizations.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.mlp_architectures import SimpleMLP, DeepMLP, ResidualBlock, ResidualMLP

# Map drug administration route to primary metabolic organ
ROUTE_TO_ORGAN = {
    'oral': 'liver',
    'sublingual': 'liver',
    'intravenous': 'kidneys',
    'intramuscular': 'muscle',
    'subcutaneous': 'skin',
    'inhalation': 'lungs',
}


def _select_organ(organ_data, route):
    target_name = ROUTE_TO_ORGAN.get(str(route).lower(), 'liver')
    match = organ_data[organ_data['organ_name'] == target_name]
    if len(match) > 0:
        return match.iloc[0]
    return organ_data.iloc[0]


def create_features(sample_idx, drug_data, organ_data, clinical_data, disease_data):
    """Create 32-dim feature vector matching train.py."""
    clinical_sample = clinical_data.iloc[sample_idx % len(clinical_data)]
    drug_name = clinical_sample['drug_name']

    drug_info = drug_data[drug_data['drug_name'] == drug_name]
    if len(drug_info) == 0:
        drug_info = drug_data.iloc[0]
    else:
        drug_info = drug_info.iloc[0]

    route = drug_info['administration_route']
    organ = _select_organ(organ_data, route)

    organ_diseases = disease_data[
        disease_data['affected_organ'] == organ['organ_name']
    ]
    disease_impact = organ_diseases['organ_impact_score'].mean() if len(organ_diseases) > 0 else 0.0

    clearance = float(drug_info['clearance_rate'])
    vd = float(drug_info['volume_distribution'])
    bioavail = float(drug_info['bioavailability'])
    dose = float(clinical_sample['dose'])
    k_el = clearance / max(vd, 1e-6)
    c_max = dose * bioavail / max(vd, 1e-6)
    auc_24 = dose * bioavail / max(clearance, 1e-6)

    features = [
        drug_info['molecular_weight'] / 1000.0,
        drug_info['logp'] / 10.0,
        bioavail,
        drug_info['protein_binding'],
        clearance / 5.0,
        vd / 10.0,
        drug_info['half_life'] / 100.0,
        1.0 if drug_info['drug_class'] == 'small_molecule' else 0.0,
        1.0 if route == 'oral' else 0.0,
        dose / 1000.0,

        organ['volume'] / 50.0,
        organ['blood_flow'] / 2000.0,
        organ['metabolic_rate'],
        organ['health_status'],
        organ['ph'] / 14.0,
        (organ['temperature'] - 35.0) / 5.0,
        organ['water_content'],
        organ['lipid_content'],
        organ['protein_content'],
        disease_impact,

        clinical_sample['age'] / 100.0,
        clinical_sample['weight'] / 200.0,
        1.0 if clinical_sample['sex'] == 'M' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'Caucasian' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'Asian' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'African' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'Hispanic' else 0.0,
        clinical_sample.get('time_point', 0.0) / 24.0,
        1.0 if clinical_sample.get('adverse_event', False) else 0.0,

        np.clip(k_el, 0.0, 5.0),
        np.clip(c_max / 1000.0, 0.0, 10.0),
        np.clip(auc_24 / 10000.0, 0.0, 10.0),
    ]

    pk_target = clinical_sample.get('concentration', 0.0) / 1000.0
    pd_target = clinical_sample.get('response_value', 0.0) / 100.0

    return np.array(features, dtype=np.float32), np.array([pk_target, pd_target], dtype=np.float32)

def main():
    print("PK-PD Model Independent Evaluation")
    print("=====================================")
    
    # Check if models exist
    default_model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'simple_mlp_model.pth')
    if not os.path.exists(default_model_path):
        print(f"Model not found at {default_model_path}. Please run train.py first.")
        return

    # Load datasets
    print("Loading datasets for evaluation...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    drug_data = pd.read_csv(os.path.join(data_dir, 'drug_properties.csv'))
    organ_data = pd.read_csv(os.path.join(data_dir, 'organ_network.csv'))
    clinical_data = pd.read_csv(os.path.join(data_dir, 'clinical_trials.csv'))
    disease_data = pd.read_csv(os.path.join(data_dir, 'disease_data.csv'))
    
    print("Preparing test features...")
    X = []
    y = []
    num_samples = min(1000, len(clinical_data))
    
    for i in range(num_samples):
        try:
            features, targets = create_features(i, drug_data, organ_data, clinical_data, disease_data)
            X.append(features)
            y.append(targets)
        except Exception:
            continue
            
    X = np.array(X)
    y = np.array(y)
    
    # We evaluate on the test split (same random state as train.py)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Evaluating on {len(X_test)} test samples")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate Simple MLP — detect input_dim from data (supports re-trained checkpoints)
    input_dim = X.shape[1]
    model = SimpleMLP(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(default_model_path, map_location=device))
    model.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        final_output = model(X_test_tensor)
        
        final_output_np = final_output.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
        
        mse = mean_squared_error(y_test_np.flatten(), final_output_np.flatten())
        mae = mean_absolute_error(y_test_np.flatten(), final_output_np.flatten())
        r2 = r2_score(y_test_np.flatten(), final_output_np.flatten())
        
        pk_mse = mean_squared_error(y_test_np[:, 0], final_output_np[:, 0])
        pk_r2 = r2_score(y_test_np[:, 0], final_output_np[:, 0])
        pd_mse = mean_squared_error(y_test_np[:, 1], final_output_np[:, 1])
        pd_r2 = r2_score(y_test_np[:, 1], final_output_np[:, 1])

    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS (Simple MLP)")
    print("="*50)
    print(f"Overall MSE:  {mse:.4f}")
    print(f"Overall MAE:  {mae:.4f}")
    print(f"Overall RMSE: {np.sqrt(mse):.4f}")
    print(f"Overall R²:   {r2:.4f}")
    print(f"PK MSE: {pk_mse:.4f} (R²: {pk_r2:.4f})")
    print(f"PD MSE: {pd_mse:.4f} (R²: {pd_r2:.4f})")
    print("="*50)
    
    # Plotting
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_np[:100, 0], final_output_np[:100, 0], alpha=0.6, label='PK')
    plt.scatter(y_test_np[:100, 1], final_output_np[:100, 1], alpha=0.6, label='PD')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs True Values (Simple MLP)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    residuals = y_test_np - final_output_np
    plt.scatter(final_output_np[:100, 0], residuals[:100, 0], alpha=0.6, label='PK')
    plt.scatter(final_output_np[:100, 1], residuals[:100, 1], alpha=0.6, label='PD')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot (Simple MLP)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/final_evaluation_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nEvaluation completed successfully!")
    print("Results saved to 'results/final_evaluation_scatter.png'")

if __name__ == "__main__":
    main()
