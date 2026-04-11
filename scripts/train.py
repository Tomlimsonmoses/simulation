"""
Enhanced Model Training Script for PK-PD Modeling

This script trains multiple model architectures and compares their performance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # headless backend — no Tk required
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from models.mlp_architectures import SimpleMLP, DeepMLP, ResidualBlock, ResidualMLP


# Map drug administration route to primary metabolic organ
ROUTE_TO_ORGAN = {
    'oral': 'liver',
    'sublingual': 'liver',
    'intravenous': 'kidneys',     # primary systemic clearance organ
    'intramuscular': 'muscle',
    'subcutaneous': 'skin',
    'inhalation': 'lungs',
}


def _select_organ(organ_data, route):
    """Return the organ row matching the drug's primary metabolic organ."""
    target_name = ROUTE_TO_ORGAN.get(str(route).lower(), 'liver')
    match = organ_data[organ_data['organ_name'] == target_name]
    if len(match) > 0:
        return match.iloc[0]
    return organ_data.iloc[0]  # fallback to first row


def create_features(sample_idx, drug_data, organ_data, clinical_data, disease_data):
    """Create 32-dim feature vector for a single sample.

    Feature layout (32):
      Drug (10): MW, logP, bioavail, protein_bind, clearance, Vd, t½, is_small_mol, is_oral, dose
      Organ (10): volume, blood_flow, metab_rate, health, pH, temp, water, lipid, protein, disease_impact
      Patient (9): age, weight, sex, 4x ethnicity, time_point, adverse_event
        (administration_route removed — duplicate of Drug feature 8)
      PK computed (3): k_el, C_max (peak single-dose), AUC_24
    """
    # Get sample data
    clinical_sample = clinical_data.iloc[sample_idx % len(clinical_data)]
    drug_name = clinical_sample['drug_name']

    # Find drug info
    drug_info = drug_data[drug_data['drug_name'] == drug_name]
    if len(drug_info) == 0:
        drug_info = drug_data.iloc[0]
    else:
        drug_info = drug_info.iloc[0]

    # Select organ matching drug's metabolic pathway (fixes always-liver bug)
    route = drug_info['administration_route']
    organ = _select_organ(organ_data, route)

    # Disease impact on the selected organ
    organ_diseases = disease_data[
        disease_data['affected_organ'] == organ['organ_name']
    ]
    disease_impact = organ_diseases['organ_impact_score'].mean() if len(organ_diseases) > 0 else 0.0

    # Pre-computed PK features
    clearance = float(drug_info['clearance_rate'])
    vd = float(drug_info['volume_distribution'])
    bioavail = float(drug_info['bioavailability'])
    dose = float(clinical_sample['dose'])
    k_el = clearance / max(vd, 1e-6)                        # elimination rate constant (h⁻¹)
    c_max = dose * bioavail / max(vd, 1e-6)                 # peak single-dose concentration (mg/L)
    auc_24 = dose * bioavail / max(clearance, 1e-6)         # AUC over 24 h

    features = [
        # Drug features (10)
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

        # Organ features (10)
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

        # Patient features (9)  — administration_route removed (was duplicate of feature 8)
        clinical_sample['age'] / 100.0,
        clinical_sample['weight'] / 200.0,
        1.0 if clinical_sample['sex'] == 'M' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'Caucasian' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'Asian' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'African' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'Hispanic' else 0.0,
        clinical_sample.get('time_point', 0.0) / 24.0,
        1.0 if clinical_sample.get('adverse_event', False) else 0.0,

        # Pre-computed PK features (3)
        np.clip(k_el, 0.0, 5.0),
        np.clip(c_max / 1000.0, 0.0, 10.0),
        np.clip(auc_24 / 10000.0, 0.0, 10.0),
    ]

    # Target values
    pk_target = clinical_sample.get('concentration', 0.0) / 1000.0
    pd_target = clinical_sample.get('response_value', 0.0) / 100.0

    return np.array(features, dtype=np.float32), np.array([pk_target, pd_target], dtype=np.float32)


def train_model(model, X_train, y_train, X_test, y_test, model_name, epochs=50, batch_size=32):
    """Train a single model and return metrics."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_output = model(X_test_tensor)
            test_loss = criterion(test_output, y_test_tensor)
            val_losses.append(test_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"{model_name} - Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {test_loss.item():.4f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_output = model(X_test_tensor)
        final_loss = criterion(final_output, y_test_tensor)
        
        final_output_np = final_output.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
        
        mse = mean_squared_error(y_test_np.flatten(), final_output_np.flatten())
        mae = mean_absolute_error(y_test_np.flatten(), final_output_np.flatten())
        r2 = r2_score(y_test_np.flatten(), final_output_np.flatten())
        
        # Separate PK and PD metrics
        pk_mse = mean_squared_error(y_test_np[:, 0], final_output_np[:, 0])
        pk_r2 = r2_score(y_test_np[:, 0], final_output_np[:, 0])
        pd_mse = mean_squared_error(y_test_np[:, 1], final_output_np[:, 1])
        pd_r2 = r2_score(y_test_np[:, 1], final_output_np[:, 1])
    
    metrics = {
        'model_name': model_name,
        'training_time': training_time,
        'final_loss': final_loss.item(),
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'pk_mse': pk_mse,
        'pk_r2': pk_r2,
        'pd_mse': pd_mse,
        'pd_r2': pd_r2,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    return metrics, model


def main():
    """Main training function."""
    print("Enhanced PK-PD Model Training")
    print("="*60)
    
    # Load datasets
    print("Loading datasets...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    drug_data = pd.read_csv(os.path.join(data_dir, 'drug_properties.csv'))
    organ_data = pd.read_csv(os.path.join(data_dir, 'organ_network.csv'))
    clinical_data = pd.read_csv(os.path.join(data_dir, 'clinical_trials.csv'))
    disease_data = pd.read_csv(os.path.join(data_dir, 'disease_data.csv'))
    
    print(f"Loaded {len(drug_data)} drugs, {len(organ_data)} organs, {len(clinical_data)} clinical samples, {len(disease_data)} disease records")
    
    # Create dataset
    print("Creating feature dataset...")
    X = []
    y = []
    num_samples = min(1000, len(clinical_data))
    
    for i in range(num_samples):
        try:
            features, targets = create_features(i, drug_data, organ_data, clinical_data, disease_data)
            X.append(features)
            y.append(targets)
        except Exception as e:
            print(f"Error creating sample {i}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Successfully created {len(X)} samples with {X.shape[1]} features each")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Define models (input_dim=32: 10 drug + 10 organ + 9 patient + 3 PK computed)
    input_dim = X.shape[1]
    models = {
        'Simple MLP': SimpleMLP(input_dim=input_dim, hidden_dim=128, output_dim=2),
        'Deep MLP': DeepMLP(input_dim=input_dim, hidden_dim=256, output_dim=2),
        'Residual MLP': ResidualMLP(input_dim=input_dim, hidden_dim=128, output_dim=2)
    }
    
    # Train all models
    all_metrics = []
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        print("-" * 40)
        
        metrics, trained_model = train_model(
            model, X_train, y_train, X_test, y_test, 
            model_name, epochs=50, batch_size=32
        )
        
        all_metrics.append(metrics)
        trained_models[model_name] = trained_model
        
        # Save model
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(trained_model.state_dict(), f'checkpoints/{model_name.lower().replace(" ", "_")}_model.pth')
        
        print(f"{model_name} Results:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  PK R²: {metrics['pk_r2']:.4f}")
        print(f"  PD R²: {metrics['pd_r2']:.4f}")
        print(f"  Training Time: {metrics['training_time']:.2f}s")
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df[['model_name', 'r2', 'rmse', 'mae', 'pk_r2', 'pd_r2', 'training_time']]
    comparison_df = comparison_df.sort_values('r2', ascending=False)
    
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # R² comparison
    axes[0, 0].bar(comparison_df['model_name'], comparison_df['r2'])
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RMSE comparison
    axes[0, 1].bar(comparison_df['model_name'], comparison_df['rmse'])
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # PK vs PD R²
    x = np.arange(len(comparison_df))
    width = 0.35
    axes[1, 0].bar(x - width/2, comparison_df['pk_r2'], width, label='PK R²')
    axes[1, 0].bar(x + width/2, comparison_df['pd_r2'], width, label='PD R²')
    axes[1, 0].set_title('PK vs PD R² Scores')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(comparison_df['model_name'], rotation=45)
    axes[1, 0].legend()
    
    # Training time
    axes[1, 1].bar(comparison_df['model_name'], comparison_df['training_time'])
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot training curves for best model
    best_model = comparison_df.iloc[0]['model_name']
    best_metrics = [m for m in all_metrics if m['model_name'] == best_model][0]
    
    plt.figure(figsize=(10, 5))
    plt.plot(best_metrics['train_losses'], label='Training Loss')
    plt.plot(best_metrics['val_losses'], label='Validation Loss')
    plt.title(f'Training Curves - {best_model}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{best_model.lower().replace(" ", "_")}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nBest model: {best_model} with R² = {comparison_df.iloc[0]['r2']:.4f}")
    print("\nAll models saved successfully to checkpoints/!")
    print("Comparison plot saved to 'results/model_comparison.png'")
    
    return all_metrics, trained_models


if __name__ == "__main__":
    metrics, models = main()
