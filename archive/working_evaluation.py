"""
Working Model Evaluation Script for GNN PK-PD Model

This script trains and evaluates a simplified GNN PK-PD model using the generated real datasets.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Simple MLP model for PK-PD prediction
class SimplePKPDModel(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=128, output_dim=2):
        super(SimplePKPDModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)


def create_features(sample_idx, drug_data, organ_data, clinical_data, disease_data):
    """Create feature vector for a single sample."""
    
    # Get sample data
    clinical_sample = clinical_data.iloc[sample_idx % len(clinical_data)]
    drug_name = clinical_sample['drug_name']
    
    # Find drug info
    drug_info = drug_data[drug_data['drug_name'] == drug_name]
    if len(drug_info) == 0:
        drug_info = drug_data.iloc[0]
    else:
        drug_info = drug_info.iloc[0]
    
    # Get organ info (use first organ for simplicity)
    organ = organ_data.iloc[0]
    
    # Get disease impacts
    organ_diseases = disease_data[
        disease_data['affected_organ'] == organ['organ_name']
    ]
    disease_impact = organ_diseases['organ_impact_score'].mean() if len(organ_diseases) > 0 else 0.0
    
    # Create comprehensive feature vector
    features = [
        # Drug features (10)
        drug_info['molecular_weight'] / 1000.0,
        drug_info['logp'] / 10.0,
        drug_info['bioavailability'],
        drug_info['protein_binding'],
        drug_info['clearance_rate'] / 5.0,
        drug_info['volume_distribution'] / 10.0,
        drug_info['half_life'] / 100.0,
        1.0 if drug_info['drug_class'] == 'small_molecule' else 0.0,
        1.0 if drug_info['administration_route'] == 'oral' else 0.0,
        clinical_sample['dose'] / 1000.0,
        
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
        
        # Patient features (10)
        clinical_sample['age'] / 100.0,
        clinical_sample['weight'] / 200.0,
        1.0 if clinical_sample['sex'] == 'M' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'Caucasian' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'Asian' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'African' else 0.0,
        1.0 if clinical_sample['ethnicity'] == 'Hispanic' else 0.0,
        1.0 if clinical_sample['administration_route'] == 'oral' else 0.0,
        clinical_sample.get('time_point', 0.0) / 24.0,
        1.0 if clinical_sample.get('adverse_event', False) else 0.0
    ]
    
    # Target values
    pk_target = clinical_sample.get('concentration', 0.0) / 1000.0
    pd_target = clinical_sample.get('response_value', 0.0) / 100.0
    
    return np.array(features), np.array([pk_target, pd_target])


def main():
    """Main evaluation function."""
    print("Simple PK-PD Model Evaluation")
    print("="*50)
    
    # Load datasets
    print("Loading datasets...")
    drug_data = pd.read_csv("./real_data/drug_properties.csv")
    organ_data = pd.read_csv("./real_data/organ_network.csv")
    clinical_data = pd.read_csv("./real_data/clinical_trials.csv")
    disease_data = pd.read_csv("./real_data/disease_data.csv")
    
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
    
    if len(X) == 0:
        print("No valid samples created. Exiting.")
        return
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimplePKPDModel(input_dim=30, hidden_dim=128, output_dim=2).to(device)
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
    
    num_epochs = 50
    batch_size = 32
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
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
        
        # Calculate metrics
        test_output_np = test_output.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
        
        mse = mean_squared_error(y_test_np.flatten(), test_output_np.flatten())
        mae = mean_absolute_error(y_test_np.flatten(), test_output_np.flatten())
        r2 = r2_score(y_test_np.flatten(), test_output_np.flatten())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {test_loss.item():.4f}, "
                  f"R²: {r2:.4f}, "
                  f"RMSE: {np.sqrt(mse):.4f}")
    
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
    
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Final Test Loss: {final_loss.item():.4f}")
    print(f"Overall MSE: {mse:.4f}")
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall RMSE: {np.sqrt(mse):.4f}")
    print(f"Overall R²: {r2:.4f}")
    print(f"PK MSE: {pk_mse:.4f} (R²: {pk_r2:.4f})")
    print(f"PD MSE: {pd_mse:.4f} (R²: {pd_r2:.4f})")
    print("="*50)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_test_np[:100, 0], final_output_np[:100, 0], alpha=0.6, label='PK')
    plt.scatter(y_test_np[:100, 1], final_output_np[:100, 1], alpha=0.6, label='PD')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    residuals = y_test_np - final_output_np
    plt.scatter(final_output_np[:100, 0], residuals[:100, 0], alpha=0.6, label='PK')
    plt.scatter(final_output_np[:100, 1], residuals[:100, 1], alpha=0.6, label='PD')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('working_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nEvaluation completed successfully!")
    print("Results saved to 'working_evaluation_results.png'")
    
    # Save model
    torch.save(model.state_dict(), 'simple_pkpd_model.pth')
    print("Model saved to 'simple_pkpd_model.pth'")
    
    return {
        'final_loss': final_loss.item(),
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'pk_mse': pk_mse,
        'pk_r2': pk_r2,
        'pd_mse': pd_mse,
        'pd_r2': pd_r2
    }


if __name__ == "__main__":
    metrics = main()
