"""
Simplified Model Evaluation Script for GNN PK-PD Model

This script trains and evaluates a simplified GNN PK-PD model using the generated real datasets.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Simple GNN model
class SimplePKPDGNN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=2):
        super(SimplePKPDGNN, self).__init__()
        
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.conv3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        # Simple message passing
        row, col = edge_index
        
        # First layer
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        
        # Aggregate neighbor information
        out = torch.zeros_like(x)
        for i in range(len(x)):
            neighbors = edge_index[1][edge_index[0] == i]
            if len(neighbors) > 0:
                out[i] = torch.mean(x[neighbors], dim=0)
            else:
                out[i] = x[i]
        
        # Second layer
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        
        # Third layer
        out = self.relu(self.conv3(out))
        
        # Output layer
        out = self.output(out)
        
        return out


def create_simple_graph(sample_idx, drug_data, organ_data, clinical_data):
    """Create a simple graph for training."""
    
    # Get sample data
    clinical_sample = clinical_data.iloc[sample_idx % len(clinical_data)]
    drug_name = clinical_sample['drug_name']
    
    # Find drug info
    drug_info = drug_data[drug_data['drug_name'] == drug_name]
    if len(drug_info) == 0:
        drug_info = drug_data.iloc[0]
    else:
        drug_info = drug_info.iloc[0]
    
    # Create node features (simplified)
    # Organ nodes (first 5 organs for simplicity)
    organ_features = []
    for i, organ in organ_data.head(5).iterrows():
        features = [
            organ['volume'] / 50.0,
            organ['blood_flow'] / 2000.0,
            organ['metabolic_rate'],
            organ['health_status'],
            clinical_sample['age'] / 100.0,
            clinical_sample['weight'] / 200.0,
            1.0 if clinical_sample['sex'] == 'M' else 0.0,
            clinical_sample.get('concentration', 0.0) / 1000.0,
            clinical_sample.get('response_value', 0.0) / 100.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding to 20
        ]
        organ_features.append(features)
    
    # Drug node (20 features to match organ features)
    drug_features = [
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
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding to 20
    ]
    
    # Combine all nodes
    all_features = torch.tensor(organ_features + [drug_features], dtype=torch.float)
    
    # Create simple edges (fully connected)
    num_nodes = len(all_features)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Target values (PK and PD for each organ)
    targets = []
    for i in range(len(organ_features)):
        pk_target = clinical_sample.get('concentration', 0.0) / 1000.0
        pd_target = clinical_sample.get('response_value', 0.0) / 100.0
        targets.append([pk_target, pd_target])
    
    targets = torch.tensor(targets, dtype=torch.float)
    
    # Create PyG Data object
    data = Data(
        x=all_features,
        edge_index=edge_index,
        y=targets,
        num_nodes=num_nodes
    )
    
    return data


def main():
    """Main evaluation function."""
    print("Simple GNN PK-PD Model Evaluation")
    print("="*50)
    
    # Load datasets
    print("Loading datasets...")
    drug_data = pd.read_csv("./real_data/drug_properties.csv")
    organ_data = pd.read_csv("./real_data/organ_network.csv")
    clinical_data = pd.read_csv("./real_data/clinical_trials.csv")
    
    print(f"Loaded {len(drug_data)} drugs, {len(organ_data)} organs, {len(clinical_data)} clinical samples")
    
    # Create dataset
    print("Creating graph dataset...")
    dataset = []
    num_samples = min(500, len(clinical_data))  # Limit for faster training
    
    for i in range(num_samples):
        try:
            graph_data = create_simple_graph(i, drug_data, organ_data, clinical_data)
            dataset.append(graph_data)
        except Exception as e:
            print(f"Error creating sample {i}: {e}")
            continue
    
    print(f"Successfully created {len(dataset)} graph samples")
    
    if len(dataset) == 0:
        print("No valid samples created. Exiting.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimplePKPDGNN(input_dim=20, hidden_dim=64, output_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    num_epochs = 20
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index)
            
            # Calculate loss only for organ nodes (exclude drug node)
            organ_output = output[:-len(batch.y)] if len(output) > len(batch.y) else output[:len(batch.y)]
            
            loss = criterion(organ_output, batch.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index)
                
                organ_output = output[:-len(batch.y)] if len(output) > len(batch.y) else output[:len(batch.y)]
                
                loss = criterion(organ_output, batch.y)
                val_loss += loss.item()
                
                all_preds.append(organ_output.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        pred_flat = all_preds.flatten()
        target_flat = all_targets.flatten()
        
        mse = mean_squared_error(target_flat, pred_flat)
        mae = mean_absolute_error(target_flat, pred_flat)
        r2 = r2_score(target_flat, pred_flat)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"R²: {r2:.4f}, "
              f"RMSE: {np.sqrt(mse):.4f}")
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final MSE: {mse:.4f}")
    print(f"Final MAE: {mae:.4f}")
    print(f"Final RMSE: {np.sqrt(mse):.4f}")
    print(f"Final R²: {r2:.4f}")
    print("="*50)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.scatter(target_flat[:100], pred_flat[:100], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs True')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    residuals = target_flat - pred_flat
    plt.scatter(pred_flat[:100], residuals[:100], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nEvaluation completed successfully!")
    print("Results saved to 'evaluation_results.png'")
    
    return {
        'final_loss': val_losses[-1],
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'r2': r2
    }


if __name__ == "__main__":
    metrics = main()
