"""
Model Training Example

This example demonstrates how to train the GNN PK-PD model
using synthetic or real data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List

# Import project modules
from models.gnn_pkpd import PKPDGNN, PKPDDataProcessor, PKPDTrainer
from training.trainer import PKPDLossFunction, EarlyStopping, CrossValidator
from data.dataset_specs import DatasetRequirements, DataPipeline


def create_synthetic_dataset(num_samples: int = 1000) -> List[Data]:
    """Create synthetic training data for the GNN model."""
    
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    dataset = []
    data_processor = PKPDDataProcessor()
    
    for i in range(num_samples):
        # Random organ parameters
        organ_params = {}
        for organ in data_processor.organ_types:
            organ_params[organ] = {
                'volume': np.random.uniform(0.5, 2.0),
                'blood_flow': np.random.uniform(50, 2000),
                'metabolic_rate': np.random.uniform(0.5, 2.0),
                'receptor_density': np.random.uniform(0.5, 2.0),
                'permeability': np.random.uniform(1e-8, 1e-5),
                'clearance': np.random.uniform(0.1, 2.0),
                'surface_area': np.random.uniform(0.1, 5.0),
                'oxygen_consumption': np.random.uniform(5, 100),
                'ph': np.random.uniform(6.5, 7.8),
                'temperature': np.random.uniform(36.5, 37.5),
                'weight_ratio': np.random.uniform(0.01, 0.5),
                'enzyme_activity': np.random.uniform(0.5, 2.0),
                'transporter_density': np.random.uniform(0.5, 2.0),
                'binding_capacity': np.random.uniform(0.5, 2.0),
                'sensitivity': np.random.uniform(0.5, 2.0),
                'health_status': np.random.uniform(0.8, 1.0)
            }
        
        # Random drug parameters
        drug_params = {
            'molecular_weight': np.random.uniform(200, 600),
            'logp': np.random.uniform(-1, 5),
            'polar_surface_area': np.random.uniform(20, 150),
            'solubility': np.random.uniform(1, 500),
            'bioavailability': np.random.uniform(0.3, 1.0),
            'protein_binding': np.random.uniform(0.1, 0.95),
            'clearance_rate': np.random.uniform(0.05, 1.5),
            'volume_distribution': np.random.uniform(0.5, 3.0),
            'half_life': np.random.uniform(2, 24)
        }
        
        # Create features
        organ_features = data_processor.create_organ_features(organ_params)
        drug_features = data_processor.create_drug_features(drug_params)
        edge_index, edge_attr = data_processor.create_graph_edges({})
        
        # Create synthetic targets
        # PK targets (drug concentrations)
        pk_targets = torch.randn(len(data_processor.organ_types), 1) * 10 + 50
        
        # PD targets (organ responses)
        pd_targets = torch.randn(len(data_processor.organ_types), 1) * 5 + 10
        
        # Disease features
        disease_features = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)  # Healthy
        
        # Create batch assignment (single graph per sample)
        batch = torch.zeros(organ_features.shape[0] + drug_features.shape[0], dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            organ_features=organ_features,
            drug_features=drug_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            disease_features=disease_features,
            batch=batch,
            pk_targets=pk_targets,
            pd_targets=pd_targets
        )
        
        dataset.append(data)
    
    print(f"Created dataset with {len(dataset)} samples")
    return dataset


def train_model_example():
    """Example of training the GNN PK-PD model."""
    
    print("=== Model Training Example ===\n")
    
    # Create synthetic dataset
    dataset = create_synthetic_dataset(num_samples=500)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = PKPDGNN(
        organ_features_dim=16,
        drug_features_dim=8,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        dropout=0.2
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    loss_function = PKPDLossFunction(
        pk_weight=1.0,
        pd_weight=1.0,
        temporal_weight=0.5,
        regularization_weight=0.01
    )
    
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.001,
        monitor='val_loss'
    )
    
    trainer = PKPDTrainer(
        model=model,
        loss_function=loss_function,
        early_stopping=early_stopping
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        save_dir='./checkpoints',
        experiment_name='pkpd_synthetic'
    )
    
    print("Training completed!")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def plot_training_history(history: Dict[str, List[Dict[str, float]]]):
    """Plot training and validation history."""
    
    import matplotlib.pyplot as plt
    
    train_history = history['train_history']
    val_history = history['val_history']
    
    epochs = range(1, len(train_history) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    train_losses = [h['total_loss'] for h in train_history]
    val_losses = [h['total_loss'] for h in val_history]
    
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PK loss
    pk_train_losses = [h.get('pk_loss', 0) for h in train_history]
    pk_val_losses = [h.get('pk_loss', 0) for h in val_history]
    
    axes[0, 1].plot(epochs, pk_train_losses, 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, pk_val_losses, 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PK Loss')
    axes[0, 1].set_title('PK Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PD loss
    pd_train_losses = [h.get('pd_loss', 0) for h in train_history]
    pd_val_losses = [h.get('pd_loss', 0) for h in val_history]
    
    axes[1, 0].plot(epochs, pd_train_losses, 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, pd_val_losses, 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PD Loss')
    axes[1, 0].set_title('PD Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # R² scores if available
    if 'pk_r2' in val_history[0]:
        pk_r2_scores = [h.get('pk_r2', 0) for h in val_history]
        pd_r2_scores = [h.get('pd_r2', 0) for h in val_history]
        
        axes[1, 1].plot(epochs, pk_r2_scores, 'g-', label='PK R²', linewidth=2)
        axes[1, 1].plot(epochs, pd_r2_scores, 'm-', label='PD R²', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Model Performance (R²)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'R² scores\nnot available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Model Performance')
    
    plt.tight_layout()
    plt.savefig('./training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training history plot saved as ./training_history.png")


def cross_validation_example():
    """Example of cross-validation for model evaluation."""
    
    print("\n=== Cross-Validation Example ===\n")
    
    # Create larger dataset for CV
    dataset = create_synthetic_dataset(num_samples=200)
    
    # Define model parameters
    model_params = {
        'organ_features_dim': 16,
        'drug_features_dim': 8,
        'hidden_dim': 64,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.2
    }
    
    # Define trainer parameters
    trainer_params = {
        'loss_function': PKPDLossFunction(),
        'early_stopping': EarlyStopping(patience=5)
    }
    
    # Create cross-validator
    cv = CrossValidator(
        model_class=PKPDGNN,
        model_params=model_params,
        trainer_class=PKPDTrainer,
        trainer_params=trainer_params,
        cv_folds=5,
        stratify=False
    )
    
    print("Running 5-fold cross-validation...")
    
    # Run cross-validation
    cv_results = cv.cross_validate(
        dataset=dataset,
        num_epochs=20,  # Fewer epochs for CV
        random_state=42
    )
    
    # Print results
    print("\nCross-Validation Results:")
    for key, value in cv_results.items():
        if key.endswith('_mean'):
            metric_name = key.replace('_mean', '')
            std_value = cv_results.get(f'{metric_name}_std', 0)
            print(f"{metric_name}: {value:.4f} ± {std_value:.4f}")
    
    # Get summary DataFrame
    summary_df = cv.get_cv_summary()
    print("\nCross-Validation Summary:")
    print(summary_df.to_string(index=False))
    
    return cv_results


def hyperparameter_optimization_example():
    """Example of hyperparameter optimization."""
    
    print("\n=== Hyperparameter Optimization Example ===\n")
    
    # Create dataset
    dataset = create_synthetic_dataset(num_samples=100)
    
    # Split for optimization
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # Combine for CV
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    
    # Initialize optimizer
    from training.trainer import HyperparameterOptimizer
    
    optimizer = HyperparameterOptimizer(
        model_class=PKPDGNN,
        trainer_class=PKPDTrainer
    )
    
    print("Running hyperparameter optimization...")
    
    # Run optimization (limited trials for example)
    opt_results = optimizer.optimize(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_trials=10,  # Limited for demo
        cv_folds=3,
        epochs_per_trial=10
    )
    
    print("\nOptimization Results:")
    print(f"Best parameters: {opt_results['best_params']}")
    print(f"Best score: {opt_results['best_score']:.4f}")
    
    return opt_results


def evaluate_model_performance(model, test_dataset):
    """Evaluate trained model performance."""
    
    print("\n=== Model Performance Evaluation ===\n")
    
    from training.trainer import PKPDMetrics
    
    model.eval()
    
    all_pk_predictions = []
    all_pk_targets = []
    all_pd_predictions = []
    all_pd_targets = []
    
    with torch.no_grad():
        for data in test_dataset:
            # Forward pass
            predictions = model(
                data.organ_features,
                data.drug_features,
                data.edge_index,
                data.edge_attr,
                data.disease_features,
                data.batch
            )
            
            # Collect predictions and targets
            all_pk_predictions.append(predictions['pk_concentrations'])
            all_pk_targets.append(data.pk_targets)
            all_pd_predictions.append(predictions['pd_responses'])
            all_pd_targets.append(data.pd_targets)
    
    # Concatenate all results
    pk_pred_tensor = torch.cat(all_pk_predictions, dim=0)
    pk_target_tensor = torch.cat(all_pk_targets, dim=0)
    pd_pred_tensor = torch.cat(all_pd_predictions, dim=0)
    pd_target_tensor = torch.cat(all_pd_targets, dim=0)
    
    # Calculate metrics
    pk_metrics = PKPDMetrics.calculate_pk_metrics(pk_pred_tensor, pk_target_tensor)
    pd_metrics = PKPDMetrics.calculate_pd_metrics(pd_pred_tensor, pd_target_tensor)
    
    print("PK Performance Metrics:")
    for metric, value in pk_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nPD Performance Metrics:")
    for metric, value in pd_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return pk_metrics, pd_metrics


def main():
    """Run the complete training example."""
    
    print("=== GNN PK-PD Model Training Example ===\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train model
    model, history = train_model_example()
    
    # Cross-validation
    cv_results = cross_validation_example()
    
    # Hyperparameter optimization (optional - can be time-consuming)
    # opt_results = hyperparameter_optimization_example()
    
    # Create test set for final evaluation
    test_dataset = create_synthetic_dataset(num_samples=100)
    
    # Evaluate final model
    pk_metrics, pd_metrics = evaluate_model_performance(model, test_dataset)
    
    print("\n=== Training example completed! ===")
    print("Model checkpoint saved to ./checkpoints/")
    print("Training history plot saved as ./training_history.png")


if __name__ == "__main__":
    main()
