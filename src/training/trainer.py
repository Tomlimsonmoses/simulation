"""
Training Framework for GNN PK-PD Model

This module implements comprehensive training and evaluation framework including:
- Model training with custom loss functions
- Cross-validation strategies
- Hyperparameter optimization
- Early stopping and model selection
- Performance metrics for PK-PD modeling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from tqdm import tqdm
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class PKPDLossFunction(nn.Module):
    """Custom loss function for PK-PD modeling."""
    
    def __init__(
        self,
        pk_weight: float = 1.0,
        pd_weight: float = 1.0,
        temporal_weight: float = 0.5,
        regularization_weight: float = 0.01
    ):
        super(PKPDLossFunction, self).__init__()
        self.pk_weight = pk_weight
        self.pd_weight = pd_weight
        self.temporal_weight = temporal_weight
        self.regularization_weight = regularization_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate custom PK-PD loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            model: Model for regularization
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # PK loss (concentration prediction)
        if 'pk_concentrations' in predictions and 'pk_concentrations' in targets:
            pk_pred = predictions['pk_concentrations']
            pk_target = targets['pk_concentrations']
            
            pk_mse = self.mse_loss(pk_pred, pk_target)
            pk_mae = self.mae_loss(pk_pred, pk_target)
            losses['pk_mse'] = pk_mse
            losses['pk_mae'] = pk_mae
            losses['pk_loss'] = pk_mse + 0.1 * pk_mae
        
        # PD loss (response prediction)
        if 'pd_responses' in predictions and 'pd_responses' in targets:
            pd_pred = predictions['pd_responses']
            pd_target = targets['pd_responses']
            
            pd_mse = self.mse_loss(pd_pred, pd_target)
            pd_mae = self.mae_loss(pd_pred, pd_target)
            losses['pd_mse'] = pd_mse
            losses['pd_mae'] = pd_mae
            losses['pd_loss'] = pd_mse + 0.1 * pd_mae
        
        # Temporal consistency loss
        if 'organ_embeddings' in predictions:
            embeddings = predictions['organ_embeddings']
            if len(embeddings.shape) > 2:  # Time series
                # Encourage smooth temporal transitions
                temporal_diff = torch.diff(embeddings, dim=1)
                temporal_loss = torch.mean(temporal_diff ** 2)
                losses['temporal_loss'] = temporal_loss
            else:
                losses['temporal_loss'] = torch.tensor(0.0)
        
        # Regularization loss
        if model is not None:
            reg_loss = 0.0
            for param in model.parameters():
                reg_loss += torch.norm(param, 2)
            losses['regularization_loss'] = self.regularization_weight * reg_loss
        else:
            losses['regularization_loss'] = torch.tensor(0.0)
        
        # Total loss
        total_loss = (
            self.pk_weight * losses.get('pk_loss', torch.tensor(0.0)) +
            self.pd_weight * losses.get('pd_loss', torch.tensor(0.0)) +
            self.temporal_weight * losses.get('temporal_loss', torch.tensor(0.0)) +
            losses['regularization_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses


class PKPDMetrics:
    """Evaluation metrics for PK-PD modeling."""
    
    @staticmethod
    def calculate_pk_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        time_points: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Calculate PK-specific metrics."""
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Flatten for overall metrics
        pred_flat = pred_np.flatten()
        target_flat = target_np.flatten()
        
        metrics = {
            'mse': mean_squared_error(target_flat, pred_flat),
            'mae': mean_absolute_error(target_flat, pred_flat),
            'rmse': np.sqrt(mean_squared_error(target_flat, pred_flat)),
            'r2': r2_score(target_flat, pred_flat)
        }
        
        # Calculate percentage error
        if np.any(target_flat != 0):
            percent_errors = np.abs((target_flat - pred_flat) / target_flat) * 100
            metrics['mape'] = np.mean(percent_errors)
        
        # Calculate AUC error if time points available
        if time_points is not None and len(pred_np.shape) > 1:
            time_np = time_points.detach().cpu().numpy()
            auc_pred = np.trapz(pred_np, time_np, axis=1)
            auc_target = np.trapz(target_np, time_np, axis=1)
            
            auc_mse = mean_squared_error(auc_target, auc_pred)
            auc_mae = mean_absolute_error(auc_target, auc_pred)
            metrics['auc_mse'] = auc_mse
            metrics['auc_mae'] = auc_mae
        
        # Calculate Cmax and Tmax errors
        if len(pred_np.shape) > 1:
            cmax_pred = np.max(pred_np, axis=1)
            cmax_target = np.max(target_np, axis=1)
            tmax_pred = np.argmax(pred_np, axis=1)
            tmax_target = np.argmax(target_np, axis=1)
            
            metrics['cmax_mse'] = mean_squared_error(cmax_target, cmax_pred)
            metrics['cmax_mae'] = mean_absolute_error(cmax_target, cmax_pred)
            metrics['tmax_mae'] = mean_absolute_error(tmax_target, tmax_pred)
        
        return metrics
    
    @staticmethod
    def calculate_pd_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        response_type: str = 'continuous'
    ) -> Dict[str, float]:
        """Calculate PD-specific metrics."""
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        pred_flat = pred_np.flatten()
        target_flat = target_np.flatten()
        
        metrics = {
            'mse': mean_squared_error(target_flat, pred_flat),
            'mae': mean_absolute_error(target_flat, pred_flat),
            'rmse': np.sqrt(mean_squared_error(target_flat, pred_flat)),
            'r2': r2_score(target_flat, pred_flat)
        }
        
        # Response-specific metrics
        if response_type == 'binary':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            pred_binary = (pred_flat > 0.5).astype(int)
            target_binary = target_flat.astype(int)
            
            metrics.update({
                'accuracy': accuracy_score(target_binary, pred_binary),
                'precision': precision_score(target_binary, pred_binary, average='weighted', zero_division=0),
                'recall': recall_score(target_binary, pred_binary, average='weighted', zero_division=0),
                'f1': f1_score(target_binary, pred_binary, average='weighted', zero_division=0)
            })
            
            # AUC if probabilities available
            if len(np.unique(target_binary)) == 2:
                try:
                    metrics['auc'] = roc_auc_score(target_binary, pred_flat)
                except:
                    metrics['auc'] = 0.5
        
        elif response_type == 'categorical':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            pred_categorical = np.argmax(pred_np, axis=1) if len(pred_np.shape) > 1 else pred_flat.round()
            target_categorical = target_np.astype(int)
            
            metrics.update({
                'accuracy': accuracy_score(target_categorical, pred_categorical),
                'precision': precision_score(target_categorical, pred_categorical, average='weighted', zero_division=0),
                'recall': recall_score(target_categorical, pred_categorical, average='weighted', zero_division=0),
                'f1': f1_score(target_categorical, pred_categorical, average='weighted', zero_division=0)
            })
        
        return metrics
    
    @staticmethod
    def calculate_therapeutic_index_metrics(
        efficacy_predictions: torch.Tensor,
        toxicity_predictions: torch.Tensor,
        efficacy_targets: torch.Tensor,
        toxicity_targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate therapeutic index related metrics."""
        eff_pred = efficacy_predictions.detach().cpu().numpy().flatten()
        tox_pred = toxicity_predictions.detach().cpu().numpy().flatten()
        eff_target = efficacy_targets.detach().cpu().numpy().flatten()
        tox_target = toxicity_targets.detach().cpu().numpy().flatten()
        
        # Calculate therapeutic indices
        ti_pred = tox_pred / (eff_pred + 1e-8)  # Avoid division by zero
        ti_target = tox_target / (eff_target + 1e-8)
        
        metrics = {
            'ti_mse': mean_squared_error(ti_target, ti_pred),
            'ti_mae': mean_absolute_error(ti_target, ti_pred),
            'ti_correlation': np.corrcoef(ti_target, ti_pred)[0, 1]
        }
        
        # Safety margin metrics
        safety_margin_pred = tox_pred - eff_pred
        safety_margin_target = tox_target - eff_target
        
        metrics.update({
            'safety_margin_mse': mean_squared_error(safety_margin_target, safety_margin_pred),
            'safety_margin_mae': mean_absolute_error(safety_margin_target, safety_margin_pred)
        })
        
        return metrics


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
        monitor: str = 'val_loss'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, current_loss: float, model: nn.Module) -> bool:
        """Check if should stop training."""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class PKPDTrainer:
    """Main trainer class for GNN PK-PD model."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        loss_function: Optional[PKPDLossFunction] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        early_stopping: Optional[EarlyStopping] = None
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.loss_function = loss_function or PKPDLossFunction()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        
        self.training_history = []
        self.validation_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'pk_loss': 0.0,
            'pd_loss': 0.0,
            'temporal_loss': 0.0,
            'regularization_loss': 0.0
        }
        
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(
                batch.organ_features,
                batch.drug_features,
                batch.edge_index,
                batch.edge_attr,
                batch.disease_features,
                batch.batch
            )
            
            # Prepare targets
            targets = {
                'pk_concentrations': batch.pk_targets,
                'pd_responses': batch.pd_targets
            }
            
            # Calculate loss
            losses = self.loss_function(predictions, targets, self.model)
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            num_batches += 1
            
            # Log progress
            if batch_idx % log_interval == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {losses["total_loss"].item():.4f}'
                )
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {
            'total_loss': 0.0,
            'pk_loss': 0.0,
            'pd_loss': 0.0,
            'temporal_loss': 0.0,
            'regularization_loss': 0.0
        }
        
        all_predictions = {'pk_concentrations': [], 'pd_responses': []}
        all_targets = {'pk_concentrations': [], 'pd_responses': []}
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(
                    batch.organ_features,
                    batch.drug_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.disease_features,
                    batch.batch
                )
                
                # Prepare targets
                targets = {
                    'pk_concentrations': batch.pk_targets,
                    'pd_responses': batch.pd_targets
                }
                
                # Calculate loss
                losses = self.loss_function(predictions, targets, self.model)
                
                # Accumulate losses
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key].item()
                
                # Store predictions and targets for metrics
                for key in all_predictions:
                    if key in predictions and key in targets:
                        all_predictions[key].append(predictions[key])
                        all_targets[key].append(targets[key])
                
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Calculate metrics
        metrics = {}
        
        for key in all_predictions:
            if all_predictions[key] and all_targets[key]:
                pred_tensor = torch.cat(all_predictions[key], dim=0)
                target_tensor = torch.cat(all_targets[key], dim=0)
                
                if key == 'pk_concentrations':
                    pk_metrics = PKPDMetrics.calculate_pk_metrics(pred_tensor, target_tensor)
                    for metric_name, value in pk_metrics.items():
                        metrics[f'pk_{metric_name}'] = value
                elif key == 'pd_responses':
                    pd_metrics = PKPDMetrics.calculate_pd_metrics(pred_tensor, target_tensor)
                    for metric_name, value in pd_metrics.items():
                        metrics[f'pd_{metric_name}'] = value
        
        # Combine losses and metrics
        epoch_results = {**epoch_losses, **metrics}
        
        return epoch_results
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str = './checkpoints',
        experiment_name: str = 'pkpd_experiment'
    ) -> Dict[str, List[float]]:
        """Train the model."""
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f'Starting epoch {epoch + 1}/{num_epochs}')
            
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            self.training_history.append(train_losses)
            
            # Validate
            val_results = self.validate_epoch(val_loader, epoch)
            self.validation_history.append(val_results)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['total_loss'])
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_results['total_loss'] < best_val_loss:
                best_val_loss = val_results['total_loss']
                self.save_model(save_dir, f'{experiment_name}_best.pth')
            
            # Check early stopping
            if self.early_stopping:
                if self.early_stopping(val_results['total_loss'], self.model):
                    self.logger.info(f'Early stopping at epoch {epoch + 1}')
                    break
            
            # Log epoch results
            self._log_epoch_results(epoch, train_losses, val_results)
        
        # Save final model
        self.save_model(save_dir, f'{experiment_name}_final.pth')
        
        return {
            'train_history': self.training_history,
            'val_history': self.validation_history
        }
    
    def _log_epoch_results(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_results: Dict[str, float]
    ):
        """Log epoch results."""
        self.logger.info(f'Epoch {epoch + 1} Results:')
        self.logger.info(f'  Train Loss: {train_losses["total_loss"]:.4f}')
        self.logger.info(f'  Val Loss: {val_results["total_loss"]:.4f}')
        
        if 'pk_r2' in val_results:
            self.logger.info(f'  PK R²: {val_results["pk_r2"]:.4f}')
        if 'pd_r2' in val_results:
            self.logger.info(f'  PD R²: {val_results["pd_r2"]:.4f}')
    
    def save_model(self, save_dir: str, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'validation_history': self.validation_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(save_dir, filename))
    
    def load_model(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_history = checkpoint.get('training_history', [])
        self.validation_history = checkpoint.get('validation_history', [])


class CrossValidator:
    """Cross-validation utility for PK-PD model."""
    
    def __init__(
        self,
        model_class: type,
        model_params: Dict,
        trainer_class: type = PKPDTrainer,
        trainer_params: Optional[Dict] = None,
        cv_folds: int = 5,
        stratify: bool = True
    ):
        self.model_class = model_class
        self.model_params = model_params
        self.trainer_class = trainer_class
        self.trainer_params = trainer_params or {}
        self.cv_folds = cv_folds
        self.stratify = stratify
        
        self.cv_results = []
        
    def cross_validate(
        self,
        dataset,
        num_epochs: int = 50,
        random_state: int = 42
    ) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Create cross-validation splits
        if self.stratify:
            # Use stratified K-fold if labels available
            if hasattr(dataset, 'y'):
                skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=random_state)
                splits = list(skf.split(range(len(dataset)), dataset.y))
            else:
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=random_state)
                splits = list(kf.split(range(len(dataset))))
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(range(len(dataset))))
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f'Fold {fold + 1}/{self.cv_folds}')
            
            # Create train/val subsets
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
            
            # Initialize model and trainer
            model = self.model_class(**self.model_params)
            trainer = self.trainer_class(model, **self.trainer_params)
            
            # Train
            history = trainer.train(train_loader, val_loader, num_epochs)
            
            # Get final validation metrics
            final_val_metrics = trainer.validation_history[-1]
            fold_results.append(final_val_metrics)
            
            print(f'Fold {fold + 1} Val Loss: {final_val_metrics["total_loss"]:.4f}')
        
        # Aggregate results
        aggregated_results = {}
        for key in fold_results[0].keys():
            values = [fold[key] for fold in fold_results]
            aggregated_results[f'{key}_mean'] = np.mean(values)
            aggregated_results[f'{key}_std'] = np.std(values)
            aggregated_results[f'{key}_values'] = values
        
        self.cv_results = aggregated_results
        return aggregated_results
    
    def get_cv_summary(self) -> pd.DataFrame:
        """Get cross-validation summary as DataFrame."""
        if not self.cv_results:
            return pd.DataFrame()
        
        summary_data = []
        for key, value in self.cv_results.items():
            if key.endswith('_mean'):
                metric_name = key.replace('_mean', '')
                mean_val = value
                std_val = self.cv_results.get(f'{metric_name}_std', 0)
                summary_data.append({
                    'Metric': metric_name,
                    'Mean': mean_val,
                    'Std': std_val
                })
        
        return pd.DataFrame(summary_data)


class HyperparameterOptimizer:
    """Hyperparameter optimization for PK-PD model."""
    
    def __init__(
        self,
        model_class: type,
        trainer_class: type = PKPDTrainer,
        search_space: Optional[Dict] = None
    ):
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.search_space = search_space or self._default_search_space()
        
        self.optimization_history = []
        
    def _default_search_space(self) -> Dict:
        """Default hyperparameter search space."""
        return {
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
            'hidden_dim': [32, 64, 128],
            'num_layers': [2, 3, 4],
            'dropout': [0.1, 0.2, 0.3],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'pk_weight': [0.5, 1.0, 2.0],
            'pd_weight': [0.5, 1.0, 2.0]
        }
    
    def optimize(
        self,
        train_dataset,
        val_dataset,
        max_trials: int = 50,
        cv_folds: int = 3,
        epochs_per_trial: int = 20
    ) -> Dict:
        """Perform hyperparameter optimization."""
        from itertools import product
        
        # Generate parameter combinations
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        
        best_score = float('inf')
        best_params = None
        
        for trial in range(min(max_trials, len(list(product(*param_values))))):
            # Sample parameters
            param_combination = list(product(*param_values))[trial]
            params = dict(zip(param_names, param_combination))
            
            print(f'Trial {trial + 1}: {params}')
            
            # Create model with current parameters
            model_params = {
                'hidden_dim': params['hidden_dim'],
                'num_layers': params['num_layers'],
                'dropout': params['dropout']
            }
            
            model = self.model_class(**model_params)
            
            # Create trainer with current parameters
            loss_function = PKPDLossFunction(
                pk_weight=params['pk_weight'],
                pd_weight=params['pd_weight']
            )
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            
            trainer = self.trainer_class(
                model,
                loss_function=loss_function,
                optimizer=optimizer
            )
            
            # Cross-validation
            cv = CrossValidator(
                self.model_class,
                model_params,
                self.trainer_class,
                cv_folds=cv_folds
            )
            
            # Combine datasets for CV
            combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
            
            try:
                cv_results = cv.cross_validate(combined_dataset, num_epochs=epochs_per_trial)
                
                # Use validation loss as score
                score = cv_results.get('total_loss_mean', float('inf'))
                
                self.optimization_history.append({
                    'trial': trial + 1,
                    'params': params,
                    'score': score,
                    'cv_results': cv_results
                })
                
                if score < best_score:
                    best_score = score
                    best_params = params
                
                print(f'Trial {trial + 1} Score: {score:.4f}')
                
            except Exception as e:
                print(f'Trial {trial + 1} Failed: {str(e)}')
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': self.optimization_history
        }
