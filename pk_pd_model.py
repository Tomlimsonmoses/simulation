"""
PK-PD Model Interface for Clinical Decision Support
==================================================

This module provides the main PK-PD model interface for the clinical decision support system.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import pickle
import os

class PKPDModel:
    """
    Main PK-PD model class for clinical decision support
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize PK-PD model
        
        Args:
            model_path: Path to trained model file
        """
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._create_default_model()
    
    def _create_default_model(self):
        """Create a default simple model for demonstration"""
        self.model = SimpleMLPPKPD()
        self.feature_names = [
            'dose', 'age', 'weight', 'time_point',
            'drug_Acetaminophen', 'drug_Ibuprofen', 'drug_Metformin', 
            'drug_Lisinopril', 'drug_Atorvastatin', 'drug_Naproxen',
            'drug_Metoprolol', 'drug_Amlodipine', 'drug_Glipizide',
            'drug_Amoxicillin', 'drug_Ciprofloxacin', 'drug_Sertraline',
            'drug_Lorazepam', 'drug_Levothyroxine', 'drug_Prednisone',
            'gender_M', 'gender_F', 'disease_Hypertension', 'disease_Diabetes',
            'disease_Hyperlipidemia', 'disease_Arthritis', 'disease_Healthy',
            'disease_Liver_Cirrhosis', 'disease_Heart_Failure', 'disease_COPD'
        ]
        self.is_trained = True
    
    def load_model(self, model_path: str):
        """
        Load trained model from file
        
        Args:
            model_path: Path to model file
        """
        try:
            if model_path.endswith('.pth'):
                self.model = torch.load(model_path, map_location='cpu')
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.feature_names = model_data['feature_names']
            
            self.is_trained = True
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            self._create_default_model()
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make PK-PD predictions
        
        Args:
            features: Feature matrix
            
        Returns:
            Predictions array with PK and PD values
        """
        if not self.is_trained:
            raise ValueError("Model is not trained or loaded")
        
        # Convert to tensor if using PyTorch model
        if hasattr(self.model, 'eval'):
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                predictions = self.model(features_tensor)
                return predictions.numpy()
        else:
            # Simple numpy-based prediction
            return self._simple_predict(features)
    
    def _simple_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Simple prediction method for demonstration
        
        Args:
            features: Feature matrix
            
        Returns:
            Predictions array
        """
        n_samples = features.shape[0]
        predictions = np.zeros((n_samples, 2))
        
        # Simple heuristic predictions based on dose and patient factors
        for i in range(n_samples):
            feat = features[i]
            dose = feat[0] if len(feat) > 0 else 10
            age = feat[1] if len(feat) > 1 else 65
            weight = feat[2] if len(feat) > 2 else 75
            
            # PK prediction (concentration)
            pk_pred = (dose * weight) / (age * 10) + np.random.normal(0, 0.1)
            pk_pred = max(0, pk_pred)
            
            # PD prediction (response)
            pd_pred = min(100, pk_pred * 0.8 + np.random.normal(0, 0.05))
            pd_pred = max(0, pd_pred)
            
            predictions[i] = [pk_pred, pd_pred]
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_type': type(self.model).__name__,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names
        }


class SimpleMLPPKPD(nn.Module):
    """
    Simple MLP model for PK-PD prediction
    """
    
    def __init__(self, input_size=30, hidden_size1=128, hidden_size2=64):
        super(SimpleMLPPKPD, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size2, 2)  # PK and PD outputs
        )
    
    def forward(self, x):
        return self.layers(x)


# Example usage
if __name__ == "__main__":
    model = PKPDModel()
    print("PK-PD Model initialized")
    print(model.get_model_info())
