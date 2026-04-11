"""
PK-PD Model Usage Demonstration

This script demonstrates how to use the trained PK-PD models for prediction
and integration into your drug-biological system simulation.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

class PKPDModelPredictor:
    """
    Easy-to-use interface for trained PK-PD models.
    """
    
    def __init__(self, models_dir: str = "trained_models"):
        """Initialize the model predictor."""
        self.models_dir = Path(models_dir)
        self.models = {}
        
        # Load available models
        self._load_available_models()
        
        print("PK-PD Model Predictor initialized")
    
    def _load_available_models(self):
        """Load all available trained models."""
        # Load clinical outcome models
        clinical_dir = self.models_dir / "clinical_outcomes"
        if clinical_dir.exists():
            self.models['clinical_outcomes'] = {}
            
            # Load treatment success model (logistic regression - best performer)
            treatment_model_path = clinical_dir / "treatment_success_logistic.pkl"
            if treatment_model_path.exists():
                with open(treatment_model_path, 'rb') as f:
                    self.models['clinical_outcomes']['treatment_success'] = pickle.load(f)
            
            # Load adverse event risk model (random forest - best performer)
            adverse_model_path = clinical_dir / "adverse_event_risk_random_forest.pkl"
            if adverse_model_path.exists():
                with open(adverse_model_path, 'rb') as f:
                    self.models['clinical_outcomes']['adverse_event_risk'] = pickle.load(f)
            
            # Load QoL improvement model (ridge regression - best performer)
            qol_model_path = clinical_dir / "qol_improvement_ridge.pkl"
            if qol_model_path.exists():
                with open(qol_model_path, 'rb') as f:
                    self.models['clinical_outcomes']['qol_improvement'] = pickle.load(f)
        
        # Load drug classification models
        drug_dir = self.models_dir / "drug_classification"
        if drug_dir.exists():
            self.models['drug_classification'] = {}
            
            # Load drug class model (random forest - perfect accuracy)
            drug_class_path = drug_dir / "drug_class_random_forest.pkl"
            if drug_class_path.exists():
                with open(drug_class_path, 'rb') as f:
                    self.models['drug_classification']['drug_class'] = pickle.load(f)
            
            # Load bioavailability model (random forest - best performer)
            bioav_path = drug_dir / "bioavailability_random_forest.pkl"
            if bioav_path.exists():
                with open(bioav_path, 'rb') as f:
                    self.models['drug_classification']['bioavailability'] = pickle.load(f)
    
    def predict_treatment_success(self, patient_data: dict) -> dict:
        """
        Predict treatment success for a patient.
        
        Args:
            patient_data: Dictionary with patient features
                Required: age, sex, weight, dose_mg, treatment_duration, molecular_weight, bioavailability
        
        Returns:
            Dictionary with prediction and confidence
        """
        if 'treatment_success' not in self.models['clinical_outcomes']:
            return {"error": "Treatment success model not available"}
        
        model_info = self.models['clinical_outcomes']['treatment_success']
        model = model_info['model']
        scaler = model_info['scaler']
        feature_columns = model_info['feature_columns']
        
        # Prepare input features
        features = []
        for col in feature_columns:
            features.append(patient_data.get(col, 0))
        
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability[1]),
            "confidence": float(max(probability)),
            "interpretation": "High likelihood of treatment success" if prediction == 1 else "Low likelihood of treatment success"
        }
    
    def predict_adverse_event_risk(self, patient_data: dict) -> dict:
        """
        Predict adverse event risk for a patient.
        
        Args:
            patient_data: Dictionary with patient features
        
        Returns:
            Dictionary with risk assessment
        """
        if 'adverse_event_risk' not in self.models['clinical_outcomes']:
            return {"error": "Adverse event risk model not available"}
        
        model_info = self.models['clinical_outcomes']['adverse_event_risk']
        model = model_info['model']
        feature_columns = model_info['feature_columns']
        
        # Prepare input features
        features = []
        for col in feature_columns:
            features.append(patient_data.get(col, 0))
        
        X = np.array(features).reshape(1, -1)
        
        # Make prediction (Random Forest doesn't need scaling)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        return {
            "risk_level": "High" if prediction == 1 else "Low",
            "probability": float(probability[1]),
            "confidence": float(max(probability)),
            "interpretation": "High risk of adverse events" if prediction == 1 else "Low risk of adverse events"
        }
    
    def predict_qol_improvement(self, patient_data: dict) -> dict:
        """
        Predict quality of life improvement.
        
        Args:
            patient_data: Dictionary with patient features
        
        Returns:
            Dictionary with QoL improvement prediction
        """
        if 'qol_improvement' not in self.models['clinical_outcomes']:
            return {"error": "QoL improvement model not available"}
        
        model_info = self.models['clinical_outcomes']['qol_improvement']
        model = model_info['model']
        scaler = model_info['scaler']
        feature_columns = model_info['feature_columns']
        
        # Prepare input features
        features = []
        for col in feature_columns:
            features.append(patient_data.get(col, 0))
        
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        return {
            "qol_improvement": float(prediction),
            "interpretation": "Significant QoL improvement" if prediction > 0.5 else "Modest QoL improvement",
            "normalized_score": float(np.clip(prediction, 0, 1))
        }
    
    def predict_drug_class(self, drug_data: dict) -> dict:
        """
        Predict drug class based on molecular properties.
        
        Args:
            drug_data: Dictionary with drug features
        
        Returns:
            Dictionary with drug class prediction
        """
        if 'drug_class' not in self.models['drug_classification']:
            return {"error": "Drug classification model not available"}
        
        model_info = self.models['drug_classification']['drug_class']
        model = model_info['model']
        label_encoder = model_info['label_encoder']
        feature_columns = model_info['feature_columns']
        
        # Prepare input features
        features = []
        for col in feature_columns:
            features.append(drug_data.get(col, 0))
        
        X = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Decode label
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(max(probability)),
            "all_probabilities": {
                label_encoder.inverse_transform([i])[0]: float(prob) 
                for i, prob in enumerate(probability)
            }
        }
    
    def predict_bioavailability_category(self, drug_data: dict) -> dict:
        """
        Predict bioavailability category (low/medium/high).
        
        Args:
            drug_data: Dictionary with drug features
        
        Returns:
            Dictionary with bioavailability prediction
        """
        if 'bioavailability' not in self.models['drug_classification']:
            return {"error": "Bioavailability model not available"}
        
        model_info = self.models['drug_classification']['bioavailability']
        model = model_info['model']
        label_encoder = model_info['label_encoder']
        feature_columns = model_info['feature_columns']
        
        # Prepare input features
        features = []
        for col in feature_columns:
            features.append(drug_data.get(col, 0))
        
        X = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Decode label
        predicted_category = label_encoder.inverse_transform([prediction])[0]
        
        return {
            "bioavailability_category": predicted_category,
            "confidence": float(max(probability)),
            "all_probabilities": {
                label_encoder.inverse_transform([i])[0]: float(prob) 
                for i, prob in enumerate(probability)
            }
        }
    
    def comprehensive_patient_assessment(self, patient_data: dict) -> dict:
        """
        Provide a comprehensive assessment for a patient.
        
        Args:
            patient_data: Dictionary with patient and drug information
        
        Returns:
            Comprehensive assessment dictionary
        """
        assessment = {
            "patient_profile": patient_data,
            "predictions": {}
        }
        
        # Treatment success prediction
        treatment_result = self.predict_treatment_success(patient_data)
        assessment["predictions"]["treatment_success"] = treatment_result
        
        # Adverse event risk prediction
        adverse_result = self.predict_adverse_event_risk(patient_data)
        assessment["predictions"]["adverse_event_risk"] = adverse_result
        
        # QoL improvement prediction
        qol_result = self.predict_qol_improvement(patient_data)
        assessment["predictions"]["qol_improvement"] = qol_result
        
        # Overall recommendation
        success_prob = treatment_result.get("probability", 0)
        adverse_prob = adverse_result.get("probability", 0)
        
        if success_prob > 0.7 and adverse_prob < 0.3:
            recommendation = "Proceed with treatment - High success probability, low risk"
        elif success_prob > 0.5 and adverse_prob < 0.5:
            recommendation = "Consider treatment with monitoring"
        else:
            recommendation = "Alternative treatment recommended - High risk or low success probability"
        
        assessment["recommendation"] = recommendation
        
        return assessment

def demonstrate_model_usage():
    """Demonstrate how to use the trained models."""
    
    print("🚀 PK-PD Model Usage Demonstration")
    print("=" * 50)
    
    # Initialize predictor
    predictor = PKPDModelPredictor()
    
    # Example 1: Patient assessment
    print("\n📋 Example 1: Patient Assessment")
    print("-" * 30)
    
    patient_data = {
        'age': 65,
        'sex': 1,  # 1 for male, 0 for female
        'weight': 70.0,
        'dose_mg': 100.0,
        'treatment_duration': 30.0,
        'molecular_weight': 300.5,
        'bioavailability': 0.7
    }
    
    assessment = predictor.comprehensive_patient_assessment(patient_data)
    
    print(f"👤 Patient Profile:")
    print(f"   Age: {patient_data['age']} years")
    print(f"   Weight: {patient_data['weight']} kg")
    print(f"   Dose: {patient_data['dose_mg']} mg")
    print(f"   Treatment Duration: {patient_data['treatment_duration']} days")
    
    print(f"\n🎯 Predictions:")
    treatment = assessment["predictions"]["treatment_success"]
    print(f"   Treatment Success: {treatment['interpretation']} ({treatment['confidence']:.2f} confidence)")
    
    adverse = assessment["predictions"]["adverse_event_risk"]
    print(f"   Adverse Event Risk: {adverse['interpretation']} ({adverse['confidence']:.2f} confidence)")
    
    qol = assessment["predictions"]["qol_improvement"]
    print(f"   QoL Improvement: {qol['interpretation']} (score: {qol['normalized_score']:.2f})")
    
    print(f"\n💡 Recommendation: {assessment['recommendation']}")
    
    # Example 2: Drug classification
    print("\n💊 Example 2: Drug Classification")
    print("-" * 30)
    
    drug_data = {
        'molecular_weight': 300.5,
        'logp': 2.1,
        'bioavailability': 0.7,
        'protein_binding': 0.8,
        'clearance_rate': 5.2,
        'volume_distribution': 50.0,
        'half_life': 8.0
    }
    
    drug_class = predictor.predict_drug_class(drug_data)
    bioav = predictor.predict_bioavailability_category(drug_data)
    
    print(f"🔬 Drug Properties:")
    print(f"   Molecular Weight: {drug_data['molecular_weight']}")
    print(f"   LogP: {drug_data['logp']}")
    print(f"   Bioavailability: {drug_data['bioavailability']}")
    
    print(f"\n🎯 Predictions:")
    print(f"   Drug Class: {drug_class['predicted_class']} ({drug_class['confidence']:.2f} confidence)")
    print(f"   Bioavailability Category: {bioav['bioavailability_category']} ({bioav['confidence']:.2f} confidence)")
    
    # Example 3: Multiple patient scenarios
    print("\n👥 Example 3: Multiple Patient Scenarios")
    print("-" * 30)
    
    scenarios = [
        {
            "name": "Young Adult",
            "data": {
                'age': 25, 'sex': 0, 'weight': 60.0, 'dose_mg': 50.0,
                'treatment_duration': 14.0, 'molecular_weight': 250.0, 'bioavailability': 0.8
            }
        },
        {
            "name": "Elderly Patient",
            "data": {
                'age': 75, 'sex': 1, 'weight': 80.0, 'dose_mg': 150.0,
                'treatment_duration': 60.0, 'molecular_weight': 400.0, 'bioavailability': 0.5
            }
        },
        {
            "name": "Middle-aged Female",
            "data": {
                'age': 45, 'sex': 0, 'weight': 65.0, 'dose_mg': 100.0,
                'treatment_duration': 30.0, 'molecular_weight': 350.0, 'bioavailability': 0.6
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}:")
        assessment = predictor.comprehensive_patient_assessment(scenario['data'])
        
        treatment = assessment["predictions"]["treatment_success"]
        adverse = assessment["predictions"]["adverse_event_risk"]
        
        print(f"   Success Probability: {treatment['probability']:.2f}")
        print(f"   Adverse Risk: {adverse['probability']:.2f}")
        print(f"   Recommendation: {assessment['recommendation']}")
    
    print("\n✅ Model demonstration completed!")
    print("\n📖 How to use in your own code:")
    print("""
# Initialize predictor
predictor = PKPDModelPredictor()

# Make predictions
treatment_result = predictor.predict_treatment_success(patient_data)
adverse_result = predictor.predict_adverse_event_risk(patient_data)
qol_result = predictor.predict_qol_improvement(patient_data)

# Get comprehensive assessment
assessment = predictor.comprehensive_patient_assessment(patient_data)
""")

if __name__ == "__main__":
    demonstrate_model_usage()
