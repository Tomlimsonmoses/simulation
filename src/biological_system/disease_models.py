"""
Disease Simulation Module

This module implements comprehensive disease modeling including:
- Chronic diseases (diabetes, hypertension, etc.)
- Acute diseases (infection, injury)
- Genetic diseases
- Disease progression models
- Disease-drug interactions
"""

import numpy as np
import torch
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class DiseaseType(Enum):
    """Enumeration of disease types."""
    CARDIOVASCULAR = "cardiovascular"
    METABOLIC = "metabolic"
    NEUROLOGICAL = "neurological"
    RESPIRATORY = "respiratory"
    RENAL = "renal"
    HEPATIC = "hepatic"
    IMMUNE = "immune"
    ENDOCRINE = "endocrine"
    CANCER = "cancer"
    INFECTIOUS = "infectious"


class DiseaseSeverity(Enum):
    """Disease severity levels."""
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


@dataclass
class DiseaseParameters:
    """Parameters defining a disease."""
    disease_type: DiseaseType
    name: str
    severity: DiseaseSeverity
    onset_time: float  # hours from simulation start
    progression_rate: float  # Rate of disease progression
    
    # Organ-specific effects
    organ_effects: Dict[str, float] = field(default_factory=dict)
    
    # Physiological parameter changes
    physiological_changes: Dict[str, float] = field(default_factory=dict)
    
    # Biomarker changes
    biomarker_changes: Dict[str, float] = field(default_factory=dict)
    
    # Treatment response factors
    treatment_sensitivity: float = 1.0
    drug_resistance_factors: Dict[str, float] = field(default_factory=dict)


class DiseaseModel:
    """Base class for disease models."""
    
    def __init__(self, disease_params: DiseaseParameters):
        self.params = disease_params
        self.current_severity = disease_params.severity.value
        self.time_since_onset = 0.0
        
    def update_disease_state(self, dt: float) -> Dict[str, float]:
        """Update disease state over time."""
        self.time_since_onset += dt
        
        # Disease progression
        if self.params.progression_rate > 0:
            severity_increase = self.params.progression_rate * dt
            self.current_severity += severity_increase
            self.current_severity = min(self.current_severity, 4.0)  # Cap at critical
        
        return self.get_current_effects()
    
    def get_current_effects(self) -> Dict[str, float]:
        """Get current disease effects on physiology."""
        effects = {}
        
        # Scale effects by current severity
        severity_factor = self.current_severity / self.params.severity.value
        
        for organ, effect in self.params.organ_effects.items():
            effects[organ] = effect * severity_factor
        
        for param, change in self.params.physiological_changes.items():
            effects[param] = change * severity_factor
        
        return effects
    
    def apply_treatment(self, treatment_effect: float) -> float:
        """Apply treatment and return response."""
        response = treatment_effect * self.params.treatment_sensitivity
        
        # Reduce disease severity with treatment
        severity_reduction = response * 0.1  # Treatment effect
        self.current_severity = max(0.0, self.current_severity - severity_reduction)
        
        return response


class CardiovascularDisease(DiseaseModel):
    """Cardiovascular disease model including hypertension, heart failure, etc."""
    
    def __init__(self, disease_type: str, severity: DiseaseSeverity):
        if disease_type == "hypertension":
            params = DiseaseParameters(
                disease_type=DiseaseType.CARDIOVASCULAR,
                name="hypertension",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.01,
                organ_effects={
                    'heart': -0.2,  # Reduced cardiac function
                    'kidney': -0.15,  # Reduced renal function
                    'brain': -0.1,  # Reduced cerebral perfusion
                },
                physiological_changes={
                    'blood_pressure_systolic': 40.0,  # +40 mmHg
                    'blood_pressure_diastolic': 25.0,  # +25 mmHg
                    'heart_rate': 10.0,  # +10 bpm
                    'cardiac_output': -0.1,  # -10%
                },
                biomarker_changes={
                    'ldl': 1.5,  # +50%
                    'crp': 2.0,  # +100%
                    'troponin': 0.5,  # +50%
                }
            )
        elif disease_type == "heart_failure":
            params = DiseaseParameters(
                disease_type=DiseaseType.CARDIOVASCULAR,
                name="heart_failure",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.02,
                organ_effects={
                    'heart': -0.4,  # Severely reduced cardiac function
                    'kidney': -0.3,  # Reduced renal perfusion
                    'liver': -0.2,  # Congestive effects
                    'lung': -0.25,  # Pulmonary congestion
                },
                physiological_changes={
                    'blood_pressure_systolic': -20.0,  # -20 mmHg
                    'blood_pressure_diastolic': -10.0,  # -10 mmHg
                    'heart_rate': 20.0,  # +20 bpm (compensatory)
                    'cardiac_output': -0.3,  # -30%
                },
                biomarker_changes={
                    'bnp': 5.0,  # +400%
                    'crp': 1.5,  # +50%
                    'troponin': 1.0,  # +100%
                }
            )
        else:
            raise ValueError(f"Unknown cardiovascular disease: {disease_type}")
        
        super().__init__(params)


class MetabolicDisease(DiseaseModel):
    """Metabolic disease model including diabetes, obesity, etc."""
    
    def __init__(self, disease_type: str, severity: DiseaseSeverity):
        if disease_type == "diabetes_type2":
            params = DiseaseParameters(
                disease_type=DiseaseType.METABOLIC,
                name="diabetes_type2",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.015,
                organ_effects={
                    'liver': -0.2,  # Reduced hepatic function
                    'kidney': -0.3,  # Diabetic nephropathy
                    'heart': -0.15,  # Cardiovascular complications
                    'brain': -0.1,  # Cognitive effects
                },
                physiological_changes={
                    'blood_glucose': 80.0,  # +80 mg/dL
                    'insulin_resistance': 2.0,  # +200%
                    'blood_pressure_systolic': 15.0,  # +15 mmHg
                    'gfr': -0.2,  # -20%
                },
                biomarker_changes={
                    'hba1c': 3.0,  # +3%
                    'glucose': 2.5,  # +150%
                    'insulin': 1.5,  # +50%
                    'creatinine': 0.3,  # +30%
                }
            )
        elif disease_type == "obesity":
            params = DiseaseParameters(
                disease_type=DiseaseType.METABOLIC,
                name="obesity",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.01,
                organ_effects={
                    'fat': 2.0,  # Increased adipose tissue
                    'liver': -0.15,  # Fatty liver
                    'heart': -0.2,  # Increased cardiac workload
                    'lung': -0.25,  # Reduced respiratory capacity
                },
                physiological_changes={
                    'blood_pressure_systolic': 20.0,  # +20 mmHg
                    'blood_pressure_diastolic': 10.0,  # +10 mmHg
                    'heart_rate': 5.0,  # +5 bpm
                    'insulin_resistance': 1.5,  # +150%
                },
                biomarker_changes={
                    'ldl': 1.3,  # +30%
                    'triglycerides': 2.0,  # +100%
                    'leptin': 3.0,  # +200%
                    'crp': 1.8,  # +80%
                }
            )
        else:
            raise ValueError(f"Unknown metabolic disease: {disease_type}")
        
        super().__init__(params)


class NeurologicalDisease(DiseaseModel):
    """Neurological disease model including Alzheimer's, Parkinson's, etc."""
    
    def __init__(self, disease_type: str, severity: DiseaseSeverity):
        if disease_type == "alzheimer":
            params = DiseaseParameters(
                disease_type=DiseaseType.NEUROLOGICAL,
                name="alzheimer",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.025,
                organ_effects={
                    'brain': -0.4,  # Neurodegeneration
                    'heart': -0.1,  # Autonomic dysfunction
                    'gut': -0.15,  # GI dysfunction
                },
                physiological_changes={
                    'cognitive_function': -0.5,  # -50%
                    'memory': -0.6,  # -60%
                    'motor_control': -0.3,  # -30%
                    'sleep_quality': -0.4,  # -40%
                },
                biomarker_changes={
                    'amyloid_beta': 3.0,  # +200%
                    'tau': 4.0,  # +300%
                    'neurofilament': 2.5,  # +150%
                }
            )
        elif disease_type == "parkinson":
            params = DiseaseParameters(
                disease_type=DiseaseType.NEUROLOGICAL,
                name="parkinson",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.02,
                organ_effects={
                    'brain': -0.35,  # Dopaminergic neuron loss
                    'gut': -0.2,  # GI dysfunction
                    'heart': -0.1,  # Autonomic dysfunction
                },
                physiological_changes={
                    'motor_control': -0.6,  # -60%
                    'tremor': 1.0,  # +100% (new symptom)
                    'rigidity': 0.8,  # +80% (new symptom)
                    'bradykinesia': 0.7,  # +70% (new symptom)
                },
                biomarker_changes={
                    'dopamine': -0.7,  # -70%
                    'alpha_synuclein': 5.0,  # +400%
                    'neurofilament': 2.0,  # +100%
                }
            )
        else:
            raise ValueError(f"Unknown neurological disease: {disease_type}")
        
        super().__init__(params)


class RespiratoryDisease(DiseaseModel):
    """Respiratory disease model including COPD, asthma, etc."""
    
    def __init__(self, disease_type: str, severity: DiseaseSeverity):
        if disease_type == "copd":
            params = DiseaseParameters(
                disease_type=DiseaseType.RESPIRATORY,
                name="copd",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.02,
                organ_effects={
                    'lung': -0.4,  # Reduced lung function
                    'heart': -0.2,  # Cor pulmonale
                    'muscle': -0.15,  # Muscle wasting
                },
                physiological_changes={
                    'fev1': -0.5,  # -50% FEV1
                    'fvc': -0.3,  # -30% FVC
                    'blood_oxygen': -15.0,  # -15%
                    'respiratory_rate': 8.0,  # +8 breaths/min
                },
                biomarker_changes={
                    'crp': 2.5,  # +150%
                    'il6': 3.0,  # +200%
                    'neutrophils': 1.5,  # +50%
                }
            )
        elif disease_type == "asthma":
            params = DiseaseParameters(
                disease_type=DiseaseType.RESPIRATORY,
                name="asthma",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.005,
                organ_effects={
                    'lung': -0.25,  # Airway inflammation
                    'immune': 0.3,  # Immune activation
                },
                physiological_changes={
                    'fev1': -0.3,  # -30% FEV1
                    'airway_resistance': 2.0,  # +100%
                    'blood_oxygen': -5.0,  # -5%
                    'respiratory_rate': 4.0,  # +4 breaths/min
                },
                biomarker_changes={
                    'eosinophils': 2.0,  # +100%
                    'ige': 3.0,  # +200%
                    'il5': 4.0,  # +300%
                }
            )
        else:
            raise ValueError(f"Unknown respiratory disease: {disease_type}")
        
        super().__init__(params)


class RenalDisease(DiseaseModel):
    """Renal disease model including CKD, acute kidney injury, etc."""
    
    def __init__(self, disease_type: str, severity: DiseaseSeverity):
        if disease_type == "chronic_kidney_disease":
            params = DiseaseParameters(
                disease_type=DiseaseType.RENAL,
                name="chronic_kidney_disease",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.03,
                organ_effects={
                    'kidney': -0.5,  # Reduced renal function
                    'heart': -0.3,  # Cardiovascular complications
                    'bone': -0.2,  # Renal osteodystrophy
                },
                physiological_changes={
                    'gfr': -0.6,  # -60%
                    'blood_pressure_systolic': 25.0,  # +25 mmHg
                    'blood_pressure_diastolic': 15.0,  # +15 mmHg
                    'blood_ph': -0.1,  # Acidosis
                },
                biomarker_changes={
                    'creatinine': 3.0,  # +200%
                    'bun': 4.0,  # +300%
                    'potassium': 0.5,  # +50%
                    'phosphorus': 0.8,  # +80%
                }
            )
        elif disease_type == "acute_kidney_injury":
            params = DiseaseParameters(
                disease_type=DiseaseType.RENAL,
                name="acute_kidney_injury",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.1,  # Rapid progression
                organ_effects={
                    'kidney': -0.6,  # Severely reduced function
                    'heart': -0.25,  # Volume overload
                    'lung': -0.2,  # Pulmonary edema
                },
                physiological_changes={
                    'gfr': -0.8,  # -80%
                    'urine_output': -0.9,  # -90%
                    'blood_pressure_systolic': 30.0,  # +30 mmHg
                    'fluid_balance': 2.0,  # +2L positive
                },
                biomarker_changes={
                    'creatinine': 5.0,  # +400%
                    'ngal': 10.0,  # +900%
                    'kim1': 8.0,  # +700%
                }
            )
        else:
            raise ValueError(f"Unknown renal disease: {disease_type}")
        
        super().__init__(params)


class HepaticDisease(DiseaseModel):
    """Hepatic disease model including cirrhosis, hepatitis, etc."""
    
    def __init__(self, disease_type: str, severity: DiseaseSeverity):
        if disease_type == "cirrhosis":
            params = DiseaseParameters(
                disease_type=DiseaseType.HEPATIC,
                name="cirrhosis",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.025,
                organ_effects={
                    'liver': -0.6,  # Severely reduced hepatic function
                    'kidney': -0.3,  # Hepatorenal syndrome
                    'brain': -0.2,  # Hepatic encephalopathy
                    'gut': -0.25,  # Portal hypertension effects
                },
                physiological_changes={
                    'liver_function': -0.7,  # -70%
                    'albumin': -0.5,  # -50%
                    'blood_pressure_systolic': -15.0,  # -15 mmHg
                    'blood_clotting': -0.4,  # -40%
                },
                biomarker_changes={
                    'ast': 4.0,  # +300%
                    'alt': 3.5,  # +250%
                    'bilirubin': 6.0,  # +500%
                    'inr': 2.0,  # +100%
                }
            )
        elif disease_type == "hepatitis":
            params = DiseaseParameters(
                disease_type=DiseaseType.HEPATIC,
                name="hepatitis",
                severity=severity,
                onset_time=0.0,
                progression_rate=0.015,
                organ_effects={
                    'liver': -0.3,  # Reduced hepatic function
                    'immune': 0.5,  # Immune activation
                },
                physiological_changes={
                    'liver_function': -0.4,  # -40%
                    'fever': 2.0,  # +2°C
                    'fatigue': 1.5,  # +150% (new symptom)
                },
                biomarker_changes={
                    'ast': 8.0,  # +700%
                    'alt': 10.0,  # +900%
                    'bilirubin': 2.5,  # +150%
                    'viral_load': 5.0,  # +400%
                }
            )
        else:
            raise ValueError(f"Unknown hepatic disease: {disease_type}")
        
        super().__init__(params)


class DiseaseSimulator:
    """Main disease simulator that manages multiple diseases."""
    
    def __init__(self):
        self.diseases: List[DiseaseModel] = []
        self.disease_history: List[Dict] = []
        
    def add_disease(self, disease: DiseaseModel):
        """Add a disease to the simulator."""
        self.diseases.append(disease)
    
    def remove_disease(self, disease_name: str):
        """Remove a disease from the simulator."""
        self.diseases = [d for d in self.diseases if d.params.name != disease_name]
    
    def update_all_diseases(self, dt: float) -> Dict[str, Dict[str, float]]:
        """Update all diseases and return combined effects."""
        combined_effects = {}
        
        for disease in self.diseases:
            disease_effects = disease.update_disease_state(dt)
            
            # Combine effects (additive for now, could be more complex)
            for param, effect in disease_effects.items():
                if param in combined_effects:
                    combined_effects[param] += effect
                else:
                    combined_effects[param] = effect
        
        # Record disease state
        self.disease_history.append({
            'time': self.get_current_time(),
            'diseases': [
                {
                    'name': d.params.name,
                    'severity': d.current_severity,
                    'time_since_onset': d.time_since_onset
                }
                for d in self.diseases
            ],
            'combined_effects': combined_effects
        })
        
        return combined_effects
    
    def apply_treatment(self, treatment_name: str, treatment_effect: float) -> Dict[str, float]:
        """Apply treatment to all relevant diseases."""
        treatment_responses = {}
        
        for disease in self.diseases:
            response = disease.apply_treatment(treatment_effect)
            treatment_responses[disease.params.name] = response
        
        return treatment_responses
    
    def get_current_time(self) -> float:
        """Get current simulation time."""
        if self.diseases:
            return max(d.time_since_onset for d in self.diseases)
        return 0.0
    
    def get_disease_summary(self) -> Dict:
        """Get summary of current disease states."""
        return {
            'total_diseases': len(self.diseases),
            'diseases': [
                {
                    'name': d.params.name,
                    'type': d.params.disease_type.value,
                    'severity': d.current_severity,
                    'time_since_onset': d.time_since_onset,
                    'effects': d.get_current_effects()
                }
                for d in self.diseases
            ]
        }
    
    def save_disease_state(self, filename: str):
        """Save disease state to file."""
        state = {
            'diseases': [
                {
                    'name': d.params.name,
                    'type': d.params.disease_type.value,
                    'severity': d.params.severity.value,
                    'current_severity': d.current_severity,
                    'time_since_onset': d.time_since_onset,
                    'parameters': {
                        'onset_time': d.params.onset_time,
                        'progression_rate': d.params.progression_rate,
                        'organ_effects': d.params.organ_effects,
                        'physiological_changes': d.params.physiological_changes,
                        'biomarker_changes': d.params.biomarker_changes,
                        'treatment_sensitivity': d.params.treatment_sensitivity,
                        'drug_resistance_factors': d.params.drug_resistance_factors
                    }
                }
                for d in self.diseases
            ],
            'history': self.disease_history
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def create_disease(cls, disease_type: str, disease_name: str, severity: DiseaseSeverity) -> DiseaseModel:
        """Factory method to create disease models."""
        disease_classes = {
            'cardiovascular': CardiovascularDisease,
            'metabolic': MetabolicDisease,
            'neurological': NeurologicalDisease,
            'respiratory': RespiratoryDisease,
            'renal': RenalDisease,
            'hepatic': HepaticDisease,
        }
        
        if disease_type not in disease_classes:
            raise ValueError(f"Unknown disease type: {disease_type}")
        
        return disease_classes[disease_type](disease_name, severity)


class DiseaseDrugInteractions:
    """Model disease-drug interactions."""
    
    def __init__(self):
        self.interaction_matrix = self._initialize_interactions()
    
    def _initialize_interactions(self) -> Dict[str, Dict[str, float]]:
        """Initialize disease-drug interaction matrix."""
        return {
            'hypertension': {
                'ace_inhibitors': 2.0,  # Enhanced effect
                'beta_blockers': 1.5,
                'diuretics': 1.8,
                'nsaids': -0.5,  # Reduced effect, increased side effects
            },
            'diabetes_type2': {
                'metformin': 1.2,
                'insulin': 1.5,
                'sulfonylureas': 1.3,
                'beta_blockers': -0.3,  # Mask hypoglycemia symptoms
            },
            'heart_failure': {
                'ace_inhibitors': 2.5,
                'beta_blockers': 2.0,
                'diuretics': 2.2,
                'digoxin': 1.5,
                'nsaids': -1.0,  # Worsen heart failure
            },
            'chronic_kidney_disease': {
                'ace_inhibitors': 1.3,
                'diuretics': 1.2,
                'statins': 0.8,  # Reduced clearance
                'antibiotics': 0.5,  # Dose adjustment needed
            },
            'cirrhosis': {
                'statins': 0.6,  # Reduced metabolism
                'beta_blockers': 1.8,  # Enhanced effect
                'diuretics': 1.5,
                'sedatives': 2.0,  # Enhanced effect
            },
            'copd': {
                'beta_blockers': -0.8,  # Contraindicated
                'bronchodilators': 2.5,
                'steroids': 1.8,
                'antibiotics': 1.2,
            },
            'alzheimer': {
                'cholinesterase_inhibitors': 2.0,
                'memantine': 1.8,
                'antipsychotics': 1.5,
                'benzodiazepines': -0.5,  # Increased confusion
            },
        }
    
    def get_interaction_factor(self, disease_name: str, drug_class: str) -> float:
        """Get disease-drug interaction factor."""
        if disease_name in self.interaction_matrix:
            return self.interaction_matrix[disease_name].get(drug_class, 1.0)
        return 1.0
    
    def adjust_drug_dose(self, disease_name: str, drug_class: str, base_dose: float) -> float:
        """Adjust drug dose based on disease state."""
        interaction_factor = self.get_interaction_factor(disease_name, drug_class)
        
        if interaction_factor > 1.0:
            # Enhanced effect - reduce dose
            return base_dose / interaction_factor
        elif interaction_factor < 0:
            # Negative interaction - avoid or reduce significantly
            return base_dose * 0.1
        else:
            # Reduced effect - increase dose
            return base_dose / interaction_factor if interaction_factor > 0 else base_dose
