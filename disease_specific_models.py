"""
Disease-Specific PK-PD Modeling System
======================================

This module implements disease-specific modifications to PK-PD parameters,
accounting for pathophysiological changes that affect drug disposition and response.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum

class DiseaseSeverity(Enum):
    """Disease severity levels"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    END_STAGE = "end_stage"

@dataclass
class DiseaseState:
    """Represents a disease state with its characteristics"""
    name: str
    category: str
    severity: DiseaseSeverity
    pk_modifications: Dict[str, float]
    pd_modifications: Dict[str, float]
    comorbidities: List[str]
    biomarkers: Dict[str, float]
    special_considerations: List[str]

class DiseaseSpecificModeler:
    """
    Implements disease-specific PK-PD modeling
    """
    
    def __init__(self):
        """Initialize disease-specific modeler"""
        self.disease_database = self._create_disease_database()
        self.disease_interactions = self._define_disease_interactions()
        self.comorbidity_effects = self._define_comorbidity_effects()
        self.biomarker_mappings = self._define_biomarker_mappings()
    
    def _create_disease_database(self):
        """
        Create comprehensive database of disease states
        
        Returns:
            Dictionary of disease states
        """
        diseases = {
            # RENAL DISEASES
            'chronic_kidney_disease': {
                'mild': DiseaseState(
                    name='Chronic Kidney Disease - Mild',
                    category='renal',
                    severity=DiseaseSeverity.MILD,
                    pk_modifications={
                        'clearance_reduction': 0.75,  # 25% reduction
                        'volume_distribution_increase': 1.1,
                        'protein_binding_increase': 1.05,
                        'absorption_reduction': 0.95
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.1,
                        'emax_increase': 1.05,
                        'ec50_decrease': 0.9
                    },
                    comorbidities=['hypertension', 'anemia', 'bone_disease'],
                    biomarkers={'gfr': 60, 'creatinine': 1.5, 'bun': 25},
                    special_considerations=['dose_adjustment_required', 'monitor_nephrotoxic_drugs']
                ),
                'moderate': DiseaseState(
                    name='Chronic Kidney Disease - Moderate',
                    category='renal',
                    severity=DiseaseSeverity.MODERATE,
                    pk_modifications={
                        'clearance_reduction': 0.5,  # 50% reduction
                        'volume_distribution_increase': 1.2,
                        'protein_binding_increase': 1.1,
                        'absorption_reduction': 0.9
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.2,
                        'emax_increase': 1.1,
                        'ec50_decrease': 0.8
                    },
                    comorbidities=['hypertension', 'anemia', 'bone_disease', 'cardiovascular_disease'],
                    biomarkers={'gfr': 30, 'creatinine': 2.5, 'bun': 40},
                    special_considerations=['significant_dose_adjustment', 'avoid_nephrotoxic_drugs', 'monitor_electrolytes']
                ),
                'severe': DiseaseState(
                    name='Chronic Kidney Disease - Severe',
                    category='renal',
                    severity=DiseaseSeverity.SEVERE,
                    pk_modifications={
                        'clearance_reduction': 0.25,  # 75% reduction
                        'volume_distribution_increase': 1.4,
                        'protein_binding_increase': 1.15,
                        'absorption_reduction': 0.85
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.3,
                        'emax_increase': 1.15,
                        'ec50_decrease': 0.7
                    },
                    comorbidities=['hypertension', 'anemia', 'bone_disease', 'cardiovascular_disease', 'fluid_overload'],
                    biomarkers={'gfr': 15, 'creatinine': 4.0, 'bun': 60},
                    special_considerations=['major_dose_adjustment', 'dialysis_consideration', 'intensive_monitoring']
                )
            },
            
            # HEPATIC DISEASES
            'liver_cirrhosis': {
                'mild': DiseaseState(
                    name='Liver Cirrhosis - Mild (Child-Pugh A)',
                    category='hepatic',
                    severity=DiseaseSeverity.MILD,
                    pk_modifications={
                        'clearance_reduction': 0.8,
                        'volume_distribution_increase': 1.15,
                        'protein_binding_decrease': 0.9,
                        'first_pass_reduction': 0.85,
                        'metabolism_reduction': 0.8
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.1,
                        'response_delay': 1.2,
                        'emax_decrease': 0.95
                    },
                    comorbidities=['ascites', 'hepatorenal_syndrome', 'coagulopathy'],
                    biomarkers={'albumin': 3.5, 'bilirubin': 2.0, 'inr': 1.3, 'platelets': 120},
                    special_considerations=['avoid_hepatotoxic_drugs', 'monitor_liver_function', 'dose_adjust_high_extraction_drugs']
                ),
                'moderate': DiseaseState(
                    name='Liver Cirrhosis - Moderate (Child-Pugh B)',
                    category='hepatic',
                    severity=DiseaseSeverity.MODERATE,
                    pk_modifications={
                        'clearance_reduction': 0.5,
                        'volume_distribution_increase': 1.3,
                        'protein_binding_decrease': 0.75,
                        'first_pass_reduction': 0.6,
                        'metabolism_reduction': 0.5
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.2,
                        'response_delay': 1.5,
                        'emax_decrease': 0.85
                    },
                    comorbidities=['ascites', 'hepatorenal_syndrome', 'coagulopathy', 'encephalopathy'],
                    biomarkers={'albumin': 2.8, 'bilirubin': 3.5, 'inr': 1.7, 'platelets': 80},
                    special_considerations=['significant_dose_adjustment', 'avoid_high_extraction_drugs', 'monitor_for_toxicity']
                ),
                'severe': DiseaseState(
                    name='Liver Cirrhosis - Severe (Child-Pugh C)',
                    category='hepatic',
                    severity=DiseaseSeverity.SEVERE,
                    pk_modifications={
                        'clearance_reduction': 0.3,
                        'volume_distribution_increase': 1.5,
                        'protein_binding_decrease': 0.6,
                        'first_pass_reduction': 0.4,
                        'metabolism_reduction': 0.3
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.3,
                        'response_delay': 2.0,
                        'emax_decrease': 0.7
                    },
                    comorbidities=['ascites', 'hepatorenal_syndrome', 'coagulopathy', 'encephalopathy', 'variceal_bleeding'],
                    biomarkers={'albumin': 2.0, 'bilirubin': 5.0, 'inr': 2.2, 'platelets': 50},
                    special_considerations=['major_dose_adjustment', 'avoid_oral_drugs', 'intensive_care_monitoring']
                )
            },
            
            # CARDIOVASCULAR DISEASES
            'heart_failure': {
                'mild': DiseaseState(
                    name='Heart Failure - Mild (NYHA Class II)',
                    category='cardiovascular',
                    severity=DiseaseSeverity.MILD,
                    pk_modifications={
                        'clearance_reduction': 0.85,
                        'volume_distribution_increase': 1.1,
                        'absorption_reduction': 0.9,
                        'hepatic_blood_flow_reduction': 0.9
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.15,
                        'response_delay': 1.1,
                        'emax_increase': 1.05
                    },
                    comorbidities=['renal_dysfunction', 'arrhythmias', 'hypertension'],
                    biomarkers={'ef': 50, 'bnp': 200, 'creatinine': 1.3},
                    special_considerations=['monitor_cardiac_output', 'adjust_renal_eliminated_drugs']
                ),
                'moderate': DiseaseState(
                    name='Heart Failure - Moderate (NYHA Class III)',
                    category='cardiovascular',
                    severity=DiseaseSeverity.MODERATE,
                    pk_modifications={
                        'clearance_reduction': 0.7,
                        'volume_distribution_increase': 1.2,
                        'absorption_reduction': 0.85,
                        'hepatic_blood_flow_reduction': 0.8
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.25,
                        'response_delay': 1.2,
                        'emax_increase': 1.1
                    },
                    comorbidities=['renal_dysfunction', 'arrhythmias', 'hypertension', 'edema'],
                    biomarkers={'ef': 35, 'bnp': 500, 'creatinine': 1.8},
                    special_considerations=['significant_dose_adjustment', 'monitor_fluid_status', 'avoid_negative_inotropes']
                ),
                'severe': DiseaseState(
                    name='Heart Failure - Severe (NYHA Class IV)',
                    category='cardiovascular',
                    severity=DiseaseSeverity.SEVERE,
                    pk_modifications={
                        'clearance_reduction': 0.5,
                        'volume_distribution_increase': 1.4,
                        'absorption_reduction': 0.8,
                        'hepatic_blood_flow_reduction': 0.6
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.3,
                        'response_delay': 1.4,
                        'emax_increase': 1.15
                    },
                    comorbidities=['renal_dysfunction', 'arrhythmias', 'hypotension', 'edema', 'cardiogenic_shock'],
                    biomarkers={'ef': 20, 'bnp': 1000, 'creatinine': 2.5},
                    special_considerations=['major_dose_adjustment', 'icu_monitoring', 'avoid_cardiotoxic_drugs']
                )
            },
            
            # ENDOCRINE DISEASES
            'diabetes_mellitus': {
                'mild': DiseaseState(
                    name='Diabetes Mellitus - Mild',
                    category='endocrine',
                    severity=DiseaseSeverity.MILD,
                    pk_modifications={
                        'clearance_increase': 1.1,  # Increased renal clearance
                        'volume_distribution_increase': 1.05,
                        'protein_binding_decrease': 0.95
                    },
                    pd_modifications={
                        'sensitivity_decrease': 0.9,
                        'response_delay': 1.1,
                        'emax_decrease': 0.95
                    },
                    comorbidities=['hypertension', 'nephropathy', 'neuropathy'],
                    biomarkers={'hba1c': 7.0, 'fasting_glucose': 140, 'creatinine': 1.2},
                    special_considerations=['monitor_glucose', 'adjust_insulin_dosing', 'watch_for_nephrotoxicity']
                ),
                'moderate': DiseaseState(
                    name='Diabetes Mellitus - Moderate',
                    category='endocrine',
                    severity=DiseaseSeverity.MODERATE,
                    pk_modifications={
                        'clearance_increase': 1.2,
                        'volume_distribution_increase': 1.1,
                        'protein_binding_decrease': 0.9
                    },
                    pd_modifications={
                        'sensitivity_decrease': 0.85,
                        'response_delay': 1.2,
                        'emax_decrease': 0.9
                    },
                    comorbidities=['hypertension', 'nephropathy', 'neuropathy', 'retinopathy'],
                    biomarkers={'hba1c': 8.5, 'fasting_glucose': 180, 'creatinine': 1.5},
                    special_considerations=['intensive_glucose_monitoring', 'dose_adjust_for_nephropathy', 'monitor_for_complications']
                ),
                'severe': DiseaseState(
                    name='Diabetes Mellitus - Severe',
                    category='endocrine',
                    severity=DiseaseSeverity.SEVERE,
                    pk_modifications={
                        'clearance_increase': 1.3,
                        'volume_distribution_increase': 1.15,
                        'protein_binding_decrease': 0.85
                    },
                    pd_modifications={
                        'sensitivity_decrease': 0.8,
                        'response_delay': 1.3,
                        'emax_decrease': 0.85
                    },
                    comorbidities=['hypertension', 'nephropathy', 'neuropathy', 'retinopathy', 'cardiovascular_disease'],
                    biomarkers={'hba1c': 10.0, 'fasting_glucose': 250, 'creatinine': 2.0},
                    special_considerations=['continuous_glucose_monitoring', 'major_dose_adjustment', 'multidisciplinary_care']
                )
            },
            
            # RESPIRATORY DISEASES
            'copd': {
                'moderate': DiseaseState(
                    name='COPD - Moderate',
                    category='respiratory',
                    severity=DiseaseSeverity.MODERATE,
                    pk_modifications={
                        'clearance_reduction': 0.9,
                        'volume_distribution_increase': 1.05,
                        'absorption_reduction': 0.95
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.1,
                        'response_delay': 1.15,
                        'emax_increase': 1.05
                    },
                    comorbidities=['cardiovascular_disease', 'osteoporosis', 'depression'],
                    biomarkers={'fev1': 60, 'fvc': 75, 'o2_saturation': 92},
                    special_considerations=['monitor_oxygenation', 'adjust_inhaled_doses', 'avoid_respiratory_depressants']
                ),
                'severe': DiseaseState(
                    name='COPD - Severe',
                    category='respiratory',
                    severity=DiseaseSeverity.SEVERE,
                    pk_modifications={
                        'clearance_reduction': 0.8,
                        'volume_distribution_increase': 1.1,
                        'absorption_reduction': 0.9
                    },
                    pd_modifications={
                        'sensitivity_increase': 1.15,
                        'response_delay': 1.25,
                        'emax_increase': 1.1
                    },
                    comorbidities=['cardiovascular_disease', 'osteoporosis', 'depression', 'respiratory_failure'],
                    biomarkers={'fev1': 35, 'fvc': 50, 'o2_saturation': 88},
                    special_considerations=['supplemental_oxygen', 'major_dose_adjustment', 'monitor_for_cor_pulmonale']
                )
            },
            
            # NEUROLOGICAL DISEASES
            'epilepsy': {
                'mild': DiseaseState(
                    name='Epilepsy - Well-controlled',
                    category='neurological',
                    severity=DiseaseSeverity.MILD,
                    pk_modifications={
                        'enzyme_induction': 1.2,  # Increased metabolism
                        'clearance_increase': 1.15,
                        'protein_binding_decrease': 0.95
                    },
                    pd_modifications={
                        'sensitivity_decrease': 0.85,
                        'tolerance_development': 1.1,
                        'emax_decrease': 0.9
                    },
                    comorbidities=['depression', 'cognitive_impairment'],
                    biomarkers={'seizure_frequency': 1, 'drug_levels': 'therapeutic'},
                    special_considerations=['therapeutic_drug_monitoring', 'watch_for_drug_interactions', 'avoid_seizure_triggers']
                ),
                'severe': DiseaseState(
                    name='Epilepsy - Poorly controlled',
                    category='neurological',
                    severity=DiseaseSeverity.SEVERE,
                    pk_modifications={
                        'enzyme_induction': 1.4,
                        'clearance_increase': 1.3,
                        'protein_binding_decrease': 0.9
                    },
                    pd_modifications={
                        'sensitivity_decrease': 0.7,
                        'tolerance_development': 1.3,
                        'emax_decrease': 0.8
                    },
                    comorbidities=['depression', 'cognitive_impairment', 'injury_risk'],
                    biomarkers={'seizure_frequency': 10, 'drug_levels': 'variable'},
                    special_considerations=['intensive_monitoring', 'polytherapy_considerations', 'quality_of_life_focus']
                )
            }
        }
        
        return diseases
    
    def _define_disease_interactions(self):
        """
        Define interactions between different diseases
        
        Returns:
            Dictionary of disease interaction effects
        """
        interactions = {
            ('renal', 'hepatic'): {
                'clearance_multiplier': 0.4,  # Synergistic reduction
                'toxicity_risk': 2.5,
                'monitoring_intensity': 'high'
            },
            ('cardiovascular', 'renal'): {
                'clearance_multiplier': 0.6,
                'volume_multiplier': 1.3,
                'monitoring_intensity': 'high'
            },
            ('respiratory', 'cardiovascular'): {
                'absorption_multiplier': 0.85,
                'distribution_multiplier': 1.1,
                'monitoring_intensity': 'moderate'
            },
            ('endocrine', 'cardiovascular'): {
                'clearance_multiplier': 0.8,
                'sensitivity_multiplier': 1.2,
                'monitoring_intensity': 'moderate'
            }
        }
        
        return interactions
    
    def _define_comorbidity_effects(self):
        """
        Define effects of specific comorbidities
        
        Returns:
            Dictionary of comorbidity effects
        """
        comorbidities = {
            'hypertension': {
                'pk_effects': {'clearance_reduction': 0.9, 'volume_increase': 1.05},
                'pd_effects': {'sensitivity_increase': 1.1, 'response_delay': 1.05}
            },
            'anemia': {
                'pk_effects': {'volume_distribution_increase': 1.1, 'protein_binding_decrease': 0.95},
                'pd_effects': {'oxygen_carrying_decrease': 0.9, 'fatigue_increase': 1.2}
            },
            'arrhythmias': {
                'pk_effects': {'clearance_reduction': 0.85},
                'pd_effects': {'cardiac_sensitivity_increase': 1.3, 'proarrhythmic_risk': 2.0}
            },
            'edema': {
                'pk_effects': {'volume_distribution_increase': 1.2, 'clearance_reduction': 0.9},
                'pd_effects': {'response_delay': 1.15, 'peak_effect_decrease': 0.9}
            },
            'neuropathy': {
                'pk_effects': {'absorption_reduction': 0.9},
                'pd_effects': {'pain_sensitivity_increase': 1.4, 'response_delay': 1.2}
            }
        }
        
        return comorbidities
    
    def _define_biomarker_mappings(self):
        """
        Define mappings from biomarkers to disease severity
        
        Returns:
            Dictionary of biomarker mappings
        """
        mappings = {
            'renal': {
                'gfr': {
                    'severe': (0, 15),
                    'moderate': (15, 30),
                    'mild': (30, 60),
                    'normal': (60, 120)
                },
                'creatinine': {
                    'severe': (3.0, 10.0),
                    'moderate': (2.0, 3.0),
                    'mild': (1.3, 2.0),
                    'normal': (0.6, 1.3)
                }
            },
            'hepatic': {
                'albumin': {
                    'severe': (0, 2.5),
                    'moderate': (2.5, 3.5),
                    'mild': (3.5, 4.0),
                    'normal': (4.0, 6.0)
                },
                'bilirubin': {
                    'severe': (3.0, 20.0),
                    'moderate': (2.0, 3.0),
                    'mild': (1.2, 2.0),
                    'normal': (0.3, 1.2)
                }
            },
            'cardiovascular': {
                'ef': {
                    'severe': (0, 30),
                    'moderate': (30, 40),
                    'mild': (40, 55),
                    'normal': (55, 70)
                },
                'bnp': {
                    'severe': (800, 5000),
                    'moderate': (400, 800),
                    'mild': (100, 400),
                    'normal': (0, 100)
                }
            }
        }
        
        return mappings
    
    def get_disease_state(self, disease_name: str, severity: str) -> Optional[DiseaseState]:
        """
        Get disease state object
        
        Args:
            disease_name: Name of the disease
            severity: Severity level
            
        Returns:
            DiseaseState object or None if not found
        """
        if disease_name in self.disease_database:
            if severity in self.disease_database[disease_name]:
                return self.disease_database[disease_name][severity]
        
        return None
    
    def assess_disease_severity(self, disease_name: str, biomarkers: Dict[str, float]) -> str:
        """
        Assess disease severity based on biomarkers
        
        Args:
            disease_name: Name of the disease
            biomarkers: Dictionary of biomarker values
            
        Returns:
            Severity level string
        """
        if disease_name not in self.biomarker_mappings:
            return 'moderate'  # Default
        
        severity_scores = {'severe': 0, 'moderate': 0, 'mild': 0}
        
        for biomarker, value in biomarkers.items():
            if biomarker in self.biomarker_mappings[disease_name]:
                ranges = self.biomarker_mappings[disease_name][biomarker]
                for severity, (min_val, max_val) in ranges.items():
                    if min_val <= value <= max_val:
                        severity_scores[severity] += 1
        
        # Return severity with highest score
        return max(severity_scores, key=severity_scores.get)
    
    def apply_disease_modifications(self, base_pk_params: Dict, base_pd_params: Dict, 
                                  disease_states: List[DiseaseState]) -> Tuple[Dict, Dict]:
        """
        Apply disease-specific modifications to PK/PD parameters
        
        Args:
            base_pk_params: Base PK parameters
            base_pd_params: Base PD parameters
            disease_states: List of disease states
            
        Returns:
            Modified PK and PD parameters
        """
        modified_pk = base_pk_params.copy()
        modified_pd = base_pd_params.copy()
        
        # Apply individual disease modifications
        for disease_state in disease_states:
            # PK modifications
            for param, multiplier in disease_state.pk_modifications.items():
                if param in modified_pk:
                    modified_pk[param] *= multiplier
                else:
                    modified_pk[param] = multiplier
            
            # PD modifications
            for param, multiplier in disease_state.pd_modifications.items():
                if param in modified_pd:
                    modified_pd[param] *= multiplier
                else:
                    modified_pd[param] = multiplier
        
        # Apply disease interaction effects
        if len(disease_states) > 1:
            for i, disease1 in enumerate(disease_states):
                for disease2 in disease_states[i+1:]:
                    interaction_key = (disease1.category, disease2.category)
                    if interaction_key in self.disease_interactions:
                        interaction = self.disease_interactions[interaction_key]
                        
                        if 'clearance_multiplier' in interaction:
                            if 'clearance' in modified_pk:
                                modified_pk['clearance'] *= interaction['clearance_multiplier']
                        
                        if 'volume_multiplier' in interaction:
                            if 'volume_distribution' in modified_pk:
                                modified_pk['volume_distribution'] *= interaction['volume_multiplier']
        
        # Apply comorbidity effects
        all_comorbidities = []
        for disease_state in disease_states:
            all_comorbidities.extend(disease_state.comorbidities)
        
        for comorbidity in set(all_comorbidities):  # Remove duplicates
            if comorbidity in self.comorbidity_effects:
                effects = self.comorbidity_effects[comorbidity]
                
                # PK effects
                for param, multiplier in effects['pk_effects'].items():
                    if param in modified_pk:
                        modified_pk[param] *= multiplier
                
                # PD effects
                for param, multiplier in effects['pd_effects'].items():
                    if param in modified_pd:
                        modified_pd[param] *= multiplier
        
        return modified_pk, modified_pd
    
    def generate_disease_specific_recommendations(self, disease_states: List[DiseaseState], 
                                               drug_properties: Dict) -> Dict:
        """
        Generate disease-specific dosing recommendations
        
        Args:
            disease_states: List of disease states
            drug_properties: Properties of the drug
            
        Returns:
            Dictionary of recommendations
        """
        recommendations = {
            'dose_adjustment': None,
            'monitoring_parameters': [],
            'contraindications': [],
            'precautions': [],
            'drug_interactions': [],
            'special_considerations': []
        }
        
        # Collect all special considerations
        all_considerations = []
        for disease_state in disease_states:
            all_considerations.extend(disease_state.special_considerations)
        
        recommendations['special_considerations'] = list(set(all_considerations))
        
        # Determine dose adjustment based on clearance changes
        total_clearance_factor = 1.0
        for disease_state in disease_states:
            if 'clearance_reduction' in disease_state.pk_modifications:
                total_clearance_factor *= disease_state.pk_modifications['clearance_reduction']
        
        if total_clearance_factor < 0.5:
            recommendations['dose_adjustment'] = 'major_reduction_50_75_percent'
        elif total_clearance_factor < 0.7:
            recommendations['dose_adjustment'] = 'moderate_reduction_30_50_percent'
        elif total_clearance_factor < 0.9:
            recommendations['dose_adjustment'] = 'mild_reduction_10_30_percent'
        elif total_clearance_factor > 1.2:
            recommendations['dose_adjustment'] = 'increase_dose_10_30_percent'
        else:
            recommendations['dose_adjustment'] = 'no_adjustment_needed'
        
        # Add monitoring parameters based on diseases
        for disease_state in disease_states:
            if disease_state.category == 'renal':
                recommendations['monitoring_parameters'].extend(['creatinine', 'gfr', 'bun', 'electrolytes'])
            elif disease_state.category == 'hepatic':
                recommendations['monitoring_parameters'].extend(['lft', 'albumin', 'inr', 'bilirubin'])
            elif disease_state.category == 'cardiovascular':
                recommendations['monitoring_parameters'].extend(['blood_pressure', 'heart_rate', 'ef', 'bnp'])
            elif disease_state.category == 'endocrine':
                recommendations['monitoring_parameters'].extend(['glucose', 'hba1c', 'renal_function'])
        
        # Remove duplicates
        recommendations['monitoring_parameters'] = list(set(recommendations['monitoring_parameters']))
        
        # Add precautions based on drug properties and diseases
        for disease_state in disease_states:
            if (drug_properties.get('nephrotoxic', False) and 
                disease_state.category == 'renal'):
                recommendations['precautions'].append('high_nephrotoxicity_risk')
            
            if (drug_properties.get('hepatotoxic', False) and 
                disease_state.category == 'hepatic'):
                recommendations['precautions'].append('high_hepatotoxicity_risk')
            
            if (drug_properties.get('cardiotoxic', False) and 
                disease_state.category == 'cardiovascular'):
                recommendations['precautions'].append('high_cardiotoxicity_risk')
        
        return recommendations
    
    def simulate_disease_specific_response(self, drug_name: str, dose: float, 
                                        patient_factors: Dict, disease_states: List[DiseaseState],
                                        time_points: np.ndarray) -> Dict:
        """
        Simulate drug response with disease-specific modifications
        
        Args:
            drug_name: Name of the drug
            dose: Administered dose
            patient_factors: Patient characteristics
            disease_states: List of disease states
            time_points: Time points for simulation
            
        Returns:
            Dictionary with simulation results
        """
        # Import expanded drug database
        from expanded_drug_database import ExpandedDrugDatabase
        drug_db = ExpandedDrugDatabase()
        
        # Get base drug properties
        drug_props = drug_db.get_drug_properties(drug_name)
        if not drug_props:
            return None
        
        # Calculate base PK/PD parameters
        base_pk = drug_db.calculate_pk_parameters(drug_name, dose, patient_factors)
        base_pd = drug_db.calculate_pd_parameters(drug_name, patient_factors)
        
        # Apply disease modifications
        modified_pk, modified_pd = self.apply_disease_modifications(
            base_pk, base_pd, disease_states
        )
        
        # Simulate with modified parameters
        concentrations = []
        responses = []
        
        # Get modified PK parameters
        absorption = modified_pk.get('absorption', base_pk['absorption'])
        vd = modified_pk.get('distribution', base_pk['distribution'])
        cl = modified_pk.get('clearance', base_pk['clearance'])
        half_life = modified_pk.get('half_life', base_pk['half_life'])
        
        # Initial concentration
        c0 = (dose * absorption) / vd
        
        # Get modified PD parameters
        emax = modified_pd.get('emax', base_pd['emax'])
        ec50 = modified_pd.get('ec50', base_pd['ec50'])
        hill = modified_pd.get('hill_coefficient', base_pd['hill_coefficient'])
        onset = modified_pd.get('onset_time', base_pd['onset_time'])
        response_delay = modified_pd.get('response_delay', 1.0)
        
        # Calculate concentration and response at each time point
        for t in time_points:
            # PK calculation
            if t < half_life * 0.1:
                c = c0 * (1 - np.exp(-t / (half_life * 0.1)))
            else:
                c = c0 * np.exp(-t / half_life)
            concentrations.append(max(0, c))
            
            # PD calculation with disease modifications
            effective_time = t * response_delay
            if effective_time < onset:
                effect = 0
            else:
                effect = (emax * (c ** hill)) / ((ec50 ** hill) + (c ** hill))
            responses.append(effect)
        
        # Generate recommendations
        recommendations = self.generate_disease_specific_recommendations(
            disease_states, drug_props
        )
        
        return {
            'time_points': time_points,
            'concentrations': np.array(concentrations),
            'responses': np.array(responses),
            'base_pk_parameters': base_pk,
            'modified_pk_parameters': modified_pk,
            'base_pd_parameters': base_pd,
            'modified_pd_parameters': modified_pd,
            'disease_states': disease_states,
            'recommendations': recommendations,
            'drug_properties': drug_props
        }
    
    def compare_healthy_vs_disease(self, drug_name: str, dose: float, 
                                  patient_factors: Dict, disease_states: List[DiseaseState]) -> Dict:
        """
        Compare drug response between healthy and disease states
        
        Args:
            drug_name: Name of the drug
            dose: Administered dose
            patient_factors: Patient characteristics
            disease_states: List of disease states
            
        Returns:
            Dictionary with comparison results
        """
        time_points = np.linspace(0, 24, 100)
        
        # Simulate healthy response
        from expanded_drug_database import ExpandedDrugDatabase
        drug_db = ExpandedDrugDatabase()
        
        healthy_simulation = drug_db.simulate_drug_response(
            drug_name, dose, patient_factors, time_points
        )
        
        # Simulate disease response
        disease_simulation = self.simulate_disease_specific_response(
            drug_name, dose, patient_factors, disease_states, time_points
        )
        
        # Calculate comparison metrics
        comparison = {
            'healthy': {
                'peak_concentration': np.max(healthy_simulation['concentrations']),
                'peak_response': np.max(healthy_simulation['responses']),
                'auc_pk': np.trapz(healthy_simulation['concentrations'], time_points),
                'auc_pd': np.trapz(healthy_simulation['responses'], time_points),
                'time_to_peak': time_points[np.argmax(healthy_simulation['concentrations'])]
            },
            'disease': {
                'peak_concentration': np.max(disease_simulation['concentrations']),
                'peak_response': np.max(disease_simulation['responses']),
                'auc_pk': np.trapz(disease_simulation['concentrations'], time_points),
                'auc_pd': np.trapz(disease_simulation['responses'], time_points),
                'time_to_peak': time_points[np.argmax(disease_simulation['concentrations'])]
            },
            'ratios': {
                'peak_concentration_ratio': np.max(disease_simulation['concentrations']) / np.max(healthy_simulation['concentrations']),
                'peak_response_ratio': np.max(disease_simulation['responses']) / np.max(healthy_simulation['responses']),
                'auc_pk_ratio': np.trapz(disease_simulation['concentrations'], time_points) / np.trapz(healthy_simulation['concentrations'], time_points),
                'auc_pd_ratio': np.trapz(disease_simulation['responses'], time_points) / np.trapz(healthy_simulation['responses'], time_points)
            },
            'recommendations': disease_simulation['recommendations']
        }
        
        return comparison
    
    def get_disease_database_summary(self):
        """
        Get summary of disease database
        
        Returns:
            Dictionary with database summary
        """
        summary = {
            'total_diseases': len(self.disease_database),
            'disease_categories': {},
            'severity_levels': list(DiseaseSeverity),
            'total_comorbidities': len(set([
                comorbidity for disease_data in self.disease_database.values()
                for severity_data in disease_data.values()
                for comorbidity in severity_data.comorbidities
            ]))
        }
        
        # Count diseases by category
        for disease_name, severity_data in self.disease_database.items():
            category = severity_data['mild'].category  # All severities have same category
            if category not in summary['disease_categories']:
                summary['disease_categories'][category] = 0
            summary['disease_categories'][category] += 1
        
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize disease-specific modeler
    disease_modeler = DiseaseSpecificModeler()
    
    # Get database summary
    summary = disease_modeler.get_disease_database_summary()
    print("🏥 Disease-Specific Modeler Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Example: Assess CKD severity
    biomarkers = {'gfr': 25, 'creatinine': 2.8}
    severity = disease_modeler.assess_disease_severity('chronic_kidney_disease', biomarkers)
    print(f"\n🔬 CKD Severity Assessment: {severity}")
    
    # Example: Get disease state
    ckd_state = disease_modeler.get_disease_state('chronic_kidney_disease', 'moderate')
    print(f"📋 Disease State: {ckd_state.name}")
    print(f"   PK Modifications: {ckd_state.pk_modifications}")
    print(f"   PD Modifications: {ckd_state.pd_modifications}")
