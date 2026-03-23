"""
Expanded Drug Classes Database for PK-PD Modeling
==================================================

This module provides comprehensive drug class information including pharmacokinetic
and pharmacodynamic properties for enhanced PK-PD modeling capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json

class ExpandedDrugDatabase:
    """
    Comprehensive database of drug classes with PK/PD properties
    """
    
    def __init__(self):
        """Initialize expanded drug database"""
        self.drug_database = self._create_comprehensive_database()
        self.drug_classes = self._categorize_drugs()
        self.pk_parameters = self._define_pk_parameters()
        self.pd_parameters = self._define_pd_parameters()
    
    def _create_comprehensive_database(self):
        """
        Create comprehensive database with multiple drug classes
        
        Returns:
            Dictionary of drug properties
        """
        drugs = {
            # ANALGESICS & ANTI-INFLAMMATORIES
            'Acetaminophen': {
                'class': 'Analgesic',
                'subclass': 'Non-opioid analgesic',
                'mechanism': 'COX inhibition, central analgesia',
                'molecular_weight': 151.16,
                'logp': 0.5,
                'bioavailability': 0.85,
                'protein_binding': 0.25,
                'volume_distribution': 0.9,  # L/kg
                'clearance': 0.2,  # L/h/kg
                'half_life': 2.5,  # hours
                'therapeutic_range': (10, 30),  # mcg/mL
                'toxicity_threshold': 150,  # mcg/mL
                'routes': ['oral', 'iv', 'rectal'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.8,  # reduces absorption
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': True
            },
            'Ibuprofen': {
                'class': 'NSAID',
                'subclass': 'Propionic acid derivative',
                'mechanism': 'COX-1 and COX-2 inhibition',
                'molecular_weight': 206.29,
                'logp': 3.5,
                'bioavailability': 0.8,
                'protein_binding': 0.99,
                'volume_distribution': 0.15,
                'clearance': 0.05,
                'half_life': 2.0,
                'therapeutic_range': (10, 50),
                'toxicity_threshold': 400,
                'routes': ['oral', 'topical', 'iv'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.6,
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': False
            },
            'Naproxen': {
                'class': 'NSAID',
                'subclass': 'Propionic acid derivative',
                'mechanism': 'COX-1 and COX-2 inhibition',
                'molecular_weight': 230.26,
                'logp': 3.3,
                'bioavailability': 0.95,
                'protein_binding': 0.99,
                'volume_distribution': 0.16,
                'clearance': 0.04,
                'half_life': 17.0,
                'therapeutic_range': (25, 75),
                'toxicity_threshold': 500,
                'routes': ['oral', 'rectal'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.7,
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': False
            },
            
            # CARDIOVASCULAR DRUGS
            'Lisinopril': {
                'class': 'ACE Inhibitor',
                'subclass': 'Antihypertensive',
                'mechanism': 'ACE inhibition, reduced angiotensin II',
                'molecular_weight': 405.49,
                'logp': -0.7,
                'bioavailability': 0.25,
                'protein_binding': 0.1,
                'volume_distribution': 1.2,
                'clearance': 0.08,
                'half_life': 12.0,
                'therapeutic_range': (5, 40),
                'toxicity_threshold': 200,
                'routes': ['oral'],
                'metabolism': 'minimal',
                'excretion': 'renal',
                'food_effect': 0.9,
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': False
            },
            'Metoprolol': {
                'class': 'Beta Blocker',
                'subclass': 'Cardioselective beta-1 antagonist',
                'mechanism': 'Beta-1 adrenergic receptor blockade',
                'molecular_weight': 267.36,
                'logp': 1.8,
                'bioavailability': 0.5,
                'protein_binding': 0.12,
                'volume_distribution': 3.2,
                'clearance': 0.15,
                'half_life': 3.5,
                'therapeutic_range': (20, 200),
                'toxicity_threshold': 1000,
                'routes': ['oral', 'iv'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.8,
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': True
            },
            'Atorvastatin': {
                'class': 'Statin',
                'subclass': 'HMG-CoA reductase inhibitor',
                'mechanism': 'HMG-CoA reductase inhibition',
                'molecular_weight': 558.64,
                'logp': 4.0,
                'bioavailability': 0.14,
                'protein_binding': 0.98,
                'volume_distribution': 5.4,
                'clearance': 0.12,
                'half_life': 14.0,
                'therapeutic_range': (10, 80),
                'toxicity_threshold': 200,
                'routes': ['oral'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.7,
                'age_adjustment': True,
                'renal_adjustment': False,
                'hepatic_adjustment': True
            },
            'Amlodipine': {
                'class': 'Calcium Channel Blocker',
                'subclass': 'Dihydropyridine',
                'mechanism': 'L-type calcium channel blockade',
                'molecular_weight': 408.88,
                'logp': 3.0,
                'bioavailability': 0.64,
                'protein_binding': 0.97,
                'volume_distribution': 21.0,
                'clearance': 0.03,
                'half_life': 35.0,
                'therapeutic_range': (2, 10),
                'toxicity_threshold': 50,
                'routes': ['oral'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.9,
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': True
            },
            
            # ANTIDIABETIC DRUGS
            'Metformin': {
                'class': 'Biguanide',
                'subclass': 'Antidiabetic',
                'mechanism': 'AMPK activation, reduced gluconeogenesis',
                'molecular_weight': 129.16,
                'logp': -0.5,
                'bioavailability': 0.6,
                'protein_binding': 0.0,
                'volume_distribution': 3.0,
                'clearance': 0.7,
                'half_life': 6.0,
                'therapeutic_range': (0.5, 2.0),
                'toxicity_threshold': 10,
                'routes': ['oral'],
                'metabolism': 'minimal',
                'excretion': 'renal',
                'food_effect': 0.5,
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': False
            },
            'Glipizide': {
                'class': 'Sulfonylurea',
                'subclass': 'Antidiabetic',
                'mechanism': 'K_ATP channel blockade, insulin secretion',
                'molecular_weight': 445.54,
                'logp': 2.8,
                'bioavailability': 0.9,
                'protein_binding': 0.98,
                'volume_distribution': 0.2,
                'clearance': 0.04,
                'half_life': 7.0,
                'therapeutic_range': (0.05, 0.4),
                'toxicity_threshold': 2,
                'routes': ['oral'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.8,
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': True
            },
            
            # ANTIBIOTICS
            'Amoxicillin': {
                'class': 'Penicillin',
                'subclass': 'Beta-lactam antibiotic',
                'mechanism': 'Cell wall synthesis inhibition',
                'molecular_weight': 365.41,
                'logp': 0.9,
                'bioavailability': 0.75,
                'protein_binding': 0.2,
                'volume_distribution': 0.3,
                'clearance': 0.3,
                'half_life': 1.5,
                'therapeutic_range': (2, 8),
                'toxicity_threshold': 100,
                'routes': ['oral', 'iv', 'im'],
                'metabolism': 'minimal',
                'excretion': 'renal',
                'food_effect': 0.7,
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': False
            },
            'Ciprofloxacin': {
                'class': 'Fluoroquinolone',
                'subclass': 'Antibiotic',
                'mechanism': 'DNA gyrase inhibition',
                'molecular_weight': 331.34,
                'logp': 0.3,
                'bioavailability': 0.7,
                'protein_binding': 0.3,
                'volume_distribution': 2.5,
                'clearance': 0.2,
                'half_life': 4.0,
                'therapeutic_range': (1, 4),
                'toxicity_threshold': 20,
                'routes': ['oral', 'iv'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.6,
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': True
            },
            
            # PSYCHIATRIC DRUGS
            'Sertraline': {
                'class': 'SSRI',
                'subclass': 'Antidepressant',
                'mechanism': 'Serotonin reuptake inhibition',
                'molecular_weight': 306.23,
                'logp': 2.8,
                'bioavailability': 0.5,
                'protein_binding': 0.98,
                'volume_distribution': 20.0,
                'clearance': 0.05,
                'half_life': 26.0,
                'therapeutic_range': (10, 150),
                'toxicity_threshold': 800,
                'routes': ['oral'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.9,
                'age_adjustment': True,
                'renal_adjustment': False,
                'hepatic_adjustment': True
            },
            'Lorazepam': {
                'class': 'Benzodiazepine',
                'subclass': 'Anxiolytic',
                'mechanism': 'GABA_A receptor potentiation',
                'molecular_weight': 321.2,
                'logp': 2.9,
                'bioavailability': 0.85,
                'protein_binding': 0.91,
                'volume_distribution': 1.1,
                'clearance': 0.08,
                'half_life': 14.0,
                'therapeutic_range': (10, 300),
                'toxicity_threshold': 1000,
                'routes': ['oral', 'iv', 'im'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.8,
                'age_adjustment': True,
                'renal_adjustment': True,
                'hepatic_adjustment': True
            },
            
            # HORMONAL DRUGS
            'Levothyroxine': {
                'class': 'Thyroid Hormone',
                'subclass': 'Hormone replacement',
                'mechanism': 'Thyroid receptor activation',
                'molecular_weight': 776.87,
                'logp': -2.0,
                'bioavailability': 0.7,
                'protein_binding': 0.99,
                'volume_distribution': 0.15,
                'clearance': 0.1,
                'half_life': 7.0,  # in euthyroid patients
                'therapeutic_range': (0.8, 2.0),
                'toxicity_threshold': 5,
                'routes': ['oral', 'iv'],
                'metabolism': 'peripheral',
                'excretion': 'renal',
                'food_effect': 0.3,  # significantly reduced with food
                'age_adjustment': True,
                'renal_adjustment': False,
                'hepatic_adjustment': False
            },
            'Prednisone': {
                'class': 'Corticosteroid',
                'subclass': 'Glucocorticoid',
                'mechanism': 'Glucocorticoid receptor activation',
                'molecular_weight': 358.43,
                'logp': 1.6,
                'bioavailability': 0.7,
                'protein_binding': 0.9,
                'volume_distribution': 0.6,
                'clearance': 0.15,
                'half_life': 3.5,
                'therapeutic_range': (5, 60),
                'toxicity_threshold': 200,
                'routes': ['oral', 'iv', 'im'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'food_effect': 0.8,
                'age_adjustment': True,
                'renal_adjustment': False,
                'hepatic_adjustment': True
            }
        }
        
        return drugs
    
    def _categorize_drugs(self):
        """
        Categorize drugs by therapeutic class
        
        Returns:
            Dictionary of drug categories
        """
        categories = {
            'Analgesics': ['Acetaminophen'],
            'NSAIDs': ['Ibuprofen', 'Naproxen'],
            'Cardiovascular': ['Lisinopril', 'Metoprolol', 'Atorvastatin', 'Amlodipine'],
            'Antidiabetics': ['Metformin', 'Glipizide'],
            'Antibiotics': ['Amoxicillin', 'Ciprofloxacin'],
            'Psychiatric': ['Sertraline', 'Lorazepam'],
            'Hormonal': ['Levothyroxine', 'Prednisone']
        }
        
        return categories
    
    def _define_pk_parameters(self):
        """
        Define PK parameter calculation methods
        
        Returns:
            Dictionary of PK parameter functions
        """
        def calculate_absorption(drug_props, dose, patient_factors):
            """Calculate drug absorption based on properties and patient factors"""
            base_absorption = drug_props['bioavailability']
            
            # Food effect
            if 'food_effect' in drug_props and patient_factors.get('fed', False):
                base_absorption *= drug_props['food_effect']
            
            # Age effect (reduced absorption in elderly)
            if patient_factors.get('age', 50) > 65:
                base_absorption *= 0.9
            
            # Route adjustment
            route = patient_factors.get('route', 'oral')
            if route == 'iv':
                base_absorption = 1.0
            elif route == 'im':
                base_absorption *= 0.8
            
            return base_absorption
        
        def calculate_distribution(drug_props, patient_factors):
            """Calculate volume of distribution with patient adjustments"""
            base_vd = drug_props['volume_distribution']
            
            # Weight adjustment
            weight = patient_factors.get('weight', 70)  # kg
            vd_l = base_vd * weight
            
            # Age effect (increased Vd in elderly for some drugs)
            age = patient_factors.get('age', 50)
            if age > 65 and drug_props['logp'] > 2:  # Lipophilic drugs
                vd_l *= 1.2
            
            # Disease state effects
            if patient_factors.get('disease') == 'liver_cirrhosis':
                vd_l *= 1.3
            elif patient_factors.get('disease') == 'renal_failure':
                vd_l *= 0.8
            
            return vd_l
        
        def calculate_clearance(drug_props, patient_factors):
            """Calculate drug clearance with patient-specific adjustments"""
            base_cl = drug_props['clearance']
            weight = patient_factors.get('weight', 70)
            
            # Weight-based clearance
            cl_lh = base_cl * weight
            
            # Age effect (reduced clearance in elderly)
            age = patient_factors.get('age', 50)
            if age > 65:
                cl_lh *= 0.7
            
            # Renal function adjustment
            if drug_props['excretion'] == 'renal':
                creatinine_clearance = patient_factors.get('creatinine_clearance', 100)  # mL/min
                if creatinine_clearance < 30:  # Severe renal impairment
                    cl_lh *= 0.3
                elif creatinine_clearance < 60:  # Moderate renal impairment
                    cl_lh *= 0.6
            
            # Hepatic function adjustment
            if drug_props['metabolism'] == 'hepatic':
                if patient_factors.get('liver_function') == 'impaired':
                    cl_lh *= 0.5
            
            return cl_lh
        
        def calculate_half_life(drug_props, patient_factors):
            """Calculate elimination half-life"""
            vd = calculate_distribution(drug_props, patient_factors)
            cl = calculate_clearance(drug_props, patient_factors)
            
            # t1/2 = (0.693 * Vd) / Cl
            half_life = (0.693 * vd) / cl
            
            return half_life
        
        return {
            'absorption': calculate_absorption,
            'distribution': calculate_distribution,
            'clearance': calculate_clearance,
            'half_life': calculate_half_life
        }
    
    def _define_pd_parameters(self):
        """
        Define PD parameter calculation methods
        
        Returns:
            Dictionary of PD parameter functions
        """
        def calculate_emax(drug_props, patient_factors):
            """Calculate maximum effect (Emax)"""
            base_emax = 100  # Normalize to 100% maximum effect
            
            # Disease state effects
            disease = patient_factors.get('disease', 'healthy')
            if disease == 'hypertension' and drug_props['class'] in ['ACE Inhibitor', 'Beta Blocker']:
                base_emax *= 1.1  # Enhanced effect in target disease
            elif disease == 'diabetes' and drug_props['class'] == 'Biguanide':
                base_emax *= 1.15
            elif disease == 'infection' and drug_props['class'] == 'Penicillin':
                base_emax *= 1.2
            
            # Age effects
            age = patient_factors.get('age', 50)
            if age > 65:
                base_emax *= 0.9  # Reduced response in elderly
            
            # Genetic factors (simplified)
            if patient_factors.get('poor_metabolizer', False) and drug_props['metabolism'] == 'hepatic':
                base_emax *= 1.3  # Enhanced effect due to reduced metabolism
            
            return base_emax
        
        def calculate_ec50(drug_props, patient_factors):
            """Calculate concentration for 50% effect (EC50)"""
            # Base EC50 related to therapeutic range
            therapeutic_min, therapeutic_max = drug_props['therapeutic_range']
            base_ec50 = (therapeutic_min + therapeutic_max) / 2
            
            # Patient sensitivity adjustments
            disease = patient_factors.get('disease', 'healthy')
            if disease == 'hypertension' and drug_props['class'] in ['ACE Inhibitor', 'Beta Blocker']:
                base_ec50 *= 0.8  # Increased sensitivity
            elif disease == 'diabetes' and drug_props['class'] == 'Biguanide':
                base_ec50 *= 0.7
            
            # Age effects
            age = patient_factors.get('age', 50)
            if age > 65:
                base_ec50 *= 1.2  # Reduced sensitivity
            
            return base_ec50
        
        def calculate_hill_coefficient(drug_props, patient_factors):
            """Calculate Hill coefficient (steepness of dose-response)"""
            # Most drugs have Hill coefficient around 1
            base_hill = 1.0
            
            # Some drugs have steeper curves
            if drug_props['class'] in ['SSRI', 'Benzodiazepine']:
                base_hill = 1.5
            elif drug_props['class'] == 'Thyroid Hormone':
                base_hill = 2.0
            
            return base_hill
        
        def calculate_onset_time(drug_props, patient_factors):
            """Calculate time to onset of effect"""
            # Base onset time related to PK properties
            base_onset = drug_props['half_life'] * 0.5
            
            # Route effects
            route = patient_factors.get('route', 'oral')
            if route == 'iv':
                base_onset *= 0.1
            elif route == 'im':
                base_onset *= 0.3
            
            # Drug class effects
            if drug_props['class'] == 'Benzodiazepine':
                base_onset *= 0.5  # Rapid onset
            elif drug_props['class'] == 'SSRI':
                base_onset *= 14  # Delayed onset (weeks)
            
            return base_onset
        
        def calculate_duration(drug_props, patient_factors):
            """Calculate duration of effect"""
            # Base duration related to half-life
            base_duration = drug_props['half_life'] * 4
            
            # Active metabolites extend duration
            if drug_props['class'] in ['Benzodiazepine', 'Corticosteroid']:
                base_duration *= 2
            
            # Patient factors
            age = patient_factors.get('age', 50)
            if age > 65:
                base_duration *= 1.3  # Prolonged duration in elderly
            
            return base_duration
        
        return {
            'emax': calculate_emax,
            'ec50': calculate_ec50,
            'hill_coefficient': calculate_hill_coefficient,
            'onset_time': calculate_onset_time,
            'duration': calculate_duration
        }
    
    def get_drug_properties(self, drug_name):
        """
        Get properties for a specific drug
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Dictionary of drug properties
        """
        return self.drug_database.get(drug_name, None)
    
    def get_drugs_by_class(self, drug_class):
        """
        Get all drugs in a specific class
        
        Args:
            drug_class: Therapeutic class name
            
        Returns:
            List of drug names
        """
        return self.drug_classes.get(drug_class, [])
    
    def get_all_drug_names(self):
        """Get list of all available drugs"""
        return list(self.drug_database.keys())
    
    def get_all_drug_classes(self):
        """Get list of all drug classes"""
        return list(self.drug_classes.keys())
    
    def calculate_pk_parameters(self, drug_name, dose, patient_factors):
        """
        Calculate PK parameters for a specific drug and patient
        
        Args:
            drug_name: Name of the drug
            dose: Administered dose
            patient_factors: Dictionary of patient characteristics
            
        Returns:
            Dictionary of calculated PK parameters
        """
        drug_props = self.get_drug_properties(drug_name)
        if not drug_props:
            return None
        
        pk_params = {}
        for param_name, param_func in self.pk_parameters.items():
            pk_params[param_name] = param_func(drug_props, dose, patient_factors)
        
        return pk_params
    
    def calculate_pd_parameters(self, drug_name, patient_factors):
        """
        Calculate PD parameters for a specific drug and patient
        
        Args:
            drug_name: Name of the drug
            patient_factors: Dictionary of patient characteristics
            
        Returns:
            Dictionary of calculated PD parameters
        """
        drug_props = self.get_drug_properties(drug_name)
        if not drug_props:
            return None
        
        pd_params = {}
        for param_name, param_func in self.pd_parameters.items():
            pd_params[param_name] = param_func(drug_props, patient_factors)
        
        return pd_params
    
    def simulate_drug_response(self, drug_name, dose, patient_factors, time_points):
        """
        Simulate complete drug response over time
        
        Args:
            drug_name: Name of the drug
            dose: Administered dose
            patient_factors: Dictionary of patient characteristics
            time_points: Array of time points for simulation
            
        Returns:
            Dictionary with PK and PD time courses
        """
        drug_props = self.get_drug_properties(drug_name)
        if not drug_props:
            return None
        
        # Calculate PK parameters
        pk_params = self.calculate_pk_parameters(drug_name, dose, patient_factors)
        
        # Calculate PD parameters
        pd_params = self.calculate_pd_parameters(drug_name, patient_factors)
        
        # Simulate PK time course (one-compartment model)
        absorption = pk_params['absorption']
        vd = pk_params['distribution']
        cl = pk_params['clearance']
        half_life = pk_params['half_life']
        
        # Initial concentration after absorption
        c0 = (dose * absorption) / vd
        
        # Calculate concentration at each time point
        concentrations = []
        for t in time_points:
            if t < half_life * 0.1:  # Absorption phase
                c = c0 * (1 - np.exp(-t / (half_life * 0.1)))
            else:  # Elimination phase
                c = c0 * np.exp(-t / half_life)
            concentrations.append(max(0, c))
        
        # Simulate PD response (Emax model)
        emax = pd_params['emax']
        ec50 = pd_params['ec50']
        hill = pd_params['hill_coefficient']
        onset = pd_params['onset_time']
        
        responses = []
        for i, (t, c) in enumerate(zip(time_points, concentrations)):
            if t < onset:
                effect = 0  # No effect before onset
            else:
                # Emax model: Effect = (Emax * C^Hill) / (EC50^Hill + C^Hill)
                effect = (emax * (c ** hill)) / ((ec50 ** hill) + (c ** hill))
            responses.append(effect)
        
        return {
            'time_points': time_points,
            'concentrations': np.array(concentrations),
            'responses': np.array(responses),
            'pk_parameters': pk_params,
            'pd_parameters': pd_params,
            'drug_properties': drug_props
        }
    
    def generate_drug_comparison_report(self, drug_names, patient_factors):
        """
        Generate comparison report for multiple drugs
        
        Args:
            drug_names: List of drug names to compare
            patient_factors: Patient characteristics for simulation
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}
        
        for drug_name in drug_names:
            drug_props = self.get_drug_properties(drug_name)
            if drug_props:
                # Standard dose for comparison
                standard_dose = 100  # mg
                
                # Simulate response
                time_points = np.linspace(0, 24, 100)  # 24 hours
                simulation = self.simulate_drug_response(
                    drug_name, standard_dose, patient_factors, time_points
                )
                
                comparison_results[drug_name] = {
                    'properties': drug_props,
                    'simulation': simulation,
                    'peak_concentration': np.max(simulation['concentrations']),
                    'peak_response': np.max(simulation['responses']),
                    'auc_pk': np.trapz(simulation['concentrations'], time_points),
                    'auc_pd': np.trapz(simulation['responses'], time_points),
                    'time_to_peak': time_points[np.argmax(simulation['concentrations'])],
                    'duration_above_therapeutic': self._calculate_duration_above_threshold(
                        simulation['concentrations'], time_points, 
                        drug_props['therapeutic_range'][0]
                    )
                }
        
        return comparison_results
    
    def _calculate_duration_above_threshold(self, concentrations, time_points, threshold):
        """Calculate duration concentration remains above threshold"""
        above_threshold = concentrations > threshold
        if np.any(above_threshold):
            first_above = np.where(above_threshold)[0][0]
            last_above = np.where(above_threshold)[0][-1]
            return time_points[last_above] - time_points[first_above]
        return 0
    
    def export_database(self, filepath):
        """
        Export drug database to JSON file
        
        Args:
            filepath: Path to save database
        """
        export_data = {
            'drug_database': self.drug_database,
            'drug_classes': self.drug_classes,
            'export_date': pd.Timestamp.now().isoformat(),
            'total_drugs': len(self.drug_database),
            'total_classes': len(self.drug_classes)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"💾 Drug database exported to: {filepath}")
    
    def get_database_summary(self):
        """
        Get summary statistics of the drug database
        
        Returns:
            Dictionary with database summary
        """
        summary = {
            'total_drugs': len(self.drug_database),
            'total_classes': len(self.drug_classes),
            'drugs_per_class': {k: len(v) for k, v in self.drug_classes.items()},
            'available_routes': list(set([
                route for drug in self.drug_database.values() 
                for route in drug['routes']
            ])),
            'metabolism_types': list(set([
                drug['metabolism'] for drug in self.drug_database.values()
            ])),
            'excretion_types': list(set([
                drug['excretion'] for drug in self.drug_database.values()
            ]))
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize expanded drug database
    db = ExpandedDrugDatabase()
    
    # Get database summary
    summary = db.get_database_summary()
    print("📊 Expanded Drug Database Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Example simulation
    patient_factors = {
        'age': 65,
        'weight': 80,
        'disease': 'hypertension',
        'creatinine_clearance': 70,
        'fed': True,
        'route': 'oral'
    }
    
    # Simulate drug response
    time_points = np.linspace(0, 24, 100)
    simulation = db.simulate_drug_response('Lisinopril', 10, patient_factors, time_points)
    
    print(f"\n💊 Lisinopril Simulation:")
    print(f"   Peak Concentration: {np.max(simulation['concentrations']):.2f}")
    print(f"   Peak Response: {np.max(simulation['responses']):.2f}")
    print(f"   Half-life: {simulation['pk_parameters']['half_life']:.2f} hours")
