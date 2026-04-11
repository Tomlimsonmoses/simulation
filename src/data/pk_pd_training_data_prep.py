"""
PK-PD Simulation Training Data Preparation System

This system prepares comprehensive training data for PK-PD drug-biological system simulation
by integrating multiple data sources with different column naming conventions.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingDataConfig:
    """Configuration for training data preparation."""
    output_dir: str = "training_data"
    include_derived_features: bool = True
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    feature_scaling: bool = True
    handle_missing: str = "median"  # "median", "mean", "drop"
    categorical_encoding: str = "one_hot"  # "one_hot", "label", "target"

class PKPDTrainingDataPrep:
    """
    Comprehensive system for preparing training data for PK-PD simulation models.
    """
    
    def __init__(self, config: TrainingDataConfig = None):
        """Initialize the training data preparation system."""
        self.config = config or TrainingDataConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load integrated knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
        # Initialize training data structures
        self.training_data = {
            'drug_properties': {},
            'pk_parameters': {},
            'pd_responses': {},
            'organ_characteristics': {},
            'disease_characteristics': {},
            'drug_organ_interactions': {},
            'drug_disease_interactions': {},
            'organ_disease_interactions': {},
            'clinical_outcomes': {},
            'population_characteristics': {}
        }
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load the integrated PK-PD knowledge base."""
        kb_path = "pk_pd_knowledge_base.json"
        
        if Path(kb_path).exists():
            with open(kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Knowledge base not found: {kb_path}")
            return {}
    
    def prepare_all_training_data(self) -> Dict[str, Any]:
        """
        Prepare all training data for PK-PD simulation models.
        
        Returns:
            Dictionary containing all prepared training datasets
        """
        logger.info("Starting comprehensive training data preparation...")
        
        # Prepare individual datasets
        self._prepare_drug_properties_training_data()
        self._prepare_pk_parameters_training_data()
        self._prepare_pd_responses_training_data()
        self._prepare_organ_characteristics_training_data()
        self._prepare_disease_characteristics_training_data()
        self._prepare_interaction_training_data()
        self._prepare_clinical_outcomes_training_data()
        
        # Create combined training datasets
        self._create_combined_training_datasets()
        
        # Generate data splits
        self._generate_data_splits()
        
        # Export training data
        self._export_training_data()
        
        logger.info("Training data preparation completed!")
        return self.training_data
    
    def _prepare_drug_properties_training_data(self):
        """Prepare drug properties training data."""
        logger.info("Preparing drug properties training data...")
        
        # Collect all drug data from different sources
        all_drugs = []
        
        # From real_data
        if 'drugs' in self.knowledge_base:
            for source, drugs in self.knowledge_base['drugs'].items():
                for drug in drugs:
                    all_drugs.append(drug)
        
        # From ingested data
        if 'drugs' in self.knowledge_base:
            for source, drugs in self.knowledge_base['drugs'].items():
                for drug in drugs:
                    all_drugs.append(drug)
        
        # Convert to DataFrame
        df_drugs = pd.DataFrame(all_drugs)
        
        if df_drugs.empty:
            logger.warning("No drug data found")
            return
        
        # Feature engineering
        df_drugs = self._engineer_drug_features(df_drugs)
        
        # Data cleaning and preprocessing
        df_drugs = self._preprocess_drug_data(df_drugs)
        
        self.training_data['drug_properties'] = {
            'data': df_drugs,
            'features': list(df_drugs.columns),
            'target_columns': ['bioavailability', 'clearance_rate', 'volume_distribution'],
            'metadata': {
                'total_drugs': len(df_drugs),
                'features_count': len(df_drugs.columns),
                'missing_values': df_drugs.isnull().sum().to_dict()
            }
        }
    
    def _prepare_pk_parameters_training_data(self):
        """Prepare PK parameters training data."""
        logger.info("Preparing PK parameters training data...")
        
        # Collect PK parameters
        all_pk_params = []
        
        if 'pk_parameters' in self.knowledge_base:
            for source, params in self.knowledge_base['pk_parameters'].items():
                for param in params:
                    all_pk_params.append(param)
        
        df_pk = pd.DataFrame(all_pk_params)
        
        if df_pk.empty:
            logger.warning("No PK parameters data found")
            return
        
        # Feature engineering
        df_pk = self._engineer_pk_features(df_pk)
        
        # Data preprocessing
        df_pk = self._preprocess_pk_data(df_pk)
        
        self.training_data['pk_parameters'] = {
            'data': df_pk,
            'features': list(df_pk.columns),
            'target_columns': ['clearance', 'volume_distribution', 'half_life'],
            'metadata': {
                'total_records': len(df_pk),
                'population_groups': df_pk['population_group'].unique().tolist() if 'population_group' in df_pk.columns else [],
                'features_count': len(df_pk.columns)
            }
        }
    
    def _prepare_pd_responses_training_data(self):
        """Prepare PD responses training data."""
        logger.info("Preparing PD responses training data...")
        
        # Collect PD responses
        all_pd_responses = []
        
        if 'pd_responses' in self.knowledge_base:
            for source, responses in self.knowledge_base['pd_responses'].items():
                for response in responses:
                    all_pd_responses.append(response)
        
        df_pd = pd.DataFrame(all_pd_responses)
        
        if df_pd.empty:
            logger.warning("No PD responses data found")
            return
        
        # Feature engineering
        df_pd = self._engineer_pd_features(df_pd)
        
        # Data preprocessing
        df_pd = self._preprocess_pd_data(df_pd)
        
        self.training_data['pd_responses'] = {
            'data': df_pd,
            'features': list(df_pd.columns),
            'target_columns': ['response_magnitude', 'ec50', 'emax'],
            'metadata': {
                'total_responses': len(df_pd),
                'response_types': df_pd['response_type'].unique().tolist() if 'response_type' in df_pd.columns else [],
                'features_count': len(df_pd.columns)
            }
        }
    
    def _prepare_organ_characteristics_training_data(self):
        """Prepare organ characteristics training data."""
        logger.info("Preparing organ characteristics training data...")
        
        # Collect organ data
        all_organs = []
        
        if 'organs' in self.knowledge_base:
            for source, organs in self.knowledge_base['organs'].items():
                for organ in organs:
                    all_organs.append(organ)
        
        df_organs = pd.DataFrame(all_organs)
        
        if df_organs.empty:
            logger.warning("No organ data found")
            return
        
        # Feature engineering
        df_organs = self._engineer_organ_features(df_organs)
        
        # Data preprocessing
        df_organs = self._preprocess_organ_data(df_organs)
        
        self.training_data['organ_characteristics'] = {
            'data': df_organs,
            'features': list(df_organs.columns),
            'target_columns': ['blood_flow', 'metabolic_rate', 'oxygen_consumption'],
            'metadata': {
                'total_organs': len(df_organs),
                'organ_types': df_organs['organ_type'].unique().tolist() if 'organ_type' in df_organs.columns else [],
                'features_count': len(df_organs.columns)
            }
        }
    
    def _prepare_disease_characteristics_training_data(self):
        """Prepare disease characteristics training data."""
        logger.info("Preparing disease characteristics training data...")
        
        # Collect disease data
        all_diseases = []
        
        if 'diseases' in self.knowledge_base:
            for source, diseases in self.knowledge_base['diseases'].items():
                for disease in diseases:
                    all_diseases.append(disease)
        
        df_diseases = pd.DataFrame(all_diseases)
        
        if df_diseases.empty:
            logger.warning("No disease data found")
            return
        
        # Feature engineering
        df_diseases = self._engineer_disease_features(df_diseases)
        
        # Data preprocessing
        df_diseases = self._preprocess_disease_data(df_diseases)
        
        self.training_data['disease_characteristics'] = {
            'data': df_diseases,
            'features': list(df_diseases.columns),
            'target_columns': ['severity', 'prevalence', 'mortality_rate'],
            'metadata': {
                'total_diseases': len(df_diseases),
                'disease_categories': df_diseases['category'].unique().tolist() if 'category' in df_diseases.columns else [],
                'features_count': len(df_diseases.columns)
            }
        }
    
    def _prepare_interaction_training_data(self):
        """Prepare interaction training data."""
        logger.info("Preparing interaction training data...")
        
        # Drug-Organ interactions
        drug_organ_interactions = []
        if 'drug_organ_interactions' in self.knowledge_base:
            for source, interactions in self.knowledge_base['drug_organ_interactions'].items():
                drug_organ_interactions.extend(interactions)
        
        df_drug_organ = pd.DataFrame(drug_organ_interactions)
        if not df_drug_organ.empty:
            df_drug_organ = self._engineer_interaction_features(df_drug_organ, 'drug_organ')
            self.training_data['drug_organ_interactions'] = {
                'data': df_drug_organ,
                'features': list(df_drug_organ.columns),
                'target_columns': ['interaction_strength', 'clearance_contribution'],
                'metadata': {'total_interactions': len(df_drug_organ)}
            }
        
        # Drug-Disease interactions
        drug_disease_interactions = []
        if 'drug_disease_interactions' in self.knowledge_base:
            for source, interactions in self.knowledge_base['drug_disease_interactions'].items():
                drug_disease_interactions.extend(interactions)
        
        df_drug_disease = pd.DataFrame(drug_disease_interactions)
        if not df_drug_disease.empty:
            df_drug_disease = self._engineer_interaction_features(df_drug_disease, 'drug_disease')
            self.training_data['drug_disease_interactions'] = {
                'data': df_drug_disease,
                'features': list(df_drug_disease.columns),
                'target_columns': ['efficacy_score', 'side_effect_risk'],
                'metadata': {'total_interactions': len(df_drug_disease)}
            }
    
    def _prepare_clinical_outcomes_training_data(self):
        """Prepare clinical outcomes training data."""
        logger.info("Preparing clinical outcomes training data...")
        
        # Generate synthetic clinical outcomes based on available data
        clinical_outcomes = self._generate_synthetic_clinical_outcomes()
        
        df_clinical = pd.DataFrame(clinical_outcomes)
        df_clinical = self._engineer_clinical_features(df_clinical)
        
        self.training_data['clinical_outcomes'] = {
            'data': df_clinical,
            'features': list(df_clinical.columns),
            'target_columns': ['treatment_success', 'adverse_event_risk', 'qol_improvement'],
            'metadata': {
                'total_patients': len(df_clinical),
                'outcome_types': ['treatment_success', 'adverse_event_risk', 'qol_improvement']
            }
        }
    
    def _engineer_drug_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for drug data."""
        df_engineered = df.copy()
        
        # Molecular weight categories
        if 'molecular_weight' in df.columns:
            df_engineered['mw_category'] = pd.cut(
                df['molecular_weight'], 
                bins=[0, 200, 400, 600, 1000, float('inf')],
                labels=['small', 'medium', 'large', 'xlarge', 'xxlarge']
            )
        
        # LogP categories
        if 'logp' in df.columns:
            df_engineered['logp_category'] = pd.cut(
                df['logp'],
                bins=[-5, 0, 2, 4, 6, float('inf')],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        # Bioavailability categories
        if 'bioavailability' in df.columns:
            df_engineered['bioavailability_category'] = pd.cut(
                df['bioavailability'],
                bins=[0, 0.2, 0.5, 0.8, 1.0],
                labels=['poor', 'moderate', 'good', 'excellent']
            )
        
        # Drug-likeness score (Lipinski's rule)
        df_engineered['lipinski_score'] = 0
        if 'molecular_weight' in df.columns:
            df_engineered['lipinski_score'] += (df['molecular_weight'] <= 500).astype(int)
        if 'logp' in df.columns:
            df_engineered['lipinski_score'] += (df['logp'] <= 5).astype(int)
        if 'hydrogen_bond_donors' in df.columns:
            df_engineered['lipinski_score'] += (df['hydrogen_bond_donors'] <= 5).astype(int)
        if 'hydrogen_bond_acceptors' in df.columns:
            df_engineered['lipinski_score'] += (df['hydrogen_bond_acceptors'] <= 10).astype(int)
        
        # Target count
        if 'targets' in df.columns:
            df_engineered['target_count'] = df['targets'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        # Category count
        if 'categories' in df.columns:
            df_engineered['category_count'] = df['categories'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        return df_engineered
    
    def _engineer_pk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for PK parameters."""
        df_engineered = df.copy()
        
        # Clearance categories
        if 'clearance' in df.columns:
            df_engineered['clearance_category'] = pd.cut(
                df['clearance'],
                bins=[0, 1, 5, 10, 20, float('inf')],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        # Volume distribution categories
        if 'volume_distribution' in df.columns:
            df_engineered['vd_category'] = pd.cut(
                df['volume_distribution'],
                bins=[0, 5, 20, 50, 100, float('inf')],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        # Half-life categories
        if 'half_life' in df.columns:
            df_engineered['half_life_category'] = pd.cut(
                df['half_life'],
                bins=[0, 2, 6, 12, 24, float('inf')],
                labels=['very_short', 'short', 'medium', 'long', 'very_long']
            )
        
        # Elimination rate constant
        if 'half_life' in df.columns:
            df_engineered['elimination_rate'] = np.log(2) / df['half_life']
        
        # Dosing frequency suggestion
        if 'half_life' in df.columns:
            def suggest_dosing_frequency(half_life):
                if half_life <= 4:
                    return 'q4h'
                elif half_life <= 8:
                    return 'q6h'
                elif half_life <= 12:
                    return 'q8h'
                elif half_life <= 24:
                    return 'q12h'
                else:
                    return 'q24h'
            
            df_engineered['dosing_frequency'] = df['half_life'].apply(suggest_dosing_frequency)
        
        return df_engineered
    
    def _engineer_pd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for PD responses."""
        df_engineered = df.copy()
        
        # Response magnitude categories
        if 'response_magnitude' in df.columns:
            df_engineered['response_category'] = pd.cut(
                df['response_magnitude'],
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['minimal', 'moderate', 'good', 'excellent']
            )
        
        # Potency categories (based on EC50)
        if 'ec50' in df.columns:
            df_engineered['potency_category'] = pd.cut(
                df['ec50'],
                bins=[0, 0.1, 1, 10, 100, float('inf')],
                labels=['very_high', 'high', 'medium', 'low', 'very_low']
            )
        
        # Therapeutic index (simplified)
        if 'emax' in df.columns and 'ec50' in df.columns:
            df_engineered['therapeutic_index'] = df['emax'] / (df['ec50'] + 1e-6)
        
        # Onset speed
        if 'time_to_response' in df.columns:
            df_engineered['onset_speed'] = pd.cut(
                df['time_to_response'],
                bins=[0, 1, 4, 12, 24, float('inf')],
                labels=['immediate', 'fast', 'moderate', 'slow', 'very_slow']
            )
        
        return df_engineered
    
    def _engineer_organ_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for organ data."""
        df_engineered = df.copy()
        
        # Blood flow per gram
        if 'blood_flow' in df.columns and 'mass' in df.columns:
            df_engineered['blood_flow_per_gram'] = df['blood_flow'] / (df['mass'] + 1e-6)
        
        # Metabolic rate per gram
        if 'metabolic_rate' in df.columns and 'mass' in df.columns:
            df_engineered['metabolic_rate_per_gram'] = df['metabolic_rate'] / (df['mass'] + 1e-6)
        
        # Size categories
        if 'volume' in df.columns:
            df_engineered['size_category'] = pd.cut(
                df['volume'],
                bins=[0, 0.5, 2, 5, 20, float('inf')],
                labels=['tiny', 'small', 'medium', 'large', 'xlarge']
            )
        
        # Perfusion categories
        if 'blood_flow' in df.columns:
            df_engineered['perfusion_category'] = pd.cut(
                df['blood_flow'],
                bins=[0, 100, 500, 1000, 1500, float('inf')],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        return df_engineered
    
    def _engineer_disease_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for disease data."""
        df_engineered = df.copy()
        
        # Symptom count
        if 'symptoms' in df.columns:
            df_engineered['symptom_count'] = df['symptoms'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        # Pathway count
        if 'pathways' in df.columns:
            df_engineered['pathway_count'] = df['pathways'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        # Affected organ count
        if 'affected_organs' in df.columns:
            df_engineered['affected_organ_count'] = df['affected_organs'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        # Complexity score
        complexity_factors = []
        if 'symptom_count' in df_engineered.columns:
            complexity_factors.append(df_engineered['symptom_count'])
        if 'pathway_count' in df_engineered.columns:
            complexity_factors.append(df_engineered['pathway_count'])
        if 'affected_organ_count' in df_engineered.columns:
            complexity_factors.append(df_engineered['affected_organ_count'])
        
        if complexity_factors:
            df_engineered['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
        
        return df_engineered
    
    def _engineer_interaction_features(self, df: pd.DataFrame, interaction_type: str) -> pd.DataFrame:
        """Engineer features for interaction data."""
        df_engineered = df.copy()
        
        # Interaction strength categories
        if 'strength' in df.columns:
            df_engineered['strength_category'] = pd.cut(
                df['strength'],
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['weak', 'moderate', 'strong', 'very_strong']
            )
        
        # Confidence score processing
        if 'confidence' in df.columns:
            df_engineered['confidence_category'] = pd.cut(
                df['confidence'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['low', 'medium', 'high', 'very_high']
            )
        
        return df_engineered
    
    def _engineer_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for clinical outcomes."""
        df_engineered = df.copy()
        
        # Risk categories
        if 'adverse_event_risk' in df.columns:
            df_engineered['risk_category'] = pd.cut(
                df['adverse_event_risk'],
                bins=[0, 0.1, 0.3, 0.6, 1.0],
                labels=['very_low', 'low', 'moderate', 'high']
            )
        
        # Efficacy categories
        if 'treatment_success' in df.columns:
            df_engineered['efficacy_category'] = pd.cut(
                df['treatment_success'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['poor', 'moderate', 'good', 'excellent']
            )
        
        return df_engineered
    
    def _generate_synthetic_clinical_outcomes(self) -> List[Dict[str, Any]]:
        """Generate synthetic clinical outcomes based on available data."""
        clinical_outcomes = []
        
        # Get sample data for realistic synthesis
        sample_drugs = []
        if 'drugs' in self.knowledge_base:
            for source, drugs in self.knowledge_base['drugs'].items():
                sample_drugs.extend(drugs[:10])  # Take first 10 drugs
        
        sample_diseases = []
        if 'diseases' in self.knowledge_base:
            for source, diseases in self.knowledge_base['diseases'].items():
                sample_diseases.extend(diseases[:10])  # Take first 10 diseases
        
        # Generate synthetic outcomes
        np.random.seed(self.config.random_seed)
        
        for i in range(1000):  # Generate 1000 synthetic patient records
            drug = np.random.choice(sample_drugs) if sample_drugs else {'drug_id': f'DRUG_{i%100}'}
            disease = np.random.choice(sample_diseases) if sample_diseases else {'disease_id': f'DISEASE_{i%50}'}
            
            # Simulate treatment outcomes based on drug properties
            bioavailability = drug.get('bioavailability', 0.5)
            half_life = drug.get('half_life', 5.0)
            
            # Treatment success influenced by bioavailability
            treatment_success = np.clip(
                np.random.normal(bioavailability * 0.8, 0.2), 0, 1
            )
            
            # Adverse event risk inversely related to half_life (shorter half-life = more frequent dosing = higher risk)
            adverse_event_risk = np.clip(
                np.random.normal(0.3 / (half_life / 5.0), 0.1), 0, 1
            )
            
            # Quality of life improvement
            qol_improvement = np.clip(
                np.random.normal(treatment_success * 0.7, 0.15), 0, 1
            )
            
            clinical_outcomes.append({
                'patient_id': f'PATIENT_{i}',
                'drug_id': drug.get('drug_id', f'DRUG_{i%100}'),
                'disease_id': disease.get('disease_id', f'DISEASE_{i%50}'),
                'treatment_success': treatment_success,
                'adverse_event_risk': adverse_event_risk,
                'qol_improvement': qol_improvement,
                'age': np.random.normal(55, 15),
                'sex': np.random.choice(['M', 'F']),
                'weight': np.random.normal(70, 15),
                'comorbidities': np.random.randint(0, 4),
                'treatment_duration': np.random.exponential(30) + 7,  # 7-180 days
                'dose_mg': np.random.lognormal(3, 1),  # Log-normal distribution
                'compliance': np.random.beta(8, 2)  # High compliance with some variation
            })
        
        return clinical_outcomes
    
    def _preprocess_drug_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess drug data."""
        return self._general_preprocessing(df)
    
    def _preprocess_pk_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess PK data."""
        return self._general_preprocessing(df)
    
    def _preprocess_pd_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess PD data."""
        return self._general_preprocessing(df)
    
    def _preprocess_organ_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess organ data."""
        return self._general_preprocessing(df)
    
    def _preprocess_disease_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess disease data."""
        return self._general_preprocessing(df)
    
    def _general_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """General preprocessing for all datasets."""
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        if self.config.handle_missing != "drop":
            # Numeric columns
            for col in numeric_columns:
                if df_processed[col].isnull().sum() > 0:
                    if self.config.handle_missing == "median":
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    elif self.config.handle_missing == "mean":
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            
            # Categorical columns - skip list columns
            for col in categorical_columns:
                if df_processed[col].isnull().sum() > 0:
                    # Check if column contains lists
                    if df_processed[col].apply(lambda x: isinstance(x, list)).any():
                        df_processed[col] = df_processed[col].apply(lambda x: x if isinstance(x, list) else [])
                    else:
                        df_processed[col] = df_processed[col].fillna('unknown')
        
        # Feature scaling for numeric columns
        if self.config.feature_scaling:
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
        
        # Categorical encoding
        if self.config.categorical_encoding == "one_hot":
            # Only encode non-list categorical columns
            non_list_categorical = []
            for col in categorical_columns:
                # Check if column contains lists or dicts
                has_lists = df_processed[col].apply(lambda x: isinstance(x, list)).any()
                has_dicts = df_processed[col].apply(lambda x: isinstance(x, dict)).any()
                if not has_lists and not has_dicts:
                    non_list_categorical.append(col)
            
            if non_list_categorical:
                df_processed = pd.get_dummies(df_processed, columns=non_list_categorical, prefix=non_list_categorical)
        
        return df_processed
    
    def _create_combined_training_datasets(self):
        """Create combined training datasets for specific model types."""
        logger.info("Creating combined training datasets...")
        
        # PK prediction dataset (drug properties -> PK parameters)
        pk_dataset = self._create_pk_prediction_dataset()
        if pk_dataset is not None:
            self.training_data['pk_prediction'] = pk_dataset
        
        # PD prediction dataset (drug + PK + disease -> PD response)
        pd_dataset = self._create_pd_prediction_dataset()
        if pd_dataset is not None:
            self.training_data['pd_prediction'] = pd_dataset
        
        # Clinical outcome prediction dataset
        clinical_dataset = self._create_clinical_prediction_dataset()
        if clinical_dataset is not None:
            self.training_data['clinical_prediction'] = clinical_dataset
    
    def _create_pk_prediction_dataset(self) -> Optional[Dict[str, Any]]:
        """Create dataset for PK parameter prediction."""
        drug_data = self.training_data.get('drug_properties', {}).get('data')
        pk_data = self.training_data.get('pk_parameters', {}).get('data')
        
        if drug_data is None or pk_data is None:
            return None
        
        # Check for common drug_id column
        drug_id_cols = [col for col in drug_data.columns if 'drug_id' in col.lower()]
        pk_id_cols = [col for col in pk_data.columns if 'drug_id' in col.lower()]
        
        if not drug_id_cols or not pk_id_cols:
            logger.warning("No drug_id column found for PK prediction dataset")
            return None
        
        drug_id_col = drug_id_cols[0]
        pk_id_col = pk_id_cols[0]
        
        # Merge drug properties with PK parameters
        merged = pd.merge(drug_data, pk_data, left_on=drug_id_col, right_on=pk_id_col, how='inner')
        
        if merged.empty:
            return None
        
        # Identify target columns
        target_cols = ['clearance', 'volume_distribution', 'half_life']
        available_targets = [col for col in target_cols if col in merged.columns]
        
        return {
            'data': merged,
            'features': [col for col in merged.columns if col not in available_targets],
            'target_columns': available_targets,
            'metadata': {
                'total_samples': len(merged),
                'feature_count': len(merged.columns) - len(available_targets)
            }
        }
    
    def _create_pd_prediction_dataset(self) -> Optional[Dict[str, Any]]:
        """Create dataset for PD response prediction."""
        drug_data = self.training_data.get('drug_properties', {}).get('data')
        pk_data = self.training_data.get('pk_parameters', {}).get('data')
        pd_data = self.training_data.get('pd_responses', {}).get('data')
        disease_data = self.training_data.get('disease_characteristics', {}).get('data')
        
        if any(data is None for data in [drug_data, pk_data, pd_data]):
            return None
        
        # Start with PD responses
        merged = pd_data.copy()
        
        # Add drug properties
        drug_id_cols = [col for col in drug_data.columns if 'drug_id' in col.lower()]
        pd_drug_id_cols = [col for col in merged.columns if 'drug_id' in col.lower()]
        
        if drug_id_cols and pd_drug_id_cols:
            merged = pd.merge(merged, drug_data, left_on=pd_drug_id_cols[0], right_on=drug_id_cols[0], how='left')
        
        # Add PK parameters
        pk_id_cols = [col for col in pk_data.columns if 'drug_id' in col.lower()]
        if pk_id_cols and pd_drug_id_cols:
            merged = pd.merge(merged, pk_data, left_on=pd_drug_id_cols[0], right_on=pk_id_cols[0], how='left')
        
        # Add disease characteristics
        disease_id_cols = [col for col in merged.columns if 'disease_id' in col.lower()]
        if disease_data is not None and disease_id_cols:
            disease_data_id_cols = [col for col in disease_data.columns if 'disease_id' in col.lower()]
            if disease_data_id_cols:
                merged = pd.merge(merged, disease_data, left_on=disease_id_cols[0], right_on=disease_data_id_cols[0], how='left')
        
        if merged.empty:
            return None
        
        target_cols = ['response_magnitude', 'ec50', 'emax']
        available_targets = [col for col in target_cols if col in merged.columns]
        
        return {
            'data': merged,
            'features': [col for col in merged.columns if col not in available_targets],
            'target_columns': available_targets,
            'metadata': {
                'total_samples': len(merged),
                'feature_count': len(merged.columns) - len(available_targets)
            }
        }
    
    def _create_clinical_prediction_dataset(self) -> Optional[Dict[str, Any]]:
        """Create dataset for clinical outcome prediction."""
        drug_data = self.training_data.get('drug_properties', {}).get('data')
        disease_data = self.training_data.get('disease_characteristics', {}).get('data')
        clinical_data = self.training_data.get('clinical_outcomes', {}).get('data')
        
        if any(data is None for data in [drug_data, disease_data, clinical_data]):
            return None
        
        # Merge all data
        merged = clinical_data.copy()
        
        # Add drug properties
        drug_id_cols = [col for col in merged.columns if 'drug_id' in col.lower()]
        if drug_id_cols:
            drug_data_id_cols = [col for col in drug_data.columns if 'drug_id' in col.lower()]
            if drug_data_id_cols:
                merged = pd.merge(merged, drug_data, left_on=drug_id_cols[0], right_on=drug_data_id_cols[0], how='left')
        
        # Add disease characteristics
        disease_id_cols = [col for col in merged.columns if 'disease_id' in col.lower()]
        if disease_id_cols:
            disease_data_id_cols = [col for col in disease_data.columns if 'disease_id' in col.lower()]
            if disease_data_id_cols:
                merged = pd.merge(merged, disease_data, left_on=disease_id_cols[0], right_on=disease_data_id_cols[0], how='left')
        
        if merged.empty:
            return None
        
        target_cols = ['treatment_success', 'adverse_event_risk', 'qol_improvement']
        available_targets = [col for col in target_cols if col in merged.columns]
        
        return {
            'data': merged,
            'features': [col for col in merged.columns if col not in available_targets],
            'target_columns': available_targets,
            'metadata': {
                'total_samples': len(merged),
                'feature_count': len(merged.columns) - len(available_targets)
            }
        }
    
    def _generate_data_splits(self):
        """Generate train/validation/test splits for all datasets."""
        logger.info("Generating data splits...")
        
        np.random.seed(self.config.random_seed)
        
        for dataset_name, dataset_info in self.training_data.items():
            if 'data' not in dataset_info or dataset_info['data'].empty:
                continue
            
            df = dataset_info['data'].copy()
            n_samples = len(df)
            
            # Calculate split sizes
            test_size = int(n_samples * self.config.test_split)
            val_size = int(n_samples * self.config.validation_split)
            train_size = n_samples - test_size - val_size
            
            # Shuffle and split
            indices = np.random.permutation(n_samples)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Create splits
            dataset_info['splits'] = {
                'train': {
                    'data': df.iloc[train_indices],
                    'indices': train_indices.tolist(),
                    'size': len(train_indices)
                },
                'validation': {
                    'data': df.iloc[val_indices],
                    'indices': val_indices.tolist(),
                    'size': len(val_indices)
                },
                'test': {
                    'data': df.iloc[test_indices],
                    'indices': test_indices.tolist(),
                    'size': len(test_indices)
                }
            }
    
    def _export_training_data(self):
        """Export all training data."""
        logger.info("Exporting training data...")
        
        # Export summary
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'config': {
                'validation_split': self.config.validation_split,
                'test_split': self.config.test_split,
                'feature_scaling': self.config.feature_scaling,
                'handle_missing': self.config.handle_missing,
                'categorical_encoding': self.config.categorical_encoding
            },
            'datasets': {}
        }
        
        # Export each dataset
        for dataset_name, dataset_info in self.training_data.items():
            if 'data' not in dataset_info:
                continue
            
            # Create dataset directory
            dataset_dir = self.output_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # Export main data
            data_path = dataset_dir / "data.csv"
            dataset_info['data'].to_csv(data_path, index=False)
            
            # Export splits if available
            if 'splits' in dataset_info:
                splits_dir = dataset_dir / "splits"
                splits_dir.mkdir(exist_ok=True)
                
                for split_name, split_info in dataset_info['splits'].items():
                    split_path = splits_dir / f"{split_name}.csv"
                    split_info['data'].to_csv(split_path, index=False)
            
            # Add to summary
            summary['datasets'][dataset_name] = {
                'total_samples': len(dataset_info['data']),
                'features_count': len(dataset_info.get('features', [])),
                'target_columns': dataset_info.get('target_columns', []),
                'has_splits': 'splits' in dataset_info,
                'metadata': dataset_info.get('metadata', {})
            }
        
        # Export summary
        summary_path = self.output_dir / "training_data_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Export as pickle for easy loading
        pickle_path = self.output_dir / "training_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.training_data, f)
        
        logger.info(f"Training data exported to: {self.output_dir}")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"Pickle file saved to: {pickle_path}")

def main():
    """Main function to prepare training data."""
    # Initialize with custom config
    config = TrainingDataConfig(
        output_dir="pk_pd_training_data",
        include_derived_features=True,
        validation_split=0.2,
        test_split=0.1,
        feature_scaling=True,
        handle_missing="median",
        categorical_encoding="one_hot"
    )
    
    # Prepare training data
    prep = PKPDTrainingDataPrep(config)
    training_data = prep.prepare_all_training_data()
    
    print(f"✅ PK-PD Training Data Preparation Completed!")
    print(f"📁 Output Directory: {config.output_dir}")
    print(f"📊 Datasets Prepared: {len([k for k, v in training_data.items() if 'data' in v])}")
    
    # Print summary
    for dataset_name, dataset_info in training_data.items():
        if 'data' in dataset_info and not dataset_info['data'].empty:
            print(f"\n📋 {dataset_name}:")
            print(f"   Samples: {len(dataset_info['data'])}")
            print(f"   Features: {len(dataset_info.get('features', []))}")
            if 'target_columns' in dataset_info:
                print(f"   Targets: {', '.join(dataset_info['target_columns'])}")
            if 'splits' in dataset_info:
                print(f"   Splits: Train({dataset_info['splits']['train']['size']}), "
                      f"Val({dataset_info['splits']['validation']['size']}), "
                      f"Test({dataset_info['splits']['test']['size']})")

if __name__ == "__main__":
    main()
