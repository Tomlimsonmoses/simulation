"""
Simplified PK-PD Training Data Preparation

This system creates training data for PK-PD simulation models with a focus on
simplicity and reliability.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleTrainingConfig:
    """Simple configuration for training data preparation."""
    output_dir: str = "simple_training_data"
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42

class SimplePKPDTrainingData:
    """Simplified PK-PD training data preparation."""
    
    def __init__(self, config: SimpleTrainingConfig = None):
        """Initialize the training data preparation system."""
        self.config = config or SimpleTrainingConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load the integrated PK-PD knowledge base."""
        kb_path = "pk_pd_knowledge_base.json"
        
        if Path(kb_path).exists():
            with open(kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Knowledge base not found: {kb_path}")
            return {}
    
    def prepare_simple_training_data(self) -> Dict[str, Any]:
        """Prepare simplified training data."""
        logger.info("Preparing simplified PK-PD training data...")
        
        training_datasets = {}
        
        # 1. Drug properties dataset
        drug_dataset = self._prepare_drug_dataset()
        if drug_dataset:
            training_datasets['drug_properties'] = drug_dataset
        
        # 2. PK parameters dataset
        pk_dataset = self._prepare_pk_dataset()
        if pk_dataset:
            training_datasets['pk_parameters'] = pk_dataset
        
        # 3. Disease characteristics dataset
        disease_dataset = self._prepare_disease_dataset()
        if disease_dataset:
            training_datasets['disease_characteristics'] = disease_dataset
        
        # 4. Organ characteristics dataset
        organ_dataset = self._prepare_organ_dataset()
        if organ_dataset:
            training_datasets['organ_characteristics'] = organ_dataset
        
        # 5. Clinical outcomes dataset (synthetic)
        clinical_dataset = self._prepare_clinical_dataset()
        if clinical_dataset:
            training_datasets['clinical_outcomes'] = clinical_dataset
        
        # 6. Combined PK prediction dataset
        pk_prediction = self._prepare_pk_prediction_dataset()
        if pk_prediction:
            training_datasets['pk_prediction'] = pk_prediction
        
        # 7. Combined clinical prediction dataset
        clinical_prediction = self._prepare_clinical_prediction_dataset()
        if clinical_prediction:
            training_datasets['clinical_prediction'] = clinical_prediction
        
        # Export all datasets
        self._export_datasets(training_datasets)
        
        return training_datasets
    
    def _prepare_drug_dataset(self) -> Optional[Dict[str, Any]]:
        """Prepare drug properties dataset."""
        logger.info("Preparing drug dataset...")
        
        # Collect drug data
        all_drugs = []
        
        # From real_data
        if 'drugs' in self.knowledge_base:
            for source, drugs in self.knowledge_base['drugs'].items():
                for drug in drugs:
                    all_drugs.append(drug)
        
        if not all_drugs:
            logger.warning("No drug data found")
            return None
        
        df = pd.DataFrame(all_drugs)
        
        # Select and clean key columns
        key_columns = [
            'drug_id', 'drug_name', 'molecular_weight', 'logp', 'bioavailability',
            'protein_binding', 'clearance_rate', 'volume_distribution', 'half_life',
            'drug_class', 'administration_route'
        ]
        
        # Filter available columns
        available_columns = [col for col in key_columns if col in df.columns]
        df_clean = df[available_columns].copy()
        
        # Clean data
        df_clean = self._clean_numeric_data(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        
        # Add derived features
        df_clean = self._add_drug_features(df_clean)
        
        # Create splits
        splits = self._create_data_splits(df_clean)
        
        return {
            'data': df_clean,
            'features': list(df_clean.columns),
            'target_columns': ['bioavailability', 'clearance_rate', 'half_life'],
            'splits': splits,
            'metadata': {
                'total_samples': len(df_clean),
                'features_count': len(df_clean.columns)
            }
        }
    
    def _prepare_pk_dataset(self) -> Optional[Dict[str, Any]]:
        """Prepare PK parameters dataset."""
        logger.info("Preparing PK dataset...")
        
        # Collect PK data
        all_pk = []
        
        if 'pk_parameters' in self.knowledge_base:
            for source, params in self.knowledge_base['pk_parameters'].items():
                for param in params:
                    all_pk.append(param)
        
        if not all_pk:
            logger.warning("No PK data found")
            return None
        
        df = pd.DataFrame(all_pk)
        
        # Select key columns
        key_columns = [
            'drug_id', 'population_group', 'clearance', 'volume_distribution',
            'absorption_rate', 'bioavailability', 'half_life', 'time_to_peak'
        ]
        
        available_columns = [col for col in key_columns if col in df.columns]
        df_clean = df[available_columns].copy()
        
        # Clean data
        df_clean = self._clean_numeric_data(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        
        # Add derived features
        df_clean = self._add_pk_features(df_clean)
        
        # Create splits
        splits = self._create_data_splits(df_clean)
        
        return {
            'data': df_clean,
            'features': list(df_clean.columns),
            'target_columns': ['clearance', 'volume_distribution', 'half_life'],
            'splits': splits,
            'metadata': {
                'total_samples': len(df_clean),
                'population_groups': df_clean['population_group'].unique().tolist() if 'population_group' in df_clean.columns else []
            }
        }
    
    def _prepare_disease_dataset(self) -> Optional[Dict[str, Any]]:
        """Prepare disease characteristics dataset."""
        logger.info("Preparing disease dataset...")
        
        # Collect disease data
        all_diseases = []
        
        if 'diseases' in self.knowledge_base:
            for source, diseases in self.knowledge_base['diseases'].items():
                for disease in diseases:
                    all_diseases.append(disease)
        
        if not all_diseases:
            logger.warning("No disease data found")
            return None
        
        df = pd.DataFrame(all_diseases)
        
        # Select key columns
        key_columns = [
            'disease_id', 'disease_name', 'description', 'category',
            'symptoms', 'pathways', 'affected_organs'
        ]
        
        available_columns = [col for col in key_columns if col in df.columns]
        df_clean = df[available_columns].copy()
        
        # Clean data
        df_clean = self._clean_text_data(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        
        # Add derived features
        df_clean = self._add_disease_features(df_clean)
        
        # Create splits
        splits = self._create_data_splits(df_clean)
        
        return {
            'data': df_clean,
            'features': list(df_clean.columns),
            'target_columns': ['severity_score', 'complexity_score'],
            'splits': splits,
            'metadata': {
                'total_samples': len(df_clean),
                'disease_categories': df_clean['category'].unique().tolist() if 'category' in df_clean.columns else []
            }
        }
    
    def _prepare_organ_dataset(self) -> Optional[Dict[str, Any]]:
        """Prepare organ characteristics dataset."""
        logger.info("Preparing organ dataset...")
        
        # Collect organ data
        all_organs = []
        
        if 'organs' in self.knowledge_base:
            for source, organs in self.knowledge_base['organs'].items():
                for organ in organs:
                    all_organs.append(organ)
        
        if not all_organs:
            logger.warning("No organ data found")
            return None
        
        df = pd.DataFrame(all_organs)
        
        # Select key columns
        key_columns = [
            'organ_id', 'organ_name', 'organ_type', 'volume', 'mass',
            'blood_flow', 'metabolic_rate', 'oxygen_consumption'
        ]
        
        available_columns = [col for col in key_columns if col in df.columns]
        df_clean = df[available_columns].copy()
        
        # Clean data
        df_clean = self._clean_numeric_data(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        
        # Add derived features
        df_clean = self._add_organ_features(df_clean)
        
        # Create splits
        splits = self._create_data_splits(df_clean)
        
        return {
            'data': df_clean,
            'features': list(df_clean.columns),
            'target_columns': ['blood_flow', 'metabolic_rate', 'oxygen_consumption'],
            'splits': splits,
            'metadata': {
                'total_samples': len(df_clean),
                'organ_types': df_clean['organ_type'].unique().tolist() if 'organ_type' in df_clean.columns else []
            }
        }
    
    def _prepare_clinical_dataset(self) -> Optional[Dict[str, Any]]:
        """Prepare synthetic clinical outcomes dataset."""
        logger.info("Preparing clinical dataset...")
        
        # Generate synthetic clinical data
        clinical_data = self._generate_synthetic_clinical_data()
        
        if not clinical_data:
            return None
        
        df = pd.DataFrame(clinical_data)
        
        # Clean data
        df = self._clean_numeric_data(df)
        df = self._handle_missing_values(df)
        
        # Create splits
        splits = self._create_data_splits(df)
        
        return {
            'data': df,
            'features': list(df.columns),
            'target_columns': ['treatment_success', 'adverse_event_risk', 'qol_improvement'],
            'splits': splits,
            'metadata': {
                'total_samples': len(df),
                'outcome_types': ['treatment_success', 'adverse_event_risk', 'qol_improvement']
            }
        }
    
    def _prepare_pk_prediction_dataset(self) -> Optional[Dict[str, Any]]:
        """Prepare PK prediction dataset."""
        logger.info("Preparing PK prediction dataset...")
        
        drug_data = self._prepare_drug_dataset()
        pk_data = self._prepare_pk_dataset()
        
        if not drug_data or not pk_data:
            return None
        
        drug_df = drug_data['data']
        pk_df = pk_data['data']
        
        # Find common drug IDs
        drug_ids = set(drug_df['drug_id']) if 'drug_id' in drug_df.columns else set()
        pk_ids = set(pk_df['drug_id']) if 'drug_id' in pk_df.columns else set()
        
        common_ids = drug_ids & pk_ids
        
        if not common_ids:
            logger.warning("No common drug IDs found for PK prediction")
            return None
        
        # Filter datasets
        drug_filtered = drug_df[drug_df['drug_id'].isin(common_ids)]
        pk_filtered = pk_df[pk_df['drug_id'].isin(common_ids)]
        
        # Merge on drug_id
        merged = pd.merge(drug_filtered, pk_filtered, on='drug_id', how='inner')
        
        # Create splits
        splits = self._create_data_splits(merged)
        
        target_cols = ['clearance', 'volume_distribution', 'half_life']
        available_targets = [col for col in target_cols if col in merged.columns]
        
        return {
            'data': merged,
            'features': [col for col in merged.columns if col not in available_targets],
            'target_columns': available_targets,
            'splits': splits,
            'metadata': {
                'total_samples': len(merged),
                'feature_count': len(merged.columns) - len(available_targets)
            }
        }
    
    def _prepare_clinical_prediction_dataset(self) -> Optional[Dict[str, Any]]:
        """Prepare clinical prediction dataset."""
        logger.info("Preparing clinical prediction dataset...")
        
        # Use synthetic clinical data with some drug properties
        clinical_data = self._prepare_clinical_dataset()
        drug_data = self._prepare_drug_dataset()
        
        if not clinical_data or not drug_data:
            return None
        
        clinical_df = clinical_data['data']
        drug_df = drug_data['data']
        
        # Add some drug properties to clinical data
        # For simplicity, just add molecular weight and bioavailability
        drug_props = drug_df[['drug_id', 'molecular_weight', 'bioavailability']].copy()
        
        # Merge
        merged = pd.merge(clinical_df, drug_props, on='drug_id', how='left')
        
        # Fill missing drug properties with median values
        if 'molecular_weight' in merged.columns:
            merged['molecular_weight'].fillna(merged['molecular_weight'].median(), inplace=True)
        if 'bioavailability' in merged.columns:
            merged['bioavailability'].fillna(merged['bioavailability'].median(), inplace=True)
        
        # Create splits
        splits = self._create_data_splits(merged)
        
        target_cols = ['treatment_success', 'adverse_event_risk', 'qol_improvement']
        available_targets = [col for col in target_cols if col in merged.columns]
        
        return {
            'data': merged,
            'features': [col for col in merged.columns if col not in available_targets],
            'target_columns': available_targets,
            'splits': splits,
            'metadata': {
                'total_samples': len(merged),
                'feature_count': len(merged.columns) - len(available_targets)
            }
        }
    
    def _clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric data."""
        df_clean = df.copy()
        
        # Convert numeric columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text data."""
        df_clean = df.copy()
        
        # Convert text columns to string and strip
        text_columns = df_clean.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace('nan', '')
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        df_clean = df.copy()
        
        # Numeric columns - fill with median
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Text columns - fill with 'unknown'
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna('unknown')
        
        return df_clean
    
    def _add_drug_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived drug features."""
        df_features = df.copy()
        
        # Drug-likeness score (simplified)
        df_features['drug_likeness_score'] = 0
        if 'molecular_weight' in df_features.columns:
            df_features['drug_likeness_score'] += (df_features['molecular_weight'] <= 500).astype(int)
        if 'logp' in df_features.columns:
            df_features['drug_likeness_score'] += (df_features['logp'] <= 5).astype(int)
        
        # Bioavailability category
        if 'bioavailability' in df_features.columns:
            df_features['bioavailability_category'] = pd.cut(
                df_features['bioavailability'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['poor', 'moderate', 'good', 'excellent']
            )
        
        return df_features
    
    def _add_pk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived PK features."""
        df_features = df.copy()
        
        # Elimination rate constant
        if 'half_life' in df_features.columns:
            df_features['elimination_rate'] = np.log(2) / df_features['half_life']
        
        # Clearance category
        if 'clearance' in df_features.columns:
            df_features['clearance_category'] = pd.cut(
                df_features['clearance'],
                bins=[0, 1, 5, 10, 20, float('inf')],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        return df_features
    
    def _add_disease_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived disease features."""
        df_features = df.copy()
        
        # Symptom count
        if 'symptoms' in df_features.columns:
            df_features['symptom_count'] = df_features['symptoms'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) else 0
            )
        
        # Complexity score (simplified)
        complexity_score = 0
        if 'symptom_count' in df_features.columns:
            complexity_score += df_features['symptom_count']
        if 'category' in df_features.columns:
            # Give points based on category complexity
            complexity_map = {
                'cancer': 3, 'cardiovascular': 2, 'neurological': 2,
                'infectious': 1, 'respiratory': 1, 'metabolic': 1
            }
            df_features['category_complexity'] = df_features['category'].map(complexity_map).fillna(1)
            complexity_score += df_features['category_complexity']
        
        df_features['complexity_score'] = complexity_score / 2  # Normalize
        
        return df_features
    
    def _add_organ_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived organ features."""
        df_features = df.copy()
        
        # Blood flow per gram
        if 'blood_flow' in df_features.columns and 'mass' in df_features.columns:
            df_features['blood_flow_per_gram'] = df_features['blood_flow'] / (df_features['mass'] + 1e-6)
        
        # Size category
        if 'volume' in df_features.columns:
            df_features['size_category'] = pd.cut(
                df_features['volume'],
                bins=[0, 0.5, 2, 5, 20, float('inf')],
                labels=['tiny', 'small', 'medium', 'large', 'xlarge']
            )
        
        return df_features
    
    def _generate_synthetic_clinical_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic clinical outcomes data."""
        clinical_data = []
        
        # Get sample drugs
        sample_drugs = []
        if 'drugs' in self.knowledge_base:
            for source, drugs in self.knowledge_base['drugs'].items():
                sample_drugs.extend(drugs[:20])  # Take first 20 drugs
        
        np.random.seed(self.config.random_seed)
        
        for i in range(500):  # Generate 500 patient records
            drug = np.random.choice(sample_drugs) if sample_drugs else {'drug_id': f'DRUG_{i%20}'}
            
            # Simulate outcomes based on drug properties
            bioavailability = drug.get('bioavailability', 0.5)
            
            clinical_data.append({
                'patient_id': f'PATIENT_{i}',
                'drug_id': drug.get('drug_id', f'DRUG_{i%20}'),
                'age': np.random.normal(55, 15),
                'sex': np.random.choice(['M', 'F']),
                'weight': np.random.normal(70, 15),
                'dose_mg': np.random.lognormal(3, 1),
                'treatment_success': np.clip(np.random.normal(bioavailability * 0.8, 0.2), 0, 1),
                'adverse_event_risk': np.clip(np.random.normal(0.3, 0.1), 0, 1),
                'qol_improvement': np.clip(np.random.normal(0.6, 0.15), 0, 1),
                'treatment_duration': np.random.exponential(30) + 7
            })
        
        return clinical_data
    
    def _create_data_splits(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create train/validation/test splits."""
        np.random.seed(self.config.random_seed)
        
        n_samples = len(df)
        indices = np.random.permutation(n_samples)
        
        # Calculate split sizes
        test_size = int(n_samples * self.config.test_split)
        val_size = int(n_samples * self.config.validation_split)
        train_size = n_samples - test_size - val_size
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return {
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
    
    def _export_datasets(self, datasets: Dict[str, Any]):
        """Export all training datasets."""
        logger.info("Exporting training datasets...")
        
        # Create summary
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'config': {
                'validation_split': self.config.validation_split,
                'test_split': self.config.test_split,
                'random_seed': self.config.random_seed
            },
            'datasets': {}
        }
        
        # Export each dataset
        for dataset_name, dataset_info in datasets.items():
            if 'data' not in dataset_info:
                continue
            
            # Create dataset directory
            dataset_dir = self.output_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # Export main data
            data_path = dataset_dir / "data.csv"
            dataset_info['data'].to_csv(data_path, index=False)
            
            # Export splits
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
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Export as pickle
        pickle_path = self.output_dir / "training_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(datasets, f)
        
        logger.info(f"Training data exported to: {self.output_dir}")

def main():
    """Main function to prepare simplified training data."""
    config = SimpleTrainingConfig(
        output_dir="simple_pk_pd_training_data",
        validation_split=0.2,
        test_split=0.1,
        random_seed=42
    )
    
    prep = SimplePKPDTrainingData(config)
    training_data = prep.prepare_simple_training_data()
    
    print(f"✅ Simplified PK-PD Training Data Prepared!")
    print(f"📁 Output Directory: {config.output_dir}")
    print(f"📊 Datasets Prepared: {len(training_data)}")
    
    # Print summary
    for dataset_name, dataset_info in training_data.items():
        if 'data' in dataset_info:
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
