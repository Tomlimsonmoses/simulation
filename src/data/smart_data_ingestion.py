"""
Smart Data Ingestion System for PK-PD Knowledge Base

This system automatically detects and maps different column naming conventions
to create a unified, context-aware knowledge base for drug-biological system simulation.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ColumnMapping:
    """Represents a column mapping with confidence score."""
    source_column: str
    target_column: str
    confidence: float
    transformation: Optional[str] = None

class SmartDataIngestion:
    """
    Smart data ingestion system that automatically detects and maps
    column names from different data sources to standard PK-PD schema.
    """
    
    def __init__(self):
        """Initialize the smart ingestion system."""
        self.standard_schema = self._define_standard_schema()
        self.column_patterns = self._define_column_patterns()
        self.similarity_threshold = 0.7
        
    def _define_standard_schema(self) -> Dict[str, List[str]]:
        """Define the standard PK-PD schema."""
        return {
            'drugs': [
                'drug_id', 'drug_name', 'drugbank_id', 'molecular_weight', 'logp',
                'solubility', 'bioavailability', 'protein_binding', 'clearance_rate',
                'volume_distribution', 'half_life', 'absorption_rate', 'metabolism_rate',
                'excretion_rate', 'drug_class', 'administration_route', 'smiles',
                'targets', 'mechanism_of_action', 'description', 'categories',
                'side_effects', 'contraindications', 'dosage_forms', 'storage_conditions'
            ],
            'diseases': [
                'disease_id', 'disease_name', 'mesh_id', 'icd10_code', 'description',
                'category', 'symptoms', 'pathways', 'affected_organs', 'severity',
                'prevalence', 'incidence', 'mortality_rate', 'biomarkers',
                'genetic_factors', 'risk_factors', 'prognosis', 'treatment_options'
            ],
            'organs': [
                'organ_id', 'organ_name', 'organ_type', 'volume', 'mass', 'blood_flow',
                'oxygen_consumption', 'metabolic_rate', 'ph', 'temperature',
                'surface_area', 'water_content', 'lipid_content', 'protein_content',
                'health_status', 'function_score', 'receptor_density', 'enzyme_activity'
            ],
            'pk_parameters': [
                'drug_id', 'population_group', 'clearance', 'volume_distribution',
                'absorption_rate', 'elimination_rate', 'bioavailability',
                'protein_binding', 'half_life', 'time_to_peak', 'auc', 'cmax',
                'clearance_cv', 'volume_cv', 'absorption_cv', 'food_effect',
                'drug_interactions', 'renal_impairment_adjustment', 'hepatic_impairment_adjustment'
            ],
            'pd_responses': [
                'drug_id', 'disease_id', 'biomarker_id', 'response_type',
                'response_magnitude', 'time_to_response', 'duration_of_effect',
                'dose_response_curve', 'ec50', 'emax', 'hill_coefficient',
                'baseline_value', 'therapeutic_window', 'side_effect_threshold'
            ],
            'clinical_trials': [
                'trial_id', 'drug_id', 'disease_id', 'phase', 'study_design',
                'sample_size', 'duration', 'primary_endpoint', 'secondary_endpoints',
                'inclusion_criteria', 'exclusion_criteria', 'results', 'adverse_events'
            ],
            'side_effects': [
                'drug_id', 'side_effect_name', 'severity', 'frequency',
                'onset_time', 'duration', 'risk_factors', 'mechanism',
                'management', 'contraindication'
            ],
            'drug_combinations': [
                'drug1_id', 'drug2_id', 'interaction_type', 'severity',
                'mechanism', 'clinical_significance', 'management',
                'alternative_combinations'
            ]
        }
    
    def _define_column_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Define common column naming patterns for each data type."""
        return {
            'drugs': {
                'drug_id': [
                    'drug_id', 'drugid', 'drugid', 'DrugID', 'DRUG_ID', 'id', 'ID',
                    'compound_id', 'compoundid', 'molecule_id', 'chembl_id', 'pubchem_id'
                ],
                'drug_name': [
                    'drug_name', 'drugname', 'drugname', 'DrugName', 'DRUG_NAME', 'name',
                    'compound_name', 'generic_name', 'brand_name', 'molecule_name'
                ],
                'molecular_weight': [
                    'molecular_weight', 'mol_weight', 'mw', 'molecular_weight_gmol',
                    'MolecularWeight', 'MW', 'MolWeight'
                ],
                'logp': [
                    'logp', 'log_p', 'logP', 'LogP', 'LOGP', 'partition_coefficient',
                    'octanol_water_partition', 'xlogp3', 'alogp'
                ],
                'bioavailability': [
                    'bioavailability', 'f_bio', 'oral_bioavailability', 'f_absorption',
                    'Bioavailability', 'F', 'F_abs', 'oral_f'
                ],
                'clearance_rate': [
                    'clearance', 'cl', 'clearance_rate', 'CL', 'Clearance', 'Cl',
                    'total_clearance', 'systemic_clearance'
                ],
                'volume_distribution': [
                    'volume_distribution', 'vd', 'v_d', 'Vd', 'VD', 'volume_of_distribution',
                    'VolumeDistribution', 'Vd_ss', 'steady_state_volume'
                ],
                'half_life': [
                    'half_life', 't_half', 't1_2', 't12', 'half_life_h', 'HalfLife',
                    'T_half', 'T1_2', 'elimination_half_life'
                ],
                'smiles': [
                    'smiles', 'smile', 'canonical_smiles', 'isomeric_smiles',
                    'SMILES', 'structure', 'chemical_structure'
                ],
                'targets': [
                    'targets', 'target', 'target_proteins', 'protein_targets',
                    'mechanism', 'mechanism_of_action', 'moa', 'MOA'
                ],
                'categories': [
                    'categories', 'category', 'drug_class', 'class', 'therapeutic_class',
                    'atc_code', 'pharmacological_class'
                ]
            },
            'diseases': {
                'disease_id': [
                    'disease_id', 'diseaseid', 'diseaseid', 'DiseaseID', 'DISEASE_ID',
                    'condition_id', 'disorder_id', 'mesh_id', 'icd_code', 'icd10_code'
                ],
                'disease_name': [
                    'disease_name', 'diseasename', 'diseasename', 'DiseaseName',
                    'DISEASE_NAME', 'condition', 'disorder', 'illness', 'diagnosis'
                ],
                'description': [
                    'description', 'desc', 'disease_description', 'condition_description',
                    'clinical_description', 'Definition', 'summary'
                ],
                'category': [
                    'category', 'categories', 'classification', 'disease_class',
                    'medical_specialty', 'organ_system', 'disease_type'
                ],
                'symptoms': [
                    'symptoms', 'symptom', 'clinical_signs', 'manifestations',
                    'presenting_symptoms', 'signs_and_symptoms'
                ],
                'pathways': [
                    'pathways', 'pathway', 'biological_pathways', 'molecular_pathways',
                    'signaling_pathways', 'metabolic_pathways'
                ]
            },
            'organs': {
                'organ_id': [
                    'organ_id', 'organid', 'OrganID', 'ORGAN_ID', 'organ_code',
                    'organ_number', 'organ_index'
                ],
                'organ_name': [
                    'organ_name', 'organname', 'OrganName', 'ORGAN_NAME', 'organ',
                    'organ_type', 'anatomical_structure'
                ],
                'volume': [
                    'volume', 'vol', 'organ_volume', 'tissue_volume', 'Volume',
                    'organ_volume_ml', 'organ_volume_l'
                ],
                'mass': [
                    'mass', 'weight', 'organ_mass', 'tissue_mass', 'organ_weight',
                    'tissue_weight', 'organ_mass_g'
                ],
                'blood_flow': [
                    'blood_flow', 'perfusion', 'organ_blood_flow', 'tissue_perfusion',
                    'blood_supply', 'vascular_flow', 'organ_perfusion'
                ]
            },
            'pk_parameters': {
                'drug_id': [
                    'drug_id', 'drugid', 'drugid', 'DrugID', 'DRUG_ID', 'compound_id',
                    'medication_id', 'substance_id'
                ],
                'population_group': [
                    'population_group', 'population', 'group', 'cohort', 'demographics',
                    'age_group', 'patient_group', 'study_population'
                ],
                'clearance': [
                    'clearance', 'cl', 'CL', 'Clearance', 'total_clearance',
                    'systemic_clearance', 'renal_clearance', 'hepatic_clearance'
                ],
                'bioavailability': [
                    'bioavailability', 'f', 'F', 'f_abs', 'oral_bioavailability',
                    'absorption_fraction', 'systemic_availability'
                ],
                'half_life': [
                    'half_life', 't_half', 't1_2', 'T_half', 'elimination_half_life',
                    'terminal_half_life', 'effective_half_life'
                ]
            }
        }
    
    def detect_data_type(self, df: pd.DataFrame, file_name: str = None) -> Tuple[str, float]:
        """
        Automatically detect the data type of a DataFrame.
        
        Args:
            df: DataFrame to analyze
            file_name: Optional file name for additional context
            
        Returns:
            Tuple of (detected_type, confidence_score)
        """
        scores = {}
        columns = [col.lower().strip() for col in df.columns]
        
        for data_type, standard_cols in self.standard_schema.items():
            score = 0
            total_matches = 0
            
            for std_col in standard_cols:
                patterns = self.column_patterns.get(data_type, {}).get(std_col, [std_col])
                
                for pattern in patterns:
                    for col in columns:
                        similarity = self._calculate_similarity(col, pattern.lower())
                        if similarity > 0.8:
                            score += similarity
                            total_matches += 1
                            break
            
            # Normalize score by number of standard columns
            if total_matches > 0:
                scores[data_type] = score / len(standard_cols)
        
        # Add file name context if available
        if file_name:
            file_name_lower = file_name.lower()
            for data_type in self.standard_schema.keys():
                if data_type in file_name_lower:
                    scores[data_type] = scores.get(data_type, 0) + 0.2
        
        if not scores:
            return 'unknown', 0.0
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, confidence
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        return SequenceMatcher(None, str1, str2).ratio()
    
    def map_columns(self, df: pd.DataFrame, data_type: str) -> List[ColumnMapping]:
        """
        Automatically map columns to standard schema.
        
        Args:
            df: DataFrame with source columns
            data_type: Detected data type
            
        Returns:
            List of column mappings with confidence scores
        """
        mappings = []
        source_columns = [col.strip() for col in df.columns]
        
        if data_type not in self.standard_schema:
            return mappings
        
        standard_cols = self.standard_schema[data_type]
        patterns = self.column_patterns.get(data_type, {})
        
        for std_col in standard_cols:
            best_match = None
            best_score = 0.0
            
            # Check exact pattern matches first
            pattern_list = patterns.get(std_col, [std_col])
            
            for source_col in source_columns:
                for pattern in pattern_list:
                    similarity = self._calculate_similarity(
                        source_col.lower(), pattern.lower()
                    )
                    
                    if similarity > best_score and similarity > self.similarity_threshold:
                        best_match = source_col
                        best_score = similarity
            
            if best_match:
                mappings.append(ColumnMapping(
                    source_column=best_match,
                    target_column=std_col,
                    confidence=best_score
                ))
        
        return mappings
    
    def apply_mappings(self, df: pd.DataFrame, mappings: List[ColumnMapping]) -> pd.DataFrame:
        """
        Apply column mappings to a DataFrame.
        
        Args:
            df: Source DataFrame
            mappings: List of column mappings
            
        Returns:
            DataFrame with mapped columns
        """
        df_mapped = df.copy()
        
        for mapping in mappings:
            if mapping.source_column in df_mapped.columns:
                df_mapped = df_mapped.rename(columns={
                    mapping.source_column: mapping.target_column
                })
                
                # Apply transformations if specified
                if mapping.transformation:
                    df_mapped[mapping.target_column] = self._apply_transformation(
                        df_mapped[mapping.target_column], mapping.transformation
                    )
        
        return df_mapped
    
    def _apply_transformation(self, series: pd.Series, transformation: str) -> pd.Series:
        """Apply data transformation to a series."""
        if transformation == 'lowercase':
            return series.astype(str).str.lower()
        elif transformation == 'uppercase':
            return series.astype(str).str.upper()
        elif transformation == 'strip':
            return series.astype(str).str.strip()
        elif transformation == 'numeric':
            return pd.to_numeric(series, errors='coerce')
        elif transformation == 'datetime':
            return pd.to_datetime(series, errors='coerce')
        else:
            return series
    
    def ingest_file(self, file_path: str, data_type: str = None) -> Dict[str, Any]:
        """
        Ingest a data file with automatic column mapping.
        
        Args:
            file_path: Path to the data file
            data_type: Optional explicit data type
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {'error': f'File not found: {file_path}'}
            
            # Load data based on file extension
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'data' in data:
                        df = pd.DataFrame(data['data'])
                    else:
                        df = pd.DataFrame(data)
            else:
                df = pd.read_csv(file_path, encoding='utf-8')
            
            # Detect data type if not provided
            if data_type is None:
                detected_type, confidence = self.detect_data_type(df, file_path.name)
                if confidence < 0.3:
                    return {
                        'error': f'Could not confidently detect data type. Detected: {detected_type} (confidence: {confidence:.2f})'
                    }
                data_type = detected_type
            
            # Map columns
            mappings = self.map_columns(df, data_type)
            
            # Apply mappings
            df_mapped = self.apply_mappings(df, mappings)
            
            # Clean and validate data
            df_clean = self._clean_data(df_mapped, data_type)
            
            return {
                'success': True,
                'data_type': data_type,
                'original_shape': df.shape,
                'mapped_shape': df_mapped.shape,
                'clean_shape': df_clean.shape,
                'column_mappings': [
                    {
                        'source': m.source_column,
                        'target': m.target_column,
                        'confidence': m.confidence
                    } for m in mappings
                ],
                'data': df_clean.to_dict('records'),
                'statistics': self._generate_statistics(df_clean, data_type)
            }
            
        except Exception as e:
            return {'error': f'Error ingesting file: {str(e)}'}
    
    def _clean_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Clean and validate data based on data type."""
        df_clean = df.copy()
        
        # Standard data cleaning
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.strip()
                df_clean[col] = df_clean[col].replace('nan', '')
        
        # Data type specific cleaning
        if data_type == 'drugs':
            df_clean = self._clean_drug_data(df_clean)
        elif data_type == 'diseases':
            df_clean = self._clean_disease_data(df_clean)
        elif data_type == 'organs':
            df_clean = self._clean_organ_data(df_clean)
        elif data_type == 'pk_parameters':
            df_clean = self._clean_pk_data(df_clean)
        
        return df_clean
    
    def _clean_drug_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean drug-specific data."""
        # Standardize drug IDs
        if 'drug_id' in df.columns:
            df['drug_id'] = df['drug_id'].astype(str).str.strip()
        
        # Parse list fields
        list_fields = ['targets', 'categories', 'side_effects']
        for field in list_fields:
            if field in df.columns:
                df[field] = self._parse_list_field(df[field])
        
        # Validate molecular properties
        numeric_fields = ['molecular_weight', 'logp', 'bioavailability']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        return df
    
    def _clean_disease_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean disease-specific data."""
        # Standardize disease IDs
        if 'disease_id' in df.columns:
            df['disease_id'] = df['disease_id'].astype(str).str.strip()
        
        # Parse list fields
        list_fields = ['symptoms', 'pathways', 'affected_organs']
        for field in list_fields:
            if field in df.columns:
                df[field] = self._parse_list_field(df[field])
        
        return df
    
    def _clean_organ_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean organ-specific data."""
        # Standardize organ IDs
        if 'organ_id' in df.columns:
            df['organ_id'] = df['organ_id'].astype(str).str.strip()
        
        # Validate numeric fields
        numeric_fields = ['volume', 'mass', 'blood_flow', 'metabolic_rate']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        return df
    
    def _clean_pk_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean PK parameter data."""
        # Standardize drug IDs
        if 'drug_id' in df.columns:
            df['drug_id'] = df['drug_id'].astype(str).str.strip()
        
        # Validate numeric PK parameters
        numeric_fields = ['clearance', 'volume_distribution', 'half_life', 'bioavailability']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        return df
    
    def _parse_list_field(self, series: pd.Series) -> pd.Series:
        """Parse list-like fields from various formats."""
        def parse_value(val):
            if pd.isna(val) or val == '' or val == '[]':
                return []
            
            val_str = str(val).strip()
            
            # Try JSON parsing first
            try:
                parsed = json.loads(val_str)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if item]
            except:
                pass
            
            # Parse bracketed lists
            if val_str.startswith('[') and val_str.endswith(']'):
                try:
                    content = val_str[1:-1]
                    items = [item.strip().strip('"\'') for item in content.split(',')]
                    return [item for item in items if item]
                except:
                    pass
            
            # Split by common separators
            separators = [';', ',', '|', '\n', '/', '\\']
            for sep in separators:
                if sep in val_str:
                    items = [item.strip().strip('"\'') for item in val_str.split(sep)]
                    return [item for item in items if item]
            
            # Single value
            return [val_str] if val_str else []
        
        return series.apply(parse_value)
    
    def _generate_statistics(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Generate statistics for the ingested data."""
        stats = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': df.dtypes.to_dict(),
            'unique_counts': {}
        }
        
        # Missing data analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                stats['missing_data'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(df)) * 100
                }
        
        # Unique counts for key fields
        key_fields = {
            'drugs': ['drug_id', 'drug_name'],
            'diseases': ['disease_id', 'disease_name'],
            'organs': ['organ_id', 'organ_name'],
            'pk_parameters': ['drug_id', 'population_group']
        }
        
        for field in key_fields.get(data_type, []):
            if field in df.columns:
                stats['unique_counts'][field] = df[field].nunique()
        
        return stats
    
    def ingest_directory(self, directory_path: str, file_pattern: str = "*") -> Dict[str, Any]:
        """
        Ingest all files in a directory matching a pattern.
        
        Args:
            directory_path: Path to directory
            file_pattern: File pattern to match (e.g., "*.csv", "*.json")
            
        Returns:
            Dictionary with ingestion results for all files
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            return {'error': f'Directory not found: {directory_path}'}
        
        results = {}
        files_processed = 0
        files_failed = 0
        
        for file_path in directory.glob(file_pattern):
            if file_path.is_file():
                result = self.ingest_file(str(file_path))
                results[file_path.name] = result
                
                if 'error' in result:
                    files_failed += 1
                else:
                    files_processed += 1
        
        return {
            'directory': str(directory),
            'file_pattern': file_pattern,
            'files_processed': files_processed,
            'files_failed': files_failed,
            'results': results
        }

def main():
    """Main function to demonstrate smart data ingestion."""
    import os
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')

    # Initialize smart ingestion system
    ingestor = SmartDataIngestion()

    # Ingest from data/raw directory
    print("🔍 Ingesting data from data/raw directory...")
    real_data_results = ingestor.ingest_directory(os.path.join(project_root, 'data', 'raw'), "*.csv")

    # Ingest from results/simulations directory
    print("\n🔍 Ingesting data from results/simulations directory...")
    test_output_results = ingestor.ingest_directory(os.path.join(project_root, 'results', 'simulations'), "*.json")

    # Save results
    results_path = os.path.join(project_root, 'results', 'ingestion', 'smart_ingestion_results.json')
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            'real_data': real_data_results,
            'test_output': test_output_results
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ Smart ingestion completed!")
    print(f"📁 Real Data: {real_data_results['files_processed']} processed, {real_data_results['files_failed']} failed")
    print(f"📁 Test Output: {test_output_results['files_processed']} processed, {test_output_results['files_failed']} failed")
    print(f"💾 Results saved to: {results_path}")

    # Show sample mappings
    print("\n📋 Sample Column Mappings:")
    for file_name, result in test_output_results['results'].items():
        if 'column_mappings' in result and result['column_mappings']:
            print(f"\n📄 {file_name}:")
            for mapping in result['column_mappings'][:5]:  # Show first 5 mappings
                print(f"   {mapping['source']} → {mapping['target']} (confidence: {mapping['confidence']:.2f})")
            break

if __name__ == "__main__":
    main()
