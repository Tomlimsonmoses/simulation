"""
Comprehensive PK-PD Knowledge Base Integration System

This module integrates multiple data sources with different column naming conventions
to create a complete knowledge base for drug-biological system simulation.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataMapping:
    """Mapping configuration for different data sources."""
    source_name: str
    file_path: str
    data_type: str
    column_mappings: Dict[str, str]  # Maps source columns to standard columns
    required_columns: List[str]
    optional_columns: List[str]
    data_transformations: Dict[str, str] = None

class PKPDKnowledgeBase:
    """
    Comprehensive PK-PD knowledge base that integrates multiple data sources
    with different column naming conventions and creates a unified structure
    for drug-biological system simulation.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the knowledge base.
        
        Args:
            base_dir: Base directory containing data sources
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.knowledge_base = {
            'drugs': {},
            'diseases': {},
            'organs': {},
            'pk_parameters': {},
            'pd_responses': {},
            'biomarkers': {},
            'clinical_trials': {},
            'population_data': {},
            'drug_organ_interactions': {},
            'drug_disease_interactions': {},
            'organ_disease_interactions': {},
            'drug_combinations': {},
            'side_effects': {}
        }
        
        # Standard column definitions
        self.standard_columns = {
            'drugs': [
                'drug_id', 'drug_name', 'drugbank_id', 'molecular_weight', 'logp',
                'solubility', 'bioavailability', 'protein_binding', 'clearance_rate',
                'volume_distribution', 'half_life', 'absorption_rate', 'metabolism_rate',
                'excretion_rate', 'drug_class', 'administration_route', 'smiles',
                'targets', 'mechanism_of_action', 'description', 'categories'
            ],
            'diseases': [
                'disease_id', 'disease_name', 'mesh_id', 'description', 'category',
                'symptoms', 'pathways', 'affected_organs', 'severity', 'prevalence',
                'biomarkers', 'genetic_factors'
            ],
            'organs': [
                'organ_id', 'organ_name', 'organ_type', 'volume', 'mass', 'blood_flow',
                'oxygen_consumption', 'metabolic_rate', 'ph', 'temperature',
                'surface_area', 'water_content', 'lipid_content', 'protein_content',
                'health_status', 'function_score'
            ],
            'pk_parameters': [
                'drug_id', 'population_group', 'clearance', 'volume_distribution',
                'absorption_rate', 'elimination_rate', 'bioavailability',
                'protein_binding', 'half_life', 'time_to_peak', 'auc', 'cmax',
                'clearance_cv', 'volume_cv', 'absorption_cv'
            ],
            'pd_responses': [
                'drug_id', 'disease_id', 'biomarker_id', 'response_type',
                'response_magnitude', 'time_to_response', 'duration_of_effect',
                'dose_response_curve', 'ec50', 'emax', 'hill_coefficient'
            ]
        }
        
        # Initialize data mappings
        self._initialize_data_mappings()
        
    def _initialize_data_mappings(self):
        """Initialize mappings for different data sources."""
        self.data_mappings = [
            # Real data mappings
            DataMapping(
                source_name="real_drug_properties",
                file_path="data/raw/drug_properties.csv",
                data_type="drugs",
                column_mappings={
                    'drug_id': 'drug_id',
                    'drug_name': 'drug_name',
                    'molecular_weight': 'molecular_weight',
                    'logp': 'logp',
                    'solubility': 'solubility',
                    'bioavailability': 'bioavailability',
                    'protein_binding': 'protein_binding',
                    'clearance_rate': 'clearance_rate',
                    'volume_distribution': 'volume_distribution',
                    'half_life': 'half_life',
                    'absorption_rate': 'absorption_rate',
                    'excretion_rate': 'excretion_rate',
                    'drug_class': 'drug_class',
                    'administration_route': 'administration_route'
                },
                required_columns=['drug_id', 'drug_name'],
                optional_columns=['molecular_weight', 'logp', 'bioavailability']
            ),
            
            DataMapping(
                source_name="real_pk_parameters",
                file_path="data/raw/pk_parameters.csv",
                data_type="pk_parameters",
                column_mappings={
                    'drug_id': 'drug_id',
                    'population_group': 'population_group',
                    'clearance': 'clearance',
                    'volume_distribution': 'volume_distribution',
                    'absorption_rate': 'absorption_rate',
                    'elimination_rate': 'elimination_rate',
                    'bioavailability': 'bioavailability',
                    'protein_binding': 'protein_binding',
                    'half_life': 'half_life',
                    'time_to_peak': 'time_to_peak',
                    'auc': 'auc',
                    'cmax': 'cmax'
                },
                required_columns=['drug_id', 'population_group'],
                optional_columns=['clearance', 'volume_distribution', 'bioavailability']
            ),
            
            DataMapping(
                source_name="real_organ_network",
                file_path="data/raw/organ_network.csv",
                data_type="organs",
                column_mappings={
                    'organ_id': 'organ_id',
                    'organ_name': 'organ_name',
                    'organ_type': 'organ_type',
                    'volume': 'volume',
                    'mass': 'mass',
                    'blood_flow': 'blood_flow',
                    'oxygen_consumption': 'oxygen_consumption',
                    'metabolic_rate': 'metabolic_rate',
                    'ph': 'ph',
                    'temperature': 'temperature',
                    'surface_area': 'surface_area',
                    'water_content': 'water_content',
                    'lipid_content': 'lipid_content',
                    'protein_content': 'protein_content',
                    'health_status': 'health_status'
                },
                required_columns=['organ_id', 'organ_name'],
                optional_columns=['volume', 'mass', 'blood_flow']
            ),
            
            DataMapping(
                source_name="real_pd_responses",
                file_path="data/raw/pd_responses.csv",
                data_type="pd_responses",
                column_mappings={
                    'drug_id': 'drug_id',
                    'disease_id': 'disease_id',
                    'biomarker_id': 'biomarker_id',
                    'response_type': 'response_type',
                    'response_magnitude': 'response_magnitude',
                    'time_to_response': 'time_to_response',
                    'duration_of_effect': 'duration_of_effect',
                    'dose_response_curve': 'dose_response_curve',
                    'ec50': 'ec50',
                    'emax': 'emax',
                    'hill_coefficient': 'hill_coefficient'
                },
                required_columns=['drug_id', 'response_type'],
                optional_columns=['response_magnitude', 'time_to_response']
            ),
            
            # Ingested data mappings
            DataMapping(
                source_name="ingested_drugs",
                file_path="results/simulations/drugs_consolidated.json",
                data_type="drugs",
                column_mappings={
                    'DrugID': 'drug_id',
                    'DrugName': 'drug_name',
                    'DrugDescription': 'description',
                    'DrugSmile': 'smiles',
                    'DrugTarget': 'targets',
                    'DrugMechanism': 'mechanism_of_action',
                    'DrugCategories': 'categories'
                },
                required_columns=['drug_id', 'drug_name'],
                optional_columns=['description', 'smiles', 'targets']
            ),
            
            DataMapping(
                source_name="ingested_diseases",
                file_path="results/simulations/diseases_consolidated.json",
                data_type="diseases",
                column_mappings={
                    'DiseaseID': 'disease_id',
                    'DiseaseName': 'disease_name',
                    'DiseaseDescription': 'description',
                    'SlimMapping': 'category',
                    'PathwayNames': 'pathways'
                },
                required_columns=['disease_id', 'disease_name'],
                optional_columns=['description', 'category', 'pathways']
            ),
            
            DataMapping(
                source_name="ingested_mappings",
                file_path="results/simulations/mappings_consolidated.json",
                data_type="drug_disease_interactions",
                column_mappings={
                    'DrugID': 'drug_id',
                    'DiseaseID': 'disease_id',
                    'RelationshipType': 'relationship_type',
                    'Confidence': 'confidence'
                },
                required_columns=['drug_id', 'disease_id'],
                optional_columns=['relationship_type', 'confidence']
            )
        ]
    
    def load_data_source(self, mapping: DataMapping) -> pd.DataFrame:
        """
        Load data from a source with column mapping and validation.
        
        Args:
            mapping: DataMapping configuration
            
        Returns:
            Standardized DataFrame
        """
        try:
            file_path = self.base_dir / mapping.file_path
            
            if not file_path.exists():
                logger.warning(f"Data file not found: {file_path}")
                return pd.DataFrame()
            
            # Load data based on file extension
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'data' in data:
                        df = pd.DataFrame(data['data'])
                    else:
                        df = pd.DataFrame(data)
            else:
                df = pd.read_csv(file_path, encoding='utf-8')
            
            # Apply column mappings
            df_mapped = self._apply_column_mappings(df, mapping)
            
            # Validate required columns
            missing_cols = [col for col in mapping.required_columns 
                          if col not in df_mapped.columns]
            if missing_cols:
                logger.warning(f"Missing required columns in {mapping.source_name}: {missing_cols}")
            
            # Apply data transformations
            df_transformed = self._apply_data_transformations(df_mapped, mapping)
            
            logger.info(f"Loaded {len(df_transformed)} records from {mapping.source_name}")
            return df_transformed
            
        except Exception as e:
            logger.error(f"Error loading {mapping.source_name}: {str(e)}")
            return pd.DataFrame()
    
    def _apply_column_mappings(self, df: pd.DataFrame, mapping: DataMapping) -> pd.DataFrame:
        """Apply column mappings to standardize column names."""
        df_mapped = df.copy()
        
        # Apply column name mappings
        for source_col, target_col in mapping.column_mappings.items():
            if source_col in df.columns:
                df_mapped = df_mapped.rename(columns={source_col: target_col})
        
        return df_mapped
    
    def _apply_data_transformations(self, df: pd.DataFrame, mapping: DataMapping) -> pd.DataFrame:
        """Apply data transformations and cleaning."""
        df_transformed = df.copy()
        
        # Clean and standardize IDs
        if 'drug_id' in df_transformed.columns:
            df_transformed['drug_id'] = df_transformed['drug_id'].astype(str).str.strip()
        
        if 'disease_id' in df_transformed.columns:
            df_transformed['disease_id'] = df_transformed['disease_id'].astype(str).str.strip()
        
        if 'organ_id' in df_transformed.columns:
            df_transformed['organ_id'] = df_transformed['organ_id'].astype(str).str.strip()
        
        # Clean text fields
        for col in ['drug_name', 'disease_name', 'organ_name']:
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].astype(str).str.strip()
        
        # Convert list-like fields
        list_fields = ['targets', 'categories', 'pathways', 'symptoms']
        for field in list_fields:
            if field in df_transformed.columns:
                df_transformed[field] = self._parse_list_field(df_transformed[field])
        
        return df_transformed
    
    def _parse_list_field(self, series: pd.Series) -> pd.Series:
        """Parse list-like fields from string representations."""
        def parse_value(val):
            if pd.isna(val) or val == '':
                return []
            if isinstance(val, str):
                # Try to parse as JSON list first
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    pass
                
                # Try to parse bracketed lists
                if val.startswith('[') and val.endswith(']'):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, list):
                            return parsed
                    except:
                        pass
                
                # Split by common separators
                separators = [';', ',', '|', '\n']
                for sep in separators:
                    if sep in val:
                        return [item.strip() for item in val.split(sep) if item.strip()]
            
            return [str(val)] if val is not None else []
        
        return series.apply(parse_value)
    
    def integrate_all_data_sources(self) -> Dict[str, Any]:
        """
        Integrate all data sources into a comprehensive knowledge base.
        
        Returns:
            Integrated knowledge base dictionary
        """
        logger.info("Starting comprehensive data integration...")
        
        # Load all data sources
        for mapping in self.data_mappings:
            df = self.load_data_source(mapping)
            if not df.empty:
                self._integrate_data_source(df, mapping)
        
        # Create derived relationships
        self._create_derived_relationships()
        
        # Generate comprehensive statistics
        self._generate_knowledge_base_stats()
        
        logger.info("Data integration completed successfully!")
        return self.knowledge_base
    
    def _integrate_data_source(self, df: pd.DataFrame, mapping: DataMapping):
        """Integrate a single data source into the knowledge base."""
        data_type = mapping.data_type
        
        if data_type not in self.knowledge_base:
            self.knowledge_base[data_type] = {}
        
        # Convert DataFrame to dictionary format
        records = df.to_dict('records')
        
        # Merge with existing data
        if mapping.source_name not in self.knowledge_base[data_type]:
            self.knowledge_base[data_type][mapping.source_name] = records
        else:
            # Merge records, avoiding duplicates
            existing_ids = set()
            if data_type in ['drugs', 'diseases', 'organs']:
                id_field = f"{data_type.rstrip('s')}_id"
                existing_ids = {record.get(id_field) for record in 
                              self.knowledge_base[data_type][mapping.source_name]}
            
            new_records = [record for record in records 
                          if record.get(id_field) not in existing_ids]
            self.knowledge_base[data_type][mapping.source_name].extend(new_records)
    
    def _create_derived_relationships(self):
        """Create derived relationships between different data types."""
        logger.info("Creating derived relationships...")
        
        # Drug-Organ interactions (based on PK parameters)
        self._create_drug_organ_interactions()
        
        # Organ-Disease interactions (based on affected organs)
        self._create_organ_disease_interactions()
        
        # Drug combination analysis
        self._analyze_drug_combinations()
    
    def _create_drug_organ_interactions(self):
        """Create drug-organ interactions based on PK parameters."""
        drug_organ_interactions = []
        
        # Get all drugs and organs
        all_drugs = self._get_all_entities('drugs')
        all_organs = self._get_all_entities('organs')
        
        for drug in all_drugs:
            drug_id = drug.get('drug_id')
            
            # Create interactions based on PK parameters
            pk_params = self._get_drug_pk_parameters(drug_id)
            if pk_params:
                # High clearance organs (liver, kidneys)
                if pk_params.get('clearance', 0) > 5.0:
                    drug_organ_interactions.append({
                        'drug_id': drug_id,
                        'organ_id': 'ORG_02',  # liver
                        'interaction_type': 'metabolism',
                        'strength': 'high',
                        'clearance_contribution': 0.6
                    })
                    drug_organ_interactions.append({
                        'drug_id': drug_id,
                        'organ_id': 'ORG_03',  # kidneys
                        'interaction_type': 'excretion',
                        'strength': 'medium',
                        'clearance_contribution': 0.3
                    })
                
                # High volume distribution (muscle, adipose)
                if pk_params.get('volume_distribution', 0) > 2.0:
                    drug_organ_interactions.append({
                        'drug_id': drug_id,
                        'organ_id': 'ORG_05',  # muscle
                        'interaction_type': 'distribution',
                        'strength': 'high',
                        'volume_contribution': 0.4
                    })
                    drug_organ_interactions.append({
                        'drug_id': drug_id,
                        'organ_id': 'ORG_06',  # adipose
                        'interaction_type': 'distribution',
                        'strength': 'medium',
                        'volume_contribution': 0.3
                    })
        
        self.knowledge_base['drug_organ_interactions']['derived'] = drug_organ_interactions
    
    def _create_organ_disease_interactions(self):
        """Create organ-disease interactions based on disease data."""
        organ_disease_interactions = []
        
        all_diseases = self._get_all_entities('diseases')
        all_organs = self._get_all_entities('organs')
        
        for disease in all_diseases:
            disease_id = disease.get('disease_id')
            disease_name = disease.get('disease_name', '').lower()
            affected_organs = disease.get('affected_organs', [])
            
            # Map disease keywords to organs
            organ_keywords = {
                'brain': ['brain', 'neuro', 'cognitive', 'memory', 'alzheimer', 'parkinson'],
                'heart': ['heart', 'cardiac', 'cardio', 'myocardial', 'angina'],
                'liver': ['liver', 'hepatic', 'cirrhosis', 'hepatitis'],
                'kidney': ['kidney', 'renal', 'nephro', 'dialysis'],
                'lung': ['lung', 'pulmonary', 'respiratory', 'asthma', 'copd'],
                'stomach': ['stomach', 'gastric', 'ulcer', 'digestive'],
                'skin': ['skin', 'dermal', 'dermatitis', 'rash']
            }
            
            for organ_id, keywords in organ_keywords.items():
                if any(keyword in disease_name for keyword in keywords):
                    organ_disease_interactions.append({
                        'disease_id': disease_id,
                        'organ_id': organ_id,
                        'interaction_type': 'affected',
                        'severity': 'moderate',
                        'confidence': 0.8
                    })
        
        self.knowledge_base['organ_disease_interactions']['derived'] = organ_disease_interactions
    
    def _analyze_drug_combinations(self):
        """Analyze potential drug combinations and interactions."""
        drug_combinations = []
        
        all_drugs = self._get_all_entities('drugs')
        
        # Analyze combinations based on drug classes and targets
        for i, drug1 in enumerate(all_drugs):
            for drug2 in all_drugs[i+1:]:
                drug1_id = drug1.get('drug_id')
                drug2_id = drug2.get('drug_id')
                
                # Check for potential interactions
                interaction = self._assess_drug_interaction(drug1, drug2)
                if interaction:
                    drug_combinations.append({
                        'drug1_id': drug1_id,
                        'drug2_id': drug2_id,
                        'interaction_type': interaction['type'],
                        'severity': interaction['severity'],
                        'mechanism': interaction['mechanism'],
                        'confidence': interaction['confidence']
                    })
        
        self.knowledge_base['drug_combinations']['derived'] = drug_combinations
    
    def _assess_drug_interaction(self, drug1: Dict, drug2: Dict) -> Optional[Dict]:
        """Assess potential interaction between two drugs."""
        interaction = None
        
        # Check for overlapping targets (potential competition)
        targets1 = set(drug1.get('targets', []))
        targets2 = set(drug2.get('targets', []))
        
        if targets1 & targets2:
            interaction = {
                'type': 'target_competition',
                'severity': 'moderate',
                'mechanism': 'shared_target_binding',
                'confidence': 0.7
            }
        
        # Check for complementary mechanisms (potential synergy)
        mech1 = drug1.get('mechanism_of_action', '').lower()
        mech2 = drug2.get('mechanism_of_action', '').lower()
        
        synergy_keywords = ['anti-inflammatory', 'antioxidant', 'vasodilator']
        if any(keyword in mech1 for keyword in synergy_keywords) and \
           any(keyword in mech2 for keyword in synergy_keywords):
            interaction = {
                'type': 'synergistic',
                'severity': 'beneficial',
                'mechanism': 'complementary_mechanisms',
                'confidence': 0.6
            }
        
        return interaction
    
    def _get_all_entities(self, entity_type: str) -> List[Dict]:
        """Get all entities of a specific type from all sources."""
        entities = []
        if entity_type in self.knowledge_base:
            for source, records in self.knowledge_base[entity_type].items():
                entities.extend(records)
        return entities
    
    def _get_drug_pk_parameters(self, drug_id: str) -> Optional[Dict]:
        """Get PK parameters for a specific drug."""
        if 'pk_parameters' in self.knowledge_base:
            for source, records in self.knowledge_base['pk_parameters'].items():
                for record in records:
                    if record.get('drug_id') == drug_id:
                        return record
        return None
    
    def _generate_knowledge_base_stats(self):
        """Generate comprehensive statistics for the knowledge base."""
        stats = {
            'total_entities': {},
            'data_sources': {},
            'relationships': {},
            'coverage_analysis': {}
        }
        
        # Count entities by type
        for entity_type, sources in self.knowledge_base.items():
            if isinstance(sources, dict):
                total_count = sum(len(records) for records in sources.values())
                stats['total_entities'][entity_type] = total_count
                stats['data_sources'][entity_type] = {
                    source: len(records) for source, records in sources.items()
                }
        
        # Count relationships
        for rel_type in ['drug_organ_interactions', 'organ_disease_interactions', 
                         'drug_disease_interactions', 'drug_combinations']:
            if rel_type in self.knowledge_base:
                total_rels = sum(len(records) for records in self.knowledge_base[rel_type].values())
                stats['relationships'][rel_type] = total_rels
        
        self.knowledge_base['statistics'] = stats
    
    def export_knowledge_base(self, output_path: str = None) -> str:
        """
        Export the integrated knowledge base to JSON file.
        
        Args:
            output_path: Path to save the knowledge base
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = self.base_dir / "pk_pd_knowledge_base.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Knowledge base exported to: {output_path}")
        return str(output_path)
    
    def get_simulation_ready_data(self) -> Dict[str, Any]:
        """
        Get data formatted specifically for PK-PD simulation models.
        
        Returns:
            Simulation-ready data structure
        """
        simulation_data = {
            'drugs': self._prepare_drugs_for_simulation(),
            'diseases': self._prepare_diseases_for_simulation(),
            'organs': self._prepare_organs_for_simulation(),
            'pk_parameters': self._prepare_pk_parameters_for_simulation(),
            'pd_responses': self._prepare_pd_responses_for_simulation(),
            'interactions': {
                'drug_organ': self.knowledge_base.get('drug_organ_interactions', {}).get('derived', []),
                'organ_disease': self.knowledge_base.get('organ_disease_interactions', {}).get('derived', []),
                'drug_disease': self.knowledge_base.get('drug_disease_interactions', {}).get('ingested_mappings', [])
            },
            'combinations': self.knowledge_base.get('drug_combinations', {}).get('derived', [])
        }
        
        return simulation_data
    
    def _prepare_drugs_for_simulation(self) -> List[Dict]:
        """Prepare drug data for simulation."""
        all_drugs = self._get_all_entities('drugs')
        
        # Merge drug properties with PK parameters
        for drug in all_drugs:
            drug_id = drug.get('drug_id')
            pk_params = self._get_drug_pk_parameters(drug_id)
            if pk_params:
                # Merge PK parameters into drug record
                for key, value in pk_params.items():
                    if key not in drug:
                        drug[f'pk_{key}'] = value
        
        return all_drugs
    
    def _prepare_diseases_for_simulation(self) -> List[Dict]:
        """Prepare disease data for simulation."""
        return self._get_all_entities('diseases')
    
    def _prepare_organs_for_simulation(self) -> List[Dict]:
        """Prepare organ data for simulation."""
        return self._get_all_entities('organs')
    
    def _prepare_pk_parameters_for_simulation(self) -> List[Dict]:
        """Prepare PK parameters for simulation."""
        return self._get_all_entities('pk_parameters')
    
    def _prepare_pd_responses_for_simulation(self) -> List[Dict]:
        """Prepare PD responses for simulation."""
        return self._get_all_entities('pd_responses')

def main():
    """Main function to build the comprehensive PK-PD knowledge base."""
    # Initialize knowledge base
    kb = PKPDKnowledgeBase(base_dir="c:/Users/sparta/WORK/projects/MEDAI/PK-PD")
    
    # Integrate all data sources
    kb.integrate_all_data_sources()
    
    # Export knowledge base
    kb_path = kb.export_knowledge_base()
    
    # Export simulation-ready data
    simulation_data = kb.get_simulation_ready_data()
    sim_path = "c:/Users/sparta/WORK/projects/MEDAI/PK-PD/simulation_ready_data.json"
    with open(sim_path, 'w', encoding='utf-8') as f:
        json.dump(simulation_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Comprehensive PK-PD Knowledge Base created!")
    print(f"📁 Knowledge Base: {kb_path}")
    print(f"🎯 Simulation Data: {sim_path}")
    
    # Print statistics
    if 'statistics' in kb.knowledge_base:
        stats = kb.knowledge_base['statistics']
        print("\n📊 Knowledge Base Statistics:")
        print(f"   Total Drugs: {stats['total_entities'].get('drugs', 0)}")
        print(f"   Total Diseases: {stats['total_entities'].get('diseases', 0)}")
        print(f"   Total Organs: {stats['total_entities'].get('organs', 0)}")
        print(f"   PK Parameters: {stats['total_entities'].get('pk_parameters', 0)}")
        print(f"   PD Responses: {stats['total_entities'].get('pd_responses', 0)}")
        print(f"   Drug-Organ Interactions: {stats['relationships'].get('drug_organ_interactions', 0)}")
        print(f"   Drug-Disease Interactions: {stats['relationships'].get('drug_disease_interactions', 0)}")
        print(f"   Drug Combinations: {stats['relationships'].get('drug_combinations', 0)}")

if __name__ == "__main__":
    main()
