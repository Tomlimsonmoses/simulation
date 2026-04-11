"""
PK-PD Simulation System - Complete Integration and Usage Guide

This script demonstrates how to use the complete PK-PD knowledge base and training data
system for drug-biological system simulation.
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PKPDSimulationSystem:
    """
    Complete PK-PD simulation system that integrates knowledge base and training data
    for drug-biological system modeling.
    """
    
    def __init__(self):
        """Initialize the PK-PD simulation system."""
        self.base_dir = Path.cwd()
        self.knowledge_base_path = self.base_dir / "pk_pd_knowledge_base.json"
        self.training_data_path = self.base_dir / "simple_pk_pd_training_data"
        self.simulation_data_path = self.base_dir / "simulation_ready_data.json"
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
        # Load training data
        self.training_data = self._load_training_data()
        
        logger.info("PK-PD Simulation System initialized successfully!")
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load the integrated PK-PD knowledge base."""
        if self.knowledge_base_path.exists():
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Knowledge base not found: {self.knowledge_base_path}")
            return {}
    
    def _load_training_data(self) -> Dict[str, Any]:
        """Load the prepared training data."""
        pickle_path = self.training_data_path / "training_data.pkl"
        
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    logger.info(f"Successfully loaded training data: {type(data)}")
                    return data
            except Exception as e:
                logger.error(f"Error loading training data: {str(e)}")
                return {}
        else:
            logger.warning(f"Training data not found: {pickle_path}")
            return {}
    
    def get_knowledge_base_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the knowledge base."""
        if not self.knowledge_base:
            return {"error": "Knowledge base not loaded"}
        
        summary = {
            "total_entities": {},
            "data_sources": {},
            "relationships": {},
            "coverage_analysis": {}
        }
        
        # Count entities by type
        for entity_type, sources in self.knowledge_base.items():
            if isinstance(sources, dict):
                total_count = sum(len(records) for records in sources.values())
                summary["total_entities"][entity_type] = total_count
                summary["data_sources"][entity_type] = {
                    source: len(records) for source, records in sources.items()
                }
        
        # Count relationships
        for rel_type in ['drug_organ_interactions', 'organ_disease_interactions', 
                         'drug_disease_interactions', 'drug_combinations']:
            if rel_type in self.knowledge_base:
                total_rels = sum(len(records) for records in self.knowledge_base[rel_type].values())
                summary["relationships"][rel_type] = total_rels
        
        return summary
    
    def get_drug_information(self, drug_id: str) -> Dict[str, Any]:
        """Get comprehensive information about a specific drug."""
        drug_info = {
            "drug_id": drug_id,
            "properties": {},
            "pk_parameters": {},
            "pd_responses": {},
            "interactions": {},
            "clinical_data": {}
        }
        
        # Get drug properties
        if 'drugs' in self.knowledge_base:
            for source, drugs in self.knowledge_base['drugs'].items():
                for drug in drugs:
                    if str(drug.get('drug_id', '')).lower() == str(drug_id).lower():
                        drug_info["properties"] = drug
                        break
        
        # Get PK parameters
        if 'pk_parameters' in self.knowledge_base:
            for source, params in self.knowledge_base['pk_parameters'].items():
                for param in params:
                    if str(param.get('drug_id', '')).lower() == str(drug_id).lower():
                        drug_info["pk_parameters"] = param
                        break
        
        # Get PD responses
        if 'pd_responses' in self.knowledge_base:
            for source, responses in self.knowledge_base['pd_responses'].items():
                for response in responses:
                    if str(response.get('drug_id', '')).lower() == str(drug_id).lower():
                        drug_info["pd_responses"] = response
                        break
        
        # Get interactions
        if 'drug_disease_interactions' in self.knowledge_base:
            interactions = []
            for source, inters in self.knowledge_base['drug_disease_interactions'].items():
                for inter in inters:
                    if str(inter.get('drug_id', '')).lower() == str(drug_id).lower():
                        interactions.append(inter)
            drug_info["interactions"]["drug_disease"] = interactions
        
        return drug_info
    
    def get_disease_information(self, disease_id: str) -> Dict[str, Any]:
        """Get comprehensive information about a specific disease."""
        disease_info = {
            "disease_id": disease_id,
            "characteristics": {},
            "affected_organs": [],
            "related_drugs": [],
            "biomarkers": []
        }
        
        # Get disease characteristics
        if 'diseases' in self.knowledge_base:
            for source, diseases in self.knowledge_base['diseases'].items():
                for disease in diseases:
                    if str(disease.get('disease_id', '')).lower() == str(disease_id).lower():
                        disease_info["characteristics"] = disease
                        break
        
        # Get related drugs
        if 'drug_disease_interactions' in self.knowledge_base:
            related_drugs = []
            for source, inters in self.knowledge_base['drug_disease_interactions'].items():
                for inter in inters:
                    if str(inter.get('disease_id', '')).lower() == str(disease_id).lower():
                        related_drugs.append(inter)
            disease_info["related_drugs"] = related_drugs
        
        return disease_info
    
    def get_organ_information(self, organ_id: str) -> Dict[str, Any]:
        """Get comprehensive information about a specific organ."""
        organ_info = {
            "organ_id": organ_id,
            "characteristics": {},
            "related_diseases": [],
            "drug_effects": []
        }
        
        # Get organ characteristics
        if 'organs' in self.knowledge_base:
            for source, organs in self.knowledge_base['organs'].items():
                for organ in organs:
                    if str(organ.get('organ_id', '')).lower() == str(organ_id).lower():
                        organ_info["characteristics"] = organ
                        break
        
        return organ_info
    
    def list_available_drugs(self, limit: int = 10) -> List[str]:
        """List available drugs in the knowledge base."""
        drugs = []
        
        if 'drugs' in self.knowledge_base:
            for source, drug_list in self.knowledge_base['drugs'].items():
                for drug in drug_list:
                    drug_id = drug.get('drug_id')
                    drug_name = drug.get('drug_name', 'Unknown')
                    if drug_id:
                        drugs.append(f"{drug_id}: {drug_name}")
        
        return drugs[:limit]
    
    def list_available_diseases(self, limit: int = 10) -> List[str]:
        """List available diseases in the knowledge base."""
        diseases = []
        
        if 'diseases' in self.knowledge_base:
            for source, disease_list in self.knowledge_base['diseases'].items():
                for disease in disease_list:
                    disease_id = disease.get('disease_id')
                    disease_name = disease.get('disease_name', 'Unknown')
                    if disease_id:
                        diseases.append(f"{disease_id}: {disease_name}")
        
        return diseases[:limit]
    
    def list_available_organs(self) -> List[str]:
        """List available organs in the knowledge base."""
        organs = []
        
        if 'organs' in self.knowledge_base:
            for source, organ_list in self.knowledge_base['organs'].items():
                for organ in organ_list:
                    organ_id = organ.get('organ_id')
                    organ_name = organ.get('organ_name', 'Unknown')
                    if organ_id:
                        organs.append(f"{organ_id}: {organ_name}")
        
        return organs
    
    def get_training_dataset_info(self) -> Dict[str, Any]:
        """Get information about available training datasets."""
        if not self.training_data:
            return {"error": "Training data not loaded"}
        
        datasets_info = {}
        
        for dataset_name, dataset_info in self.training_data.items():
            if 'data' in dataset_info:
                datasets_info[dataset_name] = {
                    "total_samples": len(dataset_info['data']),
                    "features": len(dataset_info.get('features', [])),
                    "targets": dataset_info.get('target_columns', []),
                    "has_splits": 'splits' in dataset_info,
                    "metadata": dataset_info.get('metadata', {})
                }
        
        return datasets_info
    
    def load_training_dataset(self, dataset_name: str, split: str = 'train') -> Optional[pd.DataFrame]:
        """Load a specific training dataset split."""
        if dataset_name not in self.training_data:
            logger.error(f"Dataset '{dataset_name}' not found")
            return None
        
        dataset_info = self.training_data[dataset_name]
        
        if 'splits' not in dataset_info:
            logger.error(f"No splits available for dataset '{dataset_name}'")
            return None
        
        if split not in dataset_info['splits']:
            logger.error(f"Split '{split}' not available for dataset '{dataset_name}'")
            return None
        
        return dataset_info['splits'][split]['data']
    
    def analyze_drug_properties(self) -> Dict[str, Any]:
        """Analyze drug properties in the knowledge base."""
        if 'drugs' not in self.knowledge_base:
            return {"error": "No drug data available"}
        
        # Collect all drugs
        all_drugs = []
        for source, drugs in self.knowledge_base['drugs'].items():
            all_drugs.extend(drugs)
        
        df = pd.DataFrame(all_drugs)
        
        analysis = {
            "total_drugs": len(df),
            "columns": list(df.columns),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            analysis["numeric_summary"][col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "missing": df[col].isnull().sum()
            }
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            analysis["categorical_summary"][col] = {
                "unique_count": df[col].nunique(),
                "top_values": df[col].value_counts().head().to_dict(),
                "missing": df[col].isnull().sum()
            }
        
        return analysis
    
    def analyze_pk_parameters(self) -> Dict[str, Any]:
        """Analyze PK parameters in the knowledge base."""
        if 'pk_parameters' not in self.knowledge_base:
            return {"error": "No PK data available"}
        
        # Collect all PK parameters
        all_pk = []
        for source, params in self.knowledge_base['pk_parameters'].items():
            all_pk.extend(params)
        
        df = pd.DataFrame(all_pk)
        
        analysis = {
            "total_records": len(df),
            "population_groups": df['population_group'].unique().tolist() if 'population_group' in df.columns else [],
            "parameters": {}
        }
        
        # Analyze key PK parameters
        key_params = ['clearance', 'volume_distribution', 'half_life', 'bioavailability']
        for param in key_params:
            if param in df.columns:
                analysis["parameters"][param] = {
                    "mean": df[param].mean(),
                    "std": df[param].std(),
                    "range": [df[param].min(), df[param].max()],
                    "quartiles": df[param].quantile([0.25, 0.5, 0.75]).to_dict()
                }
        
        return analysis
    
    def generate_simulation_report(self) -> str:
        """Generate a comprehensive simulation system report."""
        report = []
        report.append("=" * 60)
        report.append("PK-PD SIMULATION SYSTEM REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Knowledge base summary
        kb_summary = self.get_knowledge_base_summary()
        if "error" not in kb_summary:
            report.append("📊 KNOWLEDGE BASE SUMMARY:")
            report.append(f"Total Drugs: {kb_summary['total_entities'].get('drugs', 0)}")
            report.append(f"Total Diseases: {kb_summary['total_entities'].get('diseases', 0)}")
            report.append(f"Total Organs: {kb_summary['total_entities'].get('organs', 0)}")
            report.append(f"PK Parameters: {kb_summary['total_entities'].get('pk_parameters', 0)}")
            report.append(f"PD Responses: {kb_summary['total_entities'].get('pd_responses', 0)}")
            report.append("")
            
            report.append("🔗 RELATIONSHIPS:")
            for rel_type, count in kb_summary['relationships'].items():
                report.append(f"{rel_type}: {count}")
            report.append("")
        
        # Training data summary
        training_info = self.get_training_dataset_info()
        if "error" not in training_info:
            report.append("🎯 TRAINING DATASETS:")
            for dataset_name, info in training_info.items():
                report.append(f"{dataset_name}:")
                report.append(f"  Samples: {info['total_samples']}")
                report.append(f"  Features: {info['features']}")
                report.append(f"  Targets: {', '.join(info['targets'])}")
                report.append("")
        
        # Sample data
        report.append("📋 SAMPLE DATA:")
        drugs = self.list_available_drugs(5)
        if drugs:
            report.append("Sample Drugs:")
            for drug in drugs:
                report.append(f"  {drug}")
            report.append("")
        
        diseases = self.list_available_diseases(5)
        if diseases:
            report.append("Sample Diseases:")
            for disease in diseases:
                report.append(f"  {disease}")
            report.append("")
        
        organs = self.list_available_organs()
        if organs:
            report.append("Available Organs:")
            for organ in organs:
                report.append(f"  {organ}")
            report.append("")
        
        report.append("=" * 60)
        report.append("SYSTEM READY FOR SIMULATION MODEL TRAINING")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_sample_data_for_modeling(self, output_dir: str = "modeling_samples") -> str:
        """Export sample datasets ready for machine learning."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        samples = {}
        
        # Export drug properties sample
        drug_data = self.load_training_dataset('drug_properties', 'train')
        if drug_data is not None:
            drug_sample = drug_data.head(100)  # First 100 samples
            drug_sample.to_csv(output_path / "drug_properties_sample.csv", index=False)
            samples["drug_properties"] = len(drug_sample)
        
        # Export PK parameters sample
        pk_data = self.load_training_dataset('pk_parameters', 'train')
        if pk_data is not None:
            pk_sample = pk_data.head(100)
            pk_sample.to_csv(output_path / "pk_parameters_sample.csv", index=False)
            samples["pk_parameters"] = len(pk_sample)
        
        # Export clinical outcomes sample
        clinical_data = self.load_training_dataset('clinical_outcomes', 'train')
        if clinical_data is not None:
            clinical_sample = clinical_data.head(100)
            clinical_sample.to_csv(output_path / "clinical_outcomes_sample.csv", index=False)
            samples["clinical_outcomes"] = len(clinical_sample)
        
        # Create summary
        summary = {
            "export_timestamp": pd.Timestamp.now().isoformat(),
            "samples_exported": samples,
            "output_directory": str(output_path)
        }
        
        with open(output_path / "export_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Sample data exported to: {output_path}")
        return str(output_path)

def main():
    """Main function to demonstrate the PK-PD simulation system."""
    print("🚀 Initializing PK-PD Simulation System...")
    
    # Initialize the system
    system = PKPDSimulationSystem()
    
    # Generate comprehensive report
    print("\n" + system.generate_simulation_report())
    
    # Show sample drug information
    print("\n🔍 SAMPLE DRUG INFORMATION:")
    drugs = system.list_available_drugs(3)
    if drugs:
        first_drug_id = drugs[0].split(':')[0]
        drug_info = system.get_drug_information(first_drug_id)
        print(f"Drug ID: {drug_info['drug_id']}")
        if drug_info['properties']:
            print(f"Name: {drug_info['properties'].get('drug_name', 'N/A')}")
            print(f"Class: {drug_info['properties'].get('drug_class', 'N/A')}")
        if drug_info['pk_parameters']:
            print(f"Clearance: {drug_info['pk_parameters'].get('clearance', 'N/A')}")
            print(f"Half-life: {drug_info['pk_parameters'].get('half_life', 'N/A')}")
    
    # Show training dataset info
    print("\n🎯 TRAINING DATASETS AVAILABLE:")
    training_info = system.get_training_dataset_info()
    print(f"Training data type: {type(training_info)}")
    print(f"Training data keys: {list(training_info.keys()) if isinstance(training_info, dict) else 'Not a dict'}")
    
    if isinstance(training_info, dict):
        for dataset_name, info in training_info.items():
            if isinstance(info, dict):
                print(f"{dataset_name}: {info.get('total_samples', 'N/A')} samples, {info.get('features', 'N/A')} features")
            else:
                print(f"{dataset_name}: {type(info)} - {str(info)[:100]}...")
    else:
        print("Training data structure is not as expected")
    
    # Export sample data for modeling
    print("\n📤 Exporting sample data for modeling...")
    sample_path = system.export_sample_data_for_modeling()
    print(f"Sample data exported to: {sample_path}")
    
    # Analyze drug properties
    print("\n📊 DRUG PROPERTIES ANALYSIS:")
    drug_analysis = system.analyze_drug_properties()
    if "error" not in drug_analysis:
        print(f"Total drugs: {drug_analysis['total_drugs']}")
        print(f"Available columns: {', '.join(drug_analysis['columns'][:5])}...")
    
    print("\n✅ PK-PD Simulation System is ready for use!")
    print("\n📖 NEXT STEPS:")
    print("1. Load training datasets using system.load_training_dataset()")
    print("2. Train PK-PD models using the prepared data")
    print("3. Use system.get_drug_information() for drug-specific data")
    print("4. Use system.get_disease_information() for disease-specific data")
    print("5. Use system.get_organ_information() for organ-specific data")

if __name__ == "__main__":
    main()
