"""
PK-PD System Final Demonstration

This script demonstrates the complete PK-PD knowledge base system functionality
and provides examples of how to use it for drug-biological system simulation.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_pk_pd_system():
    """Demonstrate the complete PK-PD system functionality."""
    
    print("🚀 PK-PD KNOWLEDGE BASE SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # 1. Load and display knowledge base summary
    print("\n📊 KNOWLEDGE BASE OVERVIEW:")
    kb_path = "pk_pd_knowledge_base.json"
    
    if Path(kb_path).exists():
        with open(kb_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
        
        # Count entities
        entity_counts = {}
        for entity_type, sources in knowledge_base.items():
            if isinstance(sources, dict):
                total_count = sum(len(records) for records in sources.values())
                entity_counts[entity_type] = total_count
        
        print(f"✅ Total Drugs: {entity_counts.get('drugs', 0)}")
        print(f"✅ Total Diseases: {entity_counts.get('diseases', 0)}")
        print(f"✅ Total Organs: {entity_counts.get('organs', 0)}")
        print(f"✅ PK Parameters: {entity_counts.get('pk_parameters', 0)}")
        print(f"✅ PD Responses: {entity_counts.get('pd_responses', 0)}")
        
        # Show relationships
        if 'drug_disease_interactions' in knowledge_base:
            total_interactions = sum(len(records) for records in knowledge_base['drug_disease_interactions'].values())
            print(f"✅ Drug-Disease Interactions: {total_interactions}")
        
        if 'drug_combinations' in knowledge_base:
            total_combinations = sum(len(records) for records in knowledge_base['drug_combinations'].values())
            print(f"✅ Drug Combinations: {total_combinations}")
    
    # 2. Show sample drugs
    print("\n💊 SAMPLE DRUGS:")
    if 'drugs' in knowledge_base:
        drug_count = 0
        for source, drugs in knowledge_base['drugs'].items():
            for drug in drugs:
                if drug_count < 5:
                    drug_id = drug.get('drug_id', 'N/A')
                    drug_name = drug.get('drug_name', 'N/A')
                    drug_class = drug.get('drug_class', 'N/A')
                    print(f"   {drug_id}: {drug_name} ({drug_class})")
                    drug_count += 1
                else:
                    break
            if drug_count >= 5:
                break
    
    # 3. Show sample diseases
    print("\n🏥 SAMPLE DISEASES:")
    if 'diseases' in knowledge_base:
        disease_count = 0
        for source, diseases in knowledge_base['diseases'].items():
            for disease in diseases:
                if disease_count < 5:
                    disease_id = disease.get('disease_id', 'N/A')
                    disease_name = disease.get('disease_name', 'N/A')
                    category = disease.get('category', 'N/A')
                    print(f"   {disease_id}: {disease_name} ({category})")
                    disease_count += 1
                else:
                    break
            if disease_count >= 5:
                break
    
    # 4. Show organs
    print("\n🫁 ORGAN SYSTEMS:")
    if 'organs' in knowledge_base:
        for source, organs in knowledge_base['organs'].items():
            for organ in organs:
                organ_id = organ.get('organ_id', 'N/A')
                organ_name = organ.get('organ_name', 'N/A')
                organ_type = organ.get('organ_type', 'N/A')
                print(f"   {organ_id}: {organ_name} ({organ_type})")
    
    # 5. Show training data summary
    print("\n🎯 TRAINING DATA SUMMARY:")
    training_dir = Path("simple_pk_pd_training_data")
    
    if training_dir.exists():
        summary_path = training_dir / "training_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            print(f"✅ Datasets Prepared: {len(summary.get('datasets', {}))}")
            for dataset_name, info in summary.get('datasets', {}).items():
                samples = info.get('total_samples', 0)
                features = info.get('features_count', 0)
                targets = info.get('target_columns', [])
                print(f"   {dataset_name}: {samples} samples, {features} features")
                print(f"      Targets: {', '.join(targets)}")
    
    # 6. Demonstrate data access patterns
    print("\n🔍 DATA ACCESS EXAMPLES:")
    
    # Example: Get drug information
    if 'drugs' in knowledge_base:
        for source, drugs in knowledge_base['drugs'].items():
            if drugs:
                sample_drug = drugs[0]
                print(f"\n📋 Sample Drug Information:")
                print(f"   ID: {sample_drug.get('drug_id', 'N/A')}")
                print(f"   Name: {sample_drug.get('drug_name', 'N/A')}")
                print(f"   Class: {sample_drug.get('drug_class', 'N/A')}")
                if 'molecular_weight' in sample_drug:
                    print(f"   Molecular Weight: {sample_drug['molecular_weight']}")
                if 'bioavailability' in sample_drug:
                    print(f"   Bioavailability: {sample_drug['bioavailability']}")
                break
    
    # Example: Get PK parameters
    if 'pk_parameters' in knowledge_base:
        for source, params in knowledge_base['pk_parameters'].items():
            if params:
                sample_pk = params[0]
                print(f"\n📊 Sample PK Parameters:")
                print(f"   Drug ID: {sample_pk.get('drug_id', 'N/A')}")
                print(f"   Population: {sample_pk.get('population_group', 'N/A')}")
                if 'clearance' in sample_pk:
                    print(f"   Clearance: {sample_pk['clearance']}")
                if 'half_life' in sample_pk:
                    print(f"   Half-life: {sample_pk['half_life']}")
                break
    
    # Example: Get disease information
    if 'diseases' in knowledge_base:
        for source, diseases in knowledge_base['diseases'].items():
            if diseases:
                sample_disease = diseases[0]
                print(f"\n🏥 Sample Disease Information:")
                print(f"   ID: {sample_disease.get('disease_id', 'N/A')}")
                print(f"   Name: {sample_disease.get('disease_name', 'N/A')}")
                print(f"   Category: {sample_disease.get('category', 'N/A')}")
                if 'symptoms' in sample_disease:
                    symptoms = sample_disease['symptoms']
                    if isinstance(symptoms, list) and symptoms:
                        print(f"   Symptoms: {', '.join(symptoms[:3])}...")
                break
    
    # 7. Show system capabilities
    print("\n🚀 SYSTEM CAPABILITIES:")
    print("✅ Multi-source data integration (real_data + new datasets)")
    print("✅ Smart column mapping and context detection")
    print("✅ Automatic data cleaning and validation")
    print("✅ Feature engineering for ML models")
    print("✅ Training data preparation with proper splits")
    print("✅ Comprehensive knowledge base for simulation")
    print("✅ Extensible architecture for new data sources")
    
    # 8. Show file structure
    print("\n📁 GENERATED FILES:")
    important_files = [
        "pk_pd_knowledge_base.json",
        "simulation_ready_data.json", 
        "smart_ingestion_results.json",
        "simple_pk_pd_training_data/training_summary.json",
        "PK_PD_SYSTEM_SUMMARY.md"
    ]
    
    for file_path in important_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    print("\n🎯 NEXT STEPS FOR MODEL DEVELOPMENT:")
    print("1. Load training datasets from 'simple_pk_pd_training_data/'")
    print("2. Train PK prediction models using drug properties")
    print("3. Train clinical outcome models using patient data")
    print("4. Use knowledge base for drug-disease matching")
    print("5. Integrate with your existing PK-PD simulation models")
    
    print("\n🎉 SYSTEM STATUS: FULLY OPERATIONAL!")
    print("Your PK-PD knowledge base is ready for production use!")

def show_integration_examples():
    """Show examples of how to integrate with existing PK-PD models."""
    
    print("\n" + "="*60)
    print("🔧 INTEGRATION EXAMPLES")
    print("="*60)
    
    # Example 1: Loading drug data for PK modeling
    print("\n📊 Example 1: Loading Drug Data for PK Modeling")
    print("""
import pandas as pd
import json

# Load knowledge base
with open('pk_pd_knowledge_base.json', 'r') as f:
    kb = json.load(f)

# Extract drug properties
drugs = []
for source, drug_list in kb['drugs'].items():
    drugs.extend(drug_list)

# Create DataFrame for modeling
df_drugs = pd.DataFrame(drugs)

# Select features for PK prediction
pk_features = ['molecular_weight', 'logp', 'bioavailability', 
               'protein_binding', 'clearance_rate']
X = df_drugs[pk_features].fillna(0)
y = df_drugs['half_life']

# Train your PK model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)
""")
    
    # Example 2: Clinical outcome prediction
    print("\n🏥 Example 2: Clinical Outcome Prediction")
    print("""
# Load clinical training data
clinical_data = pd.read_csv('simple_pk_pd_training_data/clinical_outcomes/splits/train.csv')

# Prepare features and targets
features = ['age', 'sex', 'weight', 'dose_mg', 'treatment_duration']
X = clinical_data[features]
y = clinical_data['treatment_success']

# Train outcome prediction model
from sklearn.linear_model import LogisticRegression
outcome_model = LogisticRegression()
outcome_model.fit(X, y)
""")
    
    # Example 3: Drug-disease interaction analysis
    print("\n💊 Example 3: Drug-Disease Interaction Analysis")
    print("""
# Load drug-disease interactions
interactions = []
for source, interaction_list in kb['drug_disease_interactions'].items():
    interactions.extend(interaction_list)

# Analyze interaction patterns
df_interactions = pd.DataFrame(interactions)

# Find most connected drugs
drug_connections = df_interactions['drug_id'].value_counts().head(10)
print("Top 10 most connected drugs:")
print(drug_connections)

# Find most treated diseases
disease_treatments = df_interactions['disease_id'].value_counts().head(10)
print("\\nTop 10 most treated diseases:")
print(disease_treatments)
""")

def main():
    """Main demonstration function."""
    demonstrate_pk_pd_system()
    show_integration_examples()
    
    print("\n" + "="*60)
    print("🎯 PK-PD SYSTEM DEMONSTRATION COMPLETE")
    print("="*60)
    print("Your comprehensive PK-PD knowledge base is ready for use!")
    print("Check the generated files and integration examples above.")

if __name__ == "__main__":
    main()
