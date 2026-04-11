"""
Realistic Data Generator for GNN PK-PD Model
Based on dataset_spec.py specifications
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random
from scipy import stats
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import your specifications
from dataset_specs import DatasetRequirements, DatasetType, DatasetSpecification

class RealDataGenerator:
    """
    Generates realistic synthetic data for all datasets in the GNN PK-PD model.
    Uses statistical distributions based on real physiological and pharmacological knowledge.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.requirements = DatasetRequirements()
        
    def generate_all_datasets(self, output_dir: str = "./data"):
        """Generate all datasets and save to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        datasets = {}
        
        print("Generating Drug Properties Dataset...")
        datasets['drug_properties'] = self.generate_drug_properties()
        datasets['drug_properties'].to_csv(f"{output_dir}/drug_properties.csv", index=False)
        
        print("Generating Organ Network Dataset...")
        organ_data, edge_data = self.generate_organ_network()
        datasets['organ_network'] = organ_data
        datasets['organ_edges'] = edge_data
        organ_data.to_csv(f"{output_dir}/organ_network.csv", index=False)
        edge_data.to_csv(f"{output_dir}/organ_connections.csv", index=False)
        
        print("Generating Physiological Parameters...")
        phys_data = self.generate_physiological_parameters()
        datasets['physiological_params'] = phys_data
        phys_data.to_csv(f"{output_dir}/physiological_parameters.csv", index=False)
        
        print("Generating Clinical Trials Data...")
        clinical_data = self.generate_clinical_trials()
        datasets['clinical_trials'] = clinical_data
        clinical_data.to_csv(f"{output_dir}/clinical_trials.csv", index=False)
        
        print("Generating Disease Data...")
        disease_data = self.generate_disease_data()
        datasets['disease_data'] = disease_data
        disease_data.to_csv(f"{output_dir}/disease_data.csv", index=False)
        
        print("Generating Population Data...")
        population_data = self.generate_population_data()
        datasets['population_data'] = population_data
        population_data.to_csv(f"{output_dir}/population_data.csv", index=False)
        
        print("Generating Biomarker Data...")
        biomarker_data = self.generate_biomarker_data()
        datasets['biomarker_data'] = biomarker_data
        biomarker_data.to_csv(f"{output_dir}/biomarker_data.csv", index=False)
        
        print("Generating PK Parameters...")
        pk_data = self.generate_pk_parameters()
        datasets['pk_parameters'] = pk_data
        pk_data.to_csv(f"{output_dir}/pk_parameters.csv", index=False)
        
        print("Generating PD Responses...")
        pd_data = self.generate_pd_responses()
        datasets['pd_responses'] = pd_data
        pd_data.to_csv(f"{output_dir}/pd_responses.csv", index=False)
        
        print("Generating Genomic Data...")
        genomic_data = self.generate_genomic_data()
        datasets['genomic_data'] = genomic_data
        genomic_data.to_csv(f"{output_dir}/genomic_data.csv", index=False)
        
        print(f"\nAll datasets generated and saved to {output_dir}/")
        return datasets
    
    def generate_drug_properties(self, n_drugs: int = 100) -> pd.DataFrame:
        """Generate realistic drug properties based on real drug characteristics."""
        
        # Real drug database with actual properties
        real_drugs = [
            # (name, molecular_weight, logp, pka, bioavailability, protein_binding, clearance, volume, half_life, drug_class, route)
            ("Acetaminophen", 151.16, 0.5, 9.5, 0.88, 0.20, 0.35, 0.9, 2.5, "small_molecule", "oral"),
            ("Ibuprofen", 206.28, 3.5, 4.9, 0.80, 0.99, 0.72, 0.15, 2.0, "small_molecule", "oral"),
            ("Aspirin", 180.16, 1.2, 3.5, 0.68, 0.80, 0.25, 0.17, 0.3, "small_molecule", "oral"),
            ("Metformin", 129.16, -1.43, 12.4, 0.55, 0.01, 1.2, 0.6, 5.0, "small_molecule", "oral"),
            ("Lisinopril", 405.5, -0.7, 3.5, 0.25, 0.25, 0.2, 1.3, 12.0, "small_molecule", "oral"),
            ("Atorvastatin", 558.6, 4.1, 4.5, 0.14, 0.98, 0.15, 5.4, 14.0, "small_molecule", "oral"),
            ("Simvastatin", 418.6, 4.7, 4.5, 0.05, 0.95, 0.18, 3.8, 2.0, "small_molecule", "oral"),
            ("Warfarin", 308.3, 2.7, 5.1, 0.80, 0.99, 0.08, 0.14, 36.0, "small_molecule", "oral"),
            ("Digoxin", 780.9, 2.1, 13.9, 0.70, 0.25, 0.03, 6.7, 36.0, "small_molecule", "oral"),
            ("Furosemide", 330.7, 2.3, 3.9, 0.60, 0.98, 0.18, 0.15, 2.0, "small_molecule", "oral"),
            ("Hydrochlorothiazide", 295.8, -0.1, 9.2, 0.70, 0.58, 0.35, 0.8, 8.0, "small_molecule", "oral"),
            ("Amlodipine", 408.9, 3.0, 8.6, 0.64, 0.93, 0.12, 21.0, 35.0, "small_molecule", "oral"),
            ("Metoprolol", 267.4, 1.8, 9.7, 0.50, 0.12, 1.0, 4.2, 3.5, "small_molecule", "oral"),
            ("Propranolol", 259.3, 3.0, 9.5, 0.30, 0.90, 0.84, 4.3, 4.0, "small_molecule", "oral"),
            ("Diazepam", 284.7, 2.8, 3.4, 0.93, 0.98, 0.38, 1.1, 30.0, "small_molecule", "oral"),
            ("Lorazepam", 321.2, 2.9, 1.3, 0.85, 0.85, 0.15, 1.3, 14.0, "small_molecule", "oral"),
            ("Fluoxetine", 309.3, 4.5, 9.8, 0.72, 0.94, 0.24, 3.0, 4.0, "small_molecule", "oral"),
            ("Sertraline", 306.2, 5.1, 9.5, 0.44, 0.98, 0.23, 20.0, 26.0, "small_molecule", "oral"),
            ("Citalopram", 361.9, 3.5, 9.4, 0.80, 0.80, 0.36, 12.0, 35.0, "small_molecule", "oral"),
            ("Omeprazole", 345.4, 2.1, 4.0, 0.35, 0.95, 0.5, 0.3, 1.0, "small_molecule", "oral"),
            ("Insulin", 5808.0, -2.0, 5.3, 0.55, 0.01, 0.12, 0.25, 0.15, "peptide", "subcutaneous"),
            ("Morphine", 285.3, 0.9, 8.0, 0.30, 0.35, 1.2, 3.0, 2.0, "small_molecule", "intravenous"),
            ("Fentanyl", 336.5, 4.0, 8.4, 0.50, 0.80, 0.6, 4.0, 4.0, "small_molecule", "intravenous"),
            ("Oxycodone", 315.4, 0.7, 8.5, 0.60, 0.45, 0.8, 2.6, 3.5, "small_molecule", "oral"),
            ("Tramadol", 263.4, 2.5, 9.4, 0.68, 0.20, 0.42, 3.0, 6.0, "small_molecule", "oral"),
            ("Gabapentin", 171.2, -1.1, 3.7, 0.60, 0.03, 0.12, 0.8, 6.0, "small_molecule", "oral"),
            ("Pregabalin", 159.2, -0.4, 4.2, 0.90, 0.01, 0.67, 0.5, 6.0, "small_molecule", "oral"),
            ("Levothyroxine", 776.9, 2.3, 10.3, 0.40, 0.99, 0.015, 0.15, 7.0, "small_molecule", "oral"),
            ("Prednisone", 358.4, 1.5, 13.0, 0.70, 0.77, 0.15, 0.6, 3.5, "small_molecule", "oral"),
            ("Hydrocortisone", 362.5, 1.6, 13.0, 0.72, 0.90, 0.26, 0.8, 2.0, "small_molecule", "oral"),
            ("Albuterol", 239.3, 0.6, 9.3, 0.40, 0.10, 0.48, 1.2, 4.0, "small_molecule", "inhalation"),
            ("Montelukast", 586.2, 2.8, 5.1, 0.63, 0.99, 0.07, 0.4, 5.0, "small_molecule", "oral"),
            ("Fluticasone", 500.6, 3.5, 12.6, 0.01, 0.90, 0.15, 0.6, 3.5, "small_molecule", "inhalation"),
            ("Amoxicillin", 365.4, 0.9, 2.4, 0.74, 0.20, 0.25, 0.3, 1.5, "small_molecule", "oral"),
            ("Azithromycin", 748.9, 3.2, 8.8, 0.37, 0.07, 0.65, 31.0, 68.0, "small_molecule", "oral"),
            ("Ciprofloxacin", 331.3, 0.3, 6.1, 0.70, 0.30, 0.48, 2.5, 4.0, "small_molecule", "oral"),
            ("Doxycycline", 444.4, -0.2, 3.3, 0.93, 0.93, 0.20, 0.7, 18.0, "small_molecule", "oral"),
            ("Vancomycin", 1449.3, -3.1, 2.8, 0.00, 0.55, 0.12, 0.7, 6.0, "small_molecule", "intravenous"),
            ("Rituximab", 145000.0, -2.5, 7.0, 0.00, 0.50, 0.004, 3.5, 20.0, "antibody", "intravenous"),
            ("Trastuzumab", 145000.0, -2.5, 7.0, 0.00, 0.50, 0.005, 4.0, 28.0, "antibody", "intravenous"),
            ("Adalimumab", 148000.0, -2.5, 7.0, 0.00, 0.65, 0.012, 5.0, 14.0, "antibody", "subcutaneous"),
            ("Infliximab", 149000.0, -2.5, 7.0, 0.00, 0.55, 0.010, 3.0, 10.0, "antibody", "intravenous"),
            ("Etanercept", 150000.0, -2.5, 7.0, 0.00, 0.60, 0.020, 7.0, 70.0, "biologic", "subcutaneous"),
            ("Methotrexate", 454.4, -0.1, 4.8, 0.70, 0.50, 0.10, 0.8, 8.0, "small_molecule", "oral"),
            ("Clopidogrel", 321.8, 3.0, 4.5, 0.50, 0.95, 0.15, 1.0, 8.0, "small_molecule", "oral"),
            ("Apixaban", 459.5, 2.3, 4.2, 0.50, 0.87, 0.25, 0.25, 12.0, "small_molecule", "oral"),
            ("Rivaroxaban", 435.9, 2.7, 4.2, 0.80, 0.95, 0.20, 0.30, 8.0, "small_molecule", "oral"),
            ("Dabigatran", 627.7, 2.5, 4.0, 0.03, 0.35, 0.15, 0.7, 14.0, "small_molecule", "oral"),
            ("Sitagliptin", 407.3, 2.5, 8.5, 0.87, 0.38, 0.48, 1.8, 12.0, "small_molecule", "oral"),
            ("Empagliflozin", 450.9, 1.6, 12.6, 0.75, 0.86, 0.26, 1.5, 12.0, "small_molecule", "oral"),
            ("Liraglutide", 3751.0, -1.5, 7.0, 0.55, 0.99, 0.07, 1.2, 13.0, "peptide", "subcutaneous"),
            ("Semaglutide", 4113.0, -1.5, 7.0, 0.89, 0.99, 0.05, 8.5, 7.0, "peptide", "subcutaneous"),
            ("Sacubitril", 412.5, 2.1, 4.5, 0.60, 0.97, 0.18, 2.0, 12.0, "small_molecule", "oral"),
            ("Valsartan", 435.5, 2.9, 4.7, 0.23, 0.95, 0.17, 1.5, 6.0, "small_molecule", "oral"),
            ("Losartan", 421.9, 3.3, 4.0, 0.33, 0.99, 0.50, 0.4, 2.0, "small_molecule", "oral"),
            ("Carvedilol", 406.5, 4.2, 7.8, 0.25, 0.98, 0.56, 2.0, 7.0, "small_molecule", "oral"),
            ("Spironolactone", 416.6, 2.5, 7.6, 0.25, 0.90, 0.16, 0.5, 1.5, "small_molecule", "oral"),
            ("Hydralazine", 160.2, 0.2, 7.2, 0.35, 0.87, 2.0, 0.8, 3.0, "small_molecule", "oral"),
            ("Nitroglycerin", 227.1, 2.2, 2.2, 0.40, 0.00, 1.5, 0.3, 2.0, "small_molecule", "sublingual"),
            ("Verapamil", 455.6, 3.8, 8.9, 0.20, 0.90, 1.0, 4.0, 5.0, "small_molecule", "oral"),
            ("Diltiazem", 414.5, 2.7, 7.7, 0.40, 0.78, 1.2, 3.0, 4.0, "small_molecule", "oral"),
            ("Amlodipine", 408.9, 3.0, 8.6, 0.64, 0.93, 0.12, 21.0, 35.0, "small_molecule", "oral"),
            ("Nifedipine", 346.3, 2.2, 1.5, 0.50, 0.90, 0.5, 0.8, 2.0, "small_molecule", "oral"),
            ("Clonidine", 230.2, 1.6, 8.3, 0.75, 0.20, 0.22, 2.1, 12.0, "small_molecule", "oral"),
            ("Prazosin", 383.4, 2.5, 6.5, 0.55, 0.97, 0.11, 0.6, 3.0, "small_molecule", "oral"),
            ("Terazosin", 389.4, 2.6, 6.5, 0.90, 0.90, 0.10, 0.9, 12.0, "small_molecule", "oral"),
            ("Tamsulosin", 468.6, 3.0, 8.4, 0.94, 0.98, 0.10, 0.2, 15.0, "small_molecule", "oral"),
            ("Finasteride", 372.6, 3.0, 15.0, 0.63, 0.90, 0.16, 0.8, 6.0, "small_molecule", "oral"),
            ("Tadalafil", 389.4, 2.5, 16.8, 0.36, 0.94, 0.63, 1.6, 17.5, "small_molecule", "oral"),
            ("Sildenafil", 474.6, 2.3, 8.0, 0.40, 0.96, 0.41, 1.0, 4.0, "small_molecule", "oral"),
            ("Vardenafil", 488.9, 2.8, 8.7, 0.15, 0.95, 0.56, 2.5, 5.0, "small_molecule", "oral"),
            ("Ondansetron", 294.4, 2.1, 7.4, 0.56, 0.73, 0.40, 1.6, 4.0, "small_molecule", "oral"),
            ("Promethazine", 284.4, 4.3, 9.1, 0.25, 0.93, 0.90, 9.0, 16.0, "small_molecule", "oral"),
            ("Metoclopramide", 299.8, -0.1, 9.6, 0.80, 0.30, 0.50, 3.0, 5.0, "small_molecule", "oral"),
            ("Domperidone", 425.9, 3.6, 8.8, 0.15, 0.91, 0.21, 5.0, 7.0, "small_molecule", "oral"),
            ("Loperamide", 477.0, 4.6, 8.7, 0.00, 0.97, 0.02, 3.0, 11.0, "small_molecule", "oral"),
            ("Ranitidine", 314.4, 0.3, 8.2, 0.50, 0.15, 0.70, 1.3, 3.0, "small_molecule", "oral"),
            ("Famotidine", 337.5, -0.6, 6.8, 0.45, 0.15, 0.80, 1.3, 3.0, "small_molecule", "oral"),
            ("Esomeprazole", 345.4, 2.1, 4.0, 0.50, 0.97, 0.60, 0.3, 1.5, "small_molecule", "oral"),
            ("Pantoprazole", 383.9, 2.6, 3.9, 0.77, 0.98, 0.15, 0.2, 1.5, "small_molecule", "oral"),
            ("Lansoprazole", 382.9, 2.0, 4.0, 0.80, 0.97, 0.30, 0.2, 1.5, "small_molecule", "oral"),
            ("Sucralfate", 2087.0, -2.0, 7.0, 0.05, 0.00, 0.00, 0.0, 0.0, "small_molecule", "oral"),
            ("Misoprostol", 382.5, 3.5, 4.6, 0.70, 0.90, 0.30, 0.7, 2.0, "small_molecule", "oral"),
            ("Budesonide", 430.5, 2.1, 12.6, 0.11, 0.88, 0.84, 2.7, 3.0, "small_molecule", "inhalation"),
            ("Beclomethasone", 440.9, 2.3, 12.8, 0.02, 0.87, 0.15, 0.6, 2.5, "small_molecule", "inhalation"),
            ("Ciclesonide", 540.6, 3.5, 13.0, 0.52, 0.99, 0.38, 2.5, 3.5, "small_molecule", "inhalation"),
            ("Mometasone", 520.4, 2.6, 12.9, 0.01, 0.88, 0.25, 1.0, 5.0, "small_molecule", "inhalation"),
            ("Flunisolide", 434.5, 2.0, 12.5, 0.20, 0.80, 0.55, 1.8, 4.0, "small_molecule", "inhalation"),
            ("Triamcinolone", 434.5, 2.3, 12.8, 0.23, 0.77, 0.45, 1.4, 3.5, "small_molecule", "inhalation"),
            ("Prednisolone", 360.4, 1.6, 13.0, 0.82, 0.77, 0.10, 0.6, 2.5, "small_molecule", "oral"),
            ("Methylprednisolone", 374.5, 1.8, 13.0, 0.82, 0.68, 0.15, 1.0, 2.5, "small_molecule", "oral"),
            ("Dexamethasone", 392.5, 1.8, 13.0, 0.78, 0.68, 0.17, 0.8, 4.0, "small_molecule", "oral"),
            ("Betamethasone", 392.5, 1.9, 13.0, 0.64, 0.64, 0.15, 1.4, 5.0, "small_molecule", "oral"),
            ("Hydrocortisone", 362.5, 1.6, 13.0, 0.72, 0.90, 0.26, 0.8, 2.0, "small_molecule", "oral"),
            ("Cortisone", 360.4, 1.5, 13.0, 0.00, 0.25, 0.10, 0.4, 1.0, "small_molecule", "oral"),
            ("Fludrocortisone", 380.4, 2.0, 13.0, 0.00, 0.82, 0.20, 0.4, 4.0, "small_molecule", "oral"),
            ("Testosterone", 288.4, 3.3, 10.5, 0.06, 0.50, 0.10, 0.7, 8.0, "small_molecule", "intramuscular"),
            ("Estradiol", 272.4, 3.9, 10.7, 0.05, 0.50, 0.15, 0.6, 14.0, "small_molecule", "oral"),
            ("Progesterone", 314.5, 3.9, 11.0, 0.06, 0.50, 0.10, 1.2, 8.0, "small_molecule", "oral"),
            ("Medroxyprogesterone", 386.5, 3.9, 11.0, 0.06, 0.50, 0.10, 1.2, 12.0, "small_molecule", "intramuscular"),
            ("Norethindrone", 298.4, 3.3, 10.4, 0.64, 0.61, 0.10, 4.0, 8.0, "small_molecule", "oral"),
            ("Levonorgestrel", 312.4, 3.5, 10.4, 0.90, 0.50, 0.10, 1.8, 24.0, "small_molecule", "oral"),
            ("Ethinyl estradiol", 296.4, 3.7, 10.4, 0.40, 0.98, 0.10, 5.0, 24.0, "small_molecule", "oral"),
            ("Drospirenone", 366.5, 2.8, 10.4, 0.76, 0.97, 0.10, 4.0, 30.0, "small_molecule", "oral"),
            ("Desogestrel", 310.4, 3.5, 10.4, 0.62, 0.50, 0.10, 1.8, 24.0, "small_molecule", "oral"),
            ("Gestodene", 310.4, 3.0, 10.4, 0.87, 0.50, 0.10, 1.8, 24.0, "small_molecule", "oral")
        ]
        
        # Extend with variations if more drugs needed
        if n_drugs > len(real_drugs):
            # Create variations of existing drugs
            base_drugs = real_drugs.copy()
            for i in range(n_drugs - len(real_drugs)):
                base_drug = base_drugs[i % len(base_drugs)]
                variation = (
                    f"{base_drug[0]}_variant_{i//len(real_drugs)+1}",
                    base_drug[1] * np.random.uniform(0.9, 1.1),
                    base_drug[2] + np.random.uniform(-0.5, 0.5),
                    base_drug[3] + np.random.uniform(-1, 1),
                    base_drug[4] * np.random.uniform(0.8, 1.2),
                    base_drug[5] * np.random.uniform(0.8, 1.2),
                    base_drug[6] * np.random.uniform(0.8, 1.2),
                    base_drug[7] * np.random.uniform(0.8, 1.2),
                    base_drug[8] * np.random.uniform(0.8, 1.2),
                    base_drug[9],
                    base_drug[10]
                )
                real_drugs.append(variation)
        
        real_drugs = real_drugs[:n_drugs]
        
        data = {
            'drug_id': [f'DRUG_{i:05d}' for i in range(n_drugs)],
            'drug_name': [drug[0] for drug in real_drugs],
            'molecular_weight': [drug[1] for drug in real_drugs],
            'logp': [drug[2] for drug in real_drugs],
            'pka': [drug[3] for drug in real_drugs],
            'polar_surface_area': [np.random.uniform(20, 200) for _ in range(n_drugs)],
            'solubility': [10 ** np.random.uniform(-4, 1) for _ in range(n_drugs)],
            'bioavailability': [drug[4] for drug in real_drugs],
            'protein_binding': [drug[5] for drug in real_drugs],
            'permeability': [10 ** np.random.uniform(-8, -5) for _ in range(n_drugs)],
            'clearance_rate': [drug[6] for drug in real_drugs],
            'volume_distribution': [drug[7] for drug in real_drugs],
            'half_life': [drug[8] for drug in real_drugs],
            'absorption_rate': [np.random.uniform(0.1, 8) for _ in range(n_drugs)],
            'metabolism_rate': [np.random.uniform(0.05, 4) for _ in range(n_drugs)],
            'excretion_rate': [np.random.uniform(0.02, 3) for _ in range(n_drugs)],
            'drug_class': [drug[9] for drug in real_drugs],
            'administration_route': [drug[10] for drug in real_drugs]
        }
        
        # Add realistic optional columns
        data['lipinski_violations'] = [np.random.poisson(0.5) for _ in range(n_drugs)]
        data['rotatable_bonds'] = [np.random.poisson(5) for _ in range(n_drugs)]
        data['hydrogen_bond_donors'] = [np.random.poisson(3) for _ in range(n_drugs)]
        data['hydrogen_bond_acceptors'] = [np.random.poisson(6) for _ in range(n_drugs)]
        
        df = pd.DataFrame(data)
        
        # Add realistic correlations based on actual pharmacology
        df['bioavailability'] = df['bioavailability'].clip(0, 1)
        df['protein_binding'] = df['protein_binding'].clip(0, 0.99)
        
        return df
    
    def generate_organ_network(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate realistic human organ network with vascular connections."""
        
        organs = [
            ('brain', 1.4, 750, 7.4, 37.0, 0.77, 0.12, 0.11),
            ('heart', 0.33, 250, 7.4, 37.0, 0.79, 0.05, 0.16),
            ('liver', 1.5, 1500, 7.4, 37.5, 0.71, 0.05, 0.24),
            ('kidneys', 0.31, 1100, 7.4, 37.0, 0.76, 0.06, 0.18),
            ('lungs', 0.5, 1000, 7.4, 37.0, 0.79, 0.03, 0.18),
            ('muscle', 30.0, 800, 7.4, 37.0, 0.75, 0.05, 0.20),
            ('adipose', 15.0, 300, 7.4, 36.8, 0.20, 0.80, 0.00),
            ('gut', 2.0, 1200, 7.4, 37.0, 0.80, 0.05, 0.15),
            ('skin', 3.5, 400, 7.4, 36.5, 0.64, 0.10, 0.26),
            ('spleen', 0.15, 200, 7.4, 37.0, 0.78, 0.03, 0.19),
            ('pancreas', 0.08, 100, 7.4, 37.0, 0.75, 0.04, 0.21),
            ('thyroid', 0.025, 80, 7.4, 37.0, 0.76, 0.05, 0.19)
        ]
        
        organ_data = []
        for i, (name, mass, flow, ph, temp, water, lipid, protein) in enumerate(organs):
            organ_data.append({
                'organ_id': f'ORG_{i:02d}',
                'organ_name': name,
                'organ_type': name,
                'volume': mass * 1.05,  # volume ~ mass * density
                'mass': mass,
                'blood_flow': flow,
                'oxygen_consumption': flow * 0.05,  # 5% of blood flow
                'metabolic_rate': np.random.uniform(0.3, 2.0),
                'ph': ph,
                'temperature': temp,
                'surface_area': mass ** (2/3) * 100,
                'water_content': water,
                'lipid_content': lipid,
                'protein_content': protein,
                'health_status': 1.0
            })
        
        organ_df = pd.DataFrame(organ_data)
        
        # Create realistic vascular connections
        connections = []
        vascular_network = {
            'heart': ['brain', 'liver', 'kidneys', 'muscle', 'adipose', 'gut', 'skin', 'spleen'],
            'liver': ['gut'],  # portal circulation
            'lungs': ['heart']
        }
        
        for i, source in organ_df.iterrows():
            source_name = source['organ_name']
            for j, target in organ_df.iterrows():
                target_name = target['organ_name']
                
                # Determine connection type and properties
                if source_name == 'heart' and target_name in vascular_network.get('heart', []):
                    flow_rate = target['blood_flow']
                    connection_type = 'vascular'
                elif source_name == 'liver' and target_name == 'gut':
                    flow_rate = organ_df[organ_df['organ_name'] == 'gut']['blood_flow'].iloc[0]
                    connection_type = 'portal'
                elif source_name == 'lungs' and target_name == 'heart':
                    flow_rate = 5000
                    connection_type = 'pulmonary'
                else:
                    continue
                
                connections.append({
                    'source_id': source['organ_id'],
                    'target_id': target['organ_id'],
                    'source_organ': source_name,
                    'target_organ': target_name,
                    'flow_rate': flow_rate,
                    'resistance': np.random.uniform(0.05, 0.3),
                    'permeability': np.random.uniform(0.3, 0.95),
                    'delay': flow_rate / 1000 * 0.1,  # delay proportional to flow
                    'connection_type': connection_type,
                    'vascular': 1 if connection_type in ['vascular', 'pulmonary'] else 0,
                    'lymphatic': 0,
                    'neural': 1 if source_name == 'brain' else 0,
                    'hormonal': 1 if source_name in ['thyroid', 'pancreas'] else 0
                })
        
        edge_df = pd.DataFrame(connections)
        
        return organ_df, edge_df
    
    def generate_physiological_parameters(self) -> pd.DataFrame:
        """Generate population physiological parameters."""
        
        n_subjects = 10000
        
        data = {
            'subject_id': [f'SUBJ_{i:07d}' for i in range(n_subjects)],
            'age': np.random.choice([
                np.random.uniform(0, 2),   # infants
                np.random.uniform(2, 12),  # children
                np.random.uniform(12, 65), # adults
                np.random.uniform(65, 100) # elderly
            ], n_subjects, p=[0.05, 0.15, 0.7, 0.1]),
            
            'weight': np.random.normal(70, 15, n_subjects).clip(3, 200),
            'height': np.random.normal(170, 10, n_subjects).clip(50, 220),
            'sex': np.random.choice(['M', 'F'], n_subjects),
            'ethnicity': np.random.choice(['Caucasian', 'Asian', 'African', 'Hispanic', 'Other'], 
                                         n_subjects, p=[0.6, 0.15, 0.12, 0.1, 0.03]),
            'bmi': np.random.normal(25, 5, n_subjects).clip(15, 45),
            'body_surface_area': np.random.uniform(1.5, 2.2, n_subjects),
            'organ_volume_scaling': np.random.normal(1, 0.2, n_subjects).clip(0.5, 1.5),
            'metabolic_rate_scaling': np.random.normal(1, 0.15, n_subjects).clip(0.6, 1.4),
            'clearance_scaling': np.random.normal(1, 0.25, n_subjects).clip(0.5, 1.8),
            'volume_distribution_scaling': np.random.normal(1, 0.2, n_subjects).clip(0.6, 1.5)
        }
        
        df = pd.DataFrame(data)
        
        # Add realistic relationships
        df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
        df['body_surface_area'] = 0.007184 * (df['weight'] ** 0.425) * (df['height'] ** 0.725)
        
        # Age-dependent scaling
        df['organ_volume_scaling'] = df['organ_volume_scaling'] * (1 - 0.3 * (df['age'] > 60))
        
        return df
    
    def generate_clinical_trials(self, n_subjects: int = 500, n_timepoints: int = 20) -> pd.DataFrame:
        """Generate realistic clinical trial data with PK/PD profiles using real drug parameters."""
        
        total_samples = n_subjects * n_timepoints
        
        # Generate realistic concentration-time profiles
        time_points = np.linspace(0, 24, n_timepoints)
        
        # Real drug dosing information
        drug_doses = {
            "Acetaminophen": [500, 1000],
            "Ibuprofen": [200, 400, 600, 800],
            "Aspirin": [81, 325, 650],
            "Metformin": [500, 850, 1000],
            "Lisinopril": [5, 10, 20, 40],
            "Atorvastatin": [10, 20, 40, 80],
            "Simvastatin": [5, 10, 20, 40, 80],
            "Warfarin": [1, 2, 5, 10],
            "Digoxin": [0.125, 0.25, 0.5],
            "Furosemide": [20, 40, 80],
            "Hydrochlorothiazide": [12.5, 25, 50],
            "Amlodipine": [2.5, 5, 10],
            "Metoprolol": [25, 50, 100, 200],
            "Propranolol": [10, 20, 40, 80],
            "Diazepam": [2, 5, 10],
            "Lorazepam": [0.5, 1, 2],
            "Fluoxetine": [10, 20, 40, 60],
            "Sertraline": [25, 50, 100, 200],
            "Citalopram": [10, 20, 40],
            "Omeprazole": [10, 20, 40],
            "Insulin": [5, 10, 15, 20],
            "Morphine": [2, 5, 10, 15],
            "Fentanyl": [25, 50, 100],
            "Oxycodone": [5, 10, 15, 20],
            "Tramadol": [50, 100, 200],
            "Gabapentin": [100, 300, 600, 900],
            "Pregabalin": [25, 50, 75, 150, 300],
            "Levothyroxine": [25, 50, 75, 100, 125],
            "Prednisone": [5, 10, 20, 40, 60],
            "Hydrocortisone": [10, 20, 40],
            "Albuterol": [90, 180],
            "Montelukast": [4, 5, 10],
            "Fluticasone": [44, 110, 220, 440],
            "Amoxicillin": [250, 500, 875],
            "Azithromycin": [250, 500],
            "Ciprofloxacin": [250, 500, 750],
            "Doxycycline": [50, 100, 200],
            "Vancomycin": [500, 1000, 1500],
            "Methotrexate": [2.5, 5, 10, 15, 25],
            "Clopidogrel": [75],
            "Apixaban": [2.5, 5, 10],
            "Rivaroxaban": [10, 15, 20],
            "Sitagliptin": [25, 50, 100],
            "Empagliflozin": [10, 25],
            "Liraglutide": [0.6, 1.2, 1.8],
            "Semaglutide": [0.25, 0.5, 1, 2],
            "Sacubitril": [24, 49, 97],
            "Valsartan": [40, 80, 160, 320],
            "Losartan": [25, 50, 100],
            "Carvedilol": [3.125, 6.25, 12.5, 25, 50],
            "Spironolactone": [12.5, 25, 50, 100],
            "Hydralazine": [10, 25, 50, 100],
            "Nitroglycerin": [0.4, 0.8],
            "Verapamil": [40, 80, 120, 240],
            "Diltiazem": [30, 60, 90, 120, 180, 240],
            "Nifedipine": [10, 20, 30, 60, 90],
            "Clonidine": [0.1, 0.2, 0.3],
            "Prazosin": [1, 2, 5, 10],
            "Terazosin": [1, 2, 5, 10],
            "Tamsulosin": [0.4, 0.8],
            "Finasteride": [1, 5],
            "Tadalafil": [2.5, 5, 10, 20],
            "Sildenafil": [25, 50, 100],
            "Vardenafil": [5, 10, 20],
            "Ondansetron": [4, 8],
            "Promethazine": [12.5, 25, 50],
            "Metoclopramide": [5, 10],
            "Domperidone": [10, 20],
            "Loperamide": [2, 4],
            "Ranitidine": [75, 150, 300],
            "Famotidine": [10, 20, 40],
            "Esomeprazole": [20, 40],
            "Pantoprazole": [20, 40],
            "Lansoprazole": [15, 30],
            "Misoprostol": [100, 200],
            "Budesonide": [90, 180, 360],
            "Beclomethasone": [40, 80, 160, 320],
            "Ciclesonide": [80, 160, 320],
            "Mometasone": [50, 100, 200, 400],
            "Flunisolide": [250, 500],
            "Triamcinolone": [100, 200, 400],
            "Prednisolone": [5, 10, 20, 40, 60],
            "Methylprednisolone": [2, 4, 8, 16, 32],
            "Dexamethasone": [0.5, 1, 2, 4, 6],
            "Betamethasone": [0.5, 0.6, 6],
            "Fludrocortisone": [0.05, 0.1, 0.2],
            "Testosterone": [50, 100, 200],
            "Estradiol": [0.5, 1, 2],
            "Progesterone": [100, 200, 400],
            "Medroxyprogesterone": [2.5, 5, 10],
            "Norethindrone": [0.35, 0.5, 1],
            "Levonorgestrel": [0.15, 0.3],
            "Ethinyl estradiol": [0.02, 0.03, 0.035],
            "Drospirenone": [3],
            "Desogestrel": [0.15, 0.25],
            "Gestodene": [0.05, 0.075]
        }
        
        # Get real drug names from the drug properties
        real_drug_names = list(drug_doses.keys())
        
        data = {
            'trial_id': [f'TRIAL_{i:03d}' for i in range(n_subjects) for _ in range(n_timepoints)],
            'subject_id': [f'SUBJ_{i:05d}' for i in range(n_subjects) for _ in range(n_timepoints)],
            'drug_id': np.random.choice([f'DRUG_{i:04d}' for i in range(len(real_drug_names))], total_samples),
            'drug_name': np.random.choice(real_drug_names, total_samples),
            'dose': np.random.choice([50.0, 100.0, 200.0, 400.0, 800.0], total_samples, 
                                    p=[0.1, 0.3, 0.4, 0.15, 0.05]),
            'administration_route': np.random.choice(['oral', 'iv'], total_samples, p=[0.7, 0.3]),
            'time_point': np.tile(time_points, n_subjects),
            'organ_system': np.random.choice(['cardiovascular', 'respiratory', 'renal', 
                                            'hepatic', 'neurological', 'gastrointestinal'], total_samples),
            'adverse_event': np.random.choice([True, False], total_samples, p=[0.08, 0.92]),
            'severity': np.random.choice(['mild', 'moderate', 'severe'], total_samples, p=[0.6, 0.3, 0.1]),
            'age': np.random.randint(18, 85, total_samples),
            'weight': np.random.normal(70, 15, total_samples),
            'sex': np.random.choice(['M', 'F'], total_samples),
            'ethnicity': np.random.choice(['Caucasian', 'Asian', 'African', 'Hispanic'], total_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Update doses based on real drug information
        for idx, row in df.iterrows():
            drug_name = row['drug_name']
            if drug_name in drug_doses:
                df.at[idx, 'dose'] = np.random.choice(drug_doses[drug_name])
        
        # Generate realistic PK concentrations using real drug parameters
        for idx, row in df.iterrows():
            t = row['time_point']
            dose = row['dose']
            drug_name = row['drug_name']
            route = row['administration_route']
            
            # Get real drug parameters (simplified for demonstration)
            drug_params = {
                "Acetaminophen": {"vd": 0.9, "ke": 0.28, "f": 0.88, "ka": 2.5},
                "Ibuprofen": {"vd": 0.15, "ke": 0.35, "f": 0.80, "ka": 1.2},
                "Aspirin": {"vd": 0.17, "ke": 2.31, "f": 0.68, "ka": 3.0},
                "Metformin": {"vd": 0.6, "ke": 0.14, "f": 0.55, "ka": 1.5},
                "Lisinopril": {"vd": 1.3, "ke": 0.06, "f": 0.25, "ka": 0.8},
                "Atorvastatin": {"vd": 5.4, "ke": 0.05, "f": 0.14, "ka": 0.6},
                "Warfarin": {"vd": 0.14, "ke": 0.02, "f": 0.80, "ka": 1.0},
                "Digoxin": {"vd": 6.7, "ke": 0.02, "f": 0.70, "ka": 0.5},
                "Furosemide": {"vd": 0.15, "ke": 0.35, "f": 0.60, "ka": 1.8},
                "Amlodipine": {"vd": 21.0, "ke": 0.02, "f": 0.64, "ka": 0.4},
                "Metoprolol": {"vd": 4.2, "ke": 0.20, "f": 0.50, "ka": 2.0},
                "Diazepam": {"vd": 1.1, "ke": 0.02, "f": 0.93, "ka": 1.5},
                "Fluoxetine": {"vd": 3.0, "ke": 0.17, "f": 0.72, "ka": 1.0},
                "Sertraline": {"vd": 20.0, "ke": 0.03, "f": 0.44, "ka": 0.8},
                "Omeprazole": {"vd": 0.3, "ke": 0.69, "f": 0.35, "ka": 2.5},
                "Morphine": {"vd": 3.0, "ke": 0.35, "f": 0.30, "ka": 1.5},
                "Fentanyl": {"vd": 4.0, "ke": 0.17, "f": 0.50, "ka": 1.2},
                "Oxycodone": {"vd": 2.6, "ke": 0.20, "f": 0.60, "ka": 1.8},
                "Tramadol": {"vd": 3.0, "ke": 0.12, "f": 0.68, "ka": 2.0},
                "Gabapentin": {"vd": 0.8, "ke": 0.12, "f": 0.60, "ka": 1.5},
                "Levothyroxine": {"vd": 0.15, "ke": 0.10, "f": 0.40, "ka": 1.0},
                "Prednisone": {"vd": 0.6, "ke": 0.20, "f": 0.70, "ka": 2.5},
                "Hydrocortisone": {"vd": 0.8, "ke": 0.35, "f": 0.72, "ka": 2.0},
                "Albuterol": {"vd": 1.2, "ke": 0.17, "f": 0.40, "ka": 3.0},
                "Amoxicillin": {"vd": 0.3, "ke": 0.46, "f": 0.74, "ka": 2.5},
                "Azithromycin": {"vd": 31.0, "ke": 0.01, "f": 0.37, "ka": 0.5},
                "Ciprofloxacin": {"vd": 2.5, "ke": 0.17, "f": 0.70, "ka": 2.0},
                "Doxycycline": {"vd": 0.7, "ke": 0.04, "f": 0.93, "ka": 1.0},
                "Vancomycin": {"vd": 0.7, "ke": 0.12, "f": 0.00, "ka": 0.0},
                "Methotrexate": {"vd": 0.8, "ke": 0.09, "f": 0.70, "ka": 1.5},
                "Clopidogrel": {"vd": 1.0, "ke": 0.09, "f": 0.50, "ka": 2.0},
                "Apixaban": {"vd": 0.25, "ke": 0.06, "f": 0.50, "ka": 1.0},
                "Rivaroxaban": {"vd": 0.30, "ke": 0.09, "f": 0.80, "ka": 1.2},
                "Sitagliptin": {"vd": 1.8, "ke": 0.06, "f": 0.87, "ka": 1.5},
                "Liraglutide": {"vd": 1.2, "ke": 0.05, "f": 0.55, "ka": 0.3},
                "Semaglutide": {"vd": 8.5, "ke": 0.10, "f": 0.89, "ka": 0.2},
                "Valsartan": {"vd": 1.5, "ke": 0.12, "f": 0.23, "ka": 1.0},
                "Losartan": {"vd": 0.4, "ke": 0.35, "f": 0.33, "ka": 1.5},
                "Carvedilol": {"vd": 2.0, "ke": 0.10, "f": 0.25, "ka": 1.0},
                "Verapamil": {"vd": 4.0, "ke": 0.14, "f": 0.20, "ka": 1.0},
                "Diltiazem": {"vd": 3.0, "ke": 0.23, "f": 0.40, "ka": 1.5},
                "Nifedipine": {"vd": 0.8, "ke": 0.35, "f": 0.50, "ka": 2.0},
                "Clonidine": {"vd": 2.1, "ke": 0.06, "f": 0.75, "ka": 1.0},
                "Tamsulosin": {"vd": 0.2, "ke": 0.05, "f": 0.94, "ka": 0.8},
                "Finasteride": {"vd": 0.8, "ke": 0.12, "f": 0.63, "ka": 1.0},
                "Tadalafil": {"vd": 1.6, "ke": 0.04, "f": 0.36, "ka": 0.5},
                "Sildenafil": {"vd": 1.0, "ke": 0.17, "f": 0.40, "ka": 1.0},
                "Ondansetron": {"vd": 1.6, "ke": 0.17, "f": 0.56, "ka": 2.0},
                "Metoclopramide": {"vd": 3.0, "ke": 0.14, "f": 0.80, "ka": 2.5},
                "Prednisolone": {"vd": 0.6, "ke": 0.28, "f": 0.82, "ka": 2.5},
                "Dexamethasone": {"vd": 0.8, "ke": 0.17, "f": 0.78, "ka": 2.0},
                "Testosterone": {"vd": 0.7, "ke": 0.09, "f": 0.06, "ka": 1.0},
                "Estradiol": {"vd": 0.6, "ke": 0.05, "f": 0.05, "ka": 1.0},
                "Progesterone": {"vd": 1.2, "ke": 0.09, "f": 0.06, "ka": 1.0}
            }
            
            params = drug_params.get(drug_name, {"vd": 1.0, "ke": 0.2, "f": 0.7, "ka": 1.0})
            
            if route == 'iv':
                # IV bolus: C = (Dose/Vd) * exp(-ke * t)
                concentration = (dose / params["vd"]) * np.exp(-params["ke"] * t)
            else:
                # Oral: C = (F*Dose*ka)/(Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
                if abs(params["ka"] - params["ke"]) < 0.01:
                    concentration = (params["f"] * dose * params["ka"] / params["vd"]) * t * np.exp(-params["ke"] * t)
                else:
                    concentration = (params["f"] * dose * params["ka"]) / (params["vd"] * (params["ka"] - params["ke"])) * \
                                  (np.exp(-params["ke"] * t) - np.exp(-params["ka"] * t))
            
            # Add realistic noise and ensure non-negative
            concentration = max(0, concentration * (1 + np.random.normal(0, 0.15)))
            df.at[idx, 'concentration'] = concentration
            
            # Generate response based on concentration (Emax model with drug-specific parameters)
            emax_params = {
                "Acetaminophen": {"emax": 100, "ec50": 50},
                "Ibuprofen": {"emax": 80, "ec50": 40},
                "Aspirin": {"emax": 70, "ec50": 30},
                "Metformin": {"emax": 60, "ec50": 100},
                "Lisinopril": {"emax": 40, "ec50": 20},
                "Atorvastatin": {"emax": 50, "ec50": 30},
                "Warfarin": {"emax": 120, "ec50": 60},
                "Digoxin": {"emax": 150, "ec50": 80},
                "Furosemide": {"emax": 90, "ec50": 45},
                "Amlodipine": {"emax": 35, "ec50": 15},
                "Metoprolol": {"emax": 45, "ec50": 25},
                "Diazepam": {"emax": 80, "ec50": 40},
                "Fluoxetine": {"emax": 70, "ec50": 35},
                "Sertraline": {"emax": 75, "ec50": 38},
                "Omeprazole": {"emax": 85, "ec50": 42},
                "Morphine": {"emax": 90, "ec50": 45},
                "Fentanyl": {"emax": 95, "ec50": 48},
                "Oxycodone": {"emax": 88, "ec50": 44},
                "Tramadol": {"emax": 75, "ec50": 37},
                "Gabapentin": {"emax": 65, "ec50": 32},
                "Levothyroxine": {"emax": 55, "ec50": 28},
                "Prednisone": {"emax": 100, "ec50": 50},
                "Hydrocortisone": {"emax": 95, "ec50": 47},
                "Albuterol": {"emax": 85, "ec50": 42},
                "Amoxicillin": {"emax": 80, "ec50": 40},
                "Azithromycin": {"emax": 75, "ec50": 37},
                "Ciprofloxacin": {"emax": 70, "ec50": 35},
                "Doxycycline": {"emax": 72, "ec50": 36},
                "Vancomycin": {"emax": 85, "ec50": 42},
                "Methotrexate": {"emax": 90, "ec50": 45},
                "Clopidogrel": {"emax": 65, "ec50": 32},
                "Apixaban": {"emax": 80, "ec50": 40},
                "Rivaroxaban": {"emax": 78, "ec50": 39},
                "Sitagliptin": {"emax": 70, "ec50": 35},
                "Liraglutide": {"emax": 85, "ec50": 42},
                "Semaglutide": {"emax": 88, "ec50": 44},
                "Valsartan": {"emax": 40, "ec50": 20},
                "Losartan": {"emax": 38, "ec50": 19},
                "Carvedilol": {"emax": 42, "ec50": 21},
                "Verapamil": {"emax": 45, "ec50": 22},
                "Diltiazem": {"emax": 43, "ec50": 21},
                "Nifedipine": {"emax": 41, "ec50": 20},
                "Clonidine": {"emax": 50, "ec50": 25},
                "Tamsulosin": {"emax": 60, "ec50": 30},
                "Finasteride": {"emax": 55, "ec50": 27},
                "Tadalafil": {"emax": 85, "ec50": 42},
                "Sildenafil": {"emax": 82, "ec50": 41},
                "Ondansetron": {"emax": 75, "ec50": 37},
                "Metoclopramide": {"emax": 70, "ec50": 35},
                "Prednisolone": {"emax": 95, "ec50": 47},
                "Dexamethasone": {"emax": 92, "ec50": 46},
                "Testosterone": {"emax": 80, "ec50": 40},
                "Estradiol": {"emax": 75, "ec50": 37},
                "Progesterone": {"emax": 70, "ec50": 35}
            }
            
            pd_params = emax_params.get(drug_name, {"emax": 100, "ec50": 50})
            response = (pd_params["emax"] * concentration) / (pd_params["ec50"] + concentration) + np.random.normal(0, 8)
            df.at[idx, 'response_value'] = max(0, response)
            df.at[idx, 'response_measure'] = 'biomarker_level'
        
        return df
    
    def generate_disease_data(self, n_diseases: int = 50) -> pd.DataFrame:
        """Generate realistic disease data with organ impacts using real clinical information."""
        
        # Real diseases with actual organ impacts and biomarkers
        diseases = {
            'Cardiovascular': {
                'Hypertension': {
                    'organs': ['heart', 'kidneys', 'brain'],
                    'biomarkers': ['BNP', 'Troponin', 'Creatinine', 'Blood Pressure'],
                    'severity_range': [0.3, 0.8],
                    'progression': 0.02
                },
                'Heart Failure': {
                    'organs': ['heart', 'kidneys', 'lungs'],
                    'biomarkers': ['BNP', 'NT-proBNP', 'Troponin', 'C-reactive protein'],
                    'severity_range': [0.6, 0.9],
                    'progression': 0.05
                },
                'Myocardial Infarction': {
                    'organs': ['heart', 'lungs'],
                    'biomarkers': ['Troponin I', 'Troponin T', 'CK-MB', 'LDH'],
                    'severity_range': [0.8, 1.0],
                    'progression': 0.1
                },
                'Arrhythmia': {
                    'organs': ['heart', 'brain'],
                    'biomarkers': ['NT-proBNP', 'Electrolytes', 'Thyroid hormones'],
                    'severity_range': [0.4, 0.7],
                    'progression': 0.03
                }
            },
            'Metabolic': {
                'Type 2 Diabetes': {
                    'organs': ['pancreas', 'kidneys', 'heart', 'brain'],
                    'biomarkers': ['Glucose', 'HbA1c', 'Insulin', 'C-peptide'],
                    'severity_range': [0.3, 0.8],
                    'progression': 0.04
                },
                'Obesity': {
                    'organs': ['heart', 'pancreas', 'liver', 'kidneys'],
                    'biomarkers': ['BMI', 'Leptin', 'Adiponectin', 'CRP'],
                    'severity_range': [0.2, 0.7],
                    'progression': 0.02
                },
                'Hyperlipidemia': {
                    'organs': ['heart', 'liver', 'brain'],
                    'biomarkers': ['LDL', 'HDL', 'Triglycerides', 'Total Cholesterol'],
                    'severity_range': [0.3, 0.6],
                    'progression': 0.03
                },
                'Metabolic Syndrome': {
                    'organs': ['heart', 'pancreas', 'liver', 'kidneys'],
                    'biomarkers': ['Glucose', 'Triglycerides', 'HDL', 'Blood Pressure'],
                    'severity_range': [0.4, 0.8],
                    'progression': 0.04
                }
            },
            'Neurological': {
                'Alzheimer\'s Disease': {
                    'organs': ['brain'],
                    'biomarkers': ['Amyloid-beta', 'Tau protein', 'NFL', 'GFAP'],
                    'severity_range': [0.7, 1.0],
                    'progression': 0.08
                },
                'Parkinson\'s Disease': {
                    'organs': ['brain'],
                    'biomarkers': ['Alpha-synuclein', 'Dopamine', 'NFL'],
                    'severity_range': [0.6, 0.9],
                    'progression': 0.06
                },
                'Stroke': {
                    'organs': ['brain', 'heart'],
                    'biomarkers': ['NFL', 'GFAP', 'Troponin', 'D-dimer'],
                    'severity_range': [0.7, 1.0],
                    'progression': 0.1
                },
                'Epilepsy': {
                    'organs': ['brain'],
                    'biomarkers': ['Neuron-specific enolase', 'S100B', 'IL-6'],
                    'severity_range': [0.4, 0.8],
                    'progression': 0.02
                }
            },
            'Renal': {
                'Chronic Kidney Disease': {
                    'organs': ['kidneys', 'heart', 'brain'],
                    'biomarkers': ['Creatinine', 'BUN', 'eGFR', 'Albumin'],
                    'severity_range': [0.5, 0.9],
                    'progression': 0.05
                },
                'Acute Kidney Injury': {
                    'organs': ['kidneys', 'heart'],
                    'biomarkers': ['Creatinine', 'BUN', 'KIM-1', 'NGAL'],
                    'severity_range': [0.6, 1.0],
                    'progression': 0.1
                },
                'Glomerulonephritis': {
                    'organs': ['kidneys'],
                    'biomarkers': ['Proteinuria', 'Hematuria', 'Creatinine', 'Complement'],
                    'severity_range': [0.5, 0.8],
                    'progression': 0.06
                }
            },
            'Hepatic': {
                'Non-alcoholic Fatty Liver Disease': {
                    'organs': ['liver', 'heart', 'pancreas'],
                    'biomarkers': ['ALT', 'AST', 'GGT', 'ALP'],
                    'severity_range': [0.3, 0.7],
                    'progression': 0.03
                },
                'Cirrhosis': {
                    'organs': ['liver', 'kidneys', 'brain'],
                    'biomarkers': ['Bilirubin', 'Albumin', 'INR', 'Platelets'],
                    'severity_range': [0.7, 1.0],
                    'progression': 0.08
                },
                'Hepatitis': {
                    'organs': ['liver'],
                    'biomarkers': ['ALT', 'AST', 'Bilirubin', 'Viral load'],
                    'severity_range': [0.4, 0.8],
                    'progression': 0.05
                }
            },
            'Respiratory': {
                'COPD': {
                    'organs': ['lungs', 'heart'],
                    'biomarkers': ['FEV1', 'FVC', 'CRP', 'IL-6'],
                    'severity_range': [0.4, 0.8],
                    'progression': 0.04
                },
                'Asthma': {
                    'organs': ['lungs'],
                    'biomarkers': ['Eosinophils', 'IgE', 'FEV1', 'Methacholine'],
                    'severity_range': [0.2, 0.6],
                    'progression': 0.02
                },
                'Pulmonary Fibrosis': {
                    'organs': ['lungs', 'heart'],
                    'biomarkers': ['FVC', 'DLCO', 'KL-6', 'SP-D'],
                    'severity_range': [0.6, 0.9],
                    'progression': 0.07
                }
            },
            'Oncology': {
                'Breast Cancer': {
                    'organs': ['breast', 'liver', 'lungs', 'bones'],
                    'biomarkers': ['CA 15-3', 'CEA', 'Estrogen receptor', 'HER2'],
                    'severity_range': [0.5, 1.0],
                    'progression': 0.06
                },
                'Lung Cancer': {
                    'organs': ['lungs', 'brain', 'bones', 'liver'],
                    'biomarkers': ['CEA', 'CYFRA 21-1', 'NSE', 'ProGRP'],
                    'severity_range': [0.6, 1.0],
                    'progression': 0.08
                },
                'Colorectal Cancer': {
                    'organs': ['colon', 'liver', 'lungs'],
                    'biomarkers': ['CEA', 'CA 19-9', 'FOBT', 'CT scan'],
                    'severity_range': [0.5, 0.9],
                    'progression': 0.07
                },
                'Prostate Cancer': {
                    'organs': ['prostate', 'bones'],
                    'biomarkers': ['PSA', 'Free PSA', 'PCA3', 'PHI'],
                    'severity_range': [0.4, 0.8],
                    'progression': 0.05
                }
            },
            'Autoimmune': {
                'Rheumatoid Arthritis': {
                    'organs': ['joints', 'heart', 'lungs'],
                    'biomarkers': ['RF', 'Anti-CCP', 'CRP', 'ESR'],
                    'severity_range': [0.4, 0.8],
                    'progression': 0.04
                },
                'Lupus': {
                    'organs': ['kidneys', 'heart', 'brain', 'skin'],
                    'biomarkers': ['ANA', 'Anti-dsDNA', 'Complement', 'CRP'],
                    'severity_range': [0.5, 0.9],
                    'progression': 0.05
                },
                'Multiple Sclerosis': {
                    'organs': ['brain', 'spinal cord'],
                    'biomarkers': ['Oligoclonal bands', 'MRI lesions', 'Neurofilaments'],
                    'severity_range': [0.4, 0.8],
                    'progression': 0.03
                }
            }
        }
        
        data = []
        disease_id = 0
        
        for category, disease_list in diseases.items():
            for disease_name, disease_info in disease_list.items():
                disease_id += 1
                
                for organ in disease_info['organs']:
                    # Generate realistic impact score based on disease severity
                    impact_score = np.random.uniform(disease_info['severity_range'][0], 
                                                  disease_info['severity_range'][1])
                    
                    for biomarker in disease_info['biomarkers']:
                        # Generate realistic biomarker changes
                        if biomarker in ['Creatinine', 'BUN', 'ALT', 'AST', 'LDH']:
                            biomarker_change = np.random.uniform(1.5, 5.0)  # Enzymes increase
                        elif biomarker in ['eGFR', 'HDL', 'Albumin']:
                            biomarker_change = np.random.uniform(0.3, 0.7)  # Decrease
                        elif biomarker in ['Blood Pressure', 'Glucose', 'LDL', 'Triglycerides']:
                            biomarker_change = np.random.uniform(1.2, 2.5)  # Increase
                        elif biomarker in ['CRP', 'IL-6', 'ESR']:
                            biomarker_change = np.random.uniform(2.0, 10.0)  # Inflammatory markers
                        else:
                            biomarker_change = np.random.uniform(0.5, 3.0)
                        
                        data.append({
                            'disease_id': f'DIS_{disease_id:03d}',
                            'disease_name': disease_name,
                            'disease_type': category,
                            'severity': np.random.choice(['mild', 'moderate', 'severe'], 
                                                        p=[0.3, 0.4, 0.3]),
                            'affected_organ': organ,
                            'organ_impact_score': impact_score,
                            'physiological_parameter': f'{organ}_function',
                            'parameter_change': -impact_score * np.random.uniform(0.5, 1),
                            'biomarker': biomarker,
                            'biomarker_change': biomarker_change,
                            'progression_rate': disease_info['progression'] * np.random.uniform(0.8, 1.2),
                            'treatment_response_factor': np.random.uniform(0.3, 0.9)
                        })
        
        return pd.DataFrame(data)
    
    def generate_population_data(self, n_individuals: int = 100000) -> pd.DataFrame:
        """Generate realistic population demographic data."""
        
        # Use NHANES-like distributions
        age_groups = [(0, 18, 0.22), (18, 40, 0.33), (40, 65, 0.28), (65, 100, 0.17)]
        
        ages = []
        for min_age, max_age, prob in age_groups:
            n_group = int(n_individuals * prob)
            ages.extend(np.random.uniform(min_age, max_age, n_group).tolist())
        
        ages = np.array(ages)
        np.random.shuffle(ages)
        
        # Generate correlated anthropometric data
        height_male = np.random.normal(175.6, 7.1, n_individuals)
        height_female = np.random.normal(161.8, 6.2, n_individuals)
        
        sex = np.random.choice(['M', 'F'], n_individuals, p=[0.49, 0.51])
        height = np.where(sex == 'M', height_male, height_female)
        
        # BMI distribution
        bmi_mean = 26.5 + 0.05 * (ages - 40) / 10  # slight increase with age
        bmi = np.random.normal(bmi_mean, 4.5, n_individuals).clip(15, 50)
        weight = bmi * (height/100) ** 2
        
        data = {
            'population_id': [f'POP_{i:08d}' for i in range(n_individuals)],
            'age': ages,
            'weight': weight,
            'height': height,
            'sex': sex,
            'ethnicity': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], 
                                         n_individuals, p=[0.6, 0.12, 0.06, 0.18, 0.04]),
            'bmi': bmi,
            'body_surface_area': 0.007184 * (weight ** 0.425) * (height ** 0.725),
            'organ_volume_scaling': np.random.normal(1, 0.15, n_individuals).clip(0.6, 1.4),
            'metabolic_rate_scaling': np.random.normal(1, 0.12, n_individuals).clip(0.7, 1.3),
            'clearance_scaling': np.random.normal(1, 0.2, n_individuals).clip(0.5, 1.6),
            'volume_distribution_scaling': np.random.normal(1, 0.18, n_individuals).clip(0.6, 1.5)
        }
        
        return pd.DataFrame(data)
    
    def generate_biomarker_data(self, n_biomarkers: int = 500) -> pd.DataFrame:
        """Generate realistic biomarker reference data."""
        
        biomarker_types = ['enzymatic', 'hormonal', 'inflammatory', 'metabolic', 'cellular', 'genetic']
        organs = ['heart', 'liver', 'kidneys', 'brain', 'pancreas', 'thyroid', 'adrenal', 'muscle']
        
        data = []
        
        for i in range(n_biomarkers):
            biomarker_type = np.random.choice(biomarker_types)
            organ = np.random.choice(organs)
            
            # Generate realistic normal ranges
            if biomarker_type == 'enzymatic':
                normal_min = np.random.uniform(10, 50)
                normal_max = normal_min * np.random.uniform(1.5, 3)
                half_life = np.random.uniform(2, 48)
            elif biomarker_type == 'hormonal':
                normal_min = np.random.uniform(0.1, 10)
                normal_max = normal_min * np.random.uniform(1.2, 2)
                half_life = np.random.uniform(0.5, 24)
            else:
                normal_min = np.random.uniform(1, 100)
                normal_max = normal_min * np.random.uniform(2, 5)
                half_life = np.random.uniform(6, 72)
            
            data.append({
                'biomarker_id': f'BM_{i:04d}',
                'biomarker_name': f'Biomarker_{i}',
                'biomarker_type': biomarker_type,
                'organ_source': organ,
                'normal_range_min': normal_min,
                'normal_range_max': normal_max,
                'unit': np.random.choice(['mg/dL', 'ng/mL', 'pg/mL', 'U/L', 'IU/L', 'pmol/L']),
                'assay_method': np.random.choice(['ELISA', 'LC-MS/MS', 'Immunoassay', 'PCR', 'Flow Cytometry']),
                'stability': np.random.choice(['stable', 'moderately stable', 'unstable'], p=[0.5, 0.3, 0.2]),
                'half_life': half_life,
                'disease_associations': f'Disease_{np.random.randint(1, 100)}',
                'drug_effects': f'Effect_{np.random.randint(1, 50)}'
            })
        
        return pd.DataFrame(data)
    
    def generate_pk_parameters(self, n_drugs: int = 500) -> pd.DataFrame:
        """Generate realistic PK parameters with population variability."""
        
        population_groups = ['healthy_adult', 'elderly', 'pediatric', 'renal_impairment', 
                           'hepatic_impairment', 'obese']
        
        data = []
        
        for drug_id in range(n_drugs):
            for pop_group in population_groups:
                # Base parameters with population-specific modifications
                if pop_group == 'elderly':
                    clearance_factor = 0.7
                    volume_factor = 1.0
                elif pop_group == 'pediatric':
                    clearance_factor = 1.3
                    volume_factor = 1.2
                elif pop_group == 'renal_impairment':
                    clearance_factor = 0.4
                    volume_factor = 0.9
                elif pop_group == 'hepatic_impairment':
                    clearance_factor = 0.5
                    volume_factor = 1.1
                elif pop_group == 'obese':
                    clearance_factor = 1.1
                    volume_factor = 1.4
                else:
                    clearance_factor = 1.0
                    volume_factor = 1.0
                
                # Generate realistic PK parameters
                clearance = np.random.lognormal(1.5, 0.5) * clearance_factor
                volume = np.random.lognormal(3, 0.6) * volume_factor
                absorption = np.random.gamma(2, 0.8)
                elimination = clearance / volume
                half_life = np.log(2) / elimination
                
                data.append({
                    'drug_id': f'DRUG_{drug_id:04d}',
                    'population_group': pop_group,
                    'clearance': clearance,
                    'volume_distribution': volume,
                    'absorption_rate': absorption,
                    'elimination_rate': elimination,
                    'bioavailability': np.random.uniform(0.4, 0.95),
                    'protein_binding': np.random.uniform(0, 0.99),
                    'half_life': half_life,
                    'time_to_peak': np.random.uniform(1, 4),
                    'auc': np.random.uniform(100, 1000),
                    'cmax': np.random.uniform(10, 100),
                    'clearance_cv': np.random.uniform(0.2, 0.5),
                    'volume_cv': np.random.uniform(0.2, 0.4),
                    'absorption_cv': np.random.uniform(0.3, 0.6)
                })
        
        return pd.DataFrame(data)
    
    def generate_pd_responses(self, n_drugs: int = 500) -> pd.DataFrame:
        """Generate realistic PD response parameters."""
        
        response_types = ['efficacy', 'toxicity', 'biomarker', 'symptom_score']
        organs = ['heart', 'liver', 'kidneys', 'brain', 'lung', 'skin', 'gut']
        
        data = []
        
        for drug_id in range(n_drugs):
            n_responses = np.random.randint(1, 4)
            for _ in range(n_responses):
                response_type = np.random.choice(response_types)
                target_organ = np.random.choice(organs)
                
                # Generate realistic Emax model parameters
                emax = np.random.uniform(50, 200)
                ec50 = np.random.lognormal(2, 1)
                hill = np.random.uniform(0.8, 2.5)
                baseline = np.random.uniform(10, 50)
                
                data.append({
                    'drug_id': f'DRUG_{drug_id:04d}',
                    'target_organ': target_organ,
                    'response_type': response_type,
                    'emax': emax,
                    'ec50': ec50,
                    'hill_coefficient': hill,
                    'baseline_response': baseline,
                    'time_to_onset': np.random.uniform(0.5, 8),
                    'duration_of_action': np.random.uniform(4, 24),
                    'tolerance_development': np.random.uniform(0, 0.5),
                    'sensitization_potential': np.random.uniform(0, 0.3),
                    'therapeutic_window': np.random.uniform(2, 10),
                    'toxicity_threshold': emax * np.random.uniform(0.8, 1.2)
                })
        
        return pd.DataFrame(data)
    
    def generate_genomic_data(self, n_variants: int = 1000) -> pd.DataFrame:
        """Generate realistic pharmacogenomic data."""
        
        genes = ['CYP2D6', 'CYP2C9', 'CYP2C19', 'CYP3A4', 'CYP3A5', 'VKORC1', 
                'TPMT', 'UGT1A1', 'DPYD', 'SLC01B1', 'HLA-B', 'NUDT15']
        
        effect_types = ['increased_clearance', 'decreased_clearance', 'increased_toxicity', 
                       'decreased_efficacy', 'altered_bioavailability']
        
        evidence_levels = ['Level 1A', 'Level 1B', 'Level 2A', 'Level 2B', 'Level 3']
        
        data = []
        
        for i in range(n_variants):
            gene = np.random.choice(genes)
            n_effects = np.random.randint(1, 3)
            
            for effect_num in range(n_effects):
                effect_type = np.random.choice(effect_types)
                
                data.append({
                    'variant_id': f'rs{np.random.randint(100000, 9999999)}',
                    'gene_name': gene,
                    'variant_type': np.random.choice(['SNP', 'Indel', 'CNV', 'Haplotype']),
                    'allele_frequency': np.random.uniform(0.001, 0.5),
                    'ethnic_distribution': np.random.choice(['Global', 'European', 'Asian', 'African']),
                    'drug_id': f'DRUG_{np.random.randint(0, 500):04d}',
                    'effect_on_pk': effect_type,
                    'effect_on_pd': effect_type,
                    'clinical_significance': np.random.choice(['Significant', 'Moderate', 'Preliminary']),
                    'dosing_recommendation': np.random.choice([
                        'Standard dose', 'Decrease dose', 'Increase dose', 
                        'Avoid drug', 'Consider alternative'
                    ]),
                    'evidence_level': np.random.choice(evidence_levels)
                })
        
        return pd.DataFrame(data)


def main():
    """Main function to generate all datasets."""
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw')
    generator = RealDataGenerator(random_seed=42)
    datasets = generator.generate_all_datasets(output_dir=output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    for name, df in datasets.items():
        if isinstance(df, pd.DataFrame):
            print(f"{name:20s}: {len(df):6d} rows, {len(df.columns):3d} columns")
    
    return datasets


if __name__ == "__main__":
    datasets = main()
