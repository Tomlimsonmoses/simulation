"""
PK-PD Model Simulation Script

This script runs comprehensive simulations using the trained models to predict
drug responses across different scenarios, drugs, and patient populations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from models.mlp_architectures import SimpleMLP

ROUTE_TO_ORGAN = {
    'oral': 'liver',
    'sublingual': 'liver',
    'intravenous': 'kidneys',
    'intramuscular': 'muscle',
    'subcutaneous': 'skin',
    'inhalation': 'lungs',
}


def _select_organ(organ_data, route):
    target_name = ROUTE_TO_ORGAN.get(str(route).lower(), 'liver')
    match = organ_data[organ_data['organ_name'] == target_name]
    if len(match) > 0:
        return match.iloc[0]
    return organ_data.iloc[0]


class PKPDSimulator:
    """Comprehensive PK-PD simulation system."""
    
    def __init__(self, model_path=None):
        """Initialize simulator with trained model."""
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'simple_mlp_model.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load trained model (input_dim=32 after feature engineering fixes)
        self.model = SimpleMLP(input_dim=32, hidden_dim=128, output_dim=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load datasets for simulation
        self.load_datasets()
        
    def load_datasets(self):
        """Load all required datasets."""
        print("Loading simulation datasets...")
        
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
        self.drug_data = pd.read_csv(os.path.join(data_dir, 'drug_properties.csv'))
        self.organ_data = pd.read_csv(os.path.join(data_dir, 'organ_network.csv'))
        self.clinical_data = pd.read_csv(os.path.join(data_dir, 'clinical_trials.csv'))
        self.disease_data = pd.read_csv(os.path.join(data_dir, 'disease_data.csv'))
        
        print(f"Loaded {len(self.drug_data)} drugs, {len(self.organ_data)} organs")
        print(f"Loaded {len(self.clinical_data)} clinical samples")
        print(f"Loaded {len(self.disease_data)} disease records")
    
    def create_simulation_features(self, drug_name, patient_profile, disease_state=None):
        """Create feature vector for simulation."""
        
        # Find drug info
        drug_info = self.drug_data[self.drug_data['drug_name'] == drug_name]
        if len(drug_info) == 0:
            drug_info = self.drug_data.iloc[0]
        else:
            drug_info = drug_info.iloc[0]
        
        # Select organ matching drug's metabolic pathway
        route = drug_info['administration_route']
        organ = _select_organ(self.organ_data, route)

        # Calculate disease impact
        disease_impact = 0.0
        if disease_state:
            organ_diseases = self.disease_data[
                self.disease_data['affected_organ'] == organ['organ_name']
            ]
            if len(organ_diseases) > 0:
                disease_impact = organ_diseases['organ_impact_score'].mean()

        # Pre-computed PK features
        clearance = float(drug_info['clearance_rate'])
        vd = float(drug_info['volume_distribution'])
        bioavail = float(drug_info['bioavailability'])
        dose = float(patient_profile['dose'])
        k_el = clearance / max(vd, 1e-6)
        c_max = dose * bioavail / max(vd, 1e-6)
        auc_24 = dose * bioavail / max(clearance, 1e-6)

        # Create 32-dim feature vector
        features = [
            # Drug features (10)
            drug_info['molecular_weight'] / 1000.0,
            drug_info['logp'] / 10.0,
            bioavail,
            drug_info['protein_binding'],
            clearance / 5.0,
            vd / 10.0,
            drug_info['half_life'] / 100.0,
            1.0 if drug_info['drug_class'] == 'small_molecule' else 0.0,
            1.0 if route == 'oral' else 0.0,
            dose / 1000.0,

            # Organ features (10)
            organ['volume'] / 50.0,
            organ['blood_flow'] / 2000.0,
            organ['metabolic_rate'],
            organ['health_status'],
            organ['ph'] / 14.0,
            (organ['temperature'] - 35.0) / 5.0,
            organ['water_content'],
            organ['lipid_content'],
            organ['protein_content'],
            disease_impact,

            # Patient features (9)  — administration_route removed (duplicate)
            patient_profile['age'] / 100.0,
            patient_profile['weight'] / 200.0,
            1.0 if patient_profile['sex'] == 'M' else 0.0,
            1.0 if patient_profile['ethnicity'] == 'Caucasian' else 0.0,
            1.0 if patient_profile['ethnicity'] == 'Asian' else 0.0,
            1.0 if patient_profile['ethnicity'] == 'African' else 0.0,
            1.0 if patient_profile['ethnicity'] == 'Hispanic' else 0.0,
            patient_profile.get('time_point', 0.0) / 24.0,
            0.0,  # No adverse event in simulation

            # Pre-computed PK features (3)
            np.clip(k_el, 0.0, 5.0),
            np.clip(c_max / 1000.0, 0.0, 10.0),
            np.clip(auc_24 / 10000.0, 0.0, 10.0),
        ]

        return np.array(features, dtype=np.float32)
    
    def predict_pk_pd(self, drug_name, patient_profile, disease_state=None):
        """Predict PK and PD responses for a given scenario."""
        
        features = self.create_simulation_features(drug_name, patient_profile, disease_state)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features_tensor)
            predictions = predictions.cpu().numpy()[0]
        
        # Convert back from normalized values
        pk_concentration = predictions[0] * 1000.0  # Convert back to original scale
        pd_response = predictions[1] * 100.0  # Convert back to original scale
        
        return {
            'drug_name': drug_name,
            'pk_concentration': pk_concentration,
            'pd_response': pd_response,
            'patient_profile': patient_profile,
            'disease_state': disease_state
        }
    
    def simulate_dose_response(self, drug_name, patient_profile, dose_range):
        """Simulate dose-response curve for a drug."""
        
        results = []
        for dose in dose_range:
            profile = patient_profile.copy()
            profile['dose'] = dose
            
            prediction = self.predict_pk_pd(drug_name, profile)
            results.append({
                'dose': dose,
                'pk_concentration': prediction['pk_concentration'],
                'pd_response': prediction['pd_response']
            })
        
        return pd.DataFrame(results)
    
    def simulate_time_course(self, drug_name, patient_profile, time_points):
        """Simulate PK/PD time course after drug administration."""
        
        results = []
        for time in time_points:
            profile = patient_profile.copy()
            profile['time_point'] = time
            
            prediction = self.predict_pk_pd(drug_name, profile)
            results.append({
                'time': time,
                'pk_concentration': prediction['pk_concentration'],
                'pd_response': prediction['pd_response']
            })
        
        return pd.DataFrame(results)
    
    def simulate_population_response(self, drug_name, base_profile, n_patients=100):
        """Simulate response across a patient population."""
        
        results = []
        
        for i in range(n_patients):
            # Create varied patient profiles
            profile = base_profile.copy()
            
            # Add realistic variations
            profile['age'] = np.random.normal(base_profile['age'], 10)
            profile['weight'] = np.random.normal(base_profile['weight'], 15)
            profile['sex'] = np.random.choice(['M', 'F'])
            profile['ethnicity'] = np.random.choice(['Caucasian', 'Asian', 'African', 'Hispanic'])
            
            # Ensure realistic bounds
            profile['age'] = np.clip(profile['age'], 18, 85)
            profile['weight'] = np.clip(profile['weight'], 40, 150)
            
            prediction = self.predict_pk_pd(drug_name, profile)
            results.append({
                'patient_id': i,
                'age': profile['age'],
                'weight': profile['weight'],
                'sex': profile['sex'],
                'ethnicity': profile['ethnicity'],
                'pk_concentration': prediction['pk_concentration'],
                'pd_response': prediction['pd_response']
            })
        
        return pd.DataFrame(results)
    
    def run_comprehensive_simulation(self):
        """Run comprehensive simulation suite."""
        
        print("\n" + "="*60)
        print("COMPREHENSIVE PK-PD SIMULATION")
        print("="*60)
        
        # Simulation scenarios
        scenarios = {
            'young_adult': {'age': 25, 'weight': 70, 'sex': 'M', 'ethnicity': 'Caucasian', 
                          'dose': 500, 'administration_route': 'oral'},
            'elderly': {'age': 75, 'weight': 65, 'sex': 'F', 'ethnicity': 'Asian', 
                       'dose': 300, 'administration_route': 'oral'},
            'pediatric': {'age': 12, 'weight': 40, 'sex': 'M', 'ethnicity': 'African', 
                         'dose': 200, 'administration_route': 'oral'},
            'obese': {'age': 45, 'weight': 120, 'sex': 'F', 'ethnicity': 'Hispanic', 
                     'dose': 800, 'administration_route': 'oral'}
        }
        
        # Test drugs
        test_drugs = ['Acetaminophen', 'Ibuprofen', 'Metformin', 'Lisinopril']
        
        all_results = []
        
        # 1. Single dose simulations
        print("\n1. Single Dose Simulations")
        print("-" * 40)
        
        for drug in test_drugs:
            print(f"\nSimulating {drug}...")
            
            for scenario_name, profile in scenarios.items():
                prediction = self.predict_pk_pd(drug, profile)
                
                result = {
                    'drug': drug,
                    'scenario': scenario_name,
                    'pk_concentration': prediction['pk_concentration'],
                    'pd_response': prediction['pd_response'],
                    'age': profile['age'],
                    'weight': profile['weight'],
                    'sex': profile['sex'],
                    'dose': profile['dose']
                }
                all_results.append(result)
                
                print(f"  {scenario_name:12}: PK={prediction['pk_concentration']:.2f}, "
                      f"PD={prediction['pd_response']:.2f}")
        
        # 2. Dose-response curves
        print("\n2. Dose-Response Curves")
        print("-" * 40)
        
        dose_responses = {}
        base_profile = scenarios['young_adult']
        
        for drug in test_drugs[:2]:  # Limit for visualization
            print(f"Generating dose-response for {drug}...")
            dose_range = np.linspace(100, 1000, 20)
            dose_response = self.simulate_dose_response(drug, base_profile, dose_range)
            dose_responses[drug] = dose_response
        
        # 3. Time course simulation
        print("\n3. Time Course Simulation")
        print("-" * 40)
        
        time_courses = {}
        for drug in test_drugs[:2]:
            print(f"Generating time course for {drug}...")
            time_points = np.linspace(0, 24, 24)  # 24 hours
            time_course = self.simulate_time_course(drug, base_profile, time_points)
            time_courses[drug] = time_course
        
        # 4. Population simulation
        print("\n4. Population Response Simulation")
        print("-" * 40)
        
        population_responses = {}
        for drug in test_drugs[:2]:
            print(f"Simulating population response for {drug}...")
            pop_response = self.simulate_population_response(drug, base_profile, n_patients=50)
            population_responses[drug] = pop_response
        
        # Create visualizations
        self.create_simulation_plots(all_results, dose_responses, time_courses, population_responses)
        
        return {
            'single_dose': pd.DataFrame(all_results),
            'dose_response': dose_responses,
            'time_course': time_courses,
            'population': population_responses
        }
    
    def create_simulation_plots(self, single_dose_results, dose_responses, time_courses, population_responses):
        """Create comprehensive visualization of simulation results."""
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Single dose comparison
        ax1 = plt.subplot(2, 4, 1)
        df_single = pd.DataFrame(single_dose_results)
        drugs = df_single['drug'].unique()
        scenarios = df_single['scenario'].unique()
        
        x = np.arange(len(scenarios))
        width = 0.2
        
        for i, drug in enumerate(drugs[:4]):  # Limit drugs for visibility
            drug_data = df_single[df_single['drug'] == drug]
            pk_values = [drug_data[drug_data['scenario'] == s]['pk_concentration'].values[0] 
                        for s in scenarios]
            ax1.bar(x + i*width, pk_values, width, label=drug)
        
        ax1.set_xlabel('Patient Scenario')
        ax1.set_ylabel('PK Concentration')
        ax1.set_title('PK Response by Drug and Scenario')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. PD response comparison
        ax2 = plt.subplot(2, 4, 2)
        for i, drug in enumerate(drugs[:4]):
            drug_data = df_single[df_single['drug'] == drug]
            pd_values = [drug_data[drug_data['scenario'] == s]['pd_response'].values[0] 
                        for s in scenarios]
            ax2.bar(x + i*width, pd_values, width, label=drug)
        
        ax2.set_xlabel('Patient Scenario')
        ax2.set_ylabel('PD Response')
        ax2.set_title('PD Response by Drug and Scenario')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Dose-response curves
        ax3 = plt.subplot(2, 4, 3)
        for drug, df in dose_responses.items():
            ax3.plot(df['dose'], df['pk_concentration'], label=f'{drug} (PK)', marker='o')
            ax3.plot(df['dose'], df['pd_response'], label=f'{drug} (PD)', marker='s', linestyle='--')
        
        ax3.set_xlabel('Dose (mg)')
        ax3.set_ylabel('Response')
        ax3.set_title('Dose-Response Curves')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Time course curves
        ax4 = plt.subplot(2, 4, 4)
        for drug, df in time_courses.items():
            ax4.plot(df['time'], df['pk_concentration'], label=f'{drug} (PK)', marker='o')
            ax4.plot(df['time'], df['pd_response'], label=f'{drug} (PD)', marker='s', linestyle='--')
        
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Response')
        ax4.set_title('PK/PD Time Course')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Population PK distribution
        ax5 = plt.subplot(2, 4, 5)
        for drug, df in population_responses.items():
            ax5.hist(df['pk_concentration'], alpha=0.7, label=f'{drug}', bins=15)
        
        ax5.set_xlabel('PK Concentration')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Population PK Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Population PD distribution
        ax6 = plt.subplot(2, 4, 6)
        for drug, df in population_responses.items():
            ax6.hist(df['pd_response'], alpha=0.7, label=f'{drug}', bins=15)
        
        ax6.set_xlabel('PD Response')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Population PD Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Age vs Response scatter
        ax7 = plt.subplot(2, 4, 7)
        for drug, df in population_responses.items():
            ax7.scatter(df['age'], df['pk_concentration'], alpha=0.6, label=f'{drug} (PK)')
            ax7.scatter(df['age'], df['pd_response'], alpha=0.6, label=f'{drug} (PD)', marker='s')
        
        ax7.set_xlabel('Age (years)')
        ax7.set_ylabel('Response')
        ax7.set_title('Age vs Response')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Weight vs Response scatter
        ax8 = plt.subplot(2, 4, 8)
        for drug, df in population_responses.items():
            ax8.scatter(df['weight'], df['pk_concentration'], alpha=0.6, label=f'{drug} (PK)')
            ax8.scatter(df['weight'], df['pd_response'], alpha=0.6, label=f'{drug} (PD)', marker='s')
        
        ax8.set_xlabel('Weight (kg)')
        ax8.set_ylabel('Response')
        ax8.set_title('Weight vs Response')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        os.makedirs('results', exist_ok=True)
        plt.tight_layout()
        plt.savefig('results/comprehensive_simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nSimulation plots saved to 'results/comprehensive_simulation_results.png'")


def main():
    """Main simulation function."""
    print("PK-PD Model Simulation Suite")
    print("="*60)
    
    # Initialize simulator
    simulator = PKPDSimulator()
    
    # Run comprehensive simulations
    results = simulator.run_comprehensive_simulation()
    
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    # Print summary statistics
    single_dose_df = results['single_dose']
    
    print(f"\nSingle Dose Simulations: {len(single_dose_df)} scenarios")
    print(f"Average PK Concentration: {single_dose_df['pk_concentration'].mean():.2f} ± {single_dose_df['pk_concentration'].std():.2f}")
    print(f"Average PD Response: {single_dose_df['pd_response'].mean():.2f} ± {single_dose_df['pd_response'].std():.2f}")
    
    # Population analysis
    for drug, pop_df in results['population'].items():
        print(f"\n{drug} Population Analysis (n={len(pop_df)}):")
        print(f"  PK: {pop_df['pk_concentration'].mean():.2f} ± {pop_df['pk_concentration'].std():.2f}")
        print(f"  PD: {pop_df['pd_response'].mean():.2f} ± {pop_df['pd_response'].std():.2f}")
        print(f"  Age range: {pop_df['age'].min():.0f}-{pop_df['age'].max():.0f} years")
        print(f"  Weight range: {pop_df['weight'].min():.0f}-{pop_df['weight'].max():.0f} kg")
    
    print("\nSimulation completed successfully!")
    print("All results and visualizations saved.")
    
    return results


if __name__ == "__main__":
    simulation_results = main()
