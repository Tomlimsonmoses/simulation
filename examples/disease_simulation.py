"""
Disease Simulation Example

This example demonstrates how to simulate drug responses in disease conditions
using the GNN-based PK-PD model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.runner import PKPDSimulationRunner, SimulationConfig
from biological_system.disease_models import DiseaseSimulator, DiseaseSeverity
import matplotlib.pyplot as plt
import numpy as np


def simulate_hypertension():
    """Simulate antihypertensive drug response in hypertension."""
    
    print("=== Hypertension Simulation ===\n")
    
    config = SimulationConfig(
        drug_name="lisinopril",
        dose=10.0,
        administration_route="oral",
        time_horizon=48.0,
        patient_age=65,
        patient_weight=85.0,
        patient_sex="male",
        disease_type="cardiovascular_hypertension",
        disease_severity="2",  # moderate hypertension
        pk_model_type="pbpk",
        pd_model_type="organ_specific",
        enable_disease_progression=True,
        save_results=True,
        output_dir="./hypertension_results"
    )
    
    runner = PKPDSimulationRunner(config)
    results = runner.run_simulation()
    
    print("Hypertension simulation completed!")
    
    # Analyze blood pressure response
    if 'heart_blood_pressure' in results.pd_results:
        bp_response = results.pd_results['heart_blood_pressure']
        baseline_bp = bp_response[0]
        final_bp = bp_response[-1]
        bp_reduction = baseline_bp - final_bp
        
        print(f"Blood pressure reduction: {bp_reduction:.1f} mmHg")
        print(f"Baseline BP: {baseline_bp:.1f} mmHg")
        print(f"Final BP: {final_bp:.1f} mmHg")
    
    return results


def simulate_diabetes():
    """Simulate antidiabetic drug response in type 2 diabetes."""
    
    print("\n=== Type 2 Diabetes Simulation ===\n")
    
    config = SimulationConfig(
        drug_name="metformin",
        dose=1000.0,
        administration_route="oral",
        time_horizon=72.0,
        patient_age=55,
        patient_weight=95.0,
        patient_sex="female",
        disease_type="metabolic_diabetes_type2",
        disease_severity="2",  # moderate diabetes
        pk_model_type="pbpk",
        pd_model_type="organ_specific",
        enable_disease_progression=True,
        save_results=True,
        output_dir="./diabetes_results"
    )
    
    runner = PKPDSimulationRunner(config)
    results = runner.run_simulation()
    
    print("Diabetes simulation completed!")
    
    # Analyze glucose response
    if 'liver_glucose_production' in results.pd_results:
        glucose_response = results.pd_results['liver_glucose_production']
        baseline_glucose = glucose_response[0]
        final_glucose = glucose_response[-1]
        glucose_reduction = baseline_glucose - final_glucose
        
        print(f"Glucose production reduction: {glucose_reduction:.1f} mg/dL")
        print(f"Baseline glucose: {baseline_glucose:.1f} mg/dL")
        print(f"Final glucose: {final_glucose:.1f} mg/dL")
    
    return results


def simulate_ckd():
    """Simulate drug response in chronic kidney disease."""
    
    print("\n=== Chronic Kidney Disease Simulation ===\n")
    
    config = SimulationConfig(
        drug_name="atorvastatin",
        dose=20.0,
        administration_route="oral",
        time_horizon=96.0,  # Longer simulation for CKD
        patient_age=70,
        patient_weight=75.0,
        patient_sex="male",
        disease_type="renal_chronic_kidney_disease",
        disease_severity="3",  # severe CKD
        pk_model_type="pbpk",
        pd_model_type="organ_specific",
        enable_disease_progression=True,
        enable_drug_interactions=True,
        save_results=True,
        output_dir="./ckd_results"
    )
    
    runner = PKPDSimulationRunner(config)
    results = runner.run_simulation()
    
    print("CKD simulation completed!")
    
    # Analyze kidney function
    if 'kidney_filtration_rate' in results.pd_results:
        gfr_response = results.pd_results['kidney_filtration_rate']
        baseline_gfr = gfr_response[0]
        final_gfr = gfr_response[-1]
        gfr_change = final_gfr - baseline_gfr
        
        print(f"GFR change: {gfr_change:.1f} mL/min")
        print(f"Baseline GFR: {baseline_gfr:.1f} mL/min")
        print(f"Final GFR: {final_gfr:.1f} mL/min")
    
    return results


def compare_healthy_vs_disease():
    """Compare drug response between healthy and disease conditions."""
    
    print("\n=== Healthy vs Disease Comparison ===\n")
    
    drug_name="propranolol"
    dose=40.0
    
    # Healthy simulation
    print("Running healthy simulation...")
    healthy_config = SimulationConfig(
        drug_name=drug_name,
        dose=dose,
        administration_route="oral",
        time_horizon=48.0,
        patient_age=50,
        patient_weight=70.0,
        patient_sex="male",
        disease_type=None,
        pk_model_type="pbpk",
        pd_model_type="organ_specific",
        save_results=False
    )
    
    healthy_runner = PKPDSimulationRunner(healthy_config)
    healthy_results = healthy_runner.run_simulation()
    
    # Disease simulation (heart failure)
    print("Running heart failure simulation...")
    disease_config = SimulationConfig(
        drug_name=drug_name,
        dose=dose,
        administration_route="oral",
        time_horizon=48.0,
        patient_age=50,
        patient_weight=70.0,
        patient_sex="male",
        disease_type="cardiovascular_heart_failure",
        disease_severity="2",
        pk_model_type="pbpk",
        pd_model_type="organ_specific",
        enable_disease_progression=True,
        save_results=False
    )
    
    disease_runner = PKPDSimulationRunner(disease_config)
    disease_results = disease_runner.run_simulation()
    
    # Compare PK parameters
    print("\nPK Comparison:")
    for organ in ['central', 'liver', 'kidney']:
        if organ in healthy_results.pk_results and organ in disease_results.pk_results:
            healthy_cmax = np.max(healthy_results.pk_results[organ])
            disease_cmax = np.max(disease_results.pk_results[organ])
            ratio = disease_cmax / healthy_cmax
            
            print(f"{organ.capitalize()} Cmax ratio (disease/healthy): {ratio:.2f}")
    
    # Compare PD responses
    print("\nPD Comparison:")
    if 'heart_heart_rate' in healthy_results.pd_results and 'heart_heart_rate' in disease_results.pd_results:
        healthy_hr = healthy_results.pd_results['heart_heart_rate']
        disease_hr = disease_results.pd_results['heart_heart_rate']
        
        healthy_reduction = healthy_hr[0] - healthy_hr[-1]
        disease_reduction = disease_hr[0] - disease_hr[-1]
        
        print(f"Heart rate reduction (healthy): {healthy_reduction:.1f} bpm")
        print(f"Heart rate reduction (disease): {disease_reduction:.1f} bpm")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # PK comparison
    if 'central' in healthy_results.pk_results and 'central' in disease_results.pk_results:
        axes[0, 0].plot(healthy_results.output_times, healthy_results.pk_results['central'], 
                       label='Healthy', linewidth=2)
        axes[0, 0].plot(disease_results.output_times, disease_results.pk_results['central'], 
                       label='Heart Failure', linewidth=2)
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Concentration (mg/L)')
        axes[0, 0].set_title('Central Compartment PK')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Heart rate response
    if 'heart_heart_rate' in healthy_results.pd_results and 'heart_heart_rate' in disease_results.pd_results:
        axes[0, 1].plot(healthy_results.output_times, healthy_results.pd_results['heart_heart_rate'], 
                       label='Healthy', linewidth=2)
        axes[0, 1].plot(disease_results.output_times, disease_results.pd_results['heart_heart_rate'], 
                       label='Heart Failure', linewidth=2)
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Heart Rate (bpm)')
        axes[0, 1].set_title('Heart Rate Response')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Blood pressure response
    if 'heart_blood_pressure' in healthy_results.pd_results and 'heart_blood_pressure' in disease_results.pd_results:
        axes[1, 0].plot(healthy_results.output_times, healthy_results.pd_results['heart_blood_pressure'], 
                       label='Healthy', linewidth=2)
        axes[1, 0].plot(disease_results.output_times, disease_results.pd_results['heart_blood_pressure'], 
                       label='Heart Failure', linewidth=2)
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Blood Pressure (mmHg)')
        axes[1, 0].set_title('Blood Pressure Response')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Disease progression
    if disease_results.disease_results:
        disease_effects = disease_results.disease_results.get('cardiovascular_heart_failure', {})
        if disease_effects:
            effects_list = list(disease_effects.items())
            if effects_list:
                params, values = zip(*effects_list)
                axes[1, 1].bar(params, values)
                axes[1, 1].set_ylabel('Effect Size')
                axes[1, 1].set_title('Disease Effects')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./disease_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return healthy_results, disease_results


def main():
    """Run all disease simulation examples."""
    
    print("=== Disease Simulation Examples ===\n")
    
    # Run individual disease simulations
    hypertension_results = simulate_hypertension()
    diabetes_results = simulate_diabetes()
    ckd_results = simulate_ckd()
    
    # Run comparison
    healthy_results, disease_results = compare_healthy_vs_disease()
    
    print("\n=== All disease simulations completed! ===")
    print("Results saved to respective directories:")
    print("- ./hypertension_results/")
    print("- ./diabetes_results/")
    print("- ./ckd_results/")
    print("- Comparison plot saved as ./disease_comparison.png")


if __name__ == "__main__":
    main()
