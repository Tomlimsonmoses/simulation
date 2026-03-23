"""
Basic PK-PD Simulation Example

This example demonstrates how to run a basic PK-PD simulation
using the GNN-based model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.runner import PKPDSimulationRunner, SimulationConfig, run_quick_simulation
import matplotlib.pyplot as plt


def main():
    """Run a basic simulation example."""
    
    print("=== Basic PK-PD Simulation Example ===\n")
    
    # Method 1: Using the quick simulation function
    print("1. Running quick simulation...")
    results = run_quick_simulation(
        drug_name="example_drug",
        dose=100.0,
        time_horizon=48.0,
        patient_age=45,
        patient_weight=70.0
    )
    
    print(f"Simulation completed! Results summary:")
    print(f"- Time points: {len(results.output_times)}")
    print(f"- PK organs: {list(results.pk_results.keys())}")
    print(f"- PD responses: {list(results.pd_results.keys())}")
    print(f"- Summary stats: {len(results.summary_stats)}")
    
    # Method 2: Using the full configuration
    print("\n2. Running detailed simulation with custom configuration...")
    
    config = SimulationConfig(
        drug_name="lisinopril",
        dose=10.0,
        administration_route="oral",
        time_horizon=72.0,
        patient_age=65,
        patient_weight=80.0,
        patient_sex="male",
        disease_type="cardiovascular_hypertension",
        disease_severity="2",  # moderate
        pk_model_type="pbpk",
        pd_model_type="organ_specific",
        enable_disease_progression=True,
        save_results=True,
        output_dir="./example_results",
        generate_plots=True
    )
    
    runner = PKPDSimulationRunner(config)
    results = runner.run_simulation()
    
    print(f"Detailed simulation completed!")
    print(f"- Cmax (central): {results.summary_stats.get('central_cmax', 'N/A'):.2f} mg/L")
    print(f"- Tmax (central): {results.summary_stats.get('central_tmax', 'N/A'):.2f} h")
    print(f"- AUC (central): {results.summary_stats.get('central_auc', 'N/A'):.2f} mg·h/L")
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    runner.visualize_results(results)
    
    # Generate report
    print("\n4. Generating simulation report...")
    report = runner.generate_report(results)
    
    # Save report
    with open("./example_results/simulation_report.md", "w") as f:
        f.write(report)
    
    print("Report saved to ./example_results/simulation_report.md")
    
    # Display some sample plots
    if results.pk_results:
        plt.figure(figsize=(10, 6))
        for organ, concentrations in results.pk_results.items():
            plt.plot(results.output_times, concentrations, label=organ.capitalize(), linewidth=2)
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Concentration (mg/L)')
        plt.title('Drug Concentration-Time Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
