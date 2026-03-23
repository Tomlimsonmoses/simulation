"""
Population Simulation Example

This example demonstrates how to run PK-PD simulations across a population
to capture inter-individual variability.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.runner import PKPDSimulationRunner, SimulationConfig
from biological_system.organ_network import PopulationGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_population_simulation():
    """Run simulation across a diverse population."""
    
    print("=== Population Simulation Example ===\n")
    
    # Configure simulation for population
    config = SimulationConfig(
        drug_name="warfarin",
        dose=5.0,
        administration_route="oral",
        time_horizon=168.0,  # 1 week
        pk_model_type="pbpk",
        pd_model_type="organ_specific",
        enable_population_variability=True,
        population_size=50,  # 50 virtual subjects
        save_results=True,
        output_dir="./population_results",
        generate_plots=True
    )
    
    runner = PKPDSimulationRunner(config)
    
    print(f"Running simulation for {config.population_size} virtual subjects...")
    population_results = runner.run_population_simulation()
    
    print(f"Population simulation completed! Generated {len(population_results)} individual results")
    
    return population_results


def analyze_population_variability(population_results):
    """Analyze variability across the population."""
    
    print("\n=== Population Variability Analysis ===\n")
    
    # Collect PK parameters across population
    pk_params = {
        'cmax': [],
        'tmax': [],
        'auc': [],
        'half_life': []
    }
    
    # Collect PD parameters across population
    pd_params = {
        'max_response': [],
        'response_auc': []
    }
    
    # Demographics
    demographics = {
        'age': [],
        'weight': [],
        'sex': []
    }
    
    for i, result in enumerate(population_results):
        # PK parameters
        if 'central' in result.pk_results:
            concentrations = result.pk_results['central']
            cmax = np.max(concentrations)
            tmax = result.output_times[np.argmax(concentrations)]
            auc = np.trapz(concentrations, result.output_times)
            
            pk_params['cmax'].append(cmax)
            pk_params['tmax'].append(tmax)
            pk_params['auc'].append(auc)
            
            # Estimate half-life
            try:
                # Simple half-life estimation
                peak_idx = np.argmax(concentrations)
                if peak_idx < len(concentrations) - 1:
                    decline = concentrations[peak_idx:]
                    decline_times = result.output_times[peak_idx:]
                    
                    # Find time when concentration drops to half of Cmax
                    half_conc = cmax / 2
                    half_idx = np.where(decline <= half_conc)[0]
                    if len(half_idx) > 0:
                        half_life = decline_times[half_idx[0]] - decline_times[0]
                        pk_params['half_life'].append(half_life)
                    else:
                        pk_params['half_life'].append(np.nan)
                else:
                    pk_params['half_life'].append(np.nan)
            except:
                pk_params['half_life'].append(np.nan)
        
        # PD parameters
        if 'heart_heart_rate' in result.pd_results:
            response = result.pd_results['heart_heart_rate']
            max_response = np.max(np.abs(response))
            response_auc = np.trapz(np.abs(response), result.output_times)
            
            pd_params['max_response'].append(max_response)
            pd_params['response_auc'].append(response_auc)
        
        # Demographics (from config - would be stored differently in real implementation)
        demographics['age'].append(np.random.normal(50, 15))  # Placeholder
        demographics['weight'].append(np.random.normal(70, 10))  # Placeholder
        demographics['sex'].append(np.random.choice(['M', 'F']))  # Placeholder
    
    # Convert to arrays for analysis
    for key in pk_params:
        pk_params[key] = np.array(pk_params[key])
    
    for key in pd_params:
        pd_params[key] = np.array(pd_params[key])
    
    # Print statistics
    print("PK Parameter Variability:")
    for param, values in pk_params.items():
        if len(values) > 0 and not np.all(np.isnan(values)):
            values_clean = values[~np.isnan(values)]
            print(f"{param.upper()}:")
            print(f"  Mean: {np.mean(values_clean):.2f}")
            print(f"  Std: {np.std(values_clean):.2f}")
            print(f"  CV: {np.std(values_clean)/np.mean(values_clean)*100:.1f}%")
            print(f"  Range: {np.min(values_clean):.2f} - {np.max(values_clean):.2f}")
            print()
    
    print("PD Parameter Variability:")
    for param, values in pd_params.items():
        if len(values) > 0:
            print(f"{param.upper()}:")
            print(f"  Mean: {np.mean(values):.2f}")
            print(f"  Std: {np.std(values):.2f}")
            print(f"  CV: {np.std(values)/np.mean(values)*100:.1f}%")
            print(f"  Range: {np.min(values):.2f} - {np.max(values):.2f}")
            print()
    
    return pk_params, pd_params, demographics


def visualize_population_results(population_results, pk_params, pd_params):
    """Create visualizations of population results."""
    
    print("Generating population visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PK variability plots
    if len(pk_params['cmax']) > 0:
        # Cmax distribution
        axes[0, 0].hist(pk_params['cmax'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Cmax (mg/L)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Cmax Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC distribution
        axes[0, 1].hist(pk_params['auc'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('AUC (mg·h/L)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('AUC Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Half-life distribution
        half_life_clean = pk_params['half_life'][~np.isnan(pk_params['half_life'])]
        if len(half_life_clean) > 0:
            axes[0, 2].hist(half_life_clean, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 2].set_xlabel('Half-life (hours)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Half-life Distribution')
            axes[0, 2].grid(True, alpha=0.3)
    
    # Concentration-time profiles (sample of individuals)
    sample_size = min(10, len(population_results))
    sample_indices = np.random.choice(len(population_results), sample_size, replace=False)
    
    colors = plt.cm.tab10(np.linspace(0, 1, sample_size))
    
    for i, idx in enumerate(sample_indices):
        result = population_results[idx]
        if 'central' in result.pk_results:
            axes[1, 0].plot(result.output_times, result.pk_results['central'], 
                           color=colors[i], alpha=0.7, linewidth=1.5,
                           label=f'Subject {idx+1}')
    
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Concentration (mg/L)')
    axes[1, 0].set_title('Sample Concentration-Time Profiles')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # PK parameter correlations
    if len(pk_params['cmax']) > 0 and len(pk_params['auc']) > 0:
        axes[1, 1].scatter(pk_params['cmax'], pk_params['auc'], alpha=0.6)
        axes[1, 1].set_xlabel('Cmax (mg/L)')
        axes[1, 1].set_ylabel('AUC (mg·h/L)')
        axes[1, 1].set_title('Cmax vs AUC Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(pk_params['cmax'], pk_params['auc'])[0, 1]
        axes[1, 1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # PD response variability
    if len(pd_params['max_response']) > 0:
        axes[1, 2].hist(pd_params['max_response'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Max Response')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Max Response Distribution')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./population_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed concentration-time plot
    plt.figure(figsize=(12, 8))
    
    # Plot all individuals with low opacity
    for result in population_results:
        if 'central' in result.pk_results:
            plt.plot(result.output_times, result.pk_results['central'], 
                    alpha=0.2, linewidth=0.5, color='blue')
    
    # Plot median and percentiles
    if len(population_results) > 0:
        # Interpolate all profiles to common time points
        common_times = population_results[0].output_times
        all_concentrations = []
        
        for result in population_results:
            if 'central' in result.pk_results:
                # Simple interpolation (could be more sophisticated)
                if len(result.output_times) == len(common_times):
                    all_concentrations.append(result.pk_results['central'])
        
        if all_concentrations:
            all_concentrations = np.array(all_concentrations)
            median_conc = np.median(all_concentrations, axis=0)
            p5_conc = np.percentile(all_concentrations, 5, axis=0)
            p95_conc = np.percentile(all_concentrations, 95, axis=0)
            
            plt.plot(common_times, median_conc, 'r-', linewidth=3, label='Median')
            plt.fill_between(common_times, p5_conc, p95_conc, alpha=0.3, color='red', 
                            label='5th-95th percentile')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mg/L)')
    plt.title('Population Concentration-Time Profiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./population_concentration_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()


def identify_subpopulations(population_results, pk_params, pd_params):
    """Identify distinct subpopulations based on PK/PD characteristics."""
    
    print("\n=== Subpopulation Analysis ===\n")
    
    # Create feature matrix for clustering
    features = []
    for i in range(len(population_results)):
        feature_vector = []
        
        # PK features
        if i < len(pk_params['cmax']):
            feature_vector.extend([
                pk_params['cmax'][i] if not np.isnan(pk_params['cmax'][i]) else 0,
                pk_params['auc'][i] if not np.isnan(pk_params['auc'][i]) else 0,
                pk_params['half_life'][i] if not np.isnan(pk_params['half_life'][i]) else 0
            ])
        
        # PD features
        if i < len(pd_params['max_response']):
            feature_vector.extend([
                pd_params['max_response'][i],
                pd_params['response_auc'][i]
            ])
        
        features.append(feature_vector)
    
    features = np.array(features)
    
    # Simple clustering based on clearance (fast vs slow metabolizers)
    if len(pk_params['auc']) > 0:
        auc_values = np.array(pk_params['auc'])
        auc_values = auc_values[~np.isnan(auc_values)]
        
        if len(auc_values) > 0:
            # Define thresholds
            median_auc = np.median(auc_values)
            low_auc_threshold = np.percentile(auc_values, 25)
            high_auc_threshold = np.percentile(auc_values, 75)
            
            # Classify subjects
            fast_metabolizers = auc_values < low_auc_threshold
            normal_metabolizers = (auc_values >= low_auc_threshold) & (auc_values <= high_auc_threshold)
            slow_metabolizers = auc_values > high_auc_threshold
            
            print(f"Metabolizer Classification based on AUC:")
            print(f"  Fast metabolizers: {np.sum(fast_metabolizers)} ({np.sum(fast_metabolizers)/len(auc_values)*100:.1f}%)")
            print(f"  Normal metabolizers: {np.sum(normal_metabolizers)} ({np.sum(normal_metabolizers)/len(auc_values)*100:.1f}%)")
            print(f"  Slow metabolizers: {np.sum(slow_metabolizers)} ({np.sum(slow_metabolizers)/len(auc_values)*100:.1f}%)")
            
            # Calculate statistics for each group
            if np.sum(fast_metabolizers) > 0:
                fast_auc = auc_values[fast_metabolizers]
                print(f"\nFast metabolizers - AUC: {np.mean(fast_auc):.1f} ± {np.std(fast_auc):.1f} mg·h/L")
            
            if np.sum(normal_metabolizers) > 0:
                normal_auc = auc_values[normal_metabolizers]
                print(f"Normal metabolizers - AUC: {np.mean(normal_auc):.1f} ± {np.std(normal_auc):.1f} mg·h/L")
            
            if np.sum(slow_metabolizers) > 0:
                slow_auc = auc_values[slow_metabolizers]
                print(f"Slow metabolizers - AUC: {np.mean(slow_auc):.1f} ± {np.std(slow_auc):.1f} mg·h/L")
            
            return {
                'fast_metabolizers': fast_metabolizers,
                'normal_metabolizers': normal_metabolizers,
                'slow_metabolizers': slow_metabolizers
            }
    
    return None


def generate_dosing_recommendations(population_results, pk_params):
    """Generate personalized dosing recommendations based on PK characteristics."""
    
    print("\n=== Dosing Recommendations ===\n")
    
    if len(pk_params['auc']) == 0:
        print("No PK data available for dosing recommendations")
        return
    
    # Target AUC range (example for warfarin)
    target_auc_min = 50.0  # mg·h/L
    target_auc_max = 100.0  # mg·h/L
    
    recommendations = []
    
    for i, result in enumerate(population_results):
        if i < len(pk_params['auc']):
            current_auc = pk_params['auc'][i]
            
            if np.isnan(current_auc):
                recommendations.append({
                    'subject': i,
                    'current_auc': np.nan,
                    'recommendation': 'Insufficient data',
                    'dose_adjustment': 0.0
                })
                continue
            
            # Calculate dose adjustment
            if current_auc < target_auc_min:
                # Increase dose
                dose_factor = target_auc_min / current_auc
                recommendation = 'Increase dose'
                dose_adjustment = dose_factor - 1.0
            elif current_auc > target_auc_max:
                # Decrease dose
                dose_factor = target_auc_max / current_auc
                recommendation = 'Decrease dose'
                dose_adjustment = dose_factor - 1.0
            else:
                # Within therapeutic range
                recommendation = 'Maintain dose'
                dose_adjustment = 0.0
            
            recommendations.append({
                'subject': i,
                'current_auc': current_auc,
                'recommendation': recommendation,
                'dose_adjustment': dose_adjustment
            })
    
    # Summarize recommendations
    rec_df = pd.DataFrame(recommendations)
    
    print("Dosing Recommendations Summary:")
    print(rec_df['recommendation'].value_counts())
    print(f"\nDose adjustment statistics:")
    print(f"  Mean adjustment: {rec_df['dose_adjustment'].mean():.2f}")
    print(f"  Std adjustment: {rec_df['dose_adjustment'].std():.2f}")
    print(f"  Range: {rec_df['dose_adjustment'].min():.2f} to {rec_df['dose_adjustment'].max():.2f}")
    
    # Save recommendations
    rec_df.to_csv('./population_dosing_recommendations.csv', index=False)
    print("\nRecommendations saved to ./population_dosing_recommendations.csv")
    
    return recommendations


def main():
    """Run the complete population simulation example."""
    
    print("=== Population PK-PD Simulation Example ===\n")
    
    # Run population simulation
    population_results = run_population_simulation()
    
    # Analyze variability
    pk_params, pd_params, demographics = analyze_population_variability(population_results)
    
    # Visualize results
    visualize_population_results(population_results, pk_params, pd_params)
    
    # Identify subpopulations
    subpopulations = identify_subpopulations(population_results, pk_params, pd_params)
    
    # Generate dosing recommendations
    recommendations = generate_dosing_recommendations(population_results, pk_params)
    
    print("\n=== Population simulation completed! ===")
    print("Results and visualizations saved to ./population_results/")
    print("Analysis plots saved as:")
    print("  - ./population_analysis.png")
    print("  - ./population_concentration_profiles.png")
    print("  - ./population_dosing_recommendations.csv")


if __name__ == "__main__":
    main()
