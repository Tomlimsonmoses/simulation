"""
Main Simulation Runner for GNN PK-PD Model

This module implements the main simulation interface that integrates all components:
- Virtual human biological system
- Disease simulation
- PK-PD modeling with GNN
- Configurable simulation parameters
- Results visualization and export
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
import logging
from pathlib import Path

# Import project modules
from ..models.gnn_pkpd import PKPDGNN, PKPDDataProcessor
from ..models.pk_simulator import MultiCompartmentPK, PhysiologicallyBasedPK, DrugProperties
from ..models.pd_simulator import SystemicPDModel, OrganSpecificPD
from ..biological_system.organ_network import VirtualHuman, PopulationGenerator
from ..biological_system.disease_models import DiseaseSimulator, DiseaseSimulator, DiseaseDrugInteractions
from ..data.dataset_specs import DatasetRequirements, DataPipeline


@dataclass
class SimulationConfig:
    """Configuration for PK-PD simulation."""
    # Time parameters
    time_horizon: float = 72.0  # hours
    time_step: float = 0.1  # hours
    output_interval: float = 1.0  # hours
    
    # Drug parameters
    drug_name: str = "example_drug"
    dose: float = 100.0  # mg
    administration_route: str = "iv"
    dosing_schedule: List[float] = field(default_factory=lambda: [0.0])  # hours
    
    # Patient parameters
    patient_age: int = 45
    patient_weight: float = 70.0  # kg
    patient_height: float = 170.0  # cm
    patient_sex: str = "male"
    patient_ethnicity: str = "caucasian"
    
    # Disease parameters
    disease_type: Optional[str] = None
    disease_severity: Optional[str] = None
    disease_onset_time: float = 0.0
    
    # Model parameters
    use_gnn_model: bool = True
    pk_model_type: str = "pbpk"  # "compartmental" or "pbpk"
    pd_model_type: str = "organ_specific"  # "organ_specific" or "systemic"
    
    # Simulation options
    enable_disease_progression: bool = True
    enable_drug_interactions: bool = True
    enable_population_variability: bool = False
    population_size: int = 1
    
    # Output options
    save_results: bool = True
    output_dir: str = "./simulation_results"
    generate_plots: bool = True
    export_format: List[str] = field(default_factory=lambda: ["csv", "json"])


class SimulationResults:
    """Container for simulation results."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.start_time = datetime.now()
        
        # Time points
        self.time_points = np.arange(0, config.time_horizon + config.time_step, config.time_step)
        self.output_indices = np.arange(0, len(self.time_points), int(config.output_interval / config.time_step))
        self.output_times = self.time_points[self.output_indices]
        
        # Results storage
        self.pk_results = {}
        self.pd_results = {}
        self.disease_results = {}
        self.physiological_state = []
        self.gnn_predictions = {}
        
        # Summary statistics
        self.summary_stats = {}
        
    def add_pk_result(self, organ_name: str, concentrations: np.ndarray):
        """Add PK concentration results."""
        self.pk_results[organ_name] = concentrations[self.output_indices]
    
    def add_pd_result(self, organ_name: str, responses: Dict[str, np.ndarray]):
        """Add PD response results."""
        if organ_name not in self.pd_results:
            self.pd_results[organ_name] = {}
        
        for response_name, response_values in responses.items():
            self.pd_results[organ_name][response_name] = response_values[self.output_indices]
    
    def add_disease_result(self, disease_name: str, effects: Dict[str, float]):
        """Add disease effects."""
        self.disease_results[disease_name] = effects
    
    def add_physiological_state(self, state: Dict[str, float]):
        """Add physiological state snapshot."""
        self.physiological_state.append({
            'time': len(self.physiological_state) * self.config.time_step,
            **state
        })
    
    def add_gnn_prediction(self, prediction_type: str, values: torch.Tensor):
        """Add GNN model predictions."""
        self.gnn_predictions[prediction_type] = values.detach().cpu().numpy()[self.output_indices]
    
    def calculate_summary_statistics(self):
        """Calculate summary statistics."""
        stats = {}
        
        # PK statistics
        if self.pk_results:
            for organ, concentrations in self.pk_results.items():
                stats[f'{organ}_cmax'] = np.max(concentrations)
                stats[f'{organ}_tmax'] = self.output_times[np.argmax(concentrations)]
                stats[f'{organ}_auc'] = np.trapz(concentrations, self.output_times)
                stats[f'{organ}_half_life'] = self._calculate_half_life(concentrations, self.output_times)
        
        # PD statistics
        if self.pd_results:
            for organ, responses in self.pd_results.items():
                for response_name, response_values in responses.items():
                    stats[f'{organ}_{response_name}_max'] = np.max(response_values)
                    stats[f'{organ}_{response_name}_min'] = np.min(response_values)
                    stats[f'{organ}_{response_name}_auc'] = np.trapz(response_values, self.output_times)
        
        # Disease statistics
        if self.disease_results:
            for disease, effects in self.disease_results.items():
                for param, effect in effects.items():
                    stats[f'{disease}_{param}_effect'] = effect
        
        self.summary_stats = stats
    
    def _calculate_half_life(self, concentrations: np.ndarray, times: np.ndarray) -> float:
        """Calculate elimination half-life."""
        # Find peak concentration
        peak_idx = np.argmax(concentrations)
        
        # Find points after peak for elimination phase
        elim_conc = concentrations[peak_idx:]
        elim_times = times[peak_idx:]
        
        if len(elim_conc) < 2:
            return float('inf')
        
        # Fit exponential decay
        try:
            # Use log-linear regression
            log_conc = np.log(elim_conc[elim_conc > 0])
            log_times = elim_times[elim_conc > 0]
            
            if len(log_conc) < 2:
                return float('inf')
            
            slope, _ = np.polyfit(log_times, log_conc, 1)
            half_life = -np.log(2) / slope
            return max(0, half_life)
        except:
            return float('inf')
    
    def export_results(self, output_dir: str, formats: List[str]):
        """Export results to specified formats."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        for format_type in formats:
            if format_type == "csv":
                self._export_csv(output_dir, timestamp)
            elif format_type == "json":
                self._export_json(output_dir, timestamp)
            elif format_type == "excel":
                self._export_excel(output_dir, timestamp)
    
    def _export_csv(self, output_dir: str, timestamp: str):
        """Export results to CSV files."""
        base_path = Path(output_dir) / f"simulation_{timestamp}"
        
        # PK results
        if self.pk_results:
            pk_df = pd.DataFrame({
                'time': self.output_times,
                **self.pk_results
            })
            pk_df.to_csv(base_path / "pk_results.csv", index=False)
        
        # PD results
        if self.pd_results:
            for organ, responses in self.pd_results.items():
                pd_df = pd.DataFrame({
                    'time': self.output_times,
                    **responses
                })
                pd_df.to_csv(base_path / f"pd_results_{organ}.csv", index=False)
        
        # Disease results
        if self.disease_results:
            disease_df = pd.DataFrame([self.disease_results])
            disease_df.to_csv(base_path / "disease_effects.csv", index=False)
        
        # Physiological state
        if self.physiological_state:
            phys_df = pd.DataFrame(self.physiological_state)
            phys_df.to_csv(base_path / "physiological_state.csv", index=False)
        
        # Summary statistics
        if self.summary_stats:
            summary_df = pd.DataFrame([self.summary_stats])
            summary_df.to_csv(base_path / "summary_statistics.csv", index=False)
    
    def _export_json(self, output_dir: str, timestamp: str):
        """Export results to JSON file."""
        base_path = Path(output_dir) / f"simulation_{timestamp}.json"
        
        export_data = {
            'config': self.config.__dict__,
            'time_points': self.output_times.tolist(),
            'pk_results': {k: v.tolist() for k, v in self.pk_results.items()},
            'pd_results': {
                k: {rk: rv.tolist() for rk, rv in v.items()}
                for k, v in self.pd_results.items()
            },
            'disease_results': self.disease_results,
            'physiological_state': self.physiological_state,
            'gnn_predictions': {k: v.tolist() for k, v in self.gnn_predictions.items()},
            'summary_statistics': self.summary_stats,
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'simulation_duration': (datetime.now() - self.start_time).total_seconds()
            }
        }
        
        with open(base_path, 'w') as f:
            json.dump(export_data, f, indent=2)


class PKPDSimulationRunner:
    """Main simulation runner for PK-PD modeling."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        
        # Initialize components
        self.virtual_human = None
        self.disease_simulator = None
        self.pk_model = None
        self.pd_model = None
        self.gnn_model = None
        self.data_processor = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Results
        self.results = None
        
    def initialize_simulation(self) -> bool:
        """Initialize all simulation components."""
        try:
            # Create virtual human
            self.virtual_human = VirtualHuman(
                age=self.config.patient_age,
                weight=self.config.patient_weight,
                height=self.config.patient_height,
                sex=self.config.patient_sex,
                ethnicity=self.config.patient_ethnicity
            )
            
            # Initialize disease simulator if disease specified
            if self.config.disease_type:
                self.disease_simulator = DiseaseSimulator()
                disease = DiseaseSimulator.create_disease(
                    self.config.disease_type.split('_')[0],
                    self.config.disease_type,
                    DiseaseSeverity(int(self.config.disease_severity))
                )
                self.disease_simulator.add_disease(disease)
            
            # Initialize PK model
            if self.config.pk_model_type == "compartmental":
                self.pk_model = MultiCompartmentPK()
            else:  # pbpk
                drug_props = self._create_drug_properties()
                self.pk_model = PhysiologicallyBasedPK(drug_props)
            
            # Initialize PD model
            if self.config.pd_model_type == "organ_specific":
                organ_types = [organ.value for organ in self.virtual_human.organs.keys()]
                self.pd_model = SystemicPDModel(organ_types)
            else:  # systemic
                self.pd_model = SystemicPDModel(['heart', 'liver', 'kidney', 'brain'])
            
            # Initialize GNN model if enabled
            if self.config.use_gnn_model:
                self.gnn_model = PKPDGNN()
                self.data_processor = PKPDDataProcessor()
            
            self.logger.info("Simulation components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize simulation: {str(e)}")
            return False
    
    def _create_drug_properties(self) -> DrugProperties:
        """Create drug properties from configuration."""
        # Default drug properties - in real implementation, these would come from database
        return DrugProperties(
            molecular_weight=300.0,
            logp=1.0,
            pka=7.4,
            solubility=100.0,
            bioavailability=0.8,
            protein_binding=0.7,
            permeability=1e-6,
            clearance_rate=0.1,
            volume_distribution=1.0,
            half_life=10.0,
            absorption_rate=1.0,
            metabolism_rate=0.5,
            excretion_rate=0.3
        )
    
    def run_simulation(self) -> SimulationResults:
        """Run the complete PK-PD simulation."""
        if not self.initialize_simulation():
            raise RuntimeError("Failed to initialize simulation")
        
        self.logger.info(f"Starting PK-PD simulation for {self.config.drug_name}")
        
        # Initialize results container
        self.results = SimulationResults(self.config)
        
        # Get network data for GNN
        if self.config.use_gnn_model and self.gnn_model:
            network_data = self.virtual_human.export_network_data()
        
        # Main simulation loop
        num_steps = len(self.results.time_points)
        
        for step in range(num_steps):
            current_time = self.results.time_points[step]
            
            # Update disease state
            if self.disease_simulator and self.config.enable_disease_progression:
                disease_effects = self.disease_simulator.update_all_diseases(self.config.time_step)
                
                # Apply disease effects to virtual human
                for organ_name, effect in disease_effects.items():
                    if organ_name in [o.value for o in self.virtual_human.organs.keys()]:
                        self.virtual_human.modify_organ_health(organ_name, max(0.0, 1.0 + effect))
            
            # Update physiological state
            phys_state = self.virtual_human.simulate_physiological_response(self.config.time_step)
            
            # Check if dose should be administered
            dose_amount = 0.0
            for dose_time in self.config.dosing_schedule:
                if abs(current_time - dose_time) < self.config.time_step / 2:
                    dose_amount = self.config.dose
                    break
            
            # PK simulation
            if self.config.pk_model_type == "compartmental":
                pk_result = self.pk_model.simulate(
                    dose=dose_amount,
                    dosing_regimen='bolus' if dose_amount > 0 else 'none',
                    time_points=np.array([current_time])
                )
                
                # Extract concentrations
                for i, organ_name in enumerate(['central', 'peripheral1', 'peripheral2']):
                    if i < pk_result['concentrations'].shape[1]:
                        conc = pk_result['concentrations'][0, i]
                        if organ_name not in self.results.pk_results:
                            self.results.pk_results[organ_name] = []
                        self.results.pk_results[organ_name].append(conc)
            
            else:  # pbpk
                pk_result = self.pk_model.simulate(
                    dose=dose_amount,
                    administration_route=self.config.administration_route,
                    time_points=np.array([current_time])
                )
                
                # Extract organ concentrations
                for organ_name, concentrations in pk_result['concentrations'].items():
                    if organ_name not in self.results.pk_results:
                        self.results.pk_results[organ_name] = []
                    self.results.pk_results[organ_name].append(concentrations[0])
            
            # PD simulation
            for organ_name in self.virtual_human.organs.keys():
                if organ_name.value in self.results.pk_results:
                    drug_conc = np.array([self.results.pk_results[organ_name.value][-1]])
                    
                    # Create organ-specific PD model
                    organ_pd = OrganSpecificPD(organ_name.value)
                    
                    # Simulate organ response
                    drug_receptor_affinity = {  # Simplified affinity mapping
                        'beta1': 1.0, 'beta2': 1.0, 'at1_receptor': 1.0,
                        'dopamine_d1': 1.0, 'serotonin_5ht': 1.0
                    }
                    
                    organ_response = organ_pd.simulate_organ_response(
                        drug_conc,
                        np.array([current_time]),
                        drug_receptor_affinity
                    )
                    
                    # Store PD results
                    for response_name, response_values in organ_response.items():
                        key = f"{organ_name.value}_{response_name}"
                        if key not in self.results.pd_results:
                            self.results.pd_results[key] = []
                        self.results.pd_results[key].append(response_values[0])
            
            # GNN prediction (if enabled)
            if self.config.use_gnn_model and self.gnn_model and step % 10 == 0:  # Every 10 steps for efficiency
                try:
                    # Prepare GNN input
                    organ_features = network_data['node_features']
                    drug_features = torch.tensor([[300.0, 1.0, 0.8, 0.7, 0.1, 1.0]], dtype=torch.float32)
                    edge_index = network_data['edge_index']
                    edge_attr = network_data['edge_attr']
                    
                    # Disease features
                    disease_features = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)  # Healthy
                    if self.disease_simulator:
                        disease_summary = self.disease_simulator.get_disease_summary()
                        if disease_summary['total_diseases'] > 0:
                            disease_features = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
                    
                    batch = torch.zeros(organ_features.shape[0] + 1, dtype=torch.long)
                    
                    # Forward pass
                    with torch.no_grad():
                        predictions = self.gnn_model(
                            organ_features,
                            drug_features,
                            edge_index,
                            edge_attr,
                            disease_features,
                            batch
                        )
                        
                        # Store predictions
                        self.results.add_gnn_prediction('pk_concentrations', predictions['pk_concentrations'])
                        self.results.add_gnn_prediction('pd_responses', predictions['pd_responses'])
                
                except Exception as e:
                    self.logger.warning(f"GNN prediction failed at step {step}: {str(e)}")
            
            # Store physiological state
            if step % int(self.config.output_interval / self.config.time_step) == 0:
                self.results.add_physiological_state(phys_state)
        
        # Convert lists to arrays
        for key in self.results.pk_results:
            self.results.pk_results[key] = np.array(self.results.pk_results[key])
        
        for key in self.results.pd_results:
            self.results.pd_results[key] = np.array(self.results.pd_results[key])
        
        # Calculate summary statistics
        self.results.calculate_summary_statistics()
        
        # Export results
        if self.config.save_results:
            self.results.export_results(self.config.output_dir, self.config.export_format)
        
        self.logger.info("Simulation completed successfully")
        return self.results
    
    def run_population_simulation(self) -> List[SimulationResults]:
        """Run simulation for a population of virtual humans."""
        if not self.config.enable_population_variability:
            return [self.run_simulation()]
        
        # Generate population
        pop_generator = PopulationGenerator(self.config.population_size)
        population = pop_generator.generate_population()
        
        results_list = []
        
        for i, virtual_human in enumerate(population):
            self.logger.info(f"Running simulation for individual {i + 1}/{self.config.population_size}")
            
            # Update config with individual parameters
            self.config.patient_age = virtual_human.age
            self.config.patient_weight = virtual_human.weight
            self.config.patient_height = virtual_human.height
            self.config.patient_sex = virtual_human.sex
            
            # Run simulation
            individual_result = self.run_simulation()
            results_list.append(individual_result)
        
        return results_list
    
    def visualize_results(self, results: Optional[SimulationResults] = None):
        """Generate visualization plots."""
        if results is None:
            results = self.results
        
        if results is None:
            self.logger.warning("No results to visualize")
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create output directory
        output_dir = Path(self.config.output_dir) / "plots"
        output_dir.mkdir(exist_ok=True)
        
        # PK concentration-time plots
        if results.pk_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            organ_names = list(results.pk_results.keys())
            for i, organ_name in enumerate(organ_names[:4]):
                if i < len(axes):
                    axes[i].plot(results.output_times, results.pk_results[organ_name], linewidth=2)
                    axes[i].set_xlabel('Time (hours)')
                    axes[i].set_ylabel('Concentration (mg/L)')
                    axes[i].set_title(f'{organ_name.capitalize()} Concentration')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(organ_names), 4):
                axes[i].set_visible(False)
            
            plt.suptitle('Pharmacokinetic Profiles', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / 'pk_profiles.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # PD response plots
        if results.pd_results:
            # Select a subset of responses for visualization
            response_keys = list(results.pd_results.keys())[:6]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for i, response_key in enumerate(response_keys):
                if i < len(axes):
                    axes[i].plot(results.output_times, results.pd_results[response_key], linewidth=2)
                    axes[i].set_xlabel('Time (hours)')
                    axes[i].set_ylabel('Response')
                    axes[i].set_title(response_key.replace('_', ' ').title())
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(response_keys), 6):
                axes[i].set_visible(False)
            
            plt.suptitle('Pharmacodynamic Responses', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / 'pd_responses.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Physiological state plots
        if results.physiological_state:
            phys_df = pd.DataFrame(results.physiological_state)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Heart rate
            if 'heart_rate' in phys_df.columns:
                axes[0, 0].plot(phys_df['time'], phys_df['heart_rate'], linewidth=2)
                axes[0, 0].set_xlabel('Time (hours)')
                axes[0, 0].set_ylabel('Heart Rate (bpm)')
                axes[0, 0].set_title('Heart Rate')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Blood pressure
            if 'blood_pressure_systolic' in phys_df.columns:
                axes[0, 1].plot(phys_df['time'], phys_df['blood_pressure_systolic'], 
                              label='Systolic', linewidth=2)
                axes[0, 1].plot(phys_df['time'], phys_df['blood_pressure_diastolic'], 
                              label='Diastolic', linewidth=2)
                axes[0, 1].set_xlabel('Time (hours)')
                axes[0, 1].set_ylabel('Blood Pressure (mmHg)')
                axes[0, 1].set_title('Blood Pressure')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Temperature
            if 'body_temperature' in phys_df.columns:
                axes[1, 0].plot(phys_df['time'], phys_df['body_temperature'], linewidth=2)
                axes[1, 0].set_xlabel('Time (hours)')
                axes[1, 0].set_ylabel('Temperature (°C)')
                axes[1, 0].set_title('Body Temperature')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Glucose
            if 'blood_glucose' in phys_df.columns:
                axes[1, 1].plot(phys_df['time'], phys_df['blood_glucose'], linewidth=2)
                axes[1, 1].set_xlabel('Time (hours)')
                axes[1, 1].set_ylabel('Glucose (mg/dL)')
                axes[1, 1].set_title('Blood Glucose')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle('Physiological State Over Time', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / 'physiological_state.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_report(self, results: Optional[SimulationResults] = None) -> str:
        """Generate a comprehensive simulation report."""
        if results is None:
            results = self.results
        
        if results is None:
            return "No simulation results available"
        
        report = []
        report.append("# PK-PD Simulation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Configuration summary
        report.append("## Simulation Configuration")
        report.append(f"- Drug: {self.config.drug_name}")
        report.append(f"- Dose: {self.config.dose} mg")
        report.append(f"- Administration: {self.config.administration_route}")
        report.append(f"- Time Horizon: {self.config.time_horizon} hours")
        report.append(f"- Patient: {self.config.patient_age}y, {self.config.patient_weight}kg, {self.config.patient_sex}")
        report.append("")
        
        # PK summary
        if results.pk_results:
            report.append("## Pharmacokinetic Summary")
            for organ, concentrations in results.pk_results.items():
                cmax = np.max(concentrations)
                tmax = results.output_times[np.argmax(concentrations)]
                auc = np.trapz(concentrations, results.output_times)
                
                report.append(f"### {organ.capitalize()}")
                report.append(f"- Cmax: {cmax:.2f} mg/L")
                report.append(f"- Tmax: {tmax:.2f} h")
                report.append(f"- AUC: {auc:.2f} mg·h/L")
                report.append("")
        
        # PD summary
        if results.pd_results:
            report.append("## Pharmacodynamic Summary")
            for response_key, response_values in results.pd_results.items():
                max_response = np.max(response_values)
                min_response = np.min(response_values)
                auc_response = np.trapz(response_values, results.output_times)
                
                report.append(f"### {response_key.replace('_', ' ').title()}")
                report.append(f"- Max Response: {max_response:.2f}")
                report.append(f"- Min Response: {min_response:.2f}")
                report.append(f"- AUC: {auc_response:.2f}")
                report.append("")
        
        # Disease effects
        if results.disease_results:
            report.append("## Disease Effects")
            for disease, effects in results.disease_results.items():
                report.append(f"### {disease}")
                for param, effect in effects.items():
                    report.append(f"- {param}: {effect:.2f}")
                report.append("")
        
        # Summary statistics
        if results.summary_stats:
            report.append("## Summary Statistics")
            for stat_name, stat_value in results.summary_stats.items():
                report.append(f"- {stat_name}: {stat_value:.4f}")
            report.append("")
        
        return "\n".join(report)


# Convenience function for quick simulation
def run_quick_simulation(
    drug_name: str,
    dose: float,
    time_horizon: float = 48.0,
    patient_age: int = 45,
    patient_weight: float = 70.0,
    disease_type: Optional[str] = None
) -> SimulationResults:
    """Run a quick simulation with default parameters."""
    config = SimulationConfig(
        drug_name=drug_name,
        dose=dose,
        time_horizon=time_horizon,
        patient_age=patient_age,
        patient_weight=patient_weight,
        disease_type=disease_type
    )
    
    runner = PKPDSimulationRunner(config)
    return runner.run_simulation()
