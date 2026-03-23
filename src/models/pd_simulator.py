"""
Pharmacodynamic (PD) Response Simulation Module

This module implements comprehensive pharmacodynamic modeling including:
- Receptor-ligand binding kinetics
- Signal transduction pathways
- Dose-response relationships
- Time-dependent response modeling
- Organ-specific PD effects
- Tolerance and sensitization phenomena
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class ReceptorProperties:
    """Receptor-specific pharmacodynamic properties."""
    receptor_density: float  # receptors/cell
    binding_affinity: float  # Kd (nM)
    efficacy: float  # Emax (0-1)
    basal_activity: float  # Basal activity (0-1)
    desensitization_rate: float  # Rate of receptor desensitization
    resensitization_rate: float  # Rate of receptor resensitization
    internalization_rate: float  # Rate of receptor internalization
    recycling_rate: float  # Rate of receptor recycling


@dataclass
class SignalTransduction:
    """Signal transduction pathway parameters."""
    pathway_sensitivity: float  # Sensitivity to receptor activation
    amplification_factor: float  # Signal amplification
    feedback_strength: float  # Negative feedback strength
    adaptation_rate: float  # Rate of pathway adaptation
    cross_talk_factor: float  # Cross-talk with other pathways
    delay_time: float  # Signal propagation delay (hours)


@dataclass
class PhysiologicalResponse:
    """Physiological response parameters."""
    response_type: str  # 'inhibition', 'stimulation', 'biphasic'
    max_response: float  # Maximum physiological response
    baseline_value: float  # Baseline physiological value
    response_rate: float  # Rate of response onset
    recovery_rate: float  # Rate of response recovery
    threshold_concentration: float  # Concentration threshold for response
    hysteresis_factor: float  # Hysteresis in concentration-response


class ReceptorBindingModel:
    """
    Receptor-ligand binding kinetics model.
    
    Implements various binding models including:
    - Simple competitive binding
    - Allosteric modulation
    - Cooperative binding
    - Time-dependent binding
    """
    
    def __init__(self, receptor_props: ReceptorProperties):
        self.receptor_props = receptor_props
        
    def equilibrium_binding(self, drug_concentration: np.ndarray) -> np.ndarray:
        """
        Calculate equilibrium receptor occupancy.
        
        Args:
            drug_concentration: Drug concentration (nM)
            
        Returns:
            Receptor occupancy (fraction)
        """
        kd = self.receptor_props.binding_affinity
        occupancy = drug_concentration / (kd + drug_concentration)
        return occupancy
    
    def time_dependent_binding(
        self,
        drug_concentration: np.ndarray,
        time_points: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate time-dependent receptor binding and dynamics.
        
        Args:
            drug_concentration: Drug concentration over time
            time_points: Time points
            
        Returns:
            Dictionary with binding dynamics
        """
        def binding_odes(y, t, conc_func):
            """ODEs for receptor binding dynamics."""
            occupied, desensitized, internalized = y
            
            # Current drug concentration
            c = conc_func(t)
            
            # Binding kinetics
            kon = 1.0 / self.receptor_props.binding_affinity
            koff = 1.0
            
            # Rate equations
            d_occupied_dt = kon * c * (1 - occupied - desensitized - internalized) - koff * occupied
            d_occupied_dt -= self.receptor_props.desensitization_rate * occupied
            d_occupied_dt += self.receptor_props.resensitization_rate * desensitized
            
            d_desensitized_dt = self.receptor_props.desensitization_rate * occupied
            d_desensitized_dt -= self.receptor_props.resensitization_rate * desensitized
            
            d_internalized_dt = self.receptor_props.internalization_rate * occupied
            d_internalized_dt -= self.receptor_props.recycling_rate * internalized
            
            return [d_occupied_dt, d_desensitized_dt, d_internalized_dt]
        
        # Interpolate concentration function
        conc_func = interp1d(time_points, drug_concentration, kind='linear', fill_value='extrapolate')
        
        # Initial conditions
        y0 = [0.0, 0.0, 0.0]  # [occupied, desensitized, internalized]
        
        # Solve ODEs
        solution = odeint(binding_odes, y0, time_points, args=(conc_func,))
        
        return {
            'occupied': solution[:, 0],
            'desensitized': solution[:, 1],
            'internalized': solution[:, 2],
            'total_bound': solution.sum(axis=1)
        }


class SignalTransductionModel:
    """
    Signal transduction pathway model.
    
    Models the cascade of events from receptor activation
    to physiological response.
    """
    
    def __init__(self, transduction: SignalTransduction):
        self.transduction = transduction
        
    def propagate_signal(
        self,
        receptor_activation: np.ndarray,
        time_points: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Propagate signal through transduction pathways.
        
        Args:
            receptor_activation: Receptor activation over time
            time_points: Time points
            
        Returns:
            Dictionary with signal transduction results
        """
        def transduction_odes(y, t, activation_func):
            """ODEs for signal transduction."""
            signal, response, feedback = y
            
            # Current receptor activation
            act = activation_func(t)
            
            # Signal propagation with delay
            if t < self.transduction.delay_time:
                effective_activation = 0.0
            else:
                effective_activation = activation_func(t - self.transduction.delay_time)
            
            # Signal amplification
            signal_rate = self.transduction.pathway_sensitivity * effective_activation
            signal_rate += self.transduction.amplification_factor * signal
            signal_rate -= self.transduction.adaptation_rate * signal
            signal_rate -= feedback * self.transduction.feedback_strength
            
            # Response generation
            response_rate = signal - self.transduction.adaptation_rate * response
            
            # Feedback generation
            feedback_rate = response - self.transduction.adaptation_rate * feedback
            
            return [signal_rate, response_rate, feedback_rate]
        
        # Interpolate activation function
        activation_func = interp1d(time_points, receptor_activation, kind='linear', fill_value='extrapolate')
        
        # Initial conditions
        y0 = [0.0, 0.0, 0.0]  # [signal, response, feedback]
        
        # Solve ODEs
        solution = odeint(transduction_odes, y0, time_points, args=(activation_func,))
        
        return {
            'signal': solution[:, 0],
            'response': solution[:, 1],
            'feedback': solution[:, 2],
            'net_effect': solution[:, 0] - solution[:, 2]
        }


class DoseResponseModel:
    """
    Dose-response relationship models.
    
    Implements various dose-response models including:
    - Emax model
    - Sigmoid Emax model
    - Biphasic models
    - Time-dependent dose-response
    """
    
    @staticmethod
    def emax_model(concentration: np.ndarray, emax: float, ec50: float) -> np.ndarray:
        """Simple Emax model."""
        return emax * concentration / (ec50 + concentration)
    
    @staticmethod
    def sigmoid_emax_model(
        concentration: np.ndarray,
        emax: float,
        ec50: float,
        hill_coefficient: float
    ) -> np.ndarray:
        """Sigmoid Emax model with Hill coefficient."""
        return emax * concentration ** hill_coefficient / (ec50 ** hill_coefficient + concentration ** hill_coefficient)
    
    @staticmethod
    def biphasic_model(
        concentration: np.ndarray,
        emax1: float,
        ec50_1: float,
        hill1: float,
        emax2: float,
        ec50_2: float,
        hill2: float
    ) -> np.ndarray:
        """Biphasic dose-response model."""
        response1 = emax1 * concentration ** hill1 / (ec50_1 ** hill1 + concentration ** hill1)
        response2 = emax2 * concentration ** hill2 / (ec50_2 ** hill2 + concentration ** hill2)
        return response1 + response2
    
    @staticmethod
    def fit_dose_response(
        concentrations: np.ndarray,
        responses: np.ndarray,
        model_type: str = 'sigmoid_emax'
    ) -> Dict[str, float]:
        """
        Fit dose-response model to data.
        
        Args:
            concentrations: Drug concentrations
            responses: Observed responses
            model_type: Type of model to fit
            
        Returns:
            Fitted parameters
        """
        if model_type == 'sigmoid_emax':
            model_func = DoseResponseModel.sigmoid_emax_model
            p0 = [max(responses), np.median(concentrations), 1.0]  # Initial guesses
            bounds = ([0, 0, 0], [np.inf, np.inf, 10])
        elif model_type == 'emax':
            model_func = DoseResponseModel.emax_model
            p0 = [max(responses), np.median(concentrations)]
            bounds = ([0, 0], [np.inf, np.inf])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        try:
            params, _ = curve_fit(model_func, concentrations, responses, p0=p0, bounds=bounds)
            
            if model_type == 'sigmoid_emax':
                return {'emax': params[0], 'ec50': params[1], 'hill_coefficient': params[2]}
            else:
                return {'emax': params[0], 'ec50': params[1]}
        except:
            # Return default parameters if fitting fails
            return {'emax': max(responses), 'ec50': np.median(concentrations)}


class OrganSpecificPD:
    """
    Organ-specific pharmacodynamic response model.
    
    Models PD effects for different organs with their specific
    receptors and physiological responses.
    """
    
    def __init__(self, organ_name: str):
        self.organ_name = organ_name
        self.receptors = self._initialize_receptors()
        self.responses = self._initialize_responses()
        
    def _initialize_receptors(self) -> Dict[str, ReceptorProperties]:
        """Initialize organ-specific receptors."""
        receptor_configs = {
            'heart': {
                'beta1': ReceptorProperties(1000, 10, 0.8, 0.1, 0.1, 0.05, 0.05, 0.02),
                'beta2': ReceptorProperties(500, 50, 0.6, 0.05, 0.1, 0.05, 0.05, 0.02),
            },
            'brain': {
                'dopamine_d1': ReceptorProperties(800, 20, 0.7, 0.2, 0.15, 0.1, 0.08, 0.03),
                'serotonin_5ht': ReceptorProperties(600, 30, 0.6, 0.15, 0.12, 0.08, 0.06, 0.02),
            },
            'liver': {
                'nuclear_ppar': ReceptorProperties(1200, 15, 0.9, 0.1, 0.05, 0.03, 0.02, 0.01),
                'metabolic_enzymes': ReceptorProperties(2000, 5, 0.8, 0.3, 0.02, 0.01, 0.01, 0.005),
            },
            'kidney': {
                'at1_receptor': ReceptorProperties(900, 25, 0.7, 0.1, 0.08, 0.04, 0.03, 0.01),
                'dopamine_d1': ReceptorProperties(400, 40, 0.5, 0.05, 0.1, 0.05, 0.04, 0.02),
            },
            'lung': {
                'beta2': ReceptorProperties(1500, 8, 0.9, 0.05, 0.1, 0.06, 0.05, 0.02),
                'muscarinic_m3': ReceptorProperties(800, 35, 0.6, 0.1, 0.12, 0.08, 0.06, 0.03),
            }
        }
        
        return receptor_configs.get(self.organ_name, {})
    
    def _initialize_responses(self) -> Dict[str, PhysiologicalResponse]:
        """Initialize organ-specific physiological responses."""
        response_configs = {
            'heart': {
                'heart_rate': PhysiologicalResponse('stimulation', 200, 70, 0.5, 0.3, 1.0, 0.1),
                'contractility': PhysiologicalResponse('stimulation', 150, 100, 0.4, 0.2, 2.0, 0.15),
                'blood_pressure': PhysiologicalResponse('biphasic', 50, 120, 0.3, 0.4, 5.0, 0.2),
            },
            'brain': {
                'cognitive_function': PhysiologicalResponse('stimulation', 30, 100, 0.2, 0.1, 0.5, 0.05),
                'mood': PhysiologicalResponse('biphasic', 40, 50, 0.3, 0.2, 1.0, 0.1),
                'motor_control': PhysiologicalResponse('inhibition', 60, 100, 0.4, 0.3, 2.0, 0.15),
            },
            'liver': {
                'glucose_production': PhysiologicalResponse('inhibition', 80, 100, 0.5, 0.3, 3.0, 0.2),
                'lipid_metabolism': PhysiologicalResponse('stimulation', 60, 50, 0.3, 0.2, 2.0, 0.1),
                'drug_metabolism': PhysiologicalResponse('stimulation', 120, 100, 0.6, 0.4, 1.0, 0.1),
            },
            'kidney': {
                'filtration_rate': PhysiologicalResponse('inhibition', 40, 125, 0.4, 0.2, 5.0, 0.15),
                'sodium_excretion': PhysiologicalResponse('stimulation', 100, 100, 0.5, 0.3, 2.0, 0.1),
                'water_excretion': PhysiologicalResponse('stimulation', 80, 1500, 0.4, 0.3, 3.0, 0.12),
            },
            'lung': {
                'bronchodilation': PhysiologicalResponse('stimulation', 100, 100, 0.8, 0.5, 1.0, 0.1),
                'inflammation': PhysiologicalResponse('inhibition', 70, 0, 0.6, 0.4, 2.0, 0.15),
                'mucus_production': PhysiologicalResponse('inhibition', 50, 100, 0.3, 0.2, 3.0, 0.2),
            }
        }
        
        return response_configs.get(self.organ_name, {})
    
    def simulate_organ_response(
        self,
        drug_concentration: np.ndarray,
        time_points: np.ndarray,
        drug_receptor_affinity: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        Simulate organ-specific PD response.
        
        Args:
            drug_concentration: Drug concentration over time
            time_points: Time points
            drug_receptor_affinity: Drug affinity for each receptor type
            
        Returns:
            Dictionary with organ responses over time
        """
        organ_responses = {}
        
        # Calculate receptor binding for each receptor type
        receptor_activations = {}
        for receptor_name, receptor_props in self.receptors.items():
            # Modify receptor affinity based on drug properties
            effective_affinity = drug_receptor_affinity.get(receptor_name, receptor_props.binding_affinity)
            modified_props = ReceptorProperties(
                receptor_props.receptor_density,
                effective_affinity,
                receptor_props.efficacy,
                receptor_props.basal_activity,
                receptor_props.desensitization_rate,
                receptor_props.resensitization_rate,
                receptor_props.internalization_rate,
                receptor_props.recycling_rate
            )
            
            # Calculate binding dynamics
            binding_model = ReceptorBindingModel(modified_props)
            binding_dynamics = binding_model.time_dependent_binding(drug_concentration, time_points)
            
            # Calculate net activation (considering desensitization)
            receptor_activations[receptor_name] = (
                binding_dynamics['occupied'] * receptor_props.efficacy +
                receptor_props.basal_activity
            )
        
        # Calculate physiological responses
        for response_name, response_props in self.responses.items():
            # Combine receptor effects (simplified - could be more complex)
            total_activation = 0.0
            for receptor_name, activation in receptor_activations.items():
                # Weight receptor contributions (simplified)
                weight = 1.0 / len(receptor_activations)
                total_activation += weight * activation
            
            # Apply dose-response relationship
            if response_props.response_type == 'stimulation':
                response_change = DoseResponseModel.sigmoid_emax_model(
                    total_activation * drug_concentration,
                    response_props.max_response,
                    response_props.threshold_concentration,
                    1.0  # Hill coefficient
                )
            elif response_props.response_type == 'inhibition':
                response_change = -DoseResponseModel.sigmoid_emax_model(
                    total_activation * drug_concentration,
                    response_props.max_response,
                    response_props.threshold_concentration,
                    1.0
                )
            elif response_props.response_type == 'biphasic':
                # Biphasic response - stimulation at low concentration, inhibition at high
                stim_response = DoseResponseModel.sigmoid_emax_model(
                    total_activation * drug_concentration,
                    response_props.max_response * 0.6,
                    response_props.threshold_concentration * 0.5,
                    1.0
                )
                inhib_response = -DoseResponseModel.sigmoid_emax_model(
                    total_activation * drug_concentration,
                    response_props.max_response * 0.4,
                    response_props.threshold_concentration * 2.0,
                    1.0
                )
                response_change = stim_response + inhib_response
            else:
                response_change = np.zeros_like(drug_concentration)
            
            # Apply temporal dynamics
            def response_odes(y, t, response_func):
                """ODEs for response onset and recovery."""
                current_response = y[0]
                target_response = response_func(t)
                
                d_response_dt = (
                    response_props.response_rate * (target_response - current_response) -
                    response_props.recovery_rate * (current_response - response_props.baseline_value)
                )
                
                return [d_response_dt]
            
            # Interpolate target response
            target_func = interp1d(time_points, response_change, kind='linear', fill_value='extrapolate')
            
            # Solve response dynamics
            y0 = [response_props.baseline_value]
            solution = odeint(response_odes, y0, time_points, args=(target_func,))
            
            organ_responses[response_name] = solution[:, 0]
        
        return organ_responses


class SystemicPDModel:
    """
    Systemic pharmacodynamic model integrating multiple organ responses.
    
    Models the overall systemic effects of drugs including:
    - Multi-organ interactions
    - Homeostatic regulation
    - Adverse effects
    - Therapeutic index
    """
    
    def __init__(self, organs: List[str]):
        self.organs = organs
        self.organ_models = {organ: OrganSpecificPD(organ) for organ in organs}
        self.homeostatic_params = self._initialize_homeostasis()
        
    def _initialize_homeostasis(self) -> Dict[str, Dict[str, float]]:
        """Initialize homeostatic regulation parameters."""
        return {
            'blood_pressure': {
                'setpoint': 120.0,  # mmHg
                'gain': 0.5,
                'time_constant': 2.0  # hours
            },
            'heart_rate': {
                'setpoint': 70.0,  # bpm
                'gain': 0.3,
                'time_constant': 1.0
            },
            'glucose': {
                'setpoint': 100.0,  # mg/dL
                'gain': 0.4,
                'time_constant': 1.5
            }
        }
    
    def simulate_systemic_response(
        self,
        drug_concentrations: Dict[str, np.ndarray],
        time_points: np.ndarray,
        drug_receptor_affinity: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Simulate systemic PD response across all organs.
        
        Args:
            drug_concentrations: Drug concentrations in each organ
            time_points: Time points
            drug_receptor_affinity: Drug affinity for receptors in each organ
            
        Returns:
            Dictionary with systemic responses
        """
        systemic_responses = {}
        
        # Simulate individual organ responses
        for organ_name, organ_model in self.organ_models.items():
            if organ_name in drug_concentrations:
                organ_affinity = drug_receptor_affinity.get(organ_name, {})
                organ_response = organ_model.simulate_organ_response(
                    drug_concentrations[organ_name],
                    time_points,
                    organ_affinity
                )
                systemic_responses[organ_name] = organ_response
        
        # Apply homeostatic regulation
        regulated_responses = self._apply_homeostatic_regulation(systemic_responses, time_points)
        
        return regulated_responses
    
    def _apply_homeostatic_regulation(
        self,
        organ_responses: Dict[str, Dict[str, np.ndarray]],
        time_points: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Apply homeostatic regulation to organ responses."""
        regulated = organ_responses.copy()
        
        # Example: Blood pressure regulation
        if 'heart' in organ_responses and 'kidney' in organ_responses:
            heart_bp = organ_responses['heart'].get('blood_pressure', np.zeros(len(time_points)))
            kidney_bp = organ_responses['kidney'].get('blood_pressure', np.zeros(len(time_points)))
            
            # Combined blood pressure effect
            total_bp_effect = heart_bp + kidney_bp
            
            # Homeostatic correction
            bp_setpoint = self.homeostatic_params['blood_pressure']['setpoint']
            bp_gain = self.homeostatic_params['blood_pressure']['gain']
            bp_time_const = self.homeostatic_params['blood_pressure']['time_constant']
            
            # Simple homeostatic model
            corrected_bp = bp_setpoint + total_bp_effect * np.exp(-time_points / bp_time_const) * (1 - bp_gain)
            
            regulated['systemic'] = {'blood_pressure': corrected_bp}
        
        return regulated
    
    def calculate_therapeutic_index(
        self,
        efficacy_responses: Dict[str, np.ndarray],
        toxicity_responses: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate therapeutic index based on efficacy and toxicity.
        
        Args:
            efficacy_responses: Therapeutic effect responses
            toxicity_responses: Toxicity responses
            
        Returns:
            Therapeutic indices
        """
        therapeutic_indices = {}
        
        for response_name, efficacy in efficacy_responses.items():
            if response_name in toxicity_responses:
                toxicity = toxicity_responses[response_name]
                
                # Find EC50 for efficacy and toxicity
                efficacy_ec50 = self._find_ec50(efficacy)
                toxicity_ec50 = self._find_ec50(toxicity)
                
                if toxicity_ec50 > 0:
                    therapeutic_indices[response_name] = toxicity_ec50 / efficacy_ec50
                else:
                    therapeutic_indices[response_name] = float('inf')
        
        return therapeutic_indices
    
    def _find_ec50(self, response: np.ndarray) -> float:
        """Find EC50 from response curve."""
        # Simplified EC50 calculation
        max_response = np.max(response)
        half_max = max_response / 2
        
        # Find closest point to half-maximal response
        idx = np.argmin(np.abs(response - half_max))
        
        # Return corresponding concentration (simplified)
        return float(idx) / len(response) * 100  # Arbitrary scale


class PDVisualizer:
    """Visualization tools for PD simulations."""
    
    @staticmethod
    def plot_organ_response(
        organ_responses: Dict[str, np.ndarray],
        time_points: np.ndarray,
        organ_name: str,
        title: str = None,
        save_path: Optional[str] = None
    ):
        """Plot organ-specific responses."""
        plt.figure(figsize=(12, 8))
        
        for response_name, response in organ_responses.items():
            plt.plot(time_points, response, label=response_name, linewidth=2)
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Response Magnitude')
        plt.title(title or f'{organ_name.capitalize()} Response Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_dose_response_curve(
        concentrations: np.ndarray,
        responses: np.ndarray,
        fitted_params: Dict[str, float],
        title: str = "Dose-Response Curve",
        save_path: Optional[str] = None
    ):
        """Plot dose-response curve with fitted model."""
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.scatter(concentrations, responses, alpha=0.6, label='Data')
        
        # Plot fitted curve
        conc_range = np.logspace(np.log10(min(concentrations)), np.log10(max(concentrations)), 100)
        
        if 'hill_coefficient' in fitted_params:
            fitted_response = DoseResponseModel.sigmoid_emax_model(
                conc_range,
                fitted_params['emax'],
                fitted_params['ec50'],
                fitted_params['hill_coefficient']
            )
        else:
            fitted_response = DoseResponseModel.emax_model(
                conc_range,
                fitted_params['emax'],
                fitted_params['ec50']
            )
        
        plt.plot(conc_range, fitted_response, 'r-', linewidth=2, label='Fitted Model')
        
        plt.xscale('log')
        plt.xlabel('Drug Concentration (nM)')
        plt.ylabel('Response')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_systemic_effects(
        systemic_responses: Dict[str, Dict[str, np.ndarray]],
        time_points: np.ndarray,
        title: str = "Systemic Drug Effects",
        save_path: Optional[str] = None
    ):
        """Plot systemic effects across multiple organs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        organ_idx = 0
        for organ_name, organ_responses in systemic_responses.items():
            if organ_idx >= 4:
                break
                
            ax = axes[organ_idx]
            
            for response_name, response in organ_responses.items():
                ax.plot(time_points, response, label=response_name, linewidth=2)
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Response Magnitude')
            ax.set_title(f'{organ_name.capitalize()} Effects')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            organ_idx += 1
        
        # Hide unused subplots
        for i in range(organ_idx, 4):
            axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
