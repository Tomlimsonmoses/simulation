"""
Pharmacokinetic (PK) Simulation Module

This module implements comprehensive pharmacokinetic modeling including:
- Multi-compartment models
- Physiologically-based PK (PBPK) models
- Drug absorption, distribution, metabolism, excretion (ADME)
- Population PK variability
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class DrugProperties:
    """Drug-specific pharmacokinetic properties."""
    molecular_weight: float  # g/mol
    logp: float  # Octanol-water partition coefficient
    pka: float  # Acid dissociation constant
    solubility: float  # mg/L
    bioavailability: float  # Fraction absorbed (0-1)
    protein_binding: float  # Fraction bound to proteins (0-1)
    permeability: float  # cm/s
    clearance_rate: float  # L/h/kg
    volume_distribution: float  # L/kg
    half_life: float  # hours
    absorption_rate: float  # 1/hour
    metabolism_rate: float  # 1/hour
    excretion_rate: float  # 1/hour


@dataclass
class CompartmentParameters:
    """Parameters for physiological compartments."""
    volume: float  # L
    blood_flow: float  # L/h
    permeability: float  # cm/s
    surface_area: float  # m^2
    ph: float
    temperature: float  # °C
    enzyme_activity: float  # Relative activity
    transporter_density: float  # Relative density
    partition_coefficient: float  # Tissue:plasma ratio


class MultiCompartmentPK:
    """
    Multi-compartment pharmacokinetic model.
    
    Implements traditional compartmental PK models with customizable
    number of compartments and transfer rates.
    """
    
    def __init__(
        self,
        num_compartments: int = 3,
        drug_props: DrugProperties = None,
        compartment_params: List[CompartmentParameters] = None
    ):
        self.num_compartments = num_compartments
        self.drug_props = drug_props or self._default_drug_props()
        self.compartment_params = compartment_params or self._default_compartments()
        
        # Initialize transfer rate matrix
        self.transfer_rates = self._initialize_transfer_rates()
        
    def _default_drug_props(self) -> DrugProperties:
        """Default drug properties for testing."""
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
    
    def _default_compartments(self) -> List[CompartmentParameters]:
        """Default compartment parameters."""
        return [
            CompartmentParameters(5.0, 300.0, 1e-6, 1.8, 7.4, 37.0, 1.0, 1.0, 1.0),  # Central
            CompartmentParameters(15.0, 50.0, 1e-7, 2.0, 7.4, 37.0, 0.8, 0.5, 0.8),  # Peripheral 1
            CompartmentParameters(30.0, 20.0, 1e-8, 3.0, 7.4, 37.0, 0.6, 0.3, 0.6),  # Peripheral 2
        ]
    
    def _initialize_transfer_rates(self) -> np.ndarray:
        """Initialize inter-compartmental transfer rate matrix."""
        rates = np.zeros((self.num_compartments, self.num_compartments))
        
        # Central to peripheral compartments
        for i in range(1, self.num_compartments):
            blood_flow = self.compartment_params[i].blood_flow
            volume = self.compartment_params[i].volume
            rates[0, i] = blood_flow / volume  # Central to peripheral
            rates[i, 0] = blood_flow / self.compartment_params[0].volume  # Peripheral to central
        
        # Add clearance from central compartment
        rates[0, 0] = -self.drug_props.clearance_rate
        
        return rates
    
    def pk_odes(self, y: np.ndarray, t: float, dose_input: float = 0.0) -> np.ndarray:
        """
        Differential equations for multi-compartment PK model.
        
        Args:
            y: Drug amounts in each compartment
            t: Time
            dose_input: Input dose rate (for continuous infusion)
            
        Returns:
            Derivatives of drug amounts
        """
        dydt = np.zeros(self.num_compartments)
        
        # Inter-compartmental transfers
        for i in range(self.num_compartments):
            for j in range(self.num_compartments):
                if i != j:
                    dydt[i] += self.transfer_rates[j, i] * y[j] - self.transfer_rates[i, j] * y[i]
        
        # Add input to central compartment
        dydt[0] += dose_input
        
        return dydt
    
    def simulate(
        self,
        dose: float,
        dosing_regimen: str = 'bolus',
        time_points: np.ndarray = None,
        infusion_duration: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Simulate drug concentrations over time.
        
        Args:
            dose: Drug amount (mg)
            dosing_regimen: 'bolus', 'infusion', 'multiple'
            time_points: Time points for simulation (hours)
            infusion_duration: Duration for infusion (hours)
            
        Returns:
            Dictionary with time points and concentrations
        """
        if time_points is None:
            time_points = np.linspace(0, 48, 100)  # 48 hours, 100 points
        
        # Initial conditions
        y0 = np.zeros(self.num_compartments)
        
        if dosing_regimen == 'bolus':
            y0[0] = dose * self.drug_props.bioavailability
            solution = odeint(self.pk_odes, y0, time_points)
            
        elif dosing_regimen == 'infusion':
            # Continuous infusion
            dose_rate = dose / infusion_duration
            solution = odeint(
                lambda y, t: self.pk_odes(y, t, dose_rate if t < infusion_duration else 0.0),
                y0, time_points
            )
            
        elif dosing_regimen == 'multiple':
            # Multiple dosing (every 12 hours)
            num_doses = int(time_points[-1] / 12) + 1
            solution = np.zeros((len(time_points), self.num_compartments))
            y_current = y0.copy()
            
            for i, t in enumerate(time_points):
                # Check if it's time for a dose
                if t % 12 < 0.1:  # Dose every 12 hours
                    y_current[0] += dose * self.drug_props.bioavailability
                
                # Integrate to next time point
                if i < len(time_points) - 1:
                    dt = time_points[i + 1] - t
                    y_current = odeint(self.pk_odes, y_current, [0, dt])[1]
                
                solution[i] = y_current
        
        else:
            raise ValueError(f"Unknown dosing regimen: {dosing_regimen}")
        
        # Convert amounts to concentrations
        concentrations = np.zeros_like(solution)
        for i in range(self.num_compartments):
            concentrations[:, i] = solution[:, i] / self.compartment_params[i].volume
        
        return {
            'time': time_points,
            'concentrations': concentrations,
            'amounts': solution
        }


class PhysiologicallyBasedPK:
    """
    Physiologically-Based Pharmacokinetic (PBPK) model.
    
    Implements detailed PBPK modeling with organ-specific physiology
    and drug-specific properties.
    """
    
    def __init__(self, drug_props: DrugProperties):
        self.drug_props = drug_props
        self.organs = self._initialize_organs()
        self.blood_flows = self._initialize_blood_flows()
        self.tissue_partitions = self._calculate_tissue_partitions()
        
    def _initialize_organs(self) -> Dict[str, CompartmentParameters]:
        """Initialize organ-specific parameters."""
        return {
            'brain': CompartmentParameters(1.4, 750.0, 1e-7, 0.2, 7.3, 37.0, 0.9, 0.8, 1.5),
            'heart': CompartmentParameters(0.3, 240.0, 1e-6, 0.1, 7.4, 37.0, 1.0, 1.0, 1.2),
            'liver': CompartmentParameters(1.5, 1350.0, 1e-6, 1.5, 7.4, 37.0, 1.2, 1.5, 2.0),
            'kidney': CompartmentParameters(0.3, 1100.0, 1e-6, 0.3, 7.4, 37.0, 1.1, 1.2, 1.8),
            'lung': CompartmentParameters(2.5, 4200.0, 1e-6, 2.0, 7.4, 37.0, 1.0, 1.0, 1.0),
            'muscle': CompartmentParameters(28.0, 900.0, 1e-7, 12.0, 7.4, 37.0, 0.7, 0.5, 0.8),
            'fat': CompartmentParameters(15.0, 300.0, 1e-8, 15.0, 7.4, 37.0, 0.3, 0.2, 3.0),
            'skin': CompartmentParameters(3.0, 300.0, 1e-7, 2.0, 7.4, 37.0, 0.8, 0.6, 1.0),
            'gut': CompartmentParameters(1.0, 900.0, 1e-6, 2.0, 6.5, 37.0, 1.1, 1.3, 1.5),
            'spleen': CompartmentParameters(0.2, 150.0, 1e-6, 0.1, 7.4, 37.0, 1.0, 1.0, 1.3),
        }
    
    def _initialize_blood_flows(self) -> Dict[str, float]:
        """Initialize organ blood flows (L/h) for 70kg human."""
        cardiac_output = 300.0  # L/h
        return {
            'brain': 0.12 * cardiac_output,
            'heart': 0.04 * cardiac_output,
            'liver': 0.27 * cardiac_output,  # Includes portal vein
            'kidney': 0.19 * cardiac_output,
            'lung': 1.0 * cardiac_output,  # Full cardiac output
            'muscle': 0.21 * cardiac_output,
            'fat': 0.05 * cardiac_output,
            'skin': 0.04 * cardiac_output,
            'gut': 0.15 * cardiac_output,
            'spleen': 0.03 * cardiac_output,
        }
    
    def _calculate_tissue_partitions(self) -> Dict[str, float]:
        """
        Calculate tissue:plasma partition coefficients using
        Poulin and Theil method.
        """
        partitions = {}
        
        for organ_name, organ in self.organs.items():
            # Simplified partition coefficient calculation
            # Based on drug lipophilicity (logp) and tissue composition
            water_fraction = 0.7 if organ_name != 'fat' else 0.1
            lipid_fraction = 0.05 if organ_name != 'fat' else 0.8
            protein_fraction = 0.25 if organ_name != 'fat' else 0.1
            
            # Partition coefficient based on drug properties
            k_water = 1.0
            k_lipid = 10 ** self.drug_props.logp
            k_protein = 1.0 + self.drug_props.protein_binding / (1 - self.drug_props.protein_binding)
            
            partition_coeff = (
                water_fraction * k_water +
                lipid_fraction * k_lipid +
                protein_fraction * k_protein
            )
            
            partitions[organ_name] = partition_coeff
        
        return partitions
    
    def pbpk_odes(self, y: np.ndarray, t: float, absorption_rate: float = 0.0) -> np.ndarray:
        """
        Differential equations for PBPK model.
        
        Args:
            y: Drug amounts in each organ + blood
            t: Time
            absorption_rate: Rate of drug absorption from gut
            
        Returns:
            Derivatives of drug amounts
        """
        organ_names = list(self.organs.keys())
        num_organs = len(organ_names)
        dydt = np.zeros(num_organs + 1)  # +1 for blood
        
        # Blood compartment (last element)
        blood_idx = num_organs
        blood_amount = y[blood_idx]
        blood_concentration = blood_amount / 5.0  # Blood volume ~5L
        
        # Organ-blood exchanges
        for i, organ_name in enumerate(organ_names):
            organ_amount = y[i]
            organ_concentration = organ_amount / self.organs[organ_name].volume
            
            # Permeability-limited transfer
            blood_flow = self.blood_flows[organ_name]
            permeability = self.organs[organ_name].permeability
            surface_area = self.organs[organ_name].surface_area
            
            # Rate limiting step (permeability or flow)
            perm_rate = permeability * surface_area * 3600  # Convert to L/h
            flow_rate = blood_flow
            
            transfer_rate = min(perm_rate, flow_rate)
            
            # Net transfer between blood and organ
            net_transfer = transfer_rate * (blood_concentration - organ_concentration / self.tissue_partitions[organ_name])
            
            dydt[i] = net_transfer
            dydt[blood_idx] -= net_transfer
            
            # Metabolism and excretion
            if organ_name == 'liver':
                metabolism_rate = self.drug_props.metabolism_rate * organ_amount
                dydt[i] -= metabolism_rate
            elif organ_name == 'kidney':
                excretion_rate = self.drug_props.excretion_rate * organ_amount
                dydt[i] -= excretion_rate
                dydt[blood_idx] -= excretion_rate
        
        # Absorption from gut
        if absorption_rate > 0:
            dydt[blood_idx] += absorption_rate * self.drug_props.bioavailability
        
        return dydt
    
    def simulate(
        self,
        dose: float,
        administration_route: str = 'iv',
        time_points: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate PBPK drug concentrations.
        
        Args:
            dose: Drug amount (mg)
            administration_route: 'iv', 'oral', 'subcutaneous'
            time_points: Time points for simulation
            
        Returns:
            Dictionary with simulation results
        """
        if time_points is None:
            time_points = np.linspace(0, 72, 200)  # 72 hours
        
        organ_names = list(self.organs.keys())
        num_organs = len(organ_names)
        
        # Initial conditions
        y0 = np.zeros(num_organs + 1)  # Organs + blood
        
        if administration_route == 'iv':
            y0[-1] = dose  # Direct to blood
            absorption_rate = 0.0
        elif administration_route == 'oral':
            # First-order absorption from gut
            absorption_rate = dose * self.drug_props.absorption_rate
        else:
            raise ValueError(f"Unknown administration route: {administration_route}")
        
        # Solve ODEs
        solution = odeint(self.pbpk_odes, y0, time_points, args=(absorption_rate,))
        
        # Calculate concentrations
        concentrations = {}
        for i, organ_name in enumerate(organ_names):
            concentrations[organ_name] = solution[:, i] / self.organs[organ_name].volume
        
        concentrations['blood'] = solution[:, -1] / 5.0  # Blood volume
        
        return {
            'time': time_points,
            'concentrations': concentrations,
            'amounts': solution,
            'organ_names': organ_names
        }


class PopulationPK:
    """
    Population pharmacokinetic modeling for inter-individual variability.
    """
    
    def __init__(self, base_pk_model, population_size: int = 100):
        self.base_pk_model = base_pk_model
        self.population_size = population_size
        self.covariates = self._generate_covariates()
        
    def _generate_covariates(self) -> Dict[str, np.ndarray]:
        """Generate population covariates."""
        return {
            'weight': np.random.normal(70, 10, self.population_size),  # kg
            'age': np.random.normal(45, 15, self.population_size),  # years
            'height': np.random.normal(170, 10, self.population_size),  # cm
            'sex': np.random.choice(['M', 'F'], self.population_size),
            'renal_function': np.random.normal(1.0, 0.2, self.population_size),
            'hepatic_function': np.random.normal(1.0, 0.2, self.population_size),
        }
    
    def apply_covariate_effects(self, individual_idx: int) -> DrugProperties:
        """
        Apply covariate effects to drug properties for an individual.
        
        Args:
            individual_idx: Index of individual in population
            
        Returns:
            Modified drug properties
        """
        weight = self.covariates['weight'][individual_idx]
        age = self.covariates['age'][individual_idx]
        sex = self.covariates['sex'][individual_idx]
        renal_func = self.covariates['renal_function'][individual_idx]
        hepatic_func = self.covariates['hepatic_function'][individual_idx]
        
        # Allometric scaling for weight
        weight_factor = (weight / 70) ** 0.75
        
        # Age effects
        age_factor = 1.0 - 0.01 * max(0, age - 65)
        
        # Sex effects (example: females have 20% lower clearance)
        sex_factor = 0.8 if sex == 'F' else 1.0
        
        # Organ function effects
        renal_factor = renal_func
        hepatic_factor = hepatic_func
        
        # Create modified drug properties
        modified_props = DrugProperties(
            molecular_weight=self.base_pk_model.drug_props.molecular_weight,
            logp=self.base_pk_model.drug_props.logp,
            pka=self.base_pk_model.drug_props.pka,
            solubility=self.base_pk_model.drug_props.solubility,
            bioavailability=self.base_pk_model.drug_props.bioavailability,
            protein_binding=self.base_pk_model.drug_props.protein_binding,
            permeability=self.base_pk_model.drug_props.permeability,
            clearance_rate=self.base_pk_model.drug_props.clearance_rate * weight_factor * age_factor * sex_factor * renal_factor,
            volume_distribution=self.base_pk_model.drug_props.volume_distribution * weight,
            half_life=self.base_pk_model.drug_props.half_life,
            absorption_rate=self.base_pk_model.drug_props.absorption_rate,
            metabolism_rate=self.base_pk_model.drug_props.metabolism_rate * hepatic_factor,
            excretion_rate=self.base_pk_model.drug_props.excretion_rate * renal_factor
        )
        
        return modified_props
    
    def simulate_population(
        self,
        dose: float,
        time_points: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate drug concentrations across population.
        
        Args:
            dose: Drug dose
            time_points: Time points for simulation
            
        Returns:
            Dictionary with population simulation results
        """
        if time_points is None:
            time_points = np.linspace(0, 48, 100)
        
        results = {}
        
        for i in range(self.population_size):
            # Get individual-specific drug properties
            individual_props = self.apply_covariate_effects(i)
            
            # Create individual PK model
            individual_model = MultiCompartmentPK(drug_props=individual_props)
            
            # Simulate
            individual_result = individual_model.simulate(dose, time_points=time_points)
            
            results[f'individual_{i}'] = individual_result
        
        return results


class PKVisualizer:
    """Visualization tools for PK simulations."""
    
    @staticmethod
    def plot_concentration_time(
        results: Dict[str, np.ndarray],
        title: str = "Drug Concentration-Time Profile",
        save_path: Optional[str] = None
    ):
        """Plot concentration-time profiles."""
        plt.figure(figsize=(10, 6))
        
        if 'concentrations' in results:
            # Multi-compartment results
            concentrations = results['concentrations']
            time = results['time']
            
            for i in range(concentrations.shape[1]):
                plt.plot(time, concentrations[:, i], label=f'Compartment {i+1}')
        
        elif isinstance(results.get('concentrations'), dict):
            # PBPK results
            concentrations = results['concentrations']
            time = results['time']
            
            for organ_name, conc in concentrations.items():
                plt.plot(time, conc, label=organ_name)
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Concentration (mg/L)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_population_variability(
        population_results: Dict[str, Dict[str, np.ndarray]],
        title: str = "Population PK Variability",
        save_path: Optional[str] = None
    ):
        """Plot population variability."""
        plt.figure(figsize=(12, 8))
        
        # Extract concentrations for central compartment
        all_concentrations = []
        time_points = None
        
        for individual_id, result in population_results.items():
            concentrations = result['concentrations']
            if len(concentrations.shape) == 2:
                central_conc = concentrations[:, 0]  # Central compartment
            else:
                central_conc = concentrations
            
            all_concentrations.append(central_conc)
            if time_points is None:
                time_points = result['time']
        
        all_concentrations = np.array(all_concentrations)
        
        # Plot median and percentiles
        median = np.median(all_concentrations, axis=0)
        p5 = np.percentile(all_concentrations, 5, axis=0)
        p95 = np.percentile(all_concentrations, 95, axis=0)
        
        plt.plot(time_points, median, 'b-', linewidth=2, label='Median')
        plt.fill_between(time_points, p5, p95, alpha=0.3, label='5th-95th percentile')
        
        # Plot few individual trajectories
        for i in range(min(10, len(all_concentrations))):
            plt.plot(time_points, all_concentrations[i], 'gray', alpha=0.3, linewidth=0.5)
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Concentration (mg/L)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
