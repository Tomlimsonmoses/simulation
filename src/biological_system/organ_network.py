"""
Virtual Human Biological System - Organ Network

This module implements a comprehensive virtual human biological system with:
- Multi-organ network topology
- Physiological parameter modeling
- Inter-organ communication
- Homeostatic regulation
- Individual variability
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
from enum import Enum


class OrganType(Enum):
    """Enumeration of organ types."""
    CENTRAL_NERVOUS_SYSTEM = "brain"
    CARDIOVASCULAR = "heart"
    HEPATIC = "liver"
    RENAL = "kidney"
    RESPIRATORY = "lung"
    ENDOCRINE = "endocrine"
    GASTROINTESTINAL = "gut"
    MUSCULOSKELETAL = "muscle"
    IMMUNE = "spleen"
    INTEGUMENTARY = "skin"
    ADIPOSE = "fat"
    SKELETAL = "bone"


@dataclass
class OrganParameters:
    """Physiological parameters for an organ."""
    organ_type: OrganType
    volume: float  # L
    mass: float  # kg
    blood_flow: float  # L/h
    oxygen_consumption: float  # mL/min
    metabolic_rate: float  # Relative to baseline
    ph: float
    temperature: float  # °C
    surface_area: float  # m²
    
    # Molecular properties
    water_content: float  # Fraction
    lipid_content: float  # Fraction
    protein_content: float  # Fraction
    
    # Functional properties
    enzyme_activity: Dict[str, float] = field(default_factory=dict)
    transporter_density: Dict[str, float] = field(default_factory=dict)
    receptor_density: Dict[str, float] = field(default_factory=dict)
    
    # Health status
    health_status: float = 1.0  # 0-1, 1 = healthy
    pathology_score: float = 0.0  # 0-1, severity of pathology


@dataclass
class ConnectionProperties:
    """Properties of connections between organs."""
    source_organ: OrganType
    target_organ: OrganType
    connection_type: str  # 'vascular', 'neural', 'hormonal', 'lymphatic'
    flow_rate: float  # L/h or signals/h
    resistance: float  # Resistance to flow/signal
    permeability: float  # Permeability coefficient
    delay: float  # Signal propagation delay (hours)
    
    # Molecular transport properties
    solute_permeability: Dict[str, float] = field(default_factory=dict)
    active_transport: Dict[str, float] = field(default_factory=dict)


class VirtualHuman:
    """
    Virtual human biological system.
    
    Implements a comprehensive model of human physiology with
    interconnected organs and physiological processes.
    """
    
    def __init__(
        self,
        age: int = 30,
        weight: float = 70.0,
        height: float = 170.0,
        sex: str = "male",
        ethnicity: str = "caucasian"
    ):
        self.age = age
        self.weight = weight
        self.height = height
        self.sex = sex
        self.ethnicity = ethnicity
        
        # Initialize organs
        self.organs = self._initialize_organs()
        
        # Initialize connections
        self.connections = self._initialize_connections()
        
        # Build network graph
        self.network_graph = self._build_network_graph()
        
        # Initialize physiological state
        self.physiological_state = self._initialize_physiological_state()
        
        # Time tracking
        self.current_time = 0.0
        
    def _initialize_organs(self) -> Dict[OrganType, OrganParameters]:
        """Initialize organ parameters based on anthropometric data."""
        organs = {}
        
        # Allometric scaling based on body weight
        weight_factor = (self.weight / 70) ** 0.75
        volume_factor = self.weight / 70
        
        # Brain
        organs[OrganType.CENTRAL_NERVOUS_SYSTEM] = OrganParameters(
            organ_type=OrganType.CENTRAL_NERVOUS_SYSTEM,
            volume=1.4 * volume_factor,
            mass=1.4 * volume_factor,
            blood_flow=750 * weight_factor,
            oxygen_consumption=50 * weight_factor,
            metabolic_rate=1.2,
            ph=7.3,
            temperature=37.0,
            surface_area=0.2 * volume_factor ** (2/3),
            water_content=0.8,
            lipid_content=0.12,
            protein_content=0.08,
            enzyme_activity={'cyp450': 0.1, 'mao': 0.8},
            transporter_density={'p-gp': 0.5, 'glut': 0.3},
            receptor_density={'gaba': 1.0, 'glutamate': 1.0, 'dopamine': 0.8}
        )
        
        # Heart
        organs[OrganType.CARDIOVASCULAR] = OrganParameters(
            organ_type=OrganType.CARDIOVASCULAR,
            volume=0.3 * volume_factor,
            mass=0.3 * volume_factor,
            blood_flow=240 * weight_factor,
            oxygen_consumption=20 * weight_factor,
            metabolic_rate=1.5,
            ph=7.4,
            temperature=37.0,
            surface_area=0.1 * volume_factor ** (2/3),
            water_content=0.75,
            lipid_content=0.05,
            protein_content=0.2,
            enzyme_activity={'ace': 1.0, 'beta_receptors': 1.0},
            transporter_density={'cat': 0.8, 'net': 0.6},
            receptor_density={'beta1': 1.0, 'beta2': 0.5, 'alpha1': 0.3}
        )
        
        # Liver
        organs[OrganType.HEPATIC] = OrganParameters(
            organ_type=OrganType.HEPATIC,
            volume=1.5 * volume_factor,
            mass=1.5 * volume_factor,
            blood_flow=1350 * weight_factor,
            oxygen_consumption=40 * weight_factor,
            metabolic_rate=1.8,
            ph=7.4,
            temperature=37.0,
            surface_area=1.5 * volume_factor ** (2/3),
            water_content=0.7,
            lipid_content=0.05,
            protein_content=0.25,
            enzyme_activity={'cyp450': 1.0, 'ugt': 0.8, 'sult': 0.6},
            transporter_density={'oatp': 1.0, 'ntcp': 0.8, 'mrp': 0.6},
            receptor_density={'nuclear_receptors': 1.0, 'pregnane_x': 0.8}
        )
        
        # Kidney
        organs[OrganType.RENAL] = OrganParameters(
            organ_type=OrganType.RENAL,
            volume=0.3 * volume_factor,
            mass=0.3 * volume_factor,
            blood_flow=1100 * weight_factor,
            oxygen_consumption=15 * weight_factor,
            metabolic_rate=1.3,
            ph=7.4,
            temperature=37.0,
            surface_area=0.3 * volume_factor ** (2/3),
            water_content=0.8,
            lipid_content=0.03,
            protein_content=0.17,
            enzyme_activity={'carbonic_anhydrase': 1.0, 'renin': 0.8},
            transporter_density={'oct': 1.0, 'oat': 0.8, 'aquaporin': 1.0},
            receptor_density={'at1': 1.0, 'dopamine_d1': 0.6}
        )
        
        # Lung
        organs[OrganType.RESPIRATORY] = OrganParameters(
            organ_type=OrganType.RESPIRATORY,
            volume=2.5 * volume_factor,
            mass=2.5 * volume_factor,
            blood_flow=4200 * weight_factor,  # Full cardiac output
            oxygen_consumption=10 * weight_factor,
            metabolic_rate=1.1,
            ph=7.4,
            temperature=37.0,
            surface_area=2.0 * volume_factor ** (2/3),
            water_content=0.8,
            lipid_content=0.02,
            protein_content=0.18,
            enzyme_activity={'ace': 0.5, 'mao': 0.3},
            transporter_density={'p-gp': 0.3, 'mrp': 0.2},
            receptor_density={'beta2': 1.0, 'muscarinic': 0.8}
        )
        
        # Muscle
        organs[OrganType.MUSCULOSKELETAL] = OrganParameters(
            organ_type=OrganType.MUSCULOSKELETAL,
            volume=28.0 * volume_factor,
            mass=28.0 * volume_factor,
            blood_flow=900 * weight_factor,
            oxygen_consumption=30 * weight_factor,
            metabolic_rate=0.7,
            ph=7.4,
            temperature=37.0,
            surface_area=12.0 * volume_factor ** (2/3),
            water_content=0.75,
            lipid_content=0.02,
            protein_content=0.23,
            enzyme_activity={'creatine_kinase': 1.0, 'ldh': 0.8},
            transporter_density={'glut4': 0.5, 'mct': 0.3},
            receptor_density={'androgen': 0.6, 'insulin': 0.8}
        )
        
        # Fat
        organs[OrganType.ADIPOSE] = OrganParameters(
            organ_type=OrganType.ADIPOSE,
            volume=15.0 * volume_factor,
            mass=15.0 * volume_factor,
            blood_flow=300 * weight_factor,
            oxygen_consumption=5 * weight_factor,
            metabolic_rate=0.3,
            ph=7.4,
            temperature=37.0,
            surface_area=15.0 * volume_factor ** (2/3),
            water_content=0.1,
            lipid_content=0.8,
            protein_content=0.1,
            enzyme_activity={'lipase': 1.0, 'hsl': 0.8},
            transporter_density={'fatp': 0.8, 'cd36': 0.6},
            receptor_density={'ppar': 1.0, 'adrenergic': 0.4}
        )
        
        # Gut
        organs[OrganType.GASTROINTESTINAL] = OrganParameters(
            organ_type=OrganType.GASTROINTESTINAL,
            volume=1.0 * volume_factor,
            mass=1.0 * volume_factor,
            blood_flow=900 * weight_factor,
            oxygen_consumption=12 * weight_factor,
            metabolic_rate=1.1,
            ph=6.5,
            temperature=37.0,
            surface_area=2.0 * volume_factor ** (2/3),
            water_content=0.75,
            lipid_content=0.05,
            protein_content=0.2,
            enzyme_activity={'cyp3a4': 0.8, 'ugt': 0.6},
            transporter_density={'p-gp': 1.0, 'pept1': 0.8},
            receptor_density={'serotonin': 0.6, 'dopamine': 0.4}
        )
        
        # Spleen
        organs[OrganType.IMMUNE] = OrganParameters(
            organ_type=OrganType.IMMUNE,
            volume=0.2 * volume_factor,
            mass=0.2 * volume_factor,
            blood_flow=150 * weight_factor,
            oxygen_consumption=8 * weight_factor,
            metabolic_rate=1.0,
            ph=7.4,
            temperature=37.0,
            surface_area=0.1 * volume_factor ** (2/3),
            water_content=0.78,
            lipid_content=0.02,
            protein_content=0.2,
            enzyme_activity={'complement': 1.0, 'cytokines': 0.8},
            transporter_density={'fcr': 1.0, 'cr1': 0.6},
            receptor_density={'tlr': 1.0, 'fcr': 0.8}
        )
        
        # Skin
        organs[OrganType.INTEGUMENTARY] = OrganParameters(
            organ_type=OrganType.INTEGUMENTARY,
            volume=3.0 * volume_factor,
            mass=3.0 * volume_factor,
            blood_flow=300 * weight_factor,
            oxygen_consumption=8 * weight_factor,
            metabolic_rate=0.8,
            ph=7.4,
            temperature=37.0,
            surface_area=2.0 * volume_factor ** (2/3),
            water_content=0.7,
            lipid_content=0.07,
            protein_content=0.23,
            enzyme_activity={'collagenase': 0.3, 'elastase': 0.2},
            transporter_density={'aquaporin': 0.3, 'claudin': 0.2},
            receptor_density={'histamine': 0.6, 'bradykinin': 0.4}
        )
        
        # Bone
        organs[OrganType.SKELETAL] = OrganParameters(
            organ_type=OrganType.SKELETAL,
            volume=5.0 * volume_factor,
            mass=5.0 * volume_factor,
            blood_flow=200 * weight_factor,
            oxygen_consumption=6 * weight_factor,
            metabolic_rate=0.4,
            ph=7.4,
            temperature=37.0,
            surface_area=3.0 * volume_factor ** (2/3),
            water_content=0.25,
            lipid_content=0.05,
            protein_content=0.7,
            enzyme_activity={'alkaline_phosphatase': 0.8, 'osteocalcin': 0.6},
            transporter_density={'sclerostin': 0.4, 'rank': 0.3},
            receptor_density={'vitamin_d': 1.0, 'parathyroid': 0.8}
        )
        
        # Apply age and sex modifications
        self._apply_demographic_modifications(organs)
        
        return organs
    
    def _apply_demographic_modifications(self, organs: Dict[OrganType, OrganParameters]):
        """Apply age and sex-specific modifications to organ parameters."""
        # Age-related changes
        if self.age > 65:
            age_factor = 1.0 - 0.01 * (self.age - 65)  # 1% decline per year after 65
            
            for organ in organs.values():
                organ.blood_flow *= age_factor
                organ.metabolic_rate *= age_factor
                organ.enzyme_activity = {k: v * age_factor for k, v in organ.enzyme_activity.items()}
        
        elif self.age < 18:
            # Pediatric scaling
            age_factor = self.age / 18.0
            
            for organ in organs.values():
                organ.metabolic_rate *= (1.0 + 0.5 * (1.0 - age_factor))  # Higher metabolism in young
        
        # Sex-related differences
        if self.sex == "female":
            # Females typically have higher body fat percentage
            organs[OrganType.ADIPOSE].volume *= 1.3
            organs[OrganType.ADIPOSE].mass *= 1.3
            
            # Lower muscle mass
            organs[OrganType.MUSCULOSKELETAL].volume *= 0.8
            organs[OrganType.MUSCULOSKELETAL].mass *= 0.8
            
            # Different enzyme activities
            organs[OrganType.HEPATIC].enzyme_activity['cyp450'] *= 0.8
            organs[OrganType.HEPATIC].enzyme_activity['ugt'] *= 1.2
    
    def _initialize_connections(self) -> List[ConnectionProperties]:
        """Initialize connections between organs."""
        connections = []
        
        # Vascular connections (blood flow)
        vascular_connections = [
            # From heart to organs
            ("heart", "brain", "vascular", 750, 0.1, 0.8, 0.01),
            ("heart", "liver", "vascular", 1350, 0.05, 0.9, 0.01),
            ("heart", "kidney", "vascular", 1100, 0.08, 0.85, 0.01),
            ("heart", "muscle", "vascular", 900, 0.12, 0.7, 0.02),
            ("heart", "fat", "vascular", 300, 0.15, 0.3, 0.03),
            ("heart", "gut", "vascular", 900, 0.1, 0.8, 0.01),
            ("heart", "skin", "vascular", 300, 0.2, 0.6, 0.02),
            ("heart", "spleen", "vascular", 150, 0.1, 0.8, 0.01),
            
            # Return flows to heart via lungs
            ("brain", "lung", "vascular", 750, 0.1, 0.8, 0.01),
            ("liver", "lung", "vascular", 1350, 0.05, 0.9, 0.01),
            ("kidney", "lung", "vascular", 1100, 0.08, 0.85, 0.01),
            ("muscle", "lung", "vascular", 900, 0.12, 0.7, 0.02),
            ("fat", "lung", "vascular", 300, 0.15, 0.3, 0.03),
            ("gut", "liver", "vascular", 900, 0.1, 0.8, 0.01),  # Portal circulation
            ("skin", "lung", "vascular", 300, 0.2, 0.6, 0.02),
            ("spleen", "lung", "vascular", 150, 0.1, 0.8, 0.01),
        ]
        
        for source, target, conn_type, flow, resistance, permeability, delay in vascular_connections:
            connections.append(ConnectionProperties(
                source_organ=OrganType(source),
                target_organ=OrganType(target),
                connection_type=conn_type,
                flow_rate=flow,
                resistance=resistance,
                permeability=permeability,
                delay=delay
            ))
        
        # Neural connections
        neural_connections = [
            ("brain", "heart", "neural", 1000, 0.05, 1.0, 0.001),
            ("brain", "liver", "neural", 500, 0.1, 0.8, 0.002),
            ("brain", "kidney", "neural", 600, 0.08, 0.9, 0.001),
            ("brain", "gut", "neural", 800, 0.12, 0.7, 0.001),
        ]
        
        for source, target, conn_type, flow, resistance, permeability, delay in neural_connections:
            connections.append(ConnectionProperties(
                source_organ=OrganType(source),
                target_organ=OrganType(target),
                connection_type=conn_type,
                flow_rate=flow,
                resistance=resistance,
                permeability=permeability,
                delay=delay
            ))
        
        # Hormonal connections
        hormonal_connections = [
            ("endocrine", "heart", "hormonal", 100, 0.2, 0.6, 0.1),
            ("endocrine", "kidney", "hormonal", 80, 0.15, 0.7, 0.1),
            ("endocrine", "liver", "hormonal", 120, 0.1, 0.8, 0.1),
            ("endocrine", "muscle", "hormonal", 150, 0.25, 0.5, 0.1),
        ]
        
        for source, target, conn_type, flow, resistance, permeability, delay in hormonal_connections:
            connections.append(ConnectionProperties(
                source_organ=OrganType(source),
                target_organ=OrganType(target),
                connection_type=conn_type,
                flow_rate=flow,
                resistance=resistance,
                permeability=permeability,
                delay=delay
            ))
        
        return connections
    
    def _build_network_graph(self) -> nx.DiGraph:
        """Build NetworkX graph representing organ network."""
        G = nx.DiGraph()
        
        # Add organ nodes
        for organ_type, organ_params in self.organs.items():
            G.add_node(
                organ_type.value,
                **{
                    'volume': organ_params.volume,
                    'mass': organ_params.mass,
                    'blood_flow': organ_params.blood_flow,
                    'metabolic_rate': organ_params.metabolic_rate,
                    'health_status': organ_params.health_status
                }
            )
        
        # Add connection edges
        for conn in self.connections:
            G.add_edge(
                conn.source_organ.value,
                conn.target_organ.value,
                **{
                    'type': conn.connection_type,
                    'flow_rate': conn.flow_rate,
                    'resistance': conn.resistance,
                    'permeability': conn.permeability,
                    'delay': conn.delay
                }
            )
        
        return G
    
    def _initialize_physiological_state(self) -> Dict[str, float]:
        """Initialize baseline physiological state variables."""
        return {
            'heart_rate': 70.0,  # bpm
            'blood_pressure_systolic': 120.0,  # mmHg
            'blood_pressure_diastolic': 80.0,  # mmHg
            'respiratory_rate': 12.0,  # breaths/min
            'body_temperature': 37.0,  # °C
            'blood_glucose': 100.0,  # mg/dL
            'blood_oxygen': 98.0,  # %
            'blood_ph': 7.4,
            'cardiac_output': 5.0,  # L/min
            'gfr': 125.0,  # mL/min/1.73m²
            'liver_function': 1.0,  # Relative to normal
            'kidney_function': 1.0,  # Relative to normal
            'lung_function': 1.0,  # Relative to normal
        }
    
    def get_organ_parameters(self, organ_type: Union[str, OrganType]) -> Optional[OrganParameters]:
        """Get parameters for a specific organ."""
        if isinstance(organ_type, str):
            organ_type = OrganType(organ_type)
        
        return self.organs.get(organ_type)
    
    def get_connection_properties(
        self,
        source: Union[str, OrganType],
        target: Union[str, OrganType]
    ) -> Optional[ConnectionProperties]:
        """Get connection properties between two organs."""
        if isinstance(source, str):
            source = OrganType(source)
        if isinstance(target, str):
            target = OrganType(target)
        
        for conn in self.connections:
            if conn.source_organ == source and conn.target_organ == target:
                return conn
        
        return None
    
    def modify_organ_health(self, organ_type: Union[str, OrganType], health_status: float):
        """Modify health status of an organ."""
        if isinstance(organ_type, str):
            organ_type = OrganType(organ_type)
        
        if organ_type in self.organs:
            self.organs[organ_type].health_status = health_status
            self.organs[organ_type].pathology_score = 1.0 - health_status
            
            # Update physiological parameters based on health status
            organ = self.organs[organ_type]
            organ.blood_flow *= health_status
            organ.metabolic_rate *= health_status
            organ.enzyme_activity = {k: v * health_status for k, v in organ.enzyme_activity.items()}
    
    def simulate_physiological_response(self, time_step: float = 0.1) -> Dict[str, float]:
        """
        Simulate one time step of physiological response.
        
        Args:
            time_step: Time step in hours
            
        Returns:
            Updated physiological state
        """
        # Simple homeostatic regulation model
        new_state = self.physiological_state.copy()
        
        # Heart rate regulation
        hr_target = 70.0
        hr_current = self.physiological_state['heart_rate']
        new_state['heart_rate'] += 0.1 * (hr_target - hr_current) * time_step
        
        # Blood pressure regulation
        bp_systolic_target = 120.0
        bp_diastolic_target = 80.0
        new_state['blood_pressure_systolic'] += 0.1 * (bp_systolic_target - self.physiological_state['blood_pressure_systolic']) * time_step
        new_state['blood_pressure_diastolic'] += 0.1 * (bp_diastolic_target - self.physiological_state['blood_pressure_diastolic']) * time_step
        
        # Temperature regulation
        temp_target = 37.0
        new_state['body_temperature'] += 0.5 * (temp_target - self.physiological_state['body_temperature']) * time_step
        
        # Update time
        self.current_time += time_step
        
        self.physiological_state = new_state
        return new_state
    
    def export_network_data(self) -> Dict:
        """Export network data for GNN processing."""
        # Node features
        organ_types = list(self.organs.keys())
        node_features = []
        
        for organ_type in organ_types:
            organ = self.organs[organ_type]
            features = [
                organ.volume,
                organ.mass,
                organ.blood_flow,
                organ.metabolic_rate,
                organ.ph,
                organ.temperature,
                organ.water_content,
                organ.lipid_content,
                organ.protein_content,
                organ.health_status,
                organ.pathology_score
            ]
            node_features.append(features)
        
        # Edge indices and attributes
        edge_index = []
        edge_attr = []
        
        organ_to_idx = {organ_type.value: i for i, organ_type in enumerate(organ_types)}
        
        for conn in self.connections:
            if conn.source_organ in organ_to_idx and conn.target_organ in organ_to_idx:
                source_idx = organ_to_idx[conn.source_organ.value]
                target_idx = organ_to_idx[conn.target_organ.value]
                
                edge_index.append([source_idx, target_idx])
                edge_attr.append([
                    conn.flow_rate,
                    conn.resistance,
                    conn.permeability,
                    conn.delay,
                    1.0 if conn.connection_type == 'vascular' else 0.0,
                    1.0 if conn.connection_type == 'neural' else 0.0,
                    1.0 if conn.connection_type == 'hormonal' else 0.0
                ])
        
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            'edge_attr': torch.tensor(edge_attr, dtype=torch.float32),
            'organ_types': [ot.value for ot in organ_types]
        }
    
    def save_to_file(self, filename: str):
        """Save virtual human configuration to file."""
        data = {
            'demographics': {
                'age': self.age,
                'weight': self.weight,
                'height': self.height,
                'sex': self.sex,
                'ethnicity': self.ethnicity
            },
            'organs': {
                organ_type.value: {
                    'volume': organ.volume,
                    'mass': organ.mass,
                    'blood_flow': organ.blood_flow,
                    'oxygen_consumption': organ.oxygen_consumption,
                    'metabolic_rate': organ.metabolic_rate,
                    'ph': organ.ph,
                    'temperature': organ.temperature,
                    'surface_area': organ.surface_area,
                    'water_content': organ.water_content,
                    'lipid_content': organ.lipid_content,
                    'protein_content': organ.protein_content,
                    'enzyme_activity': organ.enzyme_activity,
                    'transporter_density': organ.transporter_density,
                    'receptor_density': organ.receptor_density,
                    'health_status': organ.health_status,
                    'pathology_score': organ.pathology_score
                }
                for organ_type, organ in self.organs.items()
            },
            'connections': [
                {
                    'source_organ': conn.source_organ.value,
                    'target_organ': conn.target_organ.value,
                    'connection_type': conn.connection_type,
                    'flow_rate': conn.flow_rate,
                    'resistance': conn.resistance,
                    'permeability': conn.permeability,
                    'delay': conn.delay,
                    'solute_permeability': conn.solute_permeability,
                    'active_transport': conn.active_transport
                }
                for conn in self.connections
            ],
            'physiological_state': self.physiological_state
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'VirtualHuman':
        """Load virtual human configuration from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Create virtual human with demographics
        vh = cls(
            age=data['demographics']['age'],
            weight=data['demographics']['weight'],
            height=data['demographics']['height'],
            sex=data['demographics']['sex'],
            ethnicity=data['demographics']['ethnicity']
        )
        
        # Load organs
        for organ_name, organ_data in data['organs'].items():
            organ_type = OrganType(organ_name)
            if organ_type in vh.organs:
                organ = vh.organs[organ_type]
                organ.volume = organ_data['volume']
                organ.mass = organ_data['mass']
                organ.blood_flow = organ_data['blood_flow']
                organ.oxygen_consumption = organ_data['oxygen_consumption']
                organ.metabolic_rate = organ_data['metabolic_rate']
                organ.ph = organ_data['ph']
                organ.temperature = organ_data['temperature']
                organ.surface_area = organ_data['surface_area']
                organ.water_content = organ_data['water_content']
                organ.lipid_content = organ_data['lipid_content']
                organ.protein_content = organ_data['protein_content']
                organ.enzyme_activity = organ_data['enzyme_activity']
                organ.transporter_density = organ_data['transporter_density']
                organ.receptor_density = organ_data['receptor_density']
                organ.health_status = organ_data['health_status']
                organ.pathology_score = organ_data['pathology_score']
        
        # Load physiological state
        vh.physiological_state = data['physiological_state']
        
        return vh


class PopulationGenerator:
    """Generate population of virtual humans with variability."""
    
    def __init__(self, population_size: int = 100):
        self.population_size = population_size
        
    def generate_population(
        self,
        age_range: Tuple[int, int] = (18, 80),
        weight_range: Tuple[float, float] = (50, 100),
        sex_distribution: Dict[str, float] = {"male": 0.5, "female": 0.5}
    ) -> List[VirtualHuman]:
        """Generate a population of virtual humans."""
        population = []
        
        for i in range(self.population_size):
            # Random demographics
            age = np.random.randint(age_range[0], age_range[1] + 1)
            weight = np.random.normal(70, 10)
            weight = np.clip(weight, weight_range[0], weight_range[1])
            height = np.random.normal(170, 10)
            sex = np.random.choice(
                list(sex_distribution.keys()),
                p=list(sex_distribution.values())
            )
            
            # Create virtual human
            vh = VirtualHuman(age=age, weight=weight, height=height, sex=sex)
            
            # Add individual variability
            self._add_individual_variability(vh)
            
            population.append(vh)
        
        return population
    
    def _add_individual_variability(self, vh: VirtualHuman):
        """Add individual variability to organ parameters."""
        for organ in vh.organs.values():
            # Add 10-20% variability to key parameters
            organ.blood_flow *= np.random.normal(1.0, 0.1)
            organ.metabolic_rate *= np.random.normal(1.0, 0.15)
            organ.enzyme_activity = {
                k: v * np.random.normal(1.0, 0.2)
                for k, v in organ.enzyme_activity.items()
            }
            organ.transporter_density = {
                k: v * np.random.normal(1.0, 0.2)
                for k, v in organ.transporter_density.items()
            }
            organ.receptor_density = {
                k: v * np.random.normal(1.0, 0.15)
                for k, v in organ.receptor_density.items()
            }
