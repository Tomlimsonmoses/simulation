"""
Graph Neural Network for PK-PD Modeling

This module implements a GNN architecture that combines pharmacokinetic (PK)
and pharmacodynamic (PD) modeling to predict organ responses to drugs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional


class PKPDGNN(nn.Module):
    """
    Graph Neural Network for PK-PD modeling.
    
    Architecture:
    1. Organ nodes with physiological features
    2. Drug nodes with pharmacological properties
    3. Edge connections representing blood flow and drug transport
    4. Temporal message passing for PK dynamics
    5. PD response prediction for each organ
    """
    
    def __init__(
        self,
        organ_features_dim: int = 16,
        drug_features_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        time_steps: int = 24
    ):
        super(PKPDGNN, self).__init__()
        
        self.organ_features_dim = organ_features_dim
        self.drug_features_dim = drug_features_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.time_steps = time_steps
        
        # Input projections
        self.organ_proj = nn.Linear(organ_features_dim, hidden_dim)
        self.drug_proj = nn.Linear(drug_features_dim, hidden_dim)
        
        # Graph attention layers for spatial modeling
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim  = hidden_dim if i == 0 else hidden_dim * num_heads
            is_last = (i == num_layers - 1)
            # Last layer: concat=False → output hidden_dim (matches pk/pd heads)
            # Other layers: concat=True → output hidden_dim * num_heads
            self.gat_layers.append(
                GATConv(
                    in_dim, hidden_dim, heads=num_heads,
                    dropout=dropout, concat=(not is_last),
                )
            )
        
        # Temporal modeling for PK dynamics
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,   # last GAT layer uses concat=False → hidden_dim
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # PK prediction head
        self.pk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Drug concentration
        )
        
        # PD prediction head
        self.pd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Organ response
        )
        
        # Disease modifier
        self.disease_proj = nn.Linear(4, hidden_dim)  # disease_type, severity, stage, duration
        
    def forward(
        self,
        organ_features: torch.Tensor,
        drug_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        disease_features: torch.Tensor,
        batch: torch.Tensor,
        time_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of PK-PD GNN.
        
        Args:
            organ_features: [num_organs, organ_features_dim]
            drug_features: [num_drugs, drug_features_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_features_dim]
            disease_features: [batch_size, 4]
            batch: [num_nodes] - batch assignment for each node
            time_indices: Optional time step indices
            
        Returns:
            Dict with PK and PD predictions
        """
        batch_size = batch.max().item() + 1
        
        # Project input features
        organ_h = self.organ_proj(organ_features)
        drug_h = self.drug_proj(drug_features)
        
        # Combine organ and drug features
        node_features = torch.cat([organ_h, drug_h], dim=0)
        
        # Apply disease modifier
        disease_h = self.disease_proj(disease_features)
        disease_broadcast = disease_h[batch]  # Broadcast to nodes
        
        # Add disease effects to node features
        node_features = node_features + disease_broadcast
        
        # Spatial message passing through GAT layers (edge_attr omitted — GATConv
        # was initialised without edge_dim; passing it would raise a runtime error)
        for i, gat_layer in enumerate(self.gat_layers):
            node_features = gat_layer(node_features, edge_index)
            if i < len(self.gat_layers) - 1:
                node_features = F.elu(node_features)
                node_features = F.dropout(node_features, p=self.dropout, training=self.training)
        
        # Separate organ and drug representations
        num_organs = organ_features.shape[0]
        organ_repr = node_features[:num_organs]
        
        # Temporal modeling for PK dynamics
        if time_indices is not None:
            # Reshape for temporal processing
            organ_repr_reshaped = organ_repr.unsqueeze(1).repeat(1, self.time_steps, 1)
            temporal_out, _ = self.temporal_encoder(organ_repr_reshaped)
            organ_repr = temporal_out[:, -1, :]  # Use last time step
        
        # Global pooling for batch processing
        organ_repr_pooled = global_mean_pool(organ_repr, batch[:num_organs])
        
        # PK predictions (drug concentrations)
        pk_predictions = self.pk_head(organ_repr_pooled)
        
        # PD predictions (organ responses)
        pd_predictions = self.pd_head(organ_repr_pooled)
        
        return {
            'pk_concentrations': pk_predictions,
            'pd_responses': pd_predictions,
            'organ_embeddings': organ_repr,
            'drug_embeddings': node_features[num_organs:]
        }


class PKPDDataProcessor:
    """
    Data processor for PK-PD GNN training and inference.
    """
    
    def __init__(self):
        self.organ_types = [
            'heart', 'liver', 'kidney', 'brain', 'lung', 'spleen',
            'muscle', 'fat', 'skin', 'bone', 'intestine', 'stomach'
        ]
        
    def create_organ_features(self, organ_params: Dict) -> torch.Tensor:
        """
        Create organ feature tensor from physiological parameters.
        
        Args:
            organ_params: Dict with organ-specific parameters
            
        Returns:
            Tensor of shape [num_organs, organ_features_dim]
        """
        features = []
        for organ in self.organ_types:
            if organ in organ_params:
                params = organ_params[organ]
                # Features: volume, blood_flow, metabolic_rate, receptor_density, etc.
                organ_vec = torch.tensor([
                    params.get('volume', 1.0),
                    params.get('blood_flow', 1.0),
                    params.get('metabolic_rate', 1.0),
                    params.get('receptor_density', 1.0),
                    params.get('permeability', 1.0),
                    params.get('clearance', 1.0),
                    params.get('surface_area', 1.0),
                    params.get('oxygen_consumption', 1.0),
                    params.get('ph', 7.4),
                    params.get('temperature', 37.0),
                    params.get('weight_ratio', 1.0),
                    params.get('enzyme_activity', 1.0),
                    params.get('transporter_density', 1.0),
                    params.get('binding_capacity', 1.0),
                    params.get('sensitivity', 1.0),
                    params.get('health_status', 1.0)
                ], dtype=torch.float32)
            else:
                # Default organ features
                organ_vec = torch.ones(16, dtype=torch.float32)
            
            features.append(organ_vec)
        
        return torch.stack(features)
    
    def create_drug_features(self, drug_params: Dict) -> torch.Tensor:
        """
        Create drug feature tensor from pharmacological properties.
        
        Args:
            drug_params: Dict with drug-specific parameters
            
        Returns:
            Tensor of shape [num_drugs, drug_features_dim]
        """
        drug_vec = torch.tensor([
            drug_params.get('molecular_weight', 300.0),
            drug_params.get('logp', 1.0),
            drug_params.get('polar_surface_area', 80.0),
            drug_params.get('solubility', 1.0),
            drug_params.get('bioavailability', 1.0),
            drug_params.get('protein_binding', 0.5),
            drug_params.get('half_life', 10.0),
            drug_params.get('clearance_rate', 1.0)
        ], dtype=torch.float32)
        
        return drug_vec.unsqueeze(0)
    
    def create_graph_edges(self, organ_connections: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create graph edges representing organ connections and drug transport.
        
        Args:
            organ_connections: Dict with connection strengths between organs
            
        Returns:
            edge_index, edge_attr tensors
        """
        edges = []
        edge_attrs = []
        
        # Create organ-organ connections (blood flow)
        for i, organ1 in enumerate(self.organ_types):
            for j, organ2 in enumerate(self.organ_types):
                if i != j and organ2 in organ_connections.get(organ1, {}):
                    flow_rate = organ_connections[organ1][organ2]
                    edges.append([i, j])
                    edge_attrs.append([flow_rate, 1.0, 0.0])  # [flow_rate, connection_type, drug_transport]
        
        # Create drug-organ connections (administration and elimination)
        num_organs = len(self.organ_types)
        for i, organ in enumerate(self.organ_types):
            # Drug administration to organ
            edges.append([num_organs, i])
            edge_attrs.append([1.0, 2.0, 1.0])  # [strength, type, drug_admin]
            
            # Drug elimination from organ
            edges.append([i, num_organs])
            edge_attrs.append([1.0, 3.0, -1.0])  # [strength, type, drug_elim]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        
        return edge_index, edge_attr


class PKPDTrainer:
    """
    Trainer for PK-PD GNN model.
    """
    
    def __init__(self, model: PKPDGNN, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss functions
        self.pk_loss_fn = nn.MSELoss()
        self.pd_loss_fn = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        pk_loss = 0.0
        pd_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(
                batch.organ_features,
                batch.drug_features,
                batch.edge_index,
                batch.edge_attr,
                batch.disease_features,
                batch.batch
            )
            
            # Calculate losses
            pk_loss_batch = self.pk_loss_fn(
                outputs['pk_concentrations'],
                batch.pk_targets
            )
            pd_loss_batch = self.pd_loss_fn(
                outputs['pd_responses'],
                batch.pd_targets
            )
            
            total_loss_batch = pk_loss_batch + pd_loss_batch
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            pk_loss += pk_loss_batch.item()
            pd_loss += pd_loss_batch.item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'pk_loss': pk_loss / num_batches,
            'pd_loss': pd_loss / num_batches
        }
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate the model.
        """
        self.model.eval()
        total_loss = 0.0
        pk_loss = 0.0
        pd_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                outputs = self.model(
                    batch.organ_features,
                    batch.drug_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.disease_features,
                    batch.batch
                )
                
                pk_loss_batch = self.pk_loss_fn(
                    outputs['pk_concentrations'],
                    batch.pk_targets
                )
                pd_loss_batch = self.pd_loss_fn(
                    outputs['pd_responses'],
                    batch.pd_targets
                )
                
                total_loss_batch = pk_loss_batch + pd_loss_batch
                
                total_loss += total_loss_batch.item()
                pk_loss += pk_loss_batch.item()
                pd_loss += pd_loss_batch.item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'pk_loss': pk_loss / num_batches,
            'pd_loss': pd_loss / num_batches
        }
