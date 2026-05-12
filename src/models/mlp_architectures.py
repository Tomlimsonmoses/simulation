"""
MLP Architectures for PK-PD Modeling
======================================

Shared neural network architectures used by training, evaluation, and simulation scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """Simple MLP baseline model."""
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=2):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class DeepMLP(nn.Module):
    """Deeper MLP model."""
    def __init__(self, input_dim=32, hidden_dim=256, output_dim=2):
        super(DeepMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    """Residual block for deep networks."""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        return self.relu(out)


class ResidualMLP(nn.Module):
    """MLP with residual connections."""
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=2):
        super(ResidualMLP, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(3)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_proj(x))
        for block in self.residual_blocks:
            x = block(x)
        return self.output_proj(x)


class FeatureAttentionBlock(nn.Module):
    """Soft feature-wise attention: learns which input dimensions matter most."""
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 8)),
            nn.ReLU(),
            nn.Linear(max(dim // reduction, 8), dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class AttentionMLP(nn.Module):
    """
    Deep MLP with feature-wise attention, residual connections, and layer norm.

    Architecture:
      Input → FeatureAttention → Linear(hidden) → 4× ResidualBlock(hidden)
      → LayerNorm → Linear(hidden//2) → GELU → Dropout → Linear(output)

    This is the highest-capacity model in the suite.  Use hidden_dim=256
    for production training.
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 256, output_dim: int = 2,
                 dropout: float = 0.25, n_blocks: int = 4):
        super().__init__()
        self.attention = FeatureAttentionBlock(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        x = self.attention(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = F.gelu(x + block(x))   # residual
        return self.head(x)
