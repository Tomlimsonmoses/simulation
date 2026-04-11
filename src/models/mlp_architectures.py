"""
MLP Architectures for PK-PD Modeling
======================================

Shared neural network architectures used by training, evaluation, and simulation scripts.
"""

import torch.nn as nn


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
