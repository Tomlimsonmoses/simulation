"""
Unit tests for MLP architectures.

Tests:
  - Correct output shapes for all three architectures
  - MC Dropout: variance > 0 with dropout enabled during inference
  - StagedPKPDModel: output shape and intermediate PK output
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.mlp_architectures import SimpleMLP, DeepMLP, ResidualMLP
from models.pkpd_staged import StagedPKPDModel, Stage1PKModel, Stage2PDModel

INPUT_DIM = 32
BATCH = 16


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ModelCls,kwargs", [
    (SimpleMLP,   {"input_dim": INPUT_DIM, "hidden_dim": 64,  "output_dim": 2}),
    (DeepMLP,     {"input_dim": INPUT_DIM, "hidden_dim": 128, "output_dim": 2}),
    (ResidualMLP, {"input_dim": INPUT_DIM, "hidden_dim": 64,  "output_dim": 2}),
])
def test_output_shape(ModelCls, kwargs):
    model = ModelCls(**kwargs)
    x = torch.randn(BATCH, INPUT_DIM)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (BATCH, 2), f"Expected (B,2), got {out.shape}"


def test_simple_mlp_default_input_dim():
    """Default input_dim should now be 32 (updated from 30)."""
    model = SimpleMLP()
    # Shape of first weight matrix tells us input_dim
    first_weight_shape = next(model.parameters()).shape
    assert first_weight_shape[1] == 32, (
        f"Default input_dim should be 32, got {first_weight_shape[1]}"
    )


# ---------------------------------------------------------------------------
# Loss-decrease sanity test
# ---------------------------------------------------------------------------

def test_loss_decreases():
    """Training loss should decrease over 30 epochs on random data."""
    torch.manual_seed(0)
    model = SimpleMLP(input_dim=INPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X = torch.randn(200, INPUT_DIM)
    y = torch.rand(200, 2)

    model.train()
    first_loss = None
    for _ in range(30):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()

    assert loss.item() < first_loss, "Training loss did not decrease"


# ---------------------------------------------------------------------------
# MC Dropout test
# ---------------------------------------------------------------------------

def test_mc_dropout_variance():
    """Enabling dropout at inference must produce non-zero variance across samples."""
    model = SimpleMLP(input_dim=INPUT_DIM)
    x = torch.randn(1, INPUT_DIM)

    # Switch all Dropout layers to train mode
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    preds = []
    with torch.no_grad():
        for _ in range(50):
            preds.append(model(x).numpy()[0])
    preds = np.array(preds)

    assert preds.std(axis=0).max() > 0, "MC Dropout produced zero variance"


# ---------------------------------------------------------------------------
# StagedPKPDModel tests
# ---------------------------------------------------------------------------

def test_staged_model_output_shape():
    model = StagedPKPDModel(input_dim=INPUT_DIM)
    x = torch.randn(BATCH, INPUT_DIM)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (BATCH, 2)


def test_stage1_output_range():
    """Stage1 ends with Sigmoid so output must be in [0, 1]."""
    stage1 = Stage1PKModel(input_dim=22)
    x = torch.randn(BATCH, 22)
    stage1.eval()
    with torch.no_grad():
        out = stage1(x)
    assert out.min() >= 0.0 and out.max() <= 1.0, (
        f"Stage1 output outside [0,1]: min={out.min():.4f} max={out.max():.4f}"
    )


def test_staged_model_loss_decreases():
    """Staged model loss should systematically decrease over 40 epochs."""
    torch.manual_seed(1)
    model = StagedPKPDModel(input_dim=INPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X = torch.randn(200, INPUT_DIM)
    y = torch.rand(200, 2)

    model.train()
    first_loss = None
    for _ in range(40):
        optimizer.zero_grad()
        pred = model(X)
        loss = 0.5 * loss_fn(pred[:, 0], y[:, 0]) + 0.5 * loss_fn(pred[:, 1], y[:, 1])
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()

    assert loss.item() < first_loss, "Staged model training loss did not decrease"
