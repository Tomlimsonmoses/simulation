"""
Staged PK/PD Model Training Script
=====================================

Trains StagedPKPDModel using a two-stage loss:
    loss = alpha * PK_loss + (1 - alpha) * PD_loss

The staged architecture mirrors the biological causal chain:
    drug + patient → predicted concentration → pharmacodynamic response

Usage:
    cd F:/WORK/projects/MEDAI/PK-PD
    .\\env\\Scripts\\activate
    python scripts/train_staged.py

Outputs:
    checkpoints/staged_pkpd_model.pth
    results/simulations/staged_training_curves.png
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from models.mlp_architectures import SimpleMLP
from models.pkpd_staged import StagedPKPDModel

# Reuse the fixed create_features from train.py
# We do this via a direct import so changes are shared automatically
sys.path.insert(0, os.path.dirname(__file__))
from train import create_features  # noqa: E402


def train_staged(model, X_train, y_train, X_test, y_test,
                 epochs=80, batch_size=32, lr=1e-3,
                 pk_weight=0.5):
    """Train StagedPKPDModel with a weighted PK + PD loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)                            # (B, 2)
            pk_loss = mse(pred[:, 0], yb[:, 0])
            pd_loss = mse(pred[:, 1], yb[:, 1])
            loss = pk_weight * pk_loss + (1 - pk_weight) * pd_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(loader))

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_te.to(device))
            vl_pk = mse(val_pred[:, 0], y_te[:, 0].to(device))
            vl_pd = mse(val_pred[:, 1], y_te[:, 1].to(device))
            val_l = pk_weight * vl_pk + (1 - pk_weight) * vl_pd
            val_losses.append(val_l.item())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"train={train_losses[-1]:.5f}  val={val_losses[-1]:.5f}")

    elapsed = time.time() - t0

    # Final metrics
    model.eval()
    with torch.no_grad():
        preds = model(X_te.to(device)).cpu().numpy()
    y_np = y_test

    r2_pk = r2_score(y_np[:, 0], preds[:, 0])
    r2_pd = r2_score(y_np[:, 1], preds[:, 1])
    r2_all = r2_score(y_np.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(y_np.flatten(), preds.flatten()))

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Overall R² : {r2_all:.4f}  RMSE: {rmse:.4f}")
    print(f"  PK R²      : {r2_pk:.4f}")
    print(f"  PD R²      : {r2_pd:.4f}")

    return model, train_losses, val_losses, r2_all, r2_pk, r2_pd


def main():
    print("Staged PK/PD Model Training")
    print("=" * 60)

    # Load datasets
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    drug_data = pd.read_csv(os.path.join(data_dir, 'drug_properties.csv'))
    organ_data = pd.read_csv(os.path.join(data_dir, 'organ_network.csv'))
    clinical_data = pd.read_csv(os.path.join(data_dir, 'clinical_trials.csv'))
    disease_data = pd.read_csv(os.path.join(data_dir, 'disease_data.csv'))

    print(f"Loaded {len(drug_data)} drugs, {len(clinical_data)} clinical records")

    # Build feature matrix (32-dim, same as train.py)
    X, y = [], []
    for i in range(min(1000, len(clinical_data))):
        try:
            feats, targets = create_features(i, drug_data, organ_data, clinical_data, disease_data)
            X.append(feats)
            y.append(targets)
        except Exception as e:
            print(f"  Skipped sample {i}: {e}")
    X = np.array(X)
    y = np.array(y)
    print(f"Built {len(X)} samples × {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---- Train staged model ----
    input_dim = X.shape[1]
    model = StagedPKPDModel(input_dim=input_dim, pk_hidden=128, pd_hidden=64)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nStagedPKPDModel — {param_count:,} parameters")
    print("Training…")

    model, train_losses, val_losses, r2, r2_pk, r2_pd = train_staged(
        model, X_train, y_train, X_test, y_test,
        epochs=80, batch_size=32, pk_weight=0.5
    )

    # ---- Baseline comparison (SimpleMLP, same data) ----
    print("\nBaseline SimpleMLP for comparison…")
    baseline = SimpleMLP(input_dim=input_dim)
    device = torch.device('cpu')
    baseline = baseline.to(device)
    optim = torch.optim.Adam(baseline.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_fn = nn.MSELoss()
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32)
    y_te_t = torch.tensor(y_test,  dtype=torch.float32)

    for _ in range(80):
        baseline.train()
        optim.zero_grad()
        loss = mse_fn(baseline(X_tr_t), y_tr_t)
        loss.backward()
        optim.step()

    baseline.eval()
    with torch.no_grad():
        base_preds = baseline(X_te_t).numpy()
    r2_base = r2_score(y_test.flatten(), base_preds.flatten())
    print(f"  Baseline SimpleMLP R² : {r2_base:.4f}")
    print(f"  Staged model R²       : {r2:.4f}  "
          f"({'better' if r2 > r2_base else 'worse'} by {abs(r2 - r2_base):.4f})")

    # ---- Save checkpoint ----
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'staged_pkpd_model.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")

    # ---- Plot training curves ----
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'simulations')
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='Train loss')
    ax.plot(val_losses, label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Staged PK/PD Training  (PK R²={r2_pk:.3f}, PD R²={r2_pd:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(out_dir, 'staged_training_curves.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training curve saved → {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
