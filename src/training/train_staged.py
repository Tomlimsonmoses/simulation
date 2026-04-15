"""
Staged PK/PD Model Training Script
=====================================
Trains StagedPKPDModel with a two-stage weighted loss:
    loss = pk_weight * PK_MSE + (1 - pk_weight) * PD_MSE

This mirrors the biological causal chain:
    drug + patient → predicted concentration → pharmacodynamic response

Usage (from project root with venv active):
    python -m src.training.train_staged
    # or
    python src/training/train_staged.py

Output:
    checkpoints/staged_pkpd_model.pth
    results/staged_training_curves.png
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── path fix for direct execution ──────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.mlp_architectures import SimpleMLP
from src.models.pkpd_staged import StagedPKPDModel
from src.training.features import build_dataset


def train_staged(model, X_train, y_train, X_test, y_test,
                 epochs: int = 80, batch_size: int = 32,
                 lr: float = 1e-3, pk_weight: float = 0.5):
    """Train StagedPKPDModel with a weighted PK + PD loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X_te   = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_te   = torch.tensor(y_test, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = pk_weight * mse(pred[:, 0], yb[:, 0]) + (1 - pk_weight) * mse(pred[:, 1], yb[:, 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(loader))

        model.eval()
        with torch.no_grad():
            vp = model(X_te)
            vl = pk_weight * mse(vp[:, 0], y_te[:, 0]) + (1 - pk_weight) * mse(vp[:, 1], y_te[:, 1])
            val_losses.append(vl.item())

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  "
                  f"train={train_losses[-1]:.5f}  val={val_losses[-1]:.5f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_te).cpu().numpy()
    y_np = y_test

    r2_pk  = r2_score(y_np[:, 0], preds[:, 0])
    r2_pd  = r2_score(y_np[:, 1], preds[:, 1])
    r2_all = r2_score(y_np.flatten(), preds.flatten())
    rmse   = float(np.sqrt(mean_squared_error(y_np.flatten(), preds.flatten())))

    print(f"\n  Training time : {time.time() - t0:.1f}s")
    print(f"  Overall  R²   : {r2_all:.4f}  RMSE: {rmse:.4f}")
    print(f"  PK R²         : {r2_pk:.4f}")
    print(f"  PD R²         : {r2_pd:.4f}")

    return model, train_losses, val_losses, r2_all, r2_pk, r2_pd


def main():
    print("Staged PK/PD Model Training")
    print("=" * 60)

    data_dir = os.path.join(_ROOT, "data", "raw")
    print("Building feature dataset…")
    X, y = build_dataset(data_dir, max_samples=1000)
    print(f"  {len(X)} samples × {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_dim = X.shape[1]
    model     = StagedPKPDModel(input_dim=input_dim, pk_hidden=128, pd_hidden=64)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"\nStagedPKPDModel — {n_params:,} parameters")

    model, train_losses, val_losses, r2, r2_pk, r2_pd = train_staged(
        model, X_train, y_train, X_test, y_test, epochs=80
    )

    # Baseline comparison
    print("\nBaseline SimpleMLP…")
    base = SimpleMLP(input_dim=input_dim)
    opt  = torch.optim.Adam(base.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_fn = nn.MSELoss()
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    for _ in range(80):
        base.train()
        opt.zero_grad()
        mse_fn(base(X_tr_t), y_tr_t).backward()
        opt.step()
    base.eval()
    with torch.no_grad():
        bp = base(X_tr_t).numpy()
    r2_base = r2_score(y_train.flatten(), bp.flatten())
    print(f"  Baseline SimpleMLP R² : {r2_base:.4f}")
    print(f"  Staged model   R²     : {r2:.4f}  "
          f"({'better' if r2 > r2_base else 'worse'} by {abs(r2 - r2_base):.4f})")

    # Save checkpoint
    ckpt_dir  = os.path.join(_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "staged_pkpd_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint → {ckpt_path}")

    # Training curves
    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses,   label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(f"Staged PK/PD  (PK R²={r2_pk:.3f}, PD R²={r2_pd:.3f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plot_path = os.path.join(out_dir, "staged_training_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves → {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
