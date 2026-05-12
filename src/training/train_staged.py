"""
Staged PK/PD Model Training Script  (improved)
================================================
Trains StagedPKPDModel with a two-stage weighted loss:
    loss = pk_weight * Huber_PK + (1 - pk_weight) * Huber_PD

Improvements over v1:
  • Full dataset (real + synthetic, no 1 000-sample cap)
  • 200 epochs, ReduceLROnPlateau, early stopping (patience=20)
  • Huber loss + gradient clipping
  • Saves validation results as JSON → results/staged_training_results.json

Usage:
    python -m src.training.train_staged
    python src/training/train_staged.py
"""

import datetime
import json
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.pkpd_staged import StagedPKPDModel
from src.training.features import build_dataset


def train_staged(model, X_train, y_train, X_val, y_val,
                 epochs: int = 200,
                 batch_size: int = 256,
                 lr: float = 3e-4,
                 pk_weight: float = 0.55,
                 patience: int = 20,
                 device: str = "cpu"):
    model = model.to(device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_va = torch.tensor(y_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
    huber = nn.HuberLoss(delta=0.3)

    train_losses, val_losses = [], []
    best_val   = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = pk_weight * huber(pred[:, 0], yb[:, 0]) + \
                   (1 - pk_weight) * huber(pred[:, 1], yb[:, 1])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss  += loss.item()
            n_batches += 1
        train_losses.append(ep_loss / n_batches)

        model.eval()
        with torch.no_grad():
            vp  = model(X_va)
            vl  = pk_weight * huber(vp[:, 0], y_va[:, 0]) + \
                  (1 - pk_weight) * huber(vp[:, 1], y_va[:, 1])
            val_losses.append(vl.item())
        scheduler.step(vl.item())

        if vl.item() < best_val - 1e-6:
            best_val   = vl.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"train={train_losses[-1]:.5f}  val={val_losses[-1]:.5f}  "
                  f"lr={lr_now:.2e}  no_improve={no_improve}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds = model(X_va).cpu().numpy()
    y_np = y_val

    flat_y = y_np.flatten()
    flat_p = preds.flatten()
    metrics = {
        "model":         "staged_pkpd",
        "epochs_run":    len(train_losses),
        "training_sec":  round(time.time() - t0, 2),
        "best_val_loss": round(best_val, 6),
        "r2":            round(float(r2_score(flat_y, flat_p)), 4),
        "pk_r2":         round(float(r2_score(y_np[:, 0], preds[:, 0])), 4),
        "pd_r2":         round(float(r2_score(y_np[:, 1], preds[:, 1])), 4),
        "rmse":          round(float(np.sqrt(mean_squared_error(flat_y, flat_p))), 6),
        "mae":           round(float(mean_absolute_error(flat_y, flat_p)), 6),
        "pk_mae":        round(float(mean_absolute_error(y_np[:, 0], preds[:, 0])), 6),
        "pd_mae":        round(float(mean_absolute_error(y_np[:, 1], preds[:, 1])), 6),
        "pk_weight":     pk_weight,
        "train_loss_curve": [round(v, 6) for v in train_losses],
        "val_loss_curve":   [round(v, 6) for v in val_losses],
    }
    return model, metrics


def main():
    print("Staged PK/PD Model Training (v2)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = os.path.join(_ROOT, "data", "raw")

    syn_path = os.path.join(data_dir, "synthetic_clinical_data.csv")
    if not os.path.exists(syn_path):
        print("\nSynthetic data not found — generating 50 000 samples…")
        from src.training.generate_synthetic_data import generate_samples
        df = generate_samples(n=50_000, seed=42)
        df.to_csv(syn_path, index=False)
        print(f"  Saved {len(df):,} rows → {syn_path}")

    print("\nBuilding feature dataset (real + synthetic)…")
    X, y = build_dataset(data_dir, max_samples=None, use_synthetic=True)
    print(f"  Total: {len(X)} samples × {X.shape[1]} features")

    X_tmp,   X_test,  y_tmp,   y_test  = train_test_split(X, y,     test_size=0.10, random_state=42)
    X_train, X_val,   y_train, y_val   = train_test_split(X_tmp, y_tmp, test_size=0.111, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    input_dim = X.shape[1]
    model     = StagedPKPDModel(input_dim=input_dim, pk_hidden=256, pd_hidden=128)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"\nStagedPKPDModel — {n_params:,} parameters")

    trained_model, metrics = train_staged(
        model, X_train, y_train, X_val, y_val,
        epochs=200, batch_size=256, lr=3e-4, pk_weight=0.55,
        patience=20, device=device,
    )

    # Test evaluation
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    trained_model.eval()
    with torch.no_grad():
        tp = trained_model(X_te_t).cpu().numpy()
    flat_y = y_test.flatten()
    flat_p = tp.flatten()
    test_metrics = {
        "test_r2":    round(float(r2_score(flat_y, flat_p)), 4),
        "test_pk_r2": round(float(r2_score(y_test[:, 0], tp[:, 0])), 4),
        "test_pd_r2": round(float(r2_score(y_test[:, 1], tp[:, 1])), 4),
        "test_rmse":  round(float(np.sqrt(mean_squared_error(flat_y, flat_p))), 6),
        "test_pk_mae":round(float(mean_absolute_error(y_test[:, 0], tp[:, 0])), 6),
        "test_pd_mae":round(float(mean_absolute_error(y_test[:, 1], tp[:, 1])), 6),
    }

    print(f"\n  Val   R²={metrics['r2']:.4f}  PK={metrics['pk_r2']:.4f}  PD={metrics['pd_r2']:.4f}")
    print(f"  Test  R²={test_metrics['test_r2']:.4f}  PK={test_metrics['test_pk_r2']:.4f}  PD={test_metrics['test_pd_r2']:.4f}")

    # Checkpoint
    ckpt_dir  = os.path.join(_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "staged_pkpd_model.pth")
    torch.save(trained_model.state_dict(), ckpt_path)
    print(f"\nCheckpoint → {ckpt_path}")

    # JSON results
    results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_json = {
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device":        device,
        "dataset": {
            "n_train":   len(X_train),
            "n_val":     len(X_val),
            "n_test":    len(X_test),
            "feature_dim": int(input_dim),
        },
        "hyperparameters": {
            "epochs":    200,
            "batch_size":256,
            "lr":        3e-4,
            "pk_weight": 0.55,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=8)",
            "loss":      "Weighted HuberLoss(delta=0.3)",
            "early_stop_patience": 20,
            "grad_clip": 1.0,
        },
        "validation": metrics,
        "test":        test_metrics,
    }
    json_path = os.path.join(results_dir, "staged_training_results.json")
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"Validation JSON → {json_path}")

    # Training curves plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(metrics["train_loss_curve"], label="train", color="#00bcd4")
    ax.plot(metrics["val_loss_curve"],   label="val",   color="#ff7043")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title(f"Staged PK/PD  (Val PK R²={metrics['pk_r2']:.3f}  PD R²={metrics['pd_r2']:.3f}  "
                 f"Test R²={test_metrics['test_r2']:.3f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plot_path = os.path.join(results_dir, "staged_training_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves → {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
