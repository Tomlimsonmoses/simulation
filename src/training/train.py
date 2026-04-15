"""
PK-PD MLP Training Script
==========================
Trains SimpleMLP, DeepMLP, and ResidualMLP on clinical_trials.csv,
saves checkpoints to checkpoints/, and writes a comparison plot to
results/model_comparison.png.

Usage (from project root with venv active):
    python -m src.training.train
    # or
    python src/training/train.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── path fix for direct execution ──────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.mlp_architectures import DeepMLP, ResidualMLP, SimpleMLP
from src.training.features import build_dataset


# ── core training loop ──────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, X_test, y_test,
                model_name: str, epochs: int = 50, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_te = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test,  dtype=torch.float32).to(device)

    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(X_tr), batch_size):
            xb, yb = X_tr[i:i + batch_size], y_tr[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        train_losses.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_te), y_te).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  {model_name}  epoch {epoch+1:3d}/{epochs} "
                  f"train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds_np = model(X_te).cpu().numpy()
    y_np = y_te.cpu().numpy()

    metrics = {
        "model_name":    model_name,
        "training_time": time.time() - t0,
        "mse":           mean_squared_error(y_np.flatten(), preds_np.flatten()),
        "mae":           mean_absolute_error(y_np.flatten(), preds_np.flatten()),
        "r2":            r2_score(y_np.flatten(), preds_np.flatten()),
        "pk_r2":         r2_score(y_np[:, 0], preds_np[:, 0]),
        "pd_r2":         r2_score(y_np[:, 1], preds_np[:, 1]),
        "rmse":          float(np.sqrt(mean_squared_error(y_np.flatten(), preds_np.flatten()))),
        "train_losses":  train_losses,
        "val_losses":    val_losses,
    }
    return metrics, model


# ── entry point ─────────────────────────────────────────────────────────────────
def main():
    print("PK-PD MLP Training")
    print("=" * 60)

    data_dir = os.path.join(_ROOT, "data", "raw")
    print("Building feature dataset…")
    X, y = build_dataset(data_dir, max_samples=1000)
    print(f"  {len(X)} samples × {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  train={len(X_train)}  test={len(X_test)}")

    input_dim = X.shape[1]
    model_specs = {
        "simple_mlp":   SimpleMLP(input_dim=input_dim, hidden_dim=128, output_dim=2),
        "deep_mlp":     DeepMLP(input_dim=input_dim, hidden_dim=256, output_dim=2),
        "residual_mlp": ResidualMLP(input_dim=input_dim, hidden_dim=128, output_dim=2),
    }

    ckpt_dir = os.path.join(_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_metrics, trained = [], {}
    for key, model in model_specs.items():
        print(f"\nTraining {key}…")
        metrics, trained_model = train_model(
            model, X_train, y_train, X_test, y_test,
            model_name=key, epochs=50, batch_size=32,
        )
        all_metrics.append(metrics)
        trained[key] = trained_model
        ckpt_path = os.path.join(ckpt_dir, f"{key}_model.pth")
        torch.save(trained_model.state_dict(), ckpt_path)
        print(f"  → saved {ckpt_path}  R²={metrics['r2']:.4f}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'R²':>8} {'RMSE':>8} {'PK R²':>8} {'PD R²':>8} {'Time':>8}")
    print("-" * 60)
    for m in sorted(all_metrics, key=lambda x: -x["r2"]):
        print(f"  {m['model_name']:<18} {m['r2']:>8.4f} {m['rmse']:>8.4f} "
              f"{m['pk_r2']:>8.4f} {m['pd_r2']:>8.4f} {m['training_time']:>7.1f}s")

    # Comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    names  = [m["model_name"] for m in all_metrics]
    r2s    = [m["r2"]   for m in all_metrics]
    rmses  = [m["rmse"] for m in all_metrics]
    pk_r2s = [m["pk_r2"] for m in all_metrics]
    pd_r2s = [m["pd_r2"] for m in all_metrics]

    axes[0, 0].bar(names, r2s);     axes[0, 0].set_title("R² Score");     axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 1].bar(names, rmses);   axes[0, 1].set_title("RMSE");         axes[0, 1].tick_params(axis="x", rotation=20)
    x = np.arange(len(names))
    axes[1, 0].bar(x - 0.2, pk_r2s, 0.35, label="PK R²")
    axes[1, 0].bar(x + 0.2, pd_r2s, 0.35, label="PD R²")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(names, rotation=20)
    axes[1, 0].set_title("PK vs PD R²"); axes[1, 0].legend()

    best = max(all_metrics, key=lambda m: m["r2"])
    axes[1, 1].plot(best["train_losses"], label="train")
    axes[1, 1].plot(best["val_losses"],   label="val")
    axes[1, 1].set_title(f"Loss curves — {best['model_name']}")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "model_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison plot → {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
