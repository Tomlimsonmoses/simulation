"""
PK-PD MLP Evaluation Script
=============================
Evaluates all trained checkpoints on the held-out test split and writes
results to results/models/performance_report.txt and a scatter plot.

Usage (from project root with venv active):
    python -m src.training.evaluate
    # or
    python src/training/evaluate.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── path fix for direct execution ──────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.mlp_architectures import DeepMLP, ResidualMLP, SimpleMLP
from src.models.pkpd_staged import StagedPKPDModel
from src.training.features import build_dataset

# Checkpoints to evaluate: (filename_stem, model_factory)
_CHECKPOINTS = [
    ("simple_mlp_model",   lambda d: SimpleMLP(input_dim=d)),
    ("deep_mlp_model",     lambda d: DeepMLP(input_dim=d)),
    ("residual_mlp_model", lambda d: ResidualMLP(input_dim=d)),
    ("staged_pkpd_model",  lambda d: StagedPKPDModel(input_dim=d, pk_hidden=128, pd_hidden=64)),
]


def evaluate_checkpoint(model, ckpt_path, X_test, y_test, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval().to(device)
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()

    y_flat = y_test.flatten()
    p_flat = preds.flatten()
    return {
        "mse":    float(mean_squared_error(y_flat, p_flat)),
        "rmse":   float(np.sqrt(mean_squared_error(y_flat, p_flat))),
        "mae":    float(mean_absolute_error(y_flat, p_flat)),
        "r2":     float(r2_score(y_flat, p_flat)),
        "pk_r2":  float(r2_score(y_test[:, 0], preds[:, 0])),
        "pd_r2":  float(r2_score(y_test[:, 1], preds[:, 1])),
        "preds":  preds,
    }


def main():
    print("PK-PD Model Evaluation")
    print("=" * 60)

    ckpt_dir = os.path.join(_ROOT, "checkpoints")
    data_dir = os.path.join(_ROOT, "data", "raw")

    print("Building feature dataset…")
    X, y = build_dataset(data_dir, max_samples=1000)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Evaluating on {len(X_test)} test samples")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]

    results_dir = os.path.join(_ROOT, "results", "models")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    for name, factory in _CHECKPOINTS:
        ckpt_path = os.path.join(ckpt_dir, f"{name}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {name}.pth not found — run training first")
            continue
        model = factory(input_dim)
        res   = evaluate_checkpoint(model, ckpt_path, X_test, y_test, device)
        res["name"] = name
        all_results.append(res)
        print(f"  {name:<24}  R²={res['r2']:.4f}  RMSE={res['rmse']:.4f}  "
              f"PK_R²={res['pk_r2']:.4f}  PD_R²={res['pd_r2']:.4f}")

    if not all_results:
        print("\nNo checkpoints found. Run `python -m src.training.train` first.")
        return

    # Write performance report
    report_path = os.path.join(results_dir, "performance_report.txt")
    with open(report_path, "w") as f:
        f.write("PK-PD Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test samples : {len(X_test)}\n")
        f.write(f"Feature dims : {input_dim}\n\n")
        f.write(f"{'Model':<26} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'PK_R²':>8} {'PD_R²':>8}\n")
        f.write("-" * 60 + "\n")
        for r in sorted(all_results, key=lambda x: -x["r2"]):
            f.write(f"  {r['name']:<24} {r['r2']:>8.4f} {r['rmse']:>8.4f} "
                    f"{r['mae']:>8.4f} {r['pk_r2']:>8.4f} {r['pd_r2']:>8.4f}\n")
    print(f"\nReport → {report_path}")

    # Scatter plot (best model)
    best = max(all_results, key=lambda x: x["r2"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col, label in zip(axes, [0, 1], ["PK (concentration)", "PD (response)"]):
        ax.scatter(y_test[:, col], best["preds"][:, col], alpha=0.5, s=18)
        lim = [min(y_test[:, col].min(), best["preds"][:, col].min()),
               max(y_test[:, col].max(), best["preds"][:, col].max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title(f"{label}  (R²={r2_score(y_test[:, col], best['preds'][:, col]):.4f})")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Best model: {best['name']}", fontsize=12)
    plot_path = os.path.join(results_dir, "evaluation_scatter.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot → {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
