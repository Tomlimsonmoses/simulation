"""
GNN PK-PD Training Script
==========================
Trains PKPDGNN on clinical_trials.csv data, using OrganGraphBuilder to
construct graph samples and the existing drug/disease CSV for features.

Strategy
--------
* Each row in clinical_trials.csv is one sample (drug × patient × time-point).
* Organ features come from OrganGraphBuilder using the plasma concentration
  as a proxy for blood-borne concentration (distributed via blood-flow fractions).
* Targets: pk_target = normalised plasma concentration,
           pd_target = normalised response_value.
* GATConv layers are used WITHOUT edge_attr (model was init without edge_dim).
* Single-sample forward pass per step to avoid PyG custom-attribute batching.

Usage (from project root with venv active):
    python -m src.training.train_gnn            # trains & saves checkpoint
    python src/training/train_gnn.py            # same
"""

from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from src.models.gnn_pkpd import PKPDGNN
from src.simulation.organ_graph_builder import OrganGraphBuilder, _ORGAN_ORDER

# ── hyper-parameters ──────────────────────────────────────────────────────────
MAX_SAMPLES   = 20_000   # rows from clinical_trials.csv + synthetic
EPOCHS        = 80       # more epochs with early stopping
LR            = 5e-4     # slightly lower LR for stability
WEIGHT_DECAY  = 1e-4
GRAD_ACCUM    = 16       # larger effective batch
PATIENCE      = 15       # early stopping patience
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH     = os.path.join(_ROOT, "checkpoints", "gnn_pkpd.pth")
DATA_DIR      = os.path.join(_ROOT, "data", "raw")


# ────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ────────────────────────────────────────────────────────────────────────────

def _load_drug_props(data_dir: str) -> dict:
    """Return {drug_name: {mw, logp, ba, pb, cl, vd, hl, dose}}."""
    import csv
    path = os.path.join(data_dir, "server_drugs.csv")
    props: dict = {}
    if not os.path.exists(path):
        return props
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            props[row["drug_name"]] = {
                "mw":   float(row["molecular_weight"]),
                "logp": float(row["logp"]),
                "ba":   float(row["bioavailability"]),
                "pb":   float(row["protein_binding"]),
                "cl":   float(row["clearance"]),
                "vd":   float(row["volume_distribution"]),
                "hl":   float(row["half_life"]),
                "dose": float(row["dose_mg"]),
            }
    return props


def _load_drug_props_extended(data_dir: str) -> dict:
    """Also check drug_properties.csv for additional drugs (MW, logP, etc.)."""
    props = _load_drug_props(data_dir)
    path = os.path.join(data_dir, "drug_properties.csv")
    if not os.path.exists(path):
        return props
    import csv
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("drug_name", "")
            if name and name not in props:
                try:
                    props[name] = {
                        "mw":   float(row.get("molecular_weight", 300)),
                        "logp": float(row.get("logp", 1.0)),
                        "ba":   float(row.get("bioavailability", 0.5)),
                        "pb":   float(row.get("protein_binding", 0.5)),
                        "cl":   float(row.get("clearance_rate", 5.0)),
                        "vd":   float(row.get("volume_distribution", 50.0)),
                        "hl":   float(row.get("half_life", 8.0)),
                        "dose": float(row.get("dose_mg", 100.0)),
                    }
                except (ValueError, KeyError):
                    pass
    return props


def _build_organ_conc_proxy(plasma_conc: float, weight_kg: float = 70.0) -> dict:
    """
    Distribute plasma concentration to organs using blood-flow fractions.
    Organises organs as in _ORGAN_ORDER (12 organs).
    
    Each organ gets: organ_conc ≈ plasma × (bf_organ / bf_total)
    This is a rough proxy; the GNN learns to weight these appropriately.
    """
    blood_flows = {
        "heart": 240, "liver": 1350, "kidney": 1100, "brain": 750,
        "lung": 4200, "spleen": 150, "muscle": 900, "fat": 300,
        "skin": 300, "gut": 900, "bone": 200, "stomach": 100,
    }
    wf = weight_kg / 70.0
    total = sum(blood_flows.values())
    return {o: plasma_conc * blood_flows[o] * wf / total for o in _ORGAN_ORDER}


def _build_samples(data_dir: str, max_n: int) -> list[dict]:
    """
    Load clinical_trials.csv (+ synthetic_clinical_data.csv if available) and
    return a list of sample dicts with PK/PD targets normalised to [0, 1].
    """
    import csv

    drug_props = _load_drug_props_extended(data_dir)
    raw_rows: list[dict] = []

    def _read_csv(path: str, limit: int):
        if not os.path.exists(path):
            return
        with open(path, newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if len(raw_rows) >= limit:
                    break
                try:
                    conc     = float(row["concentration"])
                    resp     = float(row["response_value"])
                    dose     = float(row.get("dose", 100.0))
                    weight   = float(row.get("weight", 70.0))
                    sev_raw  = row.get("severity", "0.3")
                    severity = float(sev_raw) if sev_raw.replace(".", "").lstrip("-").isdigit() else 0.3
                    time_h   = float(row.get("time_point", 0.0))
                    drug_name = row.get("drug_name", "")
                    if not (np.isfinite(conc) and np.isfinite(resp)):
                        continue
                    dp = drug_props.get(drug_name, {
                        "mw": 300, "logp": 1.0, "ba": 0.5, "pb": 0.5,
                        "cl": 5.0, "vd": 50.0, "hl": 8.0, "dose": dose,
                    })
                    raw_rows.append({
                        "conc": conc, "resp": resp, "dp": dp,
                        "severity": float(np.clip(severity, 0.0, 1.0)),
                        "weight_kg": weight, "time_h": time_h, "dose": dose,
                    })
                except (ValueError, TypeError):
                    continue

    _read_csv(os.path.join(data_dir, "clinical_trials.csv"), max_n // 2)
    _read_csv(os.path.join(data_dir, "synthetic_clinical_data.csv"), max_n)

    if not raw_rows:
        raise FileNotFoundError("No training data found in data/raw/")

    max_conc = max((r["conc"] for r in raw_rows), default=1.0)
    max_resp = max((r["resp"] for r in raw_rows), default=1.0)
    max_conc = max(max_conc, 1.0)
    max_resp = max(max_resp, 1.0)

    for s in raw_rows:
        s["pk_target"] = float(np.clip(s["conc"] / max_conc, 0.0, 1.0))
        s["pd_target"] = float(np.clip(s["resp"] / max_resp, 0.0, 1.0))

    print(f"  Loaded {len(raw_rows)} samples  (max_conc={max_conc:.2f}, max_resp={max_resp:.2f})")
    return raw_rows


# ────────────────────────────────────────────────────────────────────────────
# Graph construction
# ────────────────────────────────────────────────────────────────────────────

_BUILDER = OrganGraphBuilder()


def _sample_to_data(s: dict):
    """Convert one training sample dict to a torch_geometric Data object."""
    dp = s["dp"]
    organ_conc = _build_organ_conc_proxy(s["conc"], s["weight_kg"])

    data = _BUILDER.build_graph(
        organ_conc_at_t  = organ_conc,
        drug_mw          = dp.get("mw",   300.0),
        drug_logp        = dp.get("logp",   1.0),
        drug_ba          = dp.get("ba",     0.5),
        drug_pb          = dp.get("pb",     0.5),
        drug_cl          = dp.get("cl",     5.0),
        drug_vd          = dp.get("vd",    50.0),
        drug_hl          = dp.get("hl",     8.0),
        drug_dose        = s.get("dose",  100.0),
        disease_severity = s["severity"],
        patient_weight   = s["weight_kg"],
    )
    # Attach scalar targets
    data.pk_target = torch.tensor([[s["pk_target"]]], dtype=torch.float32)
    data.pd_target = torch.tensor([[s["pd_target"]]], dtype=torch.float32)
    return data


# ────────────────────────────────────────────────────────────────────────────
# Forward pass (mirrors gnn_infer_organ_pd — no edge_attr)
# ────────────────────────────────────────────────────────────────────────────

def _forward(model: PKPDGNN, data, device: str):
    """
    Perform a single-sample forward pass that skips edge_attr (GATConv
    was not initialised with edge_dim — passing edge_attr would error).

    Returns (pk_pred [1,1], pd_pred [1,1]).
    """
    organ_features = data.organ_features.to(device)   # [12, 16]
    drug_features  = data.drug_features.to(device)    # [1,   8]
    edge_index     = data.edge_index.to(device)        # [2,   E]
    disease_feat   = data.disease_features.to(device)  # [1,   4]
    batch_vec      = data.batch.to(device)             # [13]

    # Input projections
    organ_h = model.organ_proj(organ_features)          # [12, 64]
    drug_h  = model.drug_proj(drug_features)            # [1,  64]
    node_h  = torch.cat([organ_h, drug_h], dim=0)       # [13, 64]

    # Disease broadcast
    disease_h = model.disease_proj(disease_feat)        # [1, 64]
    node_h = node_h + disease_h[batch_vec]              # [13, 64]

    # GAT layers — no edge_attr to avoid edge_dim mismatch
    for i, gat_layer in enumerate(model.gat_layers):
        node_h = gat_layer(node_h, edge_index)
        if i < len(model.gat_layers) - 1:
            node_h = F.elu(node_h)
            node_h = F.dropout(node_h, p=model.dropout, training=model.training)

    # Separate representations
    num_organs = organ_features.shape[0]    # 12
    organ_repr = node_h[:num_organs]        # [12, hidden*heads]

    # Global pool over 12 organ nodes → [1, hidden*heads]
    from torch_geometric.nn import global_mean_pool
    organ_pooled = global_mean_pool(organ_repr, batch_vec[:num_organs])

    pk_pred = model.pk_head(organ_pooled)   # [1, 1]
    pd_pred = model.pd_head(organ_pooled)   # [1, 1]
    return pk_pred, pd_pred


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────

def train_gnn(samples: list[dict], epochs: int = EPOCHS) -> tuple:
    """Returns (model, val_metrics_dict)."""
    device = torch.device(DEVICE)
    print(f"  Device: {device}")

    model = PKPDGNN(
        organ_features_dim=16,
        drug_features_dim=8,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        dropout=0.2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LR / 20
    )
    criterion = nn.HuberLoss(delta=0.2)   # robust to normalisation outliers

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(samples))
    split = int(0.80 * len(samples))
    train_idx, val_idx = idx[:split], idx[split:]
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}\n")

    best_val   = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_steps = 0.0, 0
        optimizer.zero_grad()

        for step_i, si in enumerate(rng.permutation(train_idx)):
            s = samples[si]
            try:
                data = _sample_to_data(s)
            except Exception:
                continue

            pk_pred, pd_pred = _forward(model, data, DEVICE)
            loss = criterion(pk_pred, data.pk_target.to(device)) + \
                   criterion(pd_pred, data.pd_target.to(device))
            (loss / GRAD_ACCUM).backward()
            train_loss += loss.item()
            n_steps    += 1

            if (step_i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        train_losses.append(train_loss / max(n_steps, 1))

        model.eval()
        val_loss, n_val = 0.0, 0
        pk_preds_v, pd_preds_v, pk_tgts_v, pd_tgts_v = [], [], [], []
        with torch.no_grad():
            for si in val_idx:
                s = samples[si]
                try:
                    data = _sample_to_data(s)
                except Exception:
                    continue
                pk_pred, pd_pred = _forward(model, data, DEVICE)
                val_loss += (
                    criterion(pk_pred, data.pk_target.to(device)).item()
                    + criterion(pd_pred, data.pd_target.to(device)).item()
                )
                pk_preds_v.append(pk_pred.item())
                pd_preds_v.append(pd_pred.item())
                pk_tgts_v.append(s["pk_target"])
                pd_tgts_v.append(s["pd_target"])
                n_val += 1
        val_loss /= max(n_val, 1)
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": best_state,
                "val_loss":    best_val,
                "arch": {
                    "organ_features_dim": 16,
                    "drug_features_dim":  8,
                    "hidden_dim":         64,
                    "num_layers":         3,
                    "num_heads":          4,
                    "dropout":            0.2,
                },
            }, CKPT_PATH)
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={train_losses[-1]:.5f}  val={val_loss:.5f}  "
                  f"best={best_val:.5f}  lr={lr_now:.2e}  {elapsed:.0f}s")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    pk_r2 = float(r2_score(pk_tgts_v, pk_preds_v)) if len(pk_preds_v) > 1 else 0.0
    pd_r2 = float(r2_score(pd_tgts_v, pd_preds_v)) if len(pd_preds_v) > 1 else 0.0
    all_y  = pk_tgts_v + pd_tgts_v
    all_p  = pk_preds_v + pd_preds_v
    overall_r2 = float(r2_score(all_y, all_p)) if len(all_y) > 1 else 0.0
    rmse = float(np.sqrt(mean_squared_error(all_y, all_p))) if len(all_y) > 1 else 0.0

    print(f"\n  Training complete. Best val loss: {best_val:.5f}")
    print(f"  Val R²={overall_r2:.4f}  PK R²={pk_r2:.4f}  PD R²={pd_r2:.4f}  RMSE={rmse:.4f}")
    print(f"  Checkpoint → {CKPT_PATH}")

    val_metrics = {
        "model":          "gnn_pkpd",
        "epochs_run":     len(train_losses),
        "training_sec":   round(time.time() - t0, 2),
        "best_val_loss":  round(best_val, 6),
        "val_r2":         round(overall_r2, 4),
        "val_pk_r2":      round(pk_r2, 4),
        "val_pd_r2":      round(pd_r2, 4),
        "val_rmse":       round(rmse, 6),
        "train_loss_curve": [round(v, 6) for v in train_losses],
        "val_loss_curve":   [round(v, 6) for v in val_losses],
    }
    return model, val_metrics


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    import datetime, json

    print("GNN PK-PD Training (v2)")
    print("=" * 60)

    # Generate synthetic data if missing
    syn_path = os.path.join(DATA_DIR, "synthetic_clinical_data.csv")
    if not os.path.exists(syn_path):
        print("Synthetic data not found — generating 50 000 samples…")
        sys.path.insert(0, _ROOT)
        from src.training.generate_synthetic_data import generate_samples
        df = generate_samples(n=50_000, seed=42)
        df.to_csv(syn_path, index=False)
        print(f"  Saved {len(df):,} rows → {syn_path}")

    print(f"Loading up to {MAX_SAMPLES} samples (real + synthetic)…")
    samples = _build_samples(DATA_DIR, MAX_SAMPLES)
    if not samples:
        print("ERROR: no samples loaded — check DATA_DIR path")
        return

    model, val_metrics = train_gnn(samples, epochs=EPOCHS)

    results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_json = {
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device":        DEVICE,
        "dataset": {
            "n_samples":  len(samples),
            "max_samples":MAX_SAMPLES,
        },
        "hyperparameters": {
            "epochs":     EPOCHS,
            "lr":         LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_accum": GRAD_ACCUM,
            "patience":   PATIENCE,
            "optimizer":  "AdamW",
            "scheduler":  "CosineAnnealingLR",
            "loss":       "HuberLoss(delta=0.2)",
        },
        "validation": val_metrics,
        "checkpoint": CKPT_PATH,
    }
    json_path = os.path.join(results_dir, "gnn_training_results.json")
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\nValidation JSON → {json_path}")


if __name__ == "__main__":
    main()
