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
MAX_SAMPLES   = 5_000    # rows from clinical_trials.csv
EPOCHS        = 40
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
GRAD_ACCUM    = 8        # accumulate grads over N samples to simulate batch≈8
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
    Load clinical_trials.csv and return a list of sample dicts.

    Each dict has keys: plasma_conc, response_norm, drug_props, severity,
    weight_kg, time_h.
    """
    import csv
    ct_path = os.path.join(data_dir, "clinical_trials.csv")
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"clinical_trials.csv not found at {ct_path}")

    drug_props = _load_drug_props_extended(data_dir)

    # Pre-scan max concentration for normalisation
    max_conc, max_resp = 1.0, 1.0

    samples: list[dict] = []
    with open(ct_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_n:
                break
            try:
                conc = float(row["concentration"])
                resp = float(row["response_value"])
                dose = float(row.get("dose", 100.0))
                weight = float(row.get("weight", 70.0))
                severity = float(row.get("severity", 0.0) if row.get("severity", "").replace(".", "").isdigit() else 0.3)
                time_h   = float(row.get("time_point", 0.0))
                drug_name = row.get("drug_name", "")

                if not np.isfinite(conc) or not np.isfinite(resp):
                    continue

                dp = drug_props.get(drug_name, {
                    "mw": 300, "logp": 1.0, "ba": 0.5, "pb": 0.5,
                    "cl": 5.0, "vd": 50.0, "hl": 8.0, "dose": dose,
                })

                max_conc = max(max_conc, abs(conc))
                max_resp = max(max_resp, abs(resp))
                samples.append({
                    "conc":      conc,
                    "resp":      resp,
                    "dp":        dp,
                    "severity":  float(np.clip(severity, 0.0, 1.0)),
                    "weight_kg": weight,
                    "time_h":    time_h,
                    "dose":      dose,
                })
            except (ValueError, TypeError):
                continue

    # Normalise targets
    for s in samples:
        s["pk_target"] = float(np.clip(s["conc"] / max_conc, 0.0, 1.0))
        s["pd_target"] = float(np.clip(s["resp"] / max_resp, 0.0, 1.0))

    print(f"  Loaded {len(samples)} samples  (max_conc={max_conc:.1f}, max_resp={max_resp:.1f})")
    return samples


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

def train_gnn(samples: list[dict], epochs: int = EPOCHS) -> PKPDGNN:
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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LR / 10
    )
    criterion = nn.MSELoss()

    # Split 80/20
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(samples))
    split = int(0.8 * len(samples))
    train_idx, val_idx = idx[:split], idx[split:]

    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}\n")

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
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

            pk_t  = data.pk_target.to(device)
            pd_t  = data.pd_target.to(device)
            loss  = criterion(pk_pred, pk_t) + criterion(pd_pred, pd_t)
            (loss / GRAD_ACCUM).backward()

            train_loss += loss.item()
            n_steps    += 1

            if (step_i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Final gradient step if leftover
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()
        train_loss /= max(n_steps, 1)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss, n_val = 0.0, 0
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
                n_val += 1
        val_loss /= max(n_val, 1)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
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

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"best={best_val:.5f}  {elapsed:.0f}s"
            )

    print(f"\n  Training complete. Best val loss: {best_val:.5f}")
    print(f"  Checkpoint saved → {CKPT_PATH}")
    return model


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("GNN PK-PD Training")
    print("=" * 60)
    print(f"Loading up to {MAX_SAMPLES} samples from clinical_trials.csv …")
    samples = _build_samples(DATA_DIR, MAX_SAMPLES)
    if not samples:
        print("ERROR: no samples loaded — check DATA_DIR path")
        return
    train_gnn(samples, epochs=EPOCHS)


if __name__ == "__main__":
    main()
