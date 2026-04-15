"""
Organ Graph Builder
===================

Translates mechanistic engine outputs (organ_conc, drug properties) into
torch_geometric Data objects suitable for PKPDGNN inference.

Organ ordering (12 nodes, indices 0-11) is fixed to match PKPDDataProcessor:
  heart, liver, kidney, brain, lung, spleen, muscle, fat, skin, gut, bone, stomach

Drug node is appended at index 12.

Edge topology is derived from physiological vascular connections (CSV data +
VirtualHuman connection data for completeness).

GNN inference helper
--------------------
``gnn_infer_organ_pd`` runs a safe forward pass on a loaded PKPDGNN, skipping
edge_attr since the GATConv layers were not initialised with edge_dim.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Organ ordering (must match PKPDDataProcessor.organ_types) ───────────────
_ORGAN_ORDER = [
    "heart", "liver", "kidney", "brain", "lung",
    "spleen", "muscle", "fat", "skin", "gut", "bone", "stomach",
]
_ORGAN_IDX: Dict[str, int] = {o: i for i, o in enumerate(_ORGAN_ORDER)}

# ── Static baseline physiology per organ ────────────────────────────────────
#   (volume_L, blood_flow_mL_min, metabolic_rate, ph, water, lipid, protein,
#    surface_area_m2, oxygen_consumption_mL_min_100g)
_ORGAN_PHYSIOLOGY: Dict[str, Tuple] = {
    "heart":   (0.30,  240,  1.5, 7.4, 0.76, 0.07, 0.17, 0.10, 20.0),
    "liver":   (1.50, 1350,  1.8, 7.4, 0.71, 0.05, 0.18, 1.50, 40.0),
    "kidney":  (0.30, 1100,  1.3, 7.4, 0.73, 0.05, 0.15, 0.30, 15.0),
    "brain":   (1.40,  750,  1.2, 7.3, 0.77, 0.11, 0.08, 0.20, 50.0),
    "lung":    (2.50, 4200,  1.1, 7.4, 0.77, 0.03, 0.16, 2.00, 10.0),
    "spleen":  (0.20,  150,  1.0, 7.4, 0.77, 0.02, 0.14, 0.10,  8.0),
    "muscle":  (28.0,  900,  0.7, 7.4, 0.76, 0.03, 0.16, 12.0, 30.0),
    "fat":     (15.0,  300,  0.3, 7.4, 0.17, 0.79, 0.02, 15.0,  5.0),
    "skin":    (3.0,   300,  0.8, 7.4, 0.71, 0.06, 0.12,  2.0,  8.0),
    "gut":     (1.0,   900,  1.1, 6.5, 0.74, 0.04, 0.35,  2.0, 12.0),
    "bone":    (5.0,   200,  0.4, 7.4, 0.25, 0.05, 0.70,  3.0,  6.0),
    "stomach": (0.15,  100,  0.9, 2.0, 0.74, 0.04, 0.15,  0.4,  9.0),
}

# ── Vascular edge list ───────────────────────────────────────────────────────
#   (source_organ, target_organ, flow_rate_mL_min, permeability_0_1)
_ORGAN_EDGES: list = [
    # Heart → peripheral organs
    ("heart",  "brain",   750,  0.80),
    ("heart",  "liver",  1350,  0.90),
    ("heart",  "kidney", 1100,  0.85),
    ("heart",  "muscle",  900,  0.70),
    ("heart",  "fat",     300,  0.30),
    ("heart",  "gut",     900,  0.80),
    ("heart",  "skin",    300,  0.60),
    ("heart",  "spleen",  150,  0.80),
    ("heart",  "lung",   4200,  0.90),
    # Portal circulation
    ("gut",    "liver",   900,  0.80),
    # Pulmonary return
    ("lung",   "heart",  5000,  0.95),
    # Venous return to lung
    ("brain",  "lung",    750,  0.80),
    ("liver",  "lung",   1350,  0.90),
    ("kidney", "lung",   1100,  0.85),
    ("muscle", "lung",    900,  0.70),
    ("fat",    "lung",    300,  0.30),
    ("skin",   "lung",    300,  0.60),
    ("spleen", "lung",    150,  0.80),
]

# Max normalisation constants (for stable network inputs)
_FLOW_MAX  = 5000.0  # mL/min — lungs→heart
_VOL_MAX   = 30.0    # L — muscle
_BF_MAX    = 4200.0  # mL/min — lung


class OrganGraphBuilder:
    """
    Builds PyTorch Geometric Data objects for PKPDGNN inference.

    Designed to be instantiated once and reused (no mutable state).
    Does NOT require VirtualHuman — all physiology is hardcoded from ICRP data.

    Usage::

        builder = OrganGraphBuilder()
        data = builder.build_graph(
            organ_conc_at_t={"plasma": 5.2, "brain": 12.1, ...},
            drug_mw=267.0, drug_logp=1.8, drug_ba=0.50,
            drug_pb=0.80, drug_cl=14.7, drug_vd=280.0,
            drug_hl=3.5, drug_dose=50.0,
            disease_severity=0.30,
            patient_weight=70.0,
        )
    """

    def __init__(self) -> None:
        # Pre-build static edge tensors (only depends on _ORGAN_EDGES)
        self._edge_index, self._edge_attr_static = self._build_static_edges()

    # ── Public API ───────────────────────────────────────────────────────────

    def build_graph(
        self,
        organ_conc_at_t: Dict[str, float],
        drug_mw: float = 300.0,
        drug_logp: float = 1.0,
        drug_ba: float = 0.7,
        drug_pb: float = 0.5,
        drug_cl: float = 5.0,
        drug_vd: float = 50.0,
        drug_hl: float = 10.0,
        drug_dose: float = 100.0,
        disease_severity: float = 0.0,
        patient_weight: float = 70.0,
    ):
        """
        Return a torch_geometric.data.Data object with:

        * x            : [13, 16]  — 12 organ features + 1 drug feature (all organ_features_dim)
        * drug_features: [1, 8]
        * edge_index   : [2, E]
        * edge_attr    : [E, 3]
        * disease_feat : [1, 4]   — (severity, weight_factor, 0, 0)
        * batch        : [13]     — all zeros for single-sample inference

        The function is safe to call without torch if _TORCH_AVAILABLE is False
        (returns a plain dict instead).
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            logger.debug("torch_geometric not available; returning raw dict")
            return self._build_dict_fallback(
                organ_conc_at_t, drug_mw, drug_logp, drug_ba, drug_pb,
                drug_cl, drug_vd, drug_hl, drug_dose, disease_severity, patient_weight
            )

        weight_factor = patient_weight / 70.0

        # ── Organ node features (12 × 16) ────────────────────────────────
        organ_features = self._build_organ_features(
            organ_conc_at_t, weight_factor, disease_severity
        )

        # ── Drug node features (1 × 8) ───────────────────────────────────
        drug_cl_rate = drug_cl / max(drug_vd, 1e-6)  # ke = CL/Vd
        drug_vec = torch.tensor([[
            drug_mw  / 500.0,   # MW normalised to ~500 Da range
            drug_logp / 5.0,    # logP normalised
            0.16,               # PSA placeholder (normalised)
            drug_ba,
            drug_ba,            # duplicate for solubility proxy
            drug_pb,
            drug_hl / 24.0,     # half-life in days
            drug_cl_rate,       # elimination rate constant (1/h)
        ]], dtype=torch.float32)

        # ── Disease features (1 × 4) ─────────────────────────────────────
        disease_feat = torch.tensor([[
            disease_severity,
            weight_factor,
            0.0,   # spare
            0.0,   # spare
        ]], dtype=torch.float32)

        # ── Combine for node feature matrix (organs + drug dummy row) ────
        # drug "organ-slot" row uses drug features padded to 16-dim
        drug_padded = torch.zeros(1, 16, dtype=torch.float32)
        drug_padded[0, :8] = drug_vec[0]
        x = torch.cat([organ_features, drug_padded], dim=0)   # [13, 16]

        # ── Batch vector ─────────────────────────────────────────────────
        num_nodes = x.shape[0]
        batch_vec = torch.zeros(num_nodes, dtype=torch.long)

        data = Data(
            x=x,
            edge_index=self._edge_index,
            edge_attr=self._edge_attr_static,
            drug_features=drug_vec,
            disease_features=disease_feat,
            batch=batch_vec,
            organ_features=organ_features,
        )
        return data

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_organ_features(
        self,
        organ_conc_at_t: Dict[str, float],
        weight_factor: float,
        disease_severity: float,
    ):
        """Build 12×16 organ feature tensor, hydrated with current concentrations."""
        import torch

        rows = []
        max_conc = max((v for v in organ_conc_at_t.values() if np.isfinite(v)), default=1.0)
        max_conc = max(max_conc, 0.01)

        for organ in _ORGAN_ORDER:
            # Map engine organ names: gut → gut (same), engine uses 'kidney' not 'kidneys'
            engine_key = organ  # direct match for most
            if organ == "stomach":
                engine_key = "gut"   # fallback to gut for stomach
            elif organ == "bone":
                engine_key = "spleen"  # rough proxy

            physio = _ORGAN_PHYSIOLOGY[organ]
            vol, bf, mr, ph, fw, fl, fp, sa, o2 = physio

            conc = organ_conc_at_t.get(engine_key, organ_conc_at_t.get("blood", 0.0))
            conc_norm = float(np.clip(conc / max_conc, 0.0, 1.0))

            # 16-dim feature vector
            feat = [
                vol  * weight_factor / _VOL_MAX,   # 0: volume (normalised)
                bf   * weight_factor / _BF_MAX,    # 1: blood_flow (normalised)
                mr   / 2.0,                         # 2: metabolic_rate (max ~2)
                1.0,                                # 3: receptor_density (default 1.0)
                fp,                                 # 4: permeability ≈ protein content
                mr   / 2.0,                         # 5: clearance (proxy)
                sa   * weight_factor / 15.0,        # 6: surface_area (normalised)
                o2   * weight_factor / 50.0,        # 7: oxygen_consumption (normalised)
                ph   / 7.4,                         # 8: pH (normalised)
                37.0 / 37.0,                        # 9: temperature (normalised, always 1)
                weight_factor,                      # 10: weight factor
                mr   / 2.0,                         # 11: enzyme_activity proxy
                fp,                                 # 12: transporter_density proxy
                1.0,                                # 13: binding_capacity (default)
                1.0 - 0.3 * disease_severity,       # 14: sensitivity (disease modulated)
                1.0 - 0.2 * disease_severity,       # 15: health_status (disease modulated)
            ]
            # Inject concentration as an overlay on receptor_density and binding_capacity
            feat[3]  = float(np.clip(0.5 + conc_norm, 0.0, 2.0))   # receptor occ proxy
            feat[13] = conc_norm                                       # binding_capacity = conc
            rows.append(feat)

        return torch.tensor(rows, dtype=torch.float32)

    def _build_static_edges(self):
        """
        Build static edge_index [2, E] and edge_attr [E, 3] tensors from _ORGAN_EDGES.
        Only organ→organ edges (indices 0-11) are included.
        Drug–organ edges are added at inference time when drug index is known (=12).
        """
        try:
            import torch
        except ImportError:
            return None, None

        src_list, dst_list = [], []
        attr_list = []

        for src_org, dst_org, flow, perm in _ORGAN_EDGES:
            si = _ORGAN_IDX.get(src_org)
            di = _ORGAN_IDX.get(dst_org)
            if si is None or di is None:
                continue  # skip if organ not in our 12-organ list
            src_list.append(si)
            dst_list.append(di)
            attr_list.append([
                flow / _FLOW_MAX,  # normalised flow rate
                perm,              # permeability  0-1
                1.0,               # connection type: 1.0 = vascular
            ])

        # Drug node (idx 12) ↔ each organ — bidirectional
        drug_idx = len(_ORGAN_ORDER)  # 12
        for i in range(len(_ORGAN_ORDER)):
            # Drug → organ (delivery)
            src_list.append(drug_idx)
            dst_list.append(i)
            attr_list.append([0.5, 1.0, 2.0])   # type 2 = drug admin

            # Organ → drug (elimination / distribution)
            src_list.append(i)
            dst_list.append(drug_idx)
            attr_list.append([0.5, 1.0, 3.0])   # type 3 = drug elim

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(attr_list, dtype=torch.float32)

        return edge_index, edge_attr

    def _build_dict_fallback(self, *args, **kwargs) -> dict:
        """Return raw dict when torch_geometric is unavailable."""
        return {
            "torch_available": False,
            "organ_order": _ORGAN_ORDER,
        }


# ── Safe GNN inference helper ─────────────────────────────────────────────────

def gnn_infer_organ_pd(
    model,
    data,
    device: str = "cpu",
) -> Optional[Dict[str, float]]:
    """
    Run PKPDGNN forward pass for a single sample, returning per-organ PD scores.

    Safely omits edge_attr to avoid GATConv edge_dim mismatch in the current
    PKPDGNN architecture (GATConv was initialised without edge_dim).

    Returns
    -------
    Dict[str, float]
        Organ name → normalised PD response (0-1).
        Returns None if inference fails.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None

    try:
        model.eval()
        with torch.no_grad():
            organ_features = data.organ_features.to(device)  # [12, 16]
            drug_features  = data.drug_features.to(device)   # [1,  8]
            edge_index     = data.edge_index.to(device)       # [2, E]
            disease_feat   = data.disease_features.to(device) # [1, 4]
            batch_vec      = data.batch.to(device)            # [13]

            # ── Project inputs ───────────────────────────────────────────
            organ_h = model.organ_proj(organ_features)        # [12, 64]
            drug_h  = model.drug_proj(drug_features)          # [1,  64]
            node_h  = torch.cat([organ_h, drug_h], dim=0)     # [13, 64]

            # ── Disease broadcast ────────────────────────────────────────
            disease_h = model.disease_proj(disease_feat)       # [1,  64]
            node_h = node_h + disease_h[batch_vec]             # [13, 64]

            # ── GAT layers (no edge_attr to avoid edge_dim error) ────────
            for i, gat_layer in enumerate(model.gat_layers):
                node_h = gat_layer(node_h, edge_index)  # edge_attr omitted
                if i < len(model.gat_layers) - 1:
                    node_h = F.elu(node_h)
                    node_h = F.dropout(
                        node_h, p=model.dropout, training=False
                    )

            # ── Per-organ PD from organ repr ─────────────────────────────
            num_organs = organ_features.shape[0]  # 12
            organ_repr = node_h[:num_organs]       # [12, 64*num_heads]

            pd_raw = model.pd_head(organ_repr).squeeze(-1)  # [12]
            pd_scores = torch.sigmoid(pd_raw).cpu().numpy()

        return {organ: float(pd_scores[i]) for i, organ in enumerate(_ORGAN_ORDER)}

    except Exception as exc:
        logger.debug("GNN inference failed: %s", exc)
        return None
