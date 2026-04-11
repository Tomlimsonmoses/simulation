"""
Staged PK/PD Model Architecture
=================================

Two-stage model that mirrors the biological PK→PD causal chain:

  Stage 1 (PK model):  drug + patient features  →  plasma concentration
  Stage 2 (PD model):  concentration + organ + disease features  →  response

This separation gives:
  • Interpretable intermediate output (predicted concentration)
  • Ability to train each stage with its own loss / dataset
  • Swappable stages (e.g. replace Stage 1 with a 1-compartment ODE layer later)

Feature layout expected by the full staged model (32-dim, same as mlp_architectures):
  [0:10]  Drug features
  [10:20] Organ + disease features
  [20:29] Patient features
  [29:32] Pre-computed PK features (k_el, C_ss, AUC_24)

Stage 1 input: indices [0:10, 20:32]  → 22 features  (drug + patient + pre-computed PK)
Stage 2 input: [predicted_conc] + [10:20]  → 11 features  (concentration + organ/disease)
"""

import torch
import torch.nn as nn


class Stage1PKModel(nn.Module):
    """Predicts normalised plasma concentration from drug + patient features.

    Input:  22-dim  (10 drug + 9 patient + 3 pre-computed PK)
    Output: scalar  (normalised pk_concentration, range ≈ 0–1)
    """

    def __init__(self, input_dim: int = 22, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),   # concentration is normalised to [0, 1]
        )

    def forward(self, x_drug_patient: torch.Tensor) -> torch.Tensor:
        return self.net(x_drug_patient)


class Stage2PDModel(nn.Module):
    """Predicts normalised pharmacodynamic response from concentration + organ/disease features.

    Input:  11-dim  (1 predicted/true concentration + 10 organ/disease features)
    Output: scalar  (normalised pd_response, range ≈ 0–1)
    """

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),   # response is normalised to [0, 1]
        )

    def forward(self, conc: torch.Tensor, x_organ: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conc:    (batch, 1)  predicted concentration from Stage 1
            x_organ: (batch, 10) organ + disease features
        """
        x = torch.cat([conc, x_organ], dim=1)
        return self.net(x)


class StagedPKPDModel(nn.Module):
    """End-to-end wrapper that chains Stage1 → Stage2.

    Accepts the same 32-dim feature vector as SimpleMLP so it can be
    used as a drop-in replacement.

    Returns:
        (pk_pred, pd_pred) — both shape (batch, 1)

    The two outputs allow a combined loss:
        loss = alpha * MSE(pk_pred, pk_target) + (1-alpha) * MSE(pd_pred, pd_target)
    """

    # Feature slice indices (must match train.py create_features layout)
    _DRUG_IDX = slice(0, 10)        # 10 drug features
    _ORGAN_IDX = slice(10, 20)      # 10 organ + disease features
    _PATIENT_IDX = slice(20, 29)    #  9 patient features
    _PK_IDX = slice(29, 32)         #  3 pre-computed PK features

    def __init__(self, input_dim: int = 32,
                 pk_hidden: int = 128, pd_hidden: int = 64):
        super().__init__()
        # Stage 1 sees: drug (10) + patient (9) + PK-computed (3) = 22
        self.stage1 = Stage1PKModel(input_dim=22, hidden_dim=pk_hidden)
        # Stage 2 sees: concentration (1) + organ (10) = 11
        self.stage2 = Stage2PDModel(input_dim=11, hidden_dim=pd_hidden)
        self.input_dim = input_dim

    def _split(self, x: torch.Tensor):
        x_drug = x[:, self._DRUG_IDX]
        x_organ = x[:, self._ORGAN_IDX]
        x_patient = x[:, self._PATIENT_IDX]
        x_pk = x[:, self._PK_IDX]
        return x_drug, x_organ, x_patient, x_pk

    def forward(self, x: torch.Tensor):
        x_drug, x_organ, x_patient, x_pk = self._split(x)

        # Stage 1: drug + patient + pre-computed PK → concentration
        x_stage1 = torch.cat([x_drug, x_patient, x_pk], dim=1)  # (B, 22)
        pk_pred = self.stage1(x_stage1)                           # (B, 1)

        # Stage 2: predicted concentration + organ → response
        pd_pred = self.stage2(pk_pred, x_organ)                   # (B, 1)

        # Stack to (B, 2) for compatibility with existing loss code
        return torch.cat([pk_pred, pd_pred], dim=1)
