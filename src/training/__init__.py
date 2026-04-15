"""
Training utilities for PK-PD MLP models.

CLI entry points
----------------
  python -m src.training.train          — train SimpleMLP / DeepMLP / ResidualMLP
  python -m src.training.train_staged   — train StagedPKPDModel (two-stage loss)
  python -m src.training.evaluate       — evaluate all checkpoints on test split

Shared helpers
--------------
  src.training.features.create_features()  — 32-dim feature vector
  src.training.features.build_dataset()    — load CSV → (X, y) arrays
"""

from .features import create_features, build_dataset, FEATURE_DIM, ROUTE_TO_ORGAN

__all__ = ["create_features", "build_dataset", "FEATURE_DIM", "ROUTE_TO_ORGAN"]

