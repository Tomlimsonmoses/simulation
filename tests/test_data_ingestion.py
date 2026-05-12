"""
Test script for the data ingestion system.

This script tests the complete data ingestion pipeline using the
project's own data/raw CSV files (no external downloads required).
"""

import sys
import os
import pytest
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Expected CSVs produced by the data generator / already present in data/raw
_RAW_DIR = project_root / "data" / "raw"
_REQUIRED_FILES = [
    _RAW_DIR / "drug_properties.csv",
    _RAW_DIR / "disease_data.csv",
    _RAW_DIR / "clinical_trials.csv",
]


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = ["pandas", "numpy", "pathlib"]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def test_dependencies_available():
    """All required Python packages must be importable."""
    missing = check_dependencies()
    assert not missing, f"Missing packages: {', '.join(missing)}"


def test_raw_data_files_exist():
    """Core training CSV files must exist under data/raw/."""
    missing = [str(p) for p in _REQUIRED_FILES if not p.exists()]
    if missing:
        pytest.skip(
            f"Raw data files not present (run data generator first): {missing}"
        )


def test_data_ingestion():
    """Test the data ingestion system with the project's own datasets."""
    # Skip gracefully if the raw CSV files are absent
    missing = [p for p in _REQUIRED_FILES if not p.exists()]
    if missing:
        pytest.skip(
            f"Skipping ingestion test — missing files: {[str(p) for p in missing]}"
        )

    try:
        from src.data.main_data_ingestion import DataIngestionOrchestrator
    except ImportError as exc:
        pytest.skip(f"DataIngestionOrchestrator not importable: {exc}")

    output_dir = project_root / "results" / "simulations"
    output_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = DataIngestionOrchestrator()
    orchestrator.exporter.output_dir = output_dir

    results = orchestrator.run_test_with_current_datasets()

    assert results is not None, "Orchestrator returned None"
    assert results.get("success"), (
        f"Ingestion failed. Errors: {results.get('errors', [])}"
    )
