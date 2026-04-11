"""
Data Ingestion Module for PK-PD Modeling System

This module provides a comprehensive framework for ingesting, processing,
and consolidating data from multiple sources for drug-biological system
interaction analysis.
"""

from .base_ingestor import BaseIngestor, DataSource
from .csv_ingestor import CSVIngestor
from .consolidation_engine import ConsolidationEngine, ConsolidationConfig

__all__ = [
    'BaseIngestor',
    'DataSource',
    'CSVIngestor', 
    'ConsolidationEngine',
    'ConsolidationConfig'
]
