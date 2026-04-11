"""
Base classes for data ingestion framework.

This module defines abstract base classes that provide the foundation
for implementing data source-specific ingestors.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Enumeration of supported data source types."""
    CSV = "csv"
    API = "api"
    DATABASE = "database"
    JSON = "json"
    XML = "xml"


@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str
    source_type: DataSourceType
    location: str
    schema: Optional[Dict[str, Any]] = None
    credentials: Optional[Dict[str, str]] = None
    options: Optional[Dict[str, Any]] = None
    priority: int = 1
    enabled: bool = True


@dataclass
class IngestionResult:
    """Result of data ingestion operation."""
    source_name: str
    records_processed: int
    records_valid: int
    records_invalid: int
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class BaseIngestor(ABC):
    """
    Abstract base class for all data ingestors.
    
    This class defines the interface that all ingestors must implement
    to ensure consistent behavior across different data sources.
    """
    
    def __init__(self, source: DataSource):
        """
        Initialize the ingestor with a data source configuration.
        
        Args:
            source: DataSource configuration object
        """
        self.source = source
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._validate_source()
    
    def _validate_source(self) -> None:
        """Validate the data source configuration."""
        if not self.source.name:
            raise ValueError("Data source must have a name")
        
        if not self.source.location:
            raise ValueError("Data source must have a location")
        
        if not self.source.enabled:
            self.logger.warning(f"Data source '{self.source.name}' is disabled")
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema definition for this data source.
        
        Returns:
            Dictionary describing the data schema
        """
        pass
    
    @abstractmethod
    def ingest_data(self, **kwargs) -> IngestionResult:
        """
        Ingest data from the source.
        
        Args:
            **kwargs: Additional parameters for ingestion
            
        Returns:
            IngestionResult with details of the operation
        """
        pass
    
    @abstractmethod
    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate a single record against the expected schema.
        
        Args:
            record: Dictionary representing a single data record
            
        Returns:
            True if record is valid, False otherwise
        """
        pass
    
    def get_sample_data(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get a sample of data from the source for inspection.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of sample records
        """
        self.logger.info(f"Getting sample data (limit={limit}) from {self.source.name}")
        
        if not self.connect():
            raise ConnectionError(f"Failed to connect to data source: {self.source.name}")
        
        try:
            result = self.ingest_data(limit=limit)
            return result.metadata.get('sample_data', [])
        finally:
            self.disconnect()
    
    def test_connection(self) -> bool:
        """
        Test the connection to the data source.
        
        Returns:
            True if connection test successful, False otherwise
        """
        try:
            return self.connect()
        except Exception as e:
            self.logger.error(f"Connection test failed for {self.source.name}: {str(e)}")
            return False
        finally:
            try:
                self.disconnect()
            except:
                pass


class BaseProcessor(ABC):
    """
    Abstract base class for data processors.
    
    Processors are responsible for transforming and standardizing
    data after ingestion but before consolidation.
    """
    
    def __init__(self, name: str):
        """
        Initialize the processor.
        
        Args:
            name: Name of the processor
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the input data.
        
        Args:
            data: List of records to process
            
        Returns:
            List of processed records
        """
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the schema of processed data.
        
        Returns:
            Dictionary describing the output schema
        """
        pass
    
    def validate_processed_data(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        Validate processed data against expected schema.
        
        Args:
            data: Processed data to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        schema = self.get_output_schema()
        
        for i, record in enumerate(data):
            for field, field_spec in schema.items():
                if field_spec.get('required', True) and field not in record:
                    errors.append(f"Record {i}: Missing required field '{field}'")
        
        return errors
