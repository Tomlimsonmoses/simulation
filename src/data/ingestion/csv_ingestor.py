"""
CSV data ingestor implementation.

This module provides a specialized ingestor for handling CSV files
with automatic schema detection and validation.
"""

import csv
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import chardet
import logging

from .base_ingestor import BaseIngestor, IngestionResult, DataSource

logger = logging.getLogger(__name__)


class CSVIngestor(BaseIngestor):
    """
    Ingestor for CSV data sources.
    
    Handles CSV files with automatic encoding detection,
    schema inference, and data validation.
    """
    
    def __init__(self, source: DataSource):
        """
        Initialize CSV ingestor.
        
        Args:
            source: DataSource configuration for CSV file
        """
        super().__init__(source)
        self.file_path = Path(source.location)
        self.encoding = None
        self.delimiter = None
        self.header_row = 0
        
        # Set default options for CSV
        if source.options:
            self.encoding = source.options.get('encoding')
            self.delimiter = source.options.get('delimiter', ',')
            self.header_row = source.options.get('header_row', 0)
    
    def connect(self) -> bool:
        """
        Connect to CSV file by validating file existence and detecting encoding.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self.file_path.exists():
                self.logger.error(f"CSV file not found: {self.file_path}")
                return False
            
            if not self.file_path.is_file():
                self.logger.error(f"Path is not a file: {self.file_path}")
                return False
            
            # Detect encoding if not specified
            if not self.encoding:
                self.encoding = self._detect_encoding()
                self.logger.info(f"Detected encoding: {self.encoding}")
            
            # Validate CSV structure
            if not self._validate_csv_structure():
                return False
            
            self.logger.info(f"Successfully connected to CSV file: {self.file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to CSV file: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """CSV files don't require explicit disconnection."""
        pass
    
    def _detect_encoding(self) -> str:
        """
        Detect file encoding using chardet.
        
        Returns:
            Detected encoding string
        """
        try:
            with open(self.file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {str(e)}, using utf-8")
            return 'utf-8'
    
    def _validate_csv_structure(self) -> bool:
        """
        Validate that the file has a valid CSV structure.
        
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                # Read a few lines to validate structure
                csv_reader = csv.reader(f, delimiter=self.delimiter)
                
                # Skip to header row
                for _ in range(self.header_row):
                    next(csv_reader, None)
                
                # Check if we can read the header
                header = next(csv_reader, None)
                if not header:
                    self.logger.error("No header row found in CSV file")
                    return False
                
                # Try to read a few data rows
                for _ in range(3):
                    try:
                        next(csv_reader, None)
                    except StopIteration:
                        break  # End of file reached
                
            return True
            
        except Exception as e:
            self.logger.error(f"CSV structure validation failed: {str(e)}")
            return False
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Infer schema from CSV file.
        
        Returns:
            Dictionary describing the inferred schema
        """
        if not self.connect():
            raise ConnectionError("Failed to connect to CSV file")
        
        try:
            # Use pandas for schema inference
            df = pd.read_csv(
                self.file_path,
                encoding=self.encoding,
                delimiter=self.delimiter,
                header=self.header_row,
                nrows=1000  # Sample first 1000 rows for schema inference
            )
            
            schema = {
                'columns': [],
                'total_columns': len(df.columns),
                'sample_rows': len(df),
                'file_info': {
                    'path': str(self.file_path),
                    'encoding': self.encoding,
                    'delimiter': self.delimiter,
                    'header_row': self.header_row
                }
            }
            
            for col in df.columns:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'nullable': df[col].isnull().any(),
                    'unique_count': df[col].nunique(),
                    'sample_values': df[col].dropna().head(3).tolist()
                }
                schema['columns'].append(col_info)
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Schema inference failed: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def ingest_data(self, **kwargs) -> IngestionResult:
        """
        Ingest data from CSV file.
        
        Args:
            **kwargs: Additional parameters
                - limit: Maximum number of records to ingest
                - columns: List of columns to include (None for all)
                - skip_rows: Number of rows to skip after header
                
        Returns:
            IngestionResult with details of the operation
        """
        limit = kwargs.get('limit', None)
        columns = kwargs.get('columns', None)
        skip_rows = kwargs.get('skip_rows', 0)
        
        if not self.connect():
            raise ConnectionError("Failed to connect to CSV file")
        
        errors = []
        warnings = []
        records_processed = 0
        records_valid = 0
        records_invalid = 0
        ingested_data = []
        
        try:
            # Read CSV data
            read_kwargs = {
                'encoding': self.encoding,
                'delimiter': self.delimiter,
                'header': self.header_row,
                'skiprows': skip_rows
            }
            
            if limit:
                read_kwargs['nrows'] = limit
            
            if columns:
                read_kwargs['usecols'] = columns
            
            df = pd.read_csv(self.file_path, **read_kwargs)
            records_processed = len(df)
            
            # Convert to list of dictionaries
            for index, row in df.iterrows():
                record = row.to_dict()
                
                # Validate record
                if self.validate_record(record):
                    records_valid += 1
                    ingested_data.append(record)
                else:
                    records_invalid += 1
                    errors.append(f"Invalid record at row {index + self.header_row + skip_rows + 1}")
            
            # Store sample data for inspection
            sample_data = ingested_data[:5] if ingested_data else []
            
            result = IngestionResult(
                source_name=self.source.name,
                records_processed=records_processed,
                records_valid=records_valid,
                records_invalid=records_invalid,
                errors=errors,
                warnings=warnings,
                metadata={
                    'file_path': str(self.file_path),
                    'encoding': self.encoding,
                    'delimiter': self.delimiter,
                    'columns': list(df.columns),
                    'sample_data': sample_data,
                    'ingested_data': ingested_data
                }
            )
            
            self.logger.info(f"Ingestion completed: {records_valid}/{records_processed} valid records")
            return result
            
        except Exception as e:
            error_msg = f"Data ingestion failed: {str(e)}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return IngestionResult(
                source_name=self.source.name,
                records_processed=records_processed,
                records_valid=records_valid,
                records_invalid=records_invalid,
                errors=errors,
                warnings=warnings,
                metadata={}
            )
        finally:
            self.disconnect()
    
    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate a single record.
        
        Args:
            record: Dictionary representing a single data record
            
        Returns:
            True if record is valid, False otherwise
        """
        # Basic validation - check for empty records
        if not record:
            return False
        
        # Check if all values are NaN/None
        if all(pd.isna(value) for value in record.values()):
            return False
        
        # Additional validation can be added based on schema
        return True
    
    def get_column_info(self) -> Dict[str, Any]:
        """
        Get detailed information about CSV columns.
        
        Returns:
            Dictionary with column statistics and information
        """
        schema = self.get_schema()
        
        column_info = {}
        for col in schema['columns']:
            column_info[col['name']] = {
                'data_type': col['dtype'],
                'nullable': col['nullable'],
                'unique_values': col['unique_count'],
                'sample_values': col['sample_values']
            }
        
        return column_info
