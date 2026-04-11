"""
Data source configuration management.

This module provides configuration management for individual
data sources and their ingestion parameters.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Supported data formats."""
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "excel"
    DATABASE = "database"
    API = "api"


@dataclass
class CSVOptions:
    """Options for CSV data sources."""
    encoding: str = "utf-8"
    delimiter: str = ","
    header_row: int = 0
    skip_rows: int = 0
    quote_char: str = '"'
    escape_char: Optional[str] = None


@dataclass
class JSONOptions:
    """Options for JSON data sources."""
    encoding: str = "utf-8"
    root_element: Optional[str] = None
    array_field: Optional[str] = None
    flatten_nested: bool = True


@dataclass
class DatabaseOptions:
    """Options for database data sources."""
    connection_string: str
    table_name: str
    query: Optional[str] = None
    batch_size: int = 1000


@dataclass
class APIOptions:
    """Options for API data sources."""
    base_url: str
    endpoint: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    auth_type: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit: Optional[int] = None  # requests per minute


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""
    name: str
    data_type: str  # drugs, diseases, mappings, etc.
    format: DataFormat
    location: str
    priority: int = 1
    enabled: bool = True
    description: Optional[str] = None
    tags: List[str] = None
    options: Optional[Union[CSVOptions, JSONOptions, DatabaseOptions, APIOptions]] = None
    schema_file: Optional[str] = None
    validation_rules: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.tags is None:
            self.tags = []
        
        # Set default options based on format
        if self.options is None:
            if self.format == DataFormat.CSV:
                self.options = CSVOptions()
            elif self.format == DataFormat.JSON:
                self.options = JSONOptions()
            elif self.format == DataFormat.DATABASE:
                self.options = DatabaseOptions(connection_string="", table_name="")
            elif self.format == DataFormat.API:
                self.options = APIOptions(base_url="", endpoint="")
    
    def to_data_source(self):
        """
        Convert to DataSource object for ingestion.
        
        Returns:
            DataSource object
        """
        # Import here to avoid circular import
        from ..ingestion.base_ingestor import DataSource, DataSourceType
        
        source_type = DataSourceType(self.format.value)
        
        options_dict = None
        if self.options:
            options_dict = asdict(self.options)
        
        return DataSource(
            name=self.name,
            source_type=source_type,
            location=self.location,
            schema=None,  # Will be loaded separately
            credentials=None,  # Will be handled separately
            options=options_dict,
            priority=self.priority,
            enabled=self.enabled
        )


class DataSourceConfigManager:
    """
    Manager for data source configurations.
    
    Handles loading, saving, and managing configurations
    for multiple data sources.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize data source config manager.
        
        Args:
            config_file: Path to data sources configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self.logger = logging.getLogger(__name__)
        self.data_sources: Dict[str, DataSourceConfig] = {}
        
        # Load configurations if file provided
        if self.config_file and self.config_file.exists():
            self.load_data_sources()
    
    def add_data_source(self, config: DataSourceConfig) -> None:
        """
        Add a data source configuration.
        
        Args:
            config: Data source configuration
        """
        if config.name in self.data_sources:
            self.logger.warning(f"Data source '{config.name}' already exists, overwriting")
        
        self.data_sources[config.name] = config
        self.logger.info(f"Added data source: {config.name}")
    
    def remove_data_source(self, name: str) -> None:
        """
        Remove a data source configuration.
        
        Args:
            name: Name of data source to remove
        """
        if name in self.data_sources:
            del self.data_sources[name]
            self.logger.info(f"Removed data source: {name}")
        else:
            self.logger.warning(f"Data source '{name}' not found")
    
    def get_data_source(self, name: str) -> Optional[DataSourceConfig]:
        """
        Get a data source configuration by name.
        
        Args:
            name: Name of data source
            
        Returns:
            Data source configuration or None if not found
        """
        return self.data_sources.get(name)
    
    def get_data_sources_by_type(self, data_type: str) -> List[DataSourceConfig]:
        """
        Get all data sources for a specific data type.
        
        Args:
            data_type: Type of data (drugs, diseases, etc.)
            
        Returns:
            List of data source configurations
        """
        return [
            config for config in self.data_sources.values()
            if config.data_type == data_type and config.enabled
        ]
    
    def get_enabled_data_sources(self) -> List[DataSourceConfig]:
        """
        Get all enabled data sources.
        
        Returns:
            List of enabled data source configurations
        """
        return [config for config in self.data_sources.values() if config.enabled]
    
    def get_data_sources_by_priority(self, data_type: Optional[str] = None) -> List[DataSourceConfig]:
        """
        Get data sources sorted by priority.
        
        Args:
            data_type: Optional filter by data type
            
        Returns:
            List of data source configurations sorted by priority
        """
        sources = self.get_enabled_data_sources()
        
        if data_type:
            sources = [s for s in sources if s.data_type == data_type]
        
        return sorted(sources, key=lambda x: x.priority, reverse=True)
    
    def load_data_sources(self, config_file: Optional[str] = None) -> None:
        """
        Load data source configurations from file.
        
        Args:
            config_file: Path to configuration file (overrides instance file)
        """
        file_path = Path(config_file) if config_file else self.config_file
        
        if not file_path or not file_path.exists():
            self.logger.warning(f"Data sources config file not found: {file_path}")
            return
        
        try:
            import json
            import yaml
            
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            # Load data sources
            sources_data = config_data.get('data_sources', [])
            
            for source_data in sources_data:
                config = self._dict_to_data_source_config(source_data)
                self.data_sources[config.name] = config
            
            self.logger.info(f"Loaded {len(self.data_sources)} data source configurations")
            
        except Exception as e:
            self.logger.error(f"Failed to load data sources: {str(e)}")
            raise
    
    def save_data_sources(self, config_file: Optional[str] = None) -> None:
        """
        Save data source configurations to file.
        
        Args:
            config_file: Path to save configuration (overrides instance file)
        """
        file_path = Path(config_file) if config_file else self.config_file
        
        if not file_path:
            raise ValueError("No configuration file specified")
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import json
            import yaml
            
            config_data = {
                'data_sources': [
                    self._data_source_config_to_dict(config)
                    for config in self.data_sources.values()
                ]
            }
            
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif file_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            self.logger.info(f"Data sources configuration saved to: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data sources: {str(e)}")
            raise
    
    def _dict_to_data_source_config(self, data: Dict[str, Any]) -> DataSourceConfig:
        """Convert dictionary to DataSourceConfig."""
        # Convert format string to enum
        format_str = data.get('format', 'csv')
        data_format = DataFormat(format_str)
        
        # Convert options based on format
        options_data = data.get('options')
        options = None
        
        if options_data:
            if data_format == DataFormat.CSV:
                options = CSVOptions(**options_data)
            elif data_format == DataFormat.JSON:
                options = JSONOptions(**options_data)
            elif data_format == DataFormat.DATABASE:
                options = DatabaseOptions(**options_data)
            elif data_format == DataFormat.API:
                options = APIOptions(**options_data)
        
        return DataSourceConfig(
            name=data['name'],
            data_type=data['data_type'],
            format=data_format,
            location=data['location'],
            priority=data.get('priority', 1),
            enabled=data.get('enabled', True),
            description=data.get('description'),
            tags=data.get('tags', []),
            options=options,
            schema_file=data.get('schema_file'),
            validation_rules=data.get('validation_rules')
        )
    
    def _data_source_config_to_dict(self, config: DataSourceConfig) -> Dict[str, Any]:
        """Convert DataSourceConfig to dictionary."""
        data = {
            'name': config.name,
            'data_type': config.data_type,
            'format': config.format.value,
            'location': config.location,
            'priority': config.priority,
            'enabled': config.enabled,
            'description': config.description,
            'tags': config.tags,
            'schema_file': config.schema_file,
            'validation_rules': config.validation_rules
        }
        
        if config.options:
            data['options'] = asdict(config.options)
        
        return data
    
    def validate_data_source(self, config: DataSourceConfig) -> Dict[str, Any]:
        """
        Validate a data source configuration.
        
        Args:
            config: Data source configuration to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        if not config.name:
            validation_result['errors'].append("Data source name is required")
            validation_result['is_valid'] = False
        
        if not config.data_type:
            validation_result['errors'].append("Data type is required")
            validation_result['is_valid'] = False
        
        if not config.location:
            validation_result['errors'].append("Location is required")
            validation_result['is_valid'] = False
        
        # Validate location based on format
        if config.format == DataFormat.CSV:
            location_path = Path(config.location)
            if not location_path.exists():
                validation_result['errors'].append(f"CSV file not found: {config.location}")
                validation_result['is_valid'] = False
            elif not location_path.is_file():
                validation_result['errors'].append(f"Path is not a file: {config.location}")
                validation_result['is_valid'] = False
        
        elif config.format == DataFormat.DATABASE:
            if not config.options or not config.options.connection_string:
                validation_result['errors'].append("Database connection string is required")
                validation_result['is_valid'] = False
        
        elif config.format == DataFormat.API:
            if not config.options or not config.options.base_url:
                validation_result['errors'].append("API base URL is required")
                validation_result['is_valid'] = False
        
        # Validate priority
        if config.priority < 1:
            validation_result['warnings'].append("Priority should be >= 1")
        
        return validation_result
    
    def get_data_source_summary(self) -> Dict[str, Any]:
        """
        Get summary of all data sources.
        
        Returns:
            Dictionary with data source summary
        """
        summary = {
            'total_sources': len(self.data_sources),
            'enabled_sources': len(self.get_enabled_data_sources()),
            'sources_by_type': {},
            'sources_by_format': {},
            'priority_distribution': {}
        }
        
        # Count by data type
        for config in self.data_sources.values():
            data_type = config.data_type
            summary['sources_by_type'][data_type] = summary['sources_by_type'].get(data_type, 0) + 1
        
        # Count by format
        for config in self.data_sources.values():
            format_str = config.format.value
            summary['sources_by_format'][format_str] = summary['sources_by_format'].get(format_str, 0) + 1
        
        # Priority distribution
        priorities = [config.priority for config in self.data_sources.values()]
        if priorities:
            summary['priority_distribution'] = {
                'min': min(priorities),
                'max': max(priorities),
                'average': sum(priorities) / len(priorities)
            }
        
        return summary
    
    def create_sample_config(self, file_path: str) -> None:
        """
        Create a sample data sources configuration file.
        
        Args:
            file_path: Path where to create the sample config
        """
        sample_sources = [
            DataSourceConfig(
                name="sample_drugs_csv",
                data_type="drugs",
                format=DataFormat.CSV,
                location="data/drugs.csv",
                priority=1,
                description="Sample drug data from CSV file",
                tags=["drugs", "csv", "sample"],
                options=CSVOptions(encoding="utf-8", delimiter=",", header_row=0)
            ),
            DataSourceConfig(
                name="sample_diseases_json",
                data_type="diseases",
                format=DataFormat.JSON,
                location="data/diseases.json",
                priority=2,
                description="Sample disease data from JSON file",
                tags=["diseases", "json", "sample"],
                options=JSONOptions(encoding="utf-8", flatten_nested=True)
            )
        ]
        
        # Clear existing and add samples
        self.data_sources.clear()
        for source in sample_sources:
            self.data_sources[source.name] = source
        
        # Save to file
        self.config_file = Path(file_path)
        self.save_data_sources()
        
        self.logger.info(f"Sample data sources configuration created: {file_path}")
