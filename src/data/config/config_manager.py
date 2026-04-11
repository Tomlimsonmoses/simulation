"""
Configuration manager for data ingestion system.

This module provides centralized configuration management for
data sources, processing parameters, and system settings.
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    enable_validation: bool = True
    enable_quality_check: bool = True
    enable_deduplication: bool = True
    chunk_size: int = 1000
    max_workers: int = 4
    timeout_seconds: int = 300


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    output_dir: str = "output"
    format: str = "json"
    compress: bool = False
    include_metadata: bool = True
    include_summary: bool = True
    include_schema: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class SystemConfig:
    """Overall system configuration."""
    processing: ProcessingConfig
    output: OutputConfig
    logging: LoggingConfig
    
    def __post_init__(self):
        """Ensure nested configs are proper objects."""
        if isinstance(self.processing, dict):
            self.processing = ProcessingConfig(**self.processing)
        if isinstance(self.output, dict):
            self.output = OutputConfig(**self.output)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)


class ConfigManager:
    """
    Manager for system configuration.
    
    Handles loading, saving, and managing configuration
    from various sources (files, environment variables, etc.).
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = SystemConfig(
            processing=ProcessingConfig(),
            output=OutputConfig(),
            logging=LoggingConfig()
        )
        
        # Load configuration if file provided
        if self.config_file and self.config_file.exists():
            self.load_config()
    
    def load_config(self, config_file: Optional[str] = None) -> None:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file (overrides instance file)
        """
        file_path = Path(config_file) if config_file else self.config_file
        
        if not file_path or not file_path.exists():
            self.logger.warning(f"Configuration file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            # Update configuration
            self.config = SystemConfig(**config_data)
            self.logger.info(f"Configuration loaded from: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def save_config(self, config_file: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to save configuration (overrides instance file)
        """
        file_path = Path(config_file) if config_file else self.config_file
        
        if not file_path:
            raise ValueError("No configuration file specified")
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = asdict(self.config)
            
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif file_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            self.logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            raise
    
    def get_data_sources_config(self) -> Dict[str, Any]:
        """
        Get data sources configuration.
        
        Returns:
            Dictionary with data sources configuration
        """
        # This would typically load from a separate data sources config file
        # For now, return a default configuration
        return {
            'data_sources': [
                {
                    'name': 'primary_drugs',
                    'type': 'csv',
                    'location': 'C:/Users/sparta/Downloads/archive (1)/drugsInfo.csv',
                    'data_type': 'drugs',
                    'priority': 1,
                    'enabled': True,
                    'options': {
                        'encoding': 'utf-8',
                        'delimiter': ',',
                        'header_row': 0
                    }
                },
                {
                    'name': 'primary_diseases',
                    'type': 'csv',
                    'location': 'C:/Users/sparta/Downloads/archive (1)/diseasesInfo.csv',
                    'data_type': 'diseases',
                    'priority': 1,
                    'enabled': True,
                    'options': {
                        'encoding': 'utf-8',
                        'delimiter': ',',
                        'header_row': 0
                    }
                },
                {
                    'name': 'primary_mappings',
                    'type': 'csv',
                    'location': 'C:/Users/sparta/Downloads/archive (1)/mapping.csv',
                    'data_type': 'mappings',
                    'priority': 1,
                    'enabled': True,
                    'options': {
                        'encoding': 'utf-8',
                        'delimiter': ',',
                        'header_row': 0
                    }
                }
            ]
        }
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates
        """
        try:
            # Convert dict to nested updates
            self._update_nested_dict(asdict(self.config), updates)
            
            # Recreate config object
            self.config = SystemConfig(**asdict(self.config))
            
            self.logger.info("Configuration updated")
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {str(e)}")
            raise
    
    def _update_nested_dict(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Update nested dictionary with updates."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._update_nested_dict(target[key], value)
            else:
                target[key] = value
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get processing configuration."""
        return self.config.processing
    
    def get_output_config(self) -> OutputConfig:
        """Get output configuration."""
        return self.config.output
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.config.logging
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate current configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate processing config
        proc_config = self.config.processing
        if proc_config.chunk_size <= 0:
            validation_result['errors'].append("Processing chunk_size must be positive")
            validation_result['is_valid'] = False
        
        if proc_config.max_workers <= 0:
            validation_result['errors'].append("Processing max_workers must be positive")
            validation_result['is_valid'] = False
        
        if proc_config.timeout_seconds <= 0:
            validation_result['errors'].append("Processing timeout_seconds must be positive")
            validation_result['is_valid'] = False
        
        # Validate output config
        output_config = self.config.output
        if output_config.format not in ['json', 'csv', 'xml']:
            validation_result['errors'].append(f"Unsupported output format: {output_config.format}")
            validation_result['is_valid'] = False
        
        # Validate logging config
        log_config = self.config.logging
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_config.level.upper() not in valid_log_levels:
            validation_result['errors'].append(f"Invalid logging level: {log_config.level}")
            validation_result['is_valid'] = False
        
        if log_config.max_file_size_mb <= 0:
            validation_result['errors'].append("Logging max_file_size_mb must be positive")
            validation_result['is_valid'] = False
        
        # Add warnings for potential issues
        if proc_config.max_workers > 8:
            validation_result['warnings'].append("High max_workers may cause performance issues")
        
        if proc_config.chunk_size > 10000:
            validation_result['warnings'].append("Large chunk_size may use significant memory")
        
        return validation_result
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'processing': {
                'validation_enabled': self.config.processing.enable_validation,
                'quality_check_enabled': self.config.processing.enable_quality_check,
                'deduplication_enabled': self.config.processing.enable_deduplication,
                'chunk_size': self.config.processing.chunk_size,
                'max_workers': self.config.processing.max_workers
            },
            'output': {
                'output_dir': self.config.output.output_dir,
                'format': self.config.output.format,
                'compress': self.config.output.compress,
                'include_metadata': self.config.output.include_metadata
            },
            'logging': {
                'level': self.config.logging.level,
                'file_logging': self.config.logging.file_path is not None
            }
        }
    
    def create_default_config_file(self, file_path: str) -> None:
        """
        Create a default configuration file.
        
        Args:
            file_path: Path where to create the config file
        """
        self.config_file = Path(file_path)
        self.save_config()
        self.logger.info(f"Default configuration file created: {file_path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigManager':
        """
        Create ConfigManager from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ConfigManager instance
        """
        manager = cls()
        manager.config = SystemConfig(**config_dict)
        return manager
    
    @classmethod
    def from_file(cls, file_path: str) -> 'ConfigManager':
        """
        Create ConfigManager from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            ConfigManager instance
        """
        return cls(config_file=file_path)
