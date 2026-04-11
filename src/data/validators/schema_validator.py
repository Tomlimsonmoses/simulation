"""
Schema validator for data validation.

This module provides schema validation functionality to ensure
data conforms to expected structures and types.
"""

import json
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validator for data schema compliance.
    
    Validates that data records conform to expected schemas,
    including field presence, data types, and value constraints.
    """
    
    def __init__(self, schema_dir: Optional[str] = None):
        """
        Initialize schema validator.
        
        Args:
            schema_dir: Directory containing schema files
        """
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.logger = logging.getLogger(__name__)
        self.schemas = {}
        
        # Load built-in schemas
        self._load_builtin_schemas()
        
        # Load custom schemas if directory provided
        if self.schema_dir:
            self._load_custom_schemas()
    
    def _load_builtin_schemas(self) -> None:
        """Load built-in schema definitions."""
        self.schemas = {
            'drugs': {
                'type': 'object',
                'required_fields': ['DrugID', 'DrugName'],
                'field_types': {
                    'DrugID': 'string',
                    'DrugName': 'string',
                    'DrugDescription': 'string',
                    'DrugTarget': 'string_or_list',
                    'DrugPharmacodynamics': 'string',
                    'DrugSmile': 'string',
                    'DrugMechanism': 'string',
                    'DrugConditions': 'string_or_list',
                    'DrugCategories': 'string_or_list'
                },
                'field_constraints': {
                    'DrugID': {'pattern': r'^[A-Z]{2,}\d+$', 'min_length': 3},
                    'DrugName': {'min_length': 1},
                    'DrugSmile': {'pattern': r'^[BCNOPSFI\[\]()=#0123456789+.\-@]+$'}
                }
            },
            'diseases': {
                'type': 'object',
                'required_fields': ['DiseaseID', 'DiseaseName'],
                'field_types': {
                    'DiseaseID': 'string',
                    'DiseaseName': 'string',
                    'DiseaseDescription': 'string',
                    'SlimMapping': 'string_or_list',
                    'PathwayNames': 'string'
                },
                'field_constraints': {
                    'DiseaseID': {'pattern': r'^[A-Z]+:[A-Z0-9]+$', 'min_length': 3},
                    'DiseaseName': {'min_length': 1}
                }
            },
            'mappings': {
                'type': 'object',
                'required_fields': ['DrugID', 'DiseaseID'],
                'field_types': {
                    'DrugID': 'string',
                    'DiseaseID': 'string'
                },
                'field_constraints': {
                    'DrugID': {'pattern': r'^[A-Z]{2,}\d+$', 'min_length': 3},
                    'DiseaseID': {'pattern': r'^[A-Z]+:[A-Z0-9]+$', 'min_length': 3}
                }
            }
        }
    
    def _load_custom_schemas(self) -> None:
        """Load custom schema files from directory."""
        if not self.schema_dir or not self.schema_dir.exists():
            return
        
        for schema_file in self.schema_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                
                schema_name = schema_file.stem
                self.schemas[schema_name] = schema
                self.logger.info(f"Loaded custom schema: {schema_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load schema {schema_file}: {str(e)}")
    
    def validate_record(self, record: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """
        Validate a single record against a schema.
        
        Args:
            record: Record to validate
            schema_name: Name of schema to validate against
            
        Returns:
            Dictionary with validation results
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        schema = self.schemas[schema_name]
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'field_validations': {}
        }
        
        # Check required fields
        self._validate_required_fields(record, schema, validation_result)
        
        # Check field types
        self._validate_field_types(record, schema, validation_result)
        
        # Check field constraints
        self._validate_field_constraints(record, schema, validation_result)
        
        # Overall validity
        validation_result['is_valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    def validate_dataset(self, data: List[Dict[str, Any]], schema_name: str) -> Dict[str, Any]:
        """
        Validate an entire dataset against a schema.
        
        Args:
            data: List of records to validate
            schema_name: Name of schema to validate against
            
        Returns:
            Dictionary with dataset validation results
        """
        self.logger.info(f"Validating {len(data)} records against {schema_name} schema")
        
        dataset_result = {
            'total_records': len(data),
            'valid_records': 0,
            'invalid_records': 0,
            'records_with_warnings': 0,
            'common_errors': {},
            'common_warnings': {},
            'field_completeness': {},
            'record_validations': []
        }
        
        # Track error frequencies
        error_counts = {}
        warning_counts = {}
        
        # Track field completeness
        field_totals = {}
        field_present = {}
        
        for i, record in enumerate(data):
            record_validation = self.validate_record(record, schema_name)
            record_validation['record_index'] = i
            
            if record_validation['is_valid']:
                dataset_result['valid_records'] += 1
            else:
                dataset_result['invalid_records'] += 1
            
            if record_validation['warnings']:
                dataset_result['records_with_warnings'] += 1
            
            # Count errors and warnings
            for error in record_validation['errors']:
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for warning in record_validation['warnings']:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1
            
            # Track field presence
            for field in record.keys():
                field_totals[field] = field_totals.get(field, 0) + 1
                if record.get(field) is not None and record.get(field) != '':
                    field_present[field] = field_present.get(field, 0) + 1
            
            # Store detailed validation for first few records
            if i < 10:
                dataset_result['record_validations'].append(record_validation)
        
        # Calculate field completeness
        for field in field_totals:
            completeness = field_present.get(field, 0) / field_totals[field]
            dataset_result['field_completeness'][field] = {
                'present_count': field_present.get(field, 0),
                'total_count': field_totals[field],
                'completeness_ratio': completeness
            }
        
        # Get most common errors and warnings
        dataset_result['common_errors'] = dict(
            sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        dataset_result['common_warnings'] = dict(
            sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        self.logger.info(f"Validation complete: {dataset_result['valid_records']}/{dataset_result['total_records']} valid")
        return dataset_result
    
    def _validate_required_fields(self, record: Dict[str, Any], schema: Dict[str, Any], 
                                 validation_result: Dict[str, Any]) -> None:
        """Validate required fields are present."""
        required_fields = schema.get('required_fields', [])
        
        for field in required_fields:
            if field not in record or record[field] is None or record[field] == '':
                error_msg = f"Missing required field: {field}"
                validation_result['errors'].append(error_msg)
                validation_result['field_validations'][field] = {
                    'status': 'error',
                    'message': error_msg
                }
    
    def _validate_field_types(self, record: Dict[str, Any], schema: Dict[str, Any],
                             validation_result: Dict[str, Any]) -> None:
        """Validate field types."""
        field_types = schema.get('field_types', {})
        
        for field, expected_type in field_types.items():
            if field not in record:
                continue
            
            value = record[field]
            
            if expected_type == 'string':
                if not isinstance(value, str):
                    error_msg = f"Field {field} should be string, got {type(value).__name__}"
                    validation_result['errors'].append(error_msg)
                    validation_result['field_validations'][field] = {
                        'status': 'error',
                        'message': error_msg
                    }
            
            elif expected_type == 'string_or_list':
                if not isinstance(value, (str, list)):
                    error_msg = f"Field {field} should be string or list, got {type(value).__name__}"
                    validation_result['errors'].append(error_msg)
                    validation_result['field_validations'][field] = {
                        'status': 'error',
                        'message': error_msg
                    }
            
            elif expected_type == 'list':
                if not isinstance(value, list):
                    error_msg = f"Field {field} should be list, got {type(value).__name__}"
                    validation_result['errors'].append(error_msg)
                    validation_result['field_validations'][field] = {
                        'status': 'error',
                        'message': error_msg
                    }
            
            elif expected_type == 'object':
                if not isinstance(value, dict):
                    error_msg = f"Field {field} should be object, got {type(value).__name__}"
                    validation_result['errors'].append(error_msg)
                    validation_result['field_validations'][field] = {
                        'status': 'error',
                        'message': error_msg
                    }
    
    def _validate_field_constraints(self, record: Dict[str, Any], schema: Dict[str, Any],
                                  validation_result: Dict[str, Any]) -> None:
        """Validate field constraints."""
        field_constraints = schema.get('field_constraints', {})
        
        for field, constraints in field_constraints.items():
            if field not in record:
                continue
            
            value = record[field]
            
            if isinstance(value, str):
                # Pattern validation
                if 'pattern' in constraints:
                    import re
                    pattern = constraints['pattern']
                    if not re.match(pattern, value):
                        warning_msg = f"Field {field} does not match expected pattern: {pattern}"
                        validation_result['warnings'].append(warning_msg)
                        if field not in validation_result['field_validations']:
                            validation_result['field_validations'][field] = {
                                'status': 'warning',
                                'message': warning_msg
                            }
                
                # Length validation
                if 'min_length' in constraints:
                    if len(value) < constraints['min_length']:
                        error_msg = f"Field {field} is too short (min: {constraints['min_length']})"
                        validation_result['errors'].append(error_msg)
                        validation_result['field_validations'][field] = {
                            'status': 'error',
                            'message': error_msg
                        }
                
                if 'max_length' in constraints:
                    if len(value) > constraints['max_length']:
                        warning_msg = f"Field {field} is unusually long (max suggested: {constraints['max_length']})"
                        validation_result['warnings'].append(warning_msg)
                        if field not in validation_result['field_validations']:
                            validation_result['field_validations'][field] = {
                                'status': 'warning',
                                'message': warning_msg
                            }
    
    def add_custom_schema(self, schema_name: str, schema: Dict[str, Any]) -> None:
        """
        Add a custom schema definition.
        
        Args:
            schema_name: Name for the schema
            schema: Schema definition
        """
        self.schemas[schema_name] = schema
        self.logger.info(f"Added custom schema: {schema_name}")
    
    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Get a schema definition.
        
        Args:
            schema_name: Name of schema
            
        Returns:
            Schema definition
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        return self.schemas[schema_name]
    
    def list_schemas(self) -> List[str]:
        """
        Get list of available schemas.
        
        Returns:
            List of schema names
        """
        return list(self.schemas.keys())
    
    def validate_schema_compatibility(self, schema1: str, schema2: str) -> Dict[str, Any]:
        """
        Check compatibility between two schemas.
        
        Args:
            schema1: First schema name
            schema2: Second schema name
            
        Returns:
            Dictionary with compatibility analysis
        """
        if schema1 not in self.schemas or schema2 not in self.schemas:
            raise ValueError("One or both schemas not found")
        
        s1 = self.schemas[schema1]
        s2 = self.schemas[schema2]
        
        compatibility = {
            'compatible': True,
            'differences': [],
            'missing_in_s1': [],
            'missing_in_s2': [],
            'type_conflicts': [],
            'constraint_conflicts': []
        }
        
        # Compare required fields
        req1 = set(s1.get('required_fields', []))
        req2 = set(s2.get('required_fields', []))
        
        compatibility['missing_in_s1'] = list(req2 - req1)
        compatibility['missing_in_s2'] = list(req1 - req2)
        
        # Compare field types
        types1 = s1.get('field_types', {})
        types2 = s2.get('field_types', {})
        
        all_fields = set(types1.keys()) | set(types2.keys())
        
        for field in all_fields:
            if field in types1 and field in types2:
                if types1[field] != types2[field]:
                    compatibility['type_conflicts'].append({
                        'field': field,
                        'type_in_s1': types1[field],
                        'type_in_s2': types2[field]
                    })
                    compatibility['compatible'] = False
        
        # Overall compatibility
        compatibility['compatible'] = (
            len(compatibility['missing_in_s1']) == 0 and
            len(compatibility['missing_in_s2']) == 0 and
            len(compatibility['type_conflicts']) == 0
        )
        
        return compatibility
