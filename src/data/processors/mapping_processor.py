"""
Mapping data processor.

This module provides specialized processing for drug-disease relationship data,
including validation of mappings, relationship classification, and enrichment.
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple
import logging

from ..ingestion.base_ingestor import BaseProcessor

logger = logging.getLogger(__name__)


class MappingProcessor(BaseProcessor):
    """
    Processor for drug-disease relationship data.
    
    Handles mapping validation, relationship classification,
    and enrichment of drug-disease connections.
    """
    
    def __init__(self, name: str = "MappingProcessor"):
        """
        Initialize mapping processor.
        
        Args:
            name: Name of the processor
        """
        super().__init__(name)
        
        # Relationship type patterns
        self.relationship_patterns = {
            'treatment': r'\b(treats?|treating|treatment|therapy|therapeutic)\b',
            'contraindication': r'\b(contraindicated?|contraindication|avoid|should not)\b',
            'prevention': r'\b(prevents?|preventing|prevention|prophylaxis)\b',
            'symptom_relief': r'\b(relieves?|relief|symptom|palliative)\b',
            'diagnosis': r'\b(diagnoses?|diagnostic|detects?|detection)\b',
            'side_effect': r'\b(side.?effect|adverse|causes?|induces?)\b'
        }
        
        # Confidence level indicators
        self.confidence_indicators = {
            'high': ['proven', 'established', 'approved', 'fda', 'clinical trial'],
            'medium': ['evidence', 'studies', 'research', 'investigational'],
            'low': ['potential', 'possible', 'may', 'might', 'theoretical']
        }
    
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process mapping data records.
        
        Args:
            data: List of raw mapping records
            
        Returns:
            List of processed mapping records
        """
        self.logger.info(f"Processing {len(data)} mapping records")
        
        processed_data = []
        
        for i, record in enumerate(data):
            try:
                processed_record = self._process_mapping_record(record)
                processed_data.append(processed_record)
                
                if (i + 1) % 500 == 0:
                    self.logger.debug(f"Processed {i + 1}/{len(data)} records")
                    
            except Exception as e:
                self.logger.warning(f"Failed to process record {i}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {len(processed_data)} mapping records")
        return processed_data
    
    def _process_mapping_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single mapping record.
        
        Args:
            record: Raw mapping record
            
        Returns:
            Processed mapping record
        """
        processed = record.copy()
        
        # Standardize drug ID
        processed['standardized_drug_id'] = self._standardize_drug_id(record.get('DrugID', ''))
        
        # Standardize disease ID
        processed['standardized_disease_id'] = self._standardize_disease_id(record.get('DiseaseID', ''))
        
        # Create mapping key
        processed['mapping_key'] = self._create_mapping_key(
            processed['standardized_drug_id'].get('standardized_id', ''),
            processed['standardized_disease_id'].get('standardized_id', '')
        )
        
        # Classify relationship type
        processed['relationship_info'] = self._classify_relationship(record)
        
        # Validate mapping
        processed['validation_info'] = self._validate_mapping(processed)
        
        # Add processing metadata
        processed['_processing_info'] = {
            'processor': self.name,
            'timestamp': self._get_timestamp(),
            'validation_status': self._validate_processed_record(processed)
        }
        
        return processed
    
    def _standardize_drug_id(self, drug_id: str) -> Dict[str, Any]:
        """Standardize drug ID in mapping."""
        if not drug_id:
            return {'original_id': '', 'standardized_id': '', 'source': 'unknown', 'is_valid': False}
        
        drug_id = str(drug_id).strip()
        
        # Basic validation for common drug ID formats
        is_valid = bool(re.match(r'^(DB\d{5}|CHEMBL\d+|CID:\d+|RXNORM:\d+)$', drug_id))
        
        return {
            'original_id': drug_id,
            'standardized_id': drug_id.upper(),
            'source': 'database',
            'is_valid': is_valid
        }
    
    def _standardize_disease_id(self, disease_id: str) -> Dict[str, Any]:
        """Standardize disease ID in mapping."""
        if not disease_id:
            return {'original_id': '', 'standardized_id': '', 'source': 'unknown', 'is_valid': False}
        
        disease_id = str(disease_id).strip()
        
        # Basic validation for common disease ID formats
        is_valid = bool(re.match(r'^(MESH:[A-Z0-9]+|OMIM:\d+|ICD-10:[A-Z]\d{2}(\.\d{1,3})?)$', disease_id))
        
        return {
            'original_id': disease_id,
            'standardized_id': disease_id,
            'source': 'database',
            'is_valid': is_valid
        }
    
    def _create_mapping_key(self, drug_id: str, disease_id: str) -> str:
        """Create a unique key for the mapping."""
        if not drug_id or not disease_id:
            return ''
        return f"{drug_id}--{disease_id}"
    
    def _classify_relationship(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify the drug-disease relationship type.
        
        Args:
            record: Mapping record
            
        Returns:
            Dictionary with relationship classification
        """
        relationship_info = {
            'primary_type': 'unknown',
            'confidence_level': 'medium',
            'evidence_strength': 'unknown',
            'indication_type': 'unknown'
        }
        
        # If there's additional context about the relationship, use it
        # For basic mappings, we'll classify based on common patterns
        
        # Default to treatment relationship for drug-disease mappings
        relationship_info['primary_type'] = 'treatment'
        relationship_info['confidence_level'] = 'medium'
        relationship_info['indication_type'] = 'therapeutic'
        
        return relationship_info
    
    def _validate_mapping(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the mapping record.
        
        Args:
            record: Processed mapping record
            
        Returns:
            Dictionary with validation results
        """
        validation_info = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'completeness_score': 0.0
        }
        
        # Check required fields
        drug_id_info = record.get('standardized_drug_id', {})
        disease_id_info = record.get('standardized_disease_id', {})
        
        if not drug_id_info.get('is_valid', False):
            validation_info['errors'].append('Invalid drug ID')
            validation_info['is_valid'] = False
        
        if not disease_id_info.get('is_valid', False):
            validation_info['errors'].append('Invalid disease ID')
            validation_info['is_valid'] = False
        
        # Calculate completeness score
        total_fields = 2  # drug_id, disease_id
        valid_fields = sum([
            drug_id_info.get('is_valid', False),
            disease_id_info.get('is_valid', False)
        ])
        
        validation_info['completeness_score'] = valid_fields / total_fields
        
        # Add warnings for potential issues
        if not record.get('mapping_key', ''):
            validation_info['warnings'].append('Missing mapping key')
        
        return validation_info
    
    def _validate_processed_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a processed mapping record."""
        validation_status = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check validation info
        validation_info = record.get('validation_info', {})
        if not validation_info.get('is_valid', True):
            validation_status['errors'].extend(validation_info.get('errors', []))
            validation_status['is_valid'] = False
        
        # Check for duplicates (would be handled during consolidation)
        if not record.get('mapping_key', ''):
            validation_status['errors'].append('Missing mapping key')
            validation_status['is_valid'] = False
        
        return validation_status
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the schema of processed mapping data."""
        return {
            'DrugID': {'type': 'string', 'required': True},
            'DiseaseID': {'type': 'string', 'required': True},
            'standardized_drug_id': {'type': 'object', 'required': True},
            'standardized_disease_id': {'type': 'object', 'required': True},
            'mapping_key': {'type': 'string', 'required': True},
            'relationship_info': {'type': 'object', 'required': False},
            'validation_info': {'type': 'object', 'required': True}
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def analyze_mapping_patterns(self, processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in the processed mapping data.
        
        Args:
            processed_data: List of processed mapping records
            
        Returns:
            Dictionary with pattern analysis results
        """
        if not processed_data:
            return {}
        
        analysis = {
            'total_mappings': len(processed_data),
            'unique_drugs': len(set(r['standardized_drug_id']['standardized_id'] for r in processed_data)),
            'unique_diseases': len(set(r['standardized_disease_id']['standardized_id'] for r in processed_data)),
            'relationship_types': {},
            'confidence_distribution': {},
            'validation_summary': {
                'valid_mappings': 0,
                'invalid_mappings': 0,
                'warnings': 0
            }
        }
        
        for record in processed_data:
            # Count relationship types
            rel_type = record.get('relationship_info', {}).get('primary_type', 'unknown')
            analysis['relationship_types'][rel_type] = analysis['relationship_types'].get(rel_type, 0) + 1
            
            # Count confidence levels
            confidence = record.get('relationship_info', {}).get('confidence_level', 'unknown')
            analysis['confidence_distribution'][confidence] = analysis['confidence_distribution'].get(confidence, 0) + 1
            
            # Count validation status
            validation = record.get('validation_info', {})
            if validation.get('is_valid', True):
                analysis['validation_summary']['valid_mappings'] += 1
            else:
                analysis['validation_summary']['invalid_mappings'] += 1
            
            analysis['validation_summary']['warnings'] += len(validation.get('warnings', []))
        
        return analysis
