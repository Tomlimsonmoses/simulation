"""
Drug data processor.

This module provides specialized processing for drug-related data,
including standardization of drug identifiers, targets, and mechanisms.
"""

import re
from typing import Dict, List, Any, Optional, Set
import logging
from urllib.parse import quote

from ..ingestion.base_ingestor import BaseProcessor

logger = logging.getLogger(__name__)


class DrugProcessor(BaseProcessor):
    """
    Processor for drug data standardization and enrichment.
    
    Handles drug identifier normalization, target protein processing,
    mechanism standardization, and data quality validation.
    """
    
    def __init__(self, name: str = "DrugProcessor"):
        """
        Initialize drug processor.
        
        Args:
            name: Name of the processor
        """
        super().__init__(name)
        
        # Drug ID patterns for validation
        self.drug_id_patterns = {
            'drugbank': r'^DB\d{5}$',
            'chembl': r'^CHEMBL\d+$',
            'pubchem': r'^CID:\d+$',
            'rxnorm': r'^RXNORM:\d+$'
        }
        
        # Target protein ID patterns
        self.target_patterns = {
            'uniprot': r'^[A-Z0-9]{6}$',
            'gene_symbol': r'^[A-Z0-9_-]+$',
            'ensembl': r'^ENSG\d{11}$'
        }
    
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process drug data records.
        
        Args:
            data: List of raw drug records
            
        Returns:
            List of processed drug records
        """
        self.logger.info(f"Processing {len(data)} drug records")
        
        processed_data = []
        
        for i, record in enumerate(data):
            try:
                processed_record = self._process_drug_record(record)
                processed_data.append(processed_record)
                
                if (i + 1) % 100 == 0:
                    self.logger.debug(f"Processed {i + 1}/{len(data)} records")
                    
            except Exception as e:
                self.logger.warning(f"Failed to process record {i}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {len(processed_data)} drug records")
        return processed_data
    
    def _process_drug_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single drug record.
        
        Args:
            record: Raw drug record
            
        Returns:
            Processed drug record
        """
        processed = record.copy()
        
        # Standardize drug ID
        processed['standardized_drug_id'] = self._standardize_drug_id(record.get('DrugID', ''))
        
        # Process drug name
        processed['standardized_drug_name'] = self._standardize_drug_name(record.get('DrugName', ''))
        
        # Process targets
        processed['processed_targets'] = self._process_targets(record.get('DrugTarget', []))
        
        # Process mechanisms
        processed['processed_mechanisms'] = self._process_mechanisms(record.get('DrugMechanism', ''))
        
        # Process categories
        processed['processed_categories'] = self._process_categories(record.get('DrugCategories', []))
        
        # Process conditions
        processed['processed_conditions'] = self._process_conditions(record.get('DrugConditions', []))
        
        # Validate SMILES if present
        if 'DrugSmile' in record and record['DrugSmile']:
            processed['validated_smiles'] = self._validate_smiles(record['DrugSmile'])
        
        # Add processing metadata
        processed['_processing_info'] = {
            'processor': self.name,
            'timestamp': self._get_timestamp(),
            'validation_status': self._validate_processed_record(processed)
        }
        
        return processed
    
    def _standardize_drug_id(self, drug_id: str) -> Dict[str, Any]:
        """
        Standardize drug identifier and extract source information.
        
        Args:
            drug_id: Raw drug identifier
            
        Returns:
            Dictionary with standardized ID information
        """
        if not drug_id:
            return {'original_id': '', 'standardized_id': '', 'source': 'unknown'}
        
        drug_id = str(drug_id).strip()
        
        # Identify source based on pattern
        source = 'unknown'
        for source_name, pattern in self.drug_id_patterns.items():
            if re.match(pattern, drug_id):
                source = source_name
                break
        
        return {
            'original_id': drug_id,
            'standardized_id': drug_id.upper(),  # Standardize case
            'source': source,
            'is_valid': source != 'unknown'
        }
    
    def _standardize_drug_name(self, drug_name: str) -> Dict[str, Any]:
        """
        Standardize drug name.
        
        Args:
            drug_name: Raw drug name
            
        Returns:
            Dictionary with standardized name information
        """
        if not drug_name:
            return {'original_name': '', 'standardized_name': '', 'normalized_name': ''}
        
        # Clean and normalize name
        standardized = str(drug_name).strip()
        normalized = re.sub(r'[^\w\s-]', '', standardized.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return {
            'original_name': drug_name,
            'standardized_name': standardized,
            'normalized_name': normalized,
            'search_terms': self._generate_search_terms(standardized)
        }
    
    def _process_targets(self, targets: Any) -> Dict[str, Any]:
        """
        Process drug target information.
        
        Args:
            targets: Raw target data (string or list)
            
        Returns:
            Dictionary with processed target information
        """
        if not targets:
            return {
                'raw_targets': [],
                'uniprot_targets': [],
                'gene_targets': [],
                'target_count': 0,
                'target_types': []
            }
        
        # Convert to list if string
        if isinstance(targets, str):
            try:
                import ast
                targets = ast.literal_eval(targets)
            except:
                targets = [targets]
        
        if not isinstance(targets, list):
            targets = [targets]
        
        processed_targets = {
            'raw_targets': targets,
            'uniprot_targets': [],
            'gene_targets': [],
            'other_targets': [],
            'target_count': len(targets),
            'target_types': []
        }
        
        for target in targets:
            target_str = str(target).strip()
            if not target_str:
                continue
            
            # Identify target type
            if re.match(self.target_patterns['uniprot'], target_str):
                processed_targets['uniprot_targets'].append(target_str)
                processed_targets['target_types'].append('uniprot')
            elif re.match(self.target_patterns['gene_symbol'], target_str):
                processed_targets['gene_targets'].append(target_str)
                processed_targets['target_types'].append('gene')
            else:
                processed_targets['other_targets'].append(target_str)
                processed_targets['target_types'].append('other')
        
        return processed_targets
    
    def _process_mechanisms(self, mechanisms: str) -> Dict[str, Any]:
        """
        Process drug mechanism information.
        
        Args:
            mechanisms: Raw mechanism text
            
        Returns:
            Dictionary with processed mechanism information
        """
        if not mechanisms:
            return {
                'raw_mechanism': '',
                'cleaned_mechanism': '',
                'mechanism_summary': '',
                'key_actions': [],
                'mechanism_type': 'unknown'
            }
        
        # Clean mechanism text
        cleaned = str(mechanisms).strip()
        
        # Extract key actions (inhibition, activation, etc.)
        key_actions = self._extract_key_actions(cleaned)
        
        # Determine mechanism type
        mechanism_type = self._classify_mechanism(cleaned, key_actions)
        
        # Generate summary
        summary = self._generate_mechanism_summary(cleaned)
        
        return {
            'raw_mechanism': mechanisms,
            'cleaned_mechanism': cleaned,
            'mechanism_summary': summary,
            'key_actions': key_actions,
            'mechanism_type': mechanism_type
        }
    
    def _process_categories(self, categories: Any) -> Dict[str, Any]:
        """
        Process drug categories.
        
        Args:
            categories: Raw category data
            
        Returns:
            Dictionary with processed category information
        """
        if not categories:
            return {
                'raw_categories': [],
                'primary_categories': [],
                'all_categories': [],
                'category_count': 0
            }
        
        # Convert to list if needed
        if isinstance(categories, str):
            try:
                import ast
                categories = ast.literal_eval(categories)
            except:
                categories = [categories]
        
        if not isinstance(categories, list):
            categories = [categories]
        
        # Clean and deduplicate categories
        all_categories = []
        for cat in categories:
            cat_str = str(cat).strip()
            if cat_str and cat_str not in all_categories:
                all_categories.append(cat_str)
        
        # Identify primary categories (first few)
        primary_categories = all_categories[:5] if len(all_categories) > 5 else all_categories
        
        return {
            'raw_categories': categories,
            'primary_categories': primary_categories,
            'all_categories': all_categories,
            'category_count': len(all_categories)
        }
    
    def _process_conditions(self, conditions: Any) -> Dict[str, Any]:
        """
        Process drug conditions/indications.
        
        Args:
            conditions: Raw condition data
            
        Returns:
            Dictionary with processed condition information
        """
        if not conditions:
            return {
                'raw_conditions': [],
                'processed_conditions': [],
                'condition_count': 0
            }
        
        # Convert to list if needed
        if isinstance(conditions, str):
            try:
                import ast
                conditions = ast.literal_eval(conditions)
            except:
                conditions = [conditions]
        
        if not isinstance(conditions, list):
            conditions = [conditions]
        
        # Clean conditions
        processed_conditions = []
        for condition in conditions:
            condition_str = str(condition).strip()
            if condition_str and condition_str not in processed_conditions:
                processed_conditions.append(condition_str)
        
        return {
            'raw_conditions': conditions,
            'processed_conditions': processed_conditions,
            'condition_count': len(processed_conditions)
        }
    
    def _validate_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        Validate SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Dictionary with validation results
        """
        # Basic SMILES validation
        valid_chars = set('BCNOPSFI[]()=#0123456789+-.@')
        smiles_clean = str(smiles).strip()
        
        is_valid = (
            len(smiles_clean) > 0 and
            all(c in valid_chars for c in smiles_clean)
        )
        
        return {
            'original_smiles': smiles,
            'cleaned_smiles': smiles_clean,
            'is_valid': is_valid,
            'length': len(smiles_clean)
        }
    
    def _extract_key_actions(self, mechanism: str) -> List[str]:
        """Extract key actions from mechanism text."""
        actions = []
        action_patterns = [
            r'\b(inhibits?|inhibition)\b',
            r'\b(activates?|activation)\b',
            r'\b(blocks?|blockade)\b',
            r'\b(binds?|binding)\b',
            r'\b(antagonists?|antagonism)\b',
            r'\b(agonists?|agonism)\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, mechanism, re.IGNORECASE)
            actions.extend(matches)
        
        return list(set(actions))  # Remove duplicates
    
    def _classify_mechanism(self, mechanism: str, actions: List[str]) -> str:
        """Classify mechanism type based on actions."""
        mechanism_lower = mechanism.lower()
        
        if any('inhib' in action.lower() for action in actions):
            return 'inhibitor'
        elif any('activ' in action.lower() for action in actions):
            return 'activator'
        elif any('antagon' in action.lower() for action in actions):
            return 'antagonist'
        elif any('agon' in action.lower() for action in actions):
            return 'agonist'
        elif 'receptor' in mechanism_lower:
            return 'receptor_modulator'
        elif 'enzyme' in mechanism_lower:
            return 'enzyme_modulator'
        else:
            return 'other'
    
    def _generate_mechanism_summary(self, mechanism: str) -> str:
        """Generate a brief summary of the mechanism."""
        # Take first sentence or first 100 characters
        if '.' in mechanism:
            first_sentence = mechanism.split('.')[0] + '.'
            return first_sentence[:200]  # Limit length
        else:
            return mechanism[:200] + ('...' if len(mechanism) > 200 else '')
    
    def _generate_search_terms(self, drug_name: str) -> List[str]:
        """Generate search terms for drug name."""
        terms = [drug_name.lower()]
        
        # Add individual words
        words = drug_name.split()
        terms.extend([word.lower() for word in words if len(word) > 2])
        
        # Remove duplicates
        return list(set(terms))
    
    def _validate_processed_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a processed record."""
        validation_status = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        if not record.get('standardized_drug_id', {}).get('is_valid', False):
            validation_status['errors'].append('Invalid or missing drug ID')
            validation_status['is_valid'] = False
        
        if not record.get('standardized_drug_name', {}).get('standardized_name', ''):
            validation_status['warnings'].append('Missing drug name')
        
        # Check data quality
        if record.get('processed_targets', {}).get('target_count', 0) == 0:
            validation_status['warnings'].append('No targets found')
        
        return validation_status
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the schema of processed drug data."""
        return {
            'DrugID': {'type': 'string', 'required': True},
            'DrugName': {'type': 'string', 'required': True},
            'standardized_drug_id': {'type': 'object', 'required': True},
            'standardized_drug_name': {'type': 'object', 'required': True},
            'processed_targets': {'type': 'object', 'required': False},
            'processed_mechanisms': {'type': 'object', 'required': False},
            'processed_categories': {'type': 'object', 'required': False},
            'processed_conditions': {'type': 'object', 'required': False},
            'validated_smiles': {'type': 'object', 'required': False}
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
