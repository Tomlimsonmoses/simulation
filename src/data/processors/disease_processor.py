"""
Disease data processor.

This module provides specialized processing for disease-related data,
including standardization of disease identifiers, classifications,
and pathway information.
"""

import re
from typing import Dict, List, Any, Optional
import logging

from ..ingestion.base_ingestor import BaseProcessor

logger = logging.getLogger(__name__)


class DiseaseProcessor(BaseProcessor):
    """
    Processor for disease data standardization and enrichment.
    
    Handles disease identifier normalization, classification processing,
    pathway mapping, and data quality validation.
    """
    
    def __init__(self, name: str = "DiseaseProcessor"):
        """
        Initialize disease processor.
        
        Args:
            name: Name of the processor
        """
        super().__init__(name)
        
        # Disease ID patterns for validation
        self.disease_id_patterns = {
            'mesh': r'^MESH:[A-Z0-9]+$',
            'omim': r'^OMIM:\d+$',
            'icd10': r'^[A-Z]\d{2}(\.\d{1,3})?$',
            'icd9': r'^\d{3}(\.\d{1,2})?$',
            'doid': r'^DOID:\d+$'
        }
        
        # Disease category mappings
        self.category_mappings = {
            'cancer': ['neoplasm', 'tumor', 'carcinoma', 'sarcoma', 'leukemia'],
            'cardiovascular': ['heart', 'vascular', 'cardiac', 'circulation'],
            'respiratory': ['lung', 'respiratory', 'pulmonary', 'breathing'],
            'nervous_system': ['brain', 'neuro', 'neural', 'spinal'],
            'immune_system': ['immune', 'immunodeficiency', 'autoimmune'],
            'metabolic': ['metabolism', 'metabolic', 'endocrine'],
            'infectious': ['infection', 'viral', 'bacterial', 'fungal'],
            'genetic': ['genetic', 'congenital', 'hereditary', 'inborn']
        }
    
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process disease data records.
        
        Args:
            data: List of raw disease records
            
        Returns:
            List of processed disease records
        """
        self.logger.info(f"Processing {len(data)} disease records")
        
        processed_data = []
        
        for i, record in enumerate(data):
            try:
                processed_record = self._process_disease_record(record)
                processed_data.append(processed_record)
                
                if (i + 1) % 100 == 0:
                    self.logger.debug(f"Processed {i + 1}/{len(data)} records")
                    
            except Exception as e:
                self.logger.warning(f"Failed to process record {i}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {len(processed_data)} disease records")
        return processed_data
    
    def _process_disease_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single disease record.
        
        Args:
            record: Raw disease record
            
        Returns:
            Processed disease record
        """
        processed = record.copy()
        
        # Standardize disease ID
        processed['standardized_disease_id'] = self._standardize_disease_id(record.get('DiseaseID', ''))
        
        # Process disease name
        processed['standardized_disease_name'] = self._standardize_disease_name(record.get('DiseaseName', ''))
        
        # Process description
        processed['processed_description'] = self._process_description(record.get('DiseaseDescription', ''))
        
        # Process categories/classifications
        processed['processed_categories'] = self._process_categories(record.get('SlimMapping', []))
        
        # Process pathways
        processed['processed_pathways'] = self._process_pathways(record.get('PathwayNames', ''))
        
        # Add processing metadata
        processed['_processing_info'] = {
            'processor': self.name,
            'timestamp': self._get_timestamp(),
            'validation_status': self._validate_processed_record(processed)
        }
        
        return processed
    
    def _standardize_disease_id(self, disease_id: str) -> Dict[str, Any]:
        """
        Standardize disease identifier and extract source information.
        
        Args:
            disease_id: Raw disease identifier
            
        Returns:
            Dictionary with standardized ID information
        """
        if not disease_id:
            return {'original_id': '', 'standardized_id': '', 'source': 'unknown'}
        
        disease_id = str(disease_id).strip()
        
        # Identify source based on pattern
        source = 'unknown'
        for source_name, pattern in self.disease_id_patterns.items():
            if re.match(pattern, disease_id):
                source = source_name
                break
        
        return {
            'original_id': disease_id,
            'standardized_id': disease_id,
            'source': source,
            'is_valid': source != 'unknown'
        }
    
    def _standardize_disease_name(self, disease_name: str) -> Dict[str, Any]:
        """
        Standardize disease name.
        
        Args:
            disease_name: Raw disease name
            
        Returns:
            Dictionary with standardized name information
        """
        if not disease_name:
            return {'original_name': '', 'standardized_name': '', 'normalized_name': ''}
        
        # Clean and normalize name
        standardized = str(disease_name).strip()
        normalized = re.sub(r'[^\w\s-]', '', standardized.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Extract synonyms and variants
        synonyms = self._extract_synonyms(standardized)
        
        return {
            'original_name': disease_name,
            'standardized_name': standardized,
            'normalized_name': normalized,
            'synonyms': synonyms,
            'search_terms': self._generate_search_terms(standardized)
        }
    
    def _process_description(self, description: str) -> Dict[str, Any]:
        """
        Process disease description.
        
        Args:
            description: Raw disease description
            
        Returns:
            Dictionary with processed description information
        """
        if not description:
            return {
                'raw_description': '',
                'cleaned_description': '',
                'description_summary': '',
                'key_terms': [],
                'word_count': 0
            }
        
        # Clean description
        cleaned = str(description).strip()
        
        # Generate summary
        summary = self._generate_description_summary(cleaned)
        
        # Extract key medical terms
        key_terms = self._extract_medical_terms(cleaned)
        
        return {
            'raw_description': description,
            'cleaned_description': cleaned,
            'description_summary': summary,
            'key_terms': key_terms,
            'word_count': len(cleaned.split())
        }
    
    def _process_categories(self, categories: Any) -> Dict[str, Any]:
        """
        Process disease categories/classifications.
        
        Args:
            categories: Raw category data
            
        Returns:
            Dictionary with processed category information
        """
        if not categories:
            return {
                'raw_categories': [],
                'primary_categories': [],
                'broad_categories': [],
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
        
        # Identify broad categories
        broad_categories = self._map_to_broad_categories(all_categories)
        
        # Identify primary categories (first few)
        primary_categories = all_categories[:3] if len(all_categories) > 3 else all_categories
        
        return {
            'raw_categories': categories,
            'primary_categories': primary_categories,
            'broad_categories': broad_categories,
            'all_categories': all_categories,
            'category_count': len(all_categories)
        }
    
    def _process_pathways(self, pathways: str) -> Dict[str, Any]:
        """
        Process pathway information.
        
        Args:
            pathways: Raw pathway data
            
        Returns:
            Dictionary with processed pathway information
        """
        if not pathways:
            return {
                'raw_pathways': '',
                'pathway_list': [],
                'pathway_count': 0,
                'key_pathways': []
            }
        
        # Split pathways if they're comma-separated
        pathway_list = []
        if isinstance(pathways, str):
            # Try to split by common delimiters
            for delimiter in [',', ';', '|']:
                if delimiter in pathways:
                    pathway_list = [p.strip() for p in pathways.split(delimiter) if p.strip()]
                    break
            else:
                pathway_list = [pathways.strip()]
        
        # Clean and deduplicate
        pathway_list = list(set([p for p in pathway_list if p]))
        
        # Identify key pathways (important biological pathways)
        key_pathways = self._identify_key_pathways(pathway_list)
        
        return {
            'raw_pathways': pathways,
            'pathway_list': pathway_list,
            'pathway_count': len(pathway_list),
            'key_pathways': key_pathways
        }
    
    def _extract_synonyms(self, disease_name: str) -> List[str]:
        """Extract synonyms from disease name."""
        synonyms = []
        
        # Look for common patterns
        if '(' in disease_name and ')' in disease_name:
            # Extract content in parentheses as potential synonym
            paren_content = re.search(r'\(([^)]+)\)', disease_name)
            if paren_content:
                synonyms.append(paren_content.group(1))
        
        # Look for "also known as" patterns
        aka_patterns = [
            r'\baka\b',
            r'\balso known as\b',
            r'\bor\b'
        ]
        
        for pattern in aka_patterns:
            if re.search(pattern, disease_name, re.IGNORECASE):
                # Extract alternative names (simplified)
                parts = re.split(pattern, disease_name, flags=re.IGNORECASE)
                if len(parts) > 1:
                    synonyms.extend([p.strip() for p in parts[1:] if p.strip()])
        
        return synonyms
    
    def _generate_search_terms(self, disease_name: str) -> List[str]:
        """Generate search terms for disease name."""
        terms = [disease_name.lower()]
        
        # Add individual words
        words = disease_name.split()
        terms.extend([word.lower() for word in words if len(word) > 2])
        
        # Remove duplicates
        return list(set(terms))
    
    def _generate_description_summary(self, description: str) -> str:
        """Generate a brief summary of the description."""
        # Take first sentence or first 200 characters
        if '.' in description:
            first_sentence = description.split('.')[0] + '.'
            return first_sentence[:300]  # Limit length
        else:
            return description[:300] + ('...' if len(description) > 300 else '')
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terms from text."""
        # Simple medical term extraction (can be enhanced with medical dictionaries)
        medical_terms = []
        
        # Common medical term patterns
        patterns = [
            r'\b\w+osis\b',  # -osis suffix
            r'\b\w+itis\b',  # -itis suffix
            r'\b\w+emia\b',  # -emia suffix
            r'\b\w+pathy\b',  # -pathy suffix
            r'\b\w+genic\b',  # -genic suffix
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medical_terms.extend(matches)
        
        return list(set(medical_terms))
    
    def _map_to_broad_categories(self, categories: List[str]) -> List[str]:
        """Map specific categories to broad categories."""
        broad_categories = set()
        
        for category in categories:
            category_lower = category.lower()
            
            for broad_cat, keywords in self.category_mappings.items():
                if any(keyword in category_lower for keyword in keywords):
                    broad_categories.add(broad_cat)
        
        return list(broad_categories)
    
    def _identify_key_pathways(self, pathways: List[str]) -> List[str]:
        """Identify important biological pathways."""
        key_pathway_keywords = [
            'signaling pathway',
            'metabolic pathway',
            'cell cycle',
            'apoptosis',
            'dna repair',
            'immune response',
            'inflammation',
            'angiogenesis',
            'cell proliferation',
            'transcription'
        ]
        
        key_pathways = []
        
        for pathway in pathways:
            pathway_lower = pathway.lower()
            if any(keyword in pathway_lower for keyword in key_pathway_keywords):
                key_pathways.append(pathway)
        
        return key_pathways
    
    def _validate_processed_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a processed record."""
        validation_status = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        if not record.get('standardized_disease_id', {}).get('is_valid', False):
            validation_status['errors'].append('Invalid or missing disease ID')
            validation_status['is_valid'] = False
        
        if not record.get('standardized_disease_name', {}).get('standardized_name', ''):
            validation_status['warnings'].append('Missing disease name')
        
        # Check data quality
        if record.get('processed_description', {}).get('word_count', 0) < 5:
            validation_status['warnings'].append('Very short or missing description')
        
        return validation_status
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the schema of processed disease data."""
        return {
            'DiseaseID': {'type': 'string', 'required': True},
            'DiseaseName': {'type': 'string', 'required': True},
            'standardized_disease_id': {'type': 'object', 'required': True},
            'standardized_disease_name': {'type': 'object', 'required': True},
            'processed_description': {'type': 'object', 'required': False},
            'processed_categories': {'type': 'object', 'required': False},
            'processed_pathways': {'type': 'object', 'required': False}
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
