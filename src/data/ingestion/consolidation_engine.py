"""
Data consolidation engine.

This module provides the core functionality for consolidating data
from multiple sources into unified datasets.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import logging
from collections import defaultdict
import hashlib

from .base_ingestor import IngestionResult, BaseProcessor

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """Configuration for data consolidation."""
    merge_strategy: str = "merge"  # merge, overwrite, append
    conflict_resolution: str = "latest"  # latest, highest_priority, manual
    deduplication: bool = True
    validation_enabled: bool = True
    output_format: str = "json"


@dataclass
class ConsolidationResult:
    """Result of data consolidation operation."""
    total_sources: int
    total_records_processed: int
    total_records_consolidated: int
    duplicates_removed: int
    conflicts_resolved: int
    validation_errors: List[str]
    warnings: List[str]
    consolidated_data: Dict[str, List[Dict[str, Any]]]
    metadata: Dict[str, Any]


class ConsolidationEngine:
    """
    Engine for consolidating data from multiple sources.
    
    Handles merging, deduplication, conflict resolution,
    and validation of consolidated datasets.
    """
    
    def __init__(self, config: ConsolidationConfig):
        """
        Initialize consolidation engine.
        
        Args:
            config: Consolidation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.processors: Dict[str, BaseProcessor] = {}
        self.source_priorities: Dict[str, int] = {}
    
    def register_processor(self, data_type: str, processor: BaseProcessor) -> None:
        """
        Register a data processor for a specific data type.
        
        Args:
            data_type: Type of data (e.g., 'drugs', 'diseases')
            processor: Processor instance
        """
        self.processors[data_type] = processor
        self.logger.info(f"Registered processor for {data_type}")
    
    def set_source_priority(self, source_name: str, priority: int) -> None:
        """
        Set priority for a data source.
        
        Args:
            source_name: Name of the data source
            priority: Priority level (higher = more important)
        """
        self.source_priorities[source_name] = priority
        self.logger.info(f"Set priority for {source_name}: {priority}")
    
    def consolidate_data(self, ingestion_results: List[IngestionResult], 
                        data_type: str) -> ConsolidationResult:
        """
        Consolidate data from multiple ingestion results.
        
        Args:
            ingestion_results: List of ingestion results from different sources
            data_type: Type of data being consolidated
            
        Returns:
            ConsolidationResult with details of the operation
        """
        self.logger.info(f"Starting consolidation for {data_type} from {len(ingestion_results)} sources")
        
        total_records_processed = sum(r.records_processed for r in ingestion_results)
        validation_errors = []
        warnings = []
        duplicates_removed = 0
        conflicts_resolved = 0
        
        # Extract data from ingestion results
        all_data = []
        for result in ingestion_results:
            data = result.metadata.get('ingested_data', [])
            for record in data:
                record['_source'] = result.source_name
                record['_source_priority'] = self.source_priorities.get(result.source_name, 1)
            all_data.extend(data)
        
        # Process data if processor is available
        if data_type in self.processors:
            try:
                processor = self.processors[data_type]
                all_data = processor.process(all_data)
                
                # Validate processed data
                if self.config.validation_enabled:
                    errors = processor.validate_processed_data(all_data)
                    validation_errors.extend(errors)
                    
            except Exception as e:
                error_msg = f"Data processing failed for {data_type}: {str(e)}"
                self.logger.error(error_msg)
                validation_errors.append(error_msg)
        
        # Deduplicate if enabled
        if self.config.deduplication:
            all_data, duplicates_removed = self._deduplicate_data(all_data, data_type)
        
        # Resolve conflicts
        all_data, conflicts_resolved = self._resolve_conflicts(all_data, data_type)
        
        # Group data by type
        consolidated_data = self._group_data_by_type(all_data, data_type)
        
        # Create result
        result = ConsolidationResult(
            total_sources=len(ingestion_results),
            total_records_processed=total_records_processed,
            total_records_consolidated=len(all_data),
            duplicates_removed=duplicates_removed,
            conflicts_resolved=conflicts_resolved,
            validation_errors=validation_errors,
            warnings=warnings,
            consolidated_data=consolidated_data,
            metadata={
                'data_type': data_type,
                'sources': [r.source_name for r in ingestion_results],
                'consolidation_config': self.config.__dict__,
                'processing_timestamp': self._get_timestamp()
            }
        )
        
        self.logger.info(f"Consolidation completed: {len(all_data)} records consolidated")
        return result
    
    def _deduplicate_data(self, data: List[Dict[str, Any]], data_type: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        Remove duplicate records from the dataset.
        
        Args:
            data: List of records to deduplicate
            data_type: Type of data being processed
            
        Returns:
            Tuple of (deduplicated_data, duplicates_removed_count)
        """
        if not data:
            return data, 0
        
        # Use different deduplication strategies based on data type
        if data_type == 'drugs':
            return self._deduplicate_drugs(data)
        elif data_type == 'diseases':
            return self._deduplicate_diseases(data)
        elif data_type == 'mappings':
            return self._deduplicate_mappings(data)
        else:
            return self._deduplicate_generic(data)
    
    def _deduplicate_drugs(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Deduplicate drug records using DrugID."""
        seen_drugs = {}
        duplicates_removed = 0
        
        for record in data:
            drug_id = record.get('DrugID')
            if not drug_id:
                continue
            
            if drug_id in seen_drugs:
                # Keep the record from higher priority source
                existing_priority = seen_drugs[drug_id]['_source_priority']
                current_priority = record.get('_source_priority', 1)
                
                if current_priority > existing_priority:
                    seen_drugs[drug_id] = record
                    duplicates_removed += 1
                else:
                    duplicates_removed += 1
            else:
                seen_drugs[drug_id] = record
        
        return list(seen_drugs.values()), duplicates_removed
    
    def _deduplicate_diseases(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Deduplicate disease records using DiseaseID."""
        seen_diseases = {}
        duplicates_removed = 0
        
        for record in data:
            disease_id = record.get('DiseaseID')
            if not disease_id:
                continue
            
            if disease_id in seen_diseases:
                existing_priority = seen_diseases[disease_id]['_source_priority']
                current_priority = record.get('_source_priority', 1)
                
                if current_priority > existing_priority:
                    seen_diseases[disease_id] = record
                    duplicates_removed += 1
                else:
                    duplicates_removed += 1
            else:
                seen_diseases[disease_id] = record
        
        return list(seen_diseases.values()), duplicates_removed
    
    def _deduplicate_mappings(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Deduplicate mapping records using DrugID-DiseaseID combination."""
        seen_mappings = {}
        duplicates_removed = 0
        
        for record in data:
            drug_id = record.get('DrugID')
            disease_id = record.get('DiseaseID')
            
            if not drug_id or not disease_id:
                continue
            
            mapping_key = f"{drug_id}-{disease_id}"
            
            if mapping_key in seen_mappings:
                duplicates_removed += 1
            else:
                seen_mappings[mapping_key] = record
        
        return list(seen_mappings.values()), duplicates_removed
    
    def _deduplicate_generic(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Generic deduplication using content hash."""
        seen_hashes = {}
        duplicates_removed = 0
        
        for record in data:
            # Create hash of record content (excluding metadata)
            content = {k: v for k, v in record.items() if not k.startswith('_')}
            record_hash = hashlib.md5(str(content).encode()).hexdigest()
            
            if record_hash in seen_hashes:
                duplicates_removed += 1
            else:
                seen_hashes[record_hash] = record
        
        return list(seen_hashes.values()), duplicates_removed
    
    def _resolve_conflicts(self, data: List[Dict[str, Any]], data_type: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        Resolve conflicts between records from different sources.
        
        Args:
            data: List of records to process
            data_type: Type of data being processed
            
        Returns:
            Tuple of (conflict_resolved_data, conflicts_resolved_count)
        """
        conflicts_resolved = 0
        
        if self.config.conflict_resolution == "latest":
            # Keep the most recently processed record
            return data, conflicts_resolved
        
        elif self.config.conflict_resolution == "highest_priority":
            # Already handled during deduplication
            return data, conflicts_resolved
        
        # For other strategies, implement as needed
        return data, conflicts_resolved
    
    def _group_data_by_type(self, data: List[Dict[str, Any]], data_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group consolidated data by specific types.
        
        Args:
            data: List of consolidated records
            data_type: Main data type
            
        Returns:
            Dictionary with grouped data
        """
        grouped_data = {data_type: data}
        
        # Add additional grouping based on data type
        if data_type == 'drugs':
            # Group by drug categories
            categories = defaultdict(list)
            for record in data:
                drug_categories = record.get('DrugCategories', [])
                if isinstance(drug_categories, str):
                    drug_categories = [drug_categories]
                
                for category in drug_categories:
                    if category:
                        categories[f"drugs_by_{category}"].append(record)
            
            grouped_data.update(categories)
        
        elif data_type == 'diseases':
            # Group by disease categories
            categories = defaultdict(list)
            for record in data:
                disease_categories = record.get('SlimMapping', [])
                if isinstance(disease_categories, str):
                    disease_categories = [disease_categories]
                
                for category in disease_categories:
                    if category:
                        categories[f"diseases_by_{category}"].append(record)
            
            grouped_data.update(categories)
        
        return grouped_data
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_consolidation_summary(self, result: ConsolidationResult) -> Dict[str, Any]:
        """
        Get a summary of consolidation results.
        
        Args:
            result: ConsolidationResult to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'overview': {
                'total_sources': result.total_sources,
                'total_records_processed': result.total_records_processed,
                'total_records_consolidated': result.total_records_consolidated,
                'consolidation_rate': (
                    result.total_records_consolidated / result.total_records_processed 
                    if result.total_records_processed > 0 else 0
                )
            },
            'quality_metrics': {
                'duplicates_removed': result.duplicates_removed,
                'conflicts_resolved': result.conflicts_resolved,
                'validation_errors': len(result.validation_errors),
                'warnings': len(result.warnings)
            },
            'data_distribution': {
                data_type: len(records) 
                for data_type, records in result.consolidated_data.items()
            },
            'sources': result.metadata.get('sources', [])
        }
        
        return summary
