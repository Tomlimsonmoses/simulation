"""
JSON data exporter.

This module provides functionality to export consolidated data
to JSON format with proper schema structure for model integration.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import gzip

from ..ingestion.consolidation_engine import ConsolidationResult

logger = logging.getLogger(__name__)


class JSONExporter:
    """
    Exporter for JSON format output.
    
    Handles the export of consolidated data to JSON files
    with proper schema structure and metadata.
    """
    
    def __init__(self, output_dir: str, compress: bool = False):
        """
        Initialize JSON exporter.
        
        Args:
            output_dir: Directory to save output files
            compress: Whether to compress output files with gzip
        """
        self.output_dir = Path(output_dir)
        self.compress = compress
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_consolidated_data(self, consolidation_result: ConsolidationResult, 
                               data_type: str) -> Dict[str, str]:
        """
        Export consolidated data to JSON files.
        
        Args:
            consolidation_result: Result from consolidation engine
            data_type: Type of data being exported
            
        Returns:
            Dictionary mapping file types to file paths
        """
        self.logger.info(f"Exporting {data_type} data to JSON")
        
        exported_files = {}
        consolidated_data = consolidation_result.consolidated_data
        
        # Export main consolidated dataset
        main_file = self._export_main_dataset(consolidated_data, data_type, consolidation_result)
        exported_files['main'] = main_file
        
        # Export summary statistics
        summary_file = self._export_summary(consolidation_result, data_type)
        exported_files['summary'] = summary_file
        
        # Export metadata
        metadata_file = self._export_metadata(consolidation_result, data_type)
        exported_files['metadata'] = metadata_file
        
        # Export schema information
        schema_file = self._export_schema(data_type)
        exported_files['schema'] = schema_file
        
        # Export additional groupings if available (limited to prevent excessive files)
        grouping_keys = [key for key, data in consolidated_data.items() 
                       if key != data_type and data and len(data) < 100]  # Only export small groupings
        
        for key in grouping_keys:
            data = consolidated_data[key]
            grouping_file = self._export_grouping(data, key, data_type)
            exported_files[f"grouping_{key}"] = grouping_file
        
        self.logger.info(f"Exported {len(exported_files)} files for {data_type}")
        return exported_files
    
    def _export_main_dataset(self, consolidated_data: Dict[str, List[Dict[str, Any]]], 
                           data_type: str, consolidation_result: ConsolidationResult) -> str:
        """
        Export the main consolidated dataset.
        
        Args:
            consolidated_data: Dictionary with consolidated data
            data_type: Type of data
            consolidation_result: Consolidation result
            
        Returns:
            Path to exported file
        """
        main_data = consolidated_data.get(data_type, [])
        
        # Create export structure
        export_structure = {
            'metadata': {
                'data_type': data_type,
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(main_data),
                'sources': consolidation_result.metadata.get('sources', []),
                'consolidation_info': {
                    'total_sources': consolidation_result.total_sources,
                    'records_processed': consolidation_result.total_records_processed,
                    'duplicates_removed': consolidation_result.duplicates_removed,
                    'conflicts_resolved': consolidation_result.conflicts_resolved
                }
            },
            'data': main_data,
            'quality_metrics': {
                'validation_errors': consolidation_result.validation_errors,
                'warnings': consolidation_result.warnings,
                'data_quality_score': self._calculate_quality_score(consolidation_result)
            }
        }
        
        # Generate filename
        filename = f"{data_type}_consolidated.json"
        if self.compress:
            filename += ".gz"
        
        filepath = self.output_dir / filename
        
        # Write to file
        self._write_json_file(export_structure, filepath)
        
        self.logger.info(f"Exported main dataset: {len(main_data)} records to {filepath}")
        return str(filepath)
    
    def _export_summary(self, consolidation_result: ConsolidationResult, data_type: str) -> str:
        """
        Export summary statistics.
        
        Args:
            consolidation_result: Consolidation result
            data_type: Type of data
            
        Returns:
            Path to exported summary file
        """
        summary = self._generate_summary(consolidation_result, data_type)
        
        filename = f"{data_type}_summary.json"
        if self.compress:
            filename += ".gz"
        
        filepath = self.output_dir / filename
        self._write_json_file(summary, filepath)
        
        return str(filepath)
    
    def _export_metadata(self, consolidation_result: ConsolidationResult, data_type: str) -> str:
        """
        Export detailed metadata.
        
        Args:
            consolidation_result: Consolidation result
            data_type: Type of data
            
        Returns:
            Path to exported metadata file
        """
        metadata = {
            'export_info': {
                'data_type': data_type,
                'export_timestamp': datetime.now().isoformat(),
                'exporter_version': '1.0.0',
                'output_format': 'json',
                'compression': self.compress
            },
            'consolidation_metadata': consolidation_result.metadata,
            'processing_info': {
                'total_sources': consolidation_result.total_sources,
                'total_records_processed': consolidation_result.total_records_processed,
                'total_records_consolidated': consolidation_result.total_records_consolidated,
                'duplicates_removed': consolidation_result.duplicates_removed,
                'conflicts_resolved': consolidation_result.conflicts_resolved,
                'validation_errors_count': len(consolidation_result.validation_errors),
                'warnings_count': len(consolidation_result.warnings)
            },
            'data_distribution': {
                data_type: len(consolidation_result.consolidated_data.get(data_type, []))
            },
            'quality_assessment': {
                'validation_errors': consolidation_result.validation_errors,
                'warnings': consolidation_result.warnings,
                'overall_quality_score': self._calculate_quality_score(consolidation_result)
            }
        }
        
        filename = f"{data_type}_metadata.json"
        if self.compress:
            filename += ".gz"
        
        filepath = self.output_dir / filename
        self._write_json_file(metadata, filepath)
        
        return str(filepath)
    
    def _export_schema(self, data_type: str) -> str:
        """
        Export schema information for the data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            Path to exported schema file
        """
        schema = self._get_schema_for_type(data_type)
        
        filename = f"{data_type}_schema.json"
        if self.compress:
            filename += ".gz"
        
        filepath = self.output_dir / filename
        self._write_json_file(schema, filepath)
        
        return str(filepath)
    
    def _export_grouping(self, data: List[Dict[str, Any]], grouping_name: str, data_type: str) -> str:
        """
        Export grouped data.
        
        Args:
            data: Grouped data to export
            grouping_name: Name of the grouping
            data_type: Main data type
            
        Returns:
            Path to exported grouping file
        """
        grouping_structure = {
            'metadata': {
                'data_type': data_type,
                'grouping_type': grouping_name,
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(data)
            },
            'data': data
        }
        
        filename = f"{data_type}_{grouping_name}.json"
        if self.compress:
            filename += ".gz"
        
        filepath = self.output_dir / filename
        self._write_json_file(grouping_structure, filepath)
        
        return str(filepath)
    
    def _write_json_file(self, data: Dict[str, Any], filepath: Path) -> None:
        """
        Write data to JSON file.
        
        Args:
            data: Data to write
            filepath: Path to output file
        """
        try:
            if self.compress:
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                    
        except Exception as e:
            self.logger.error(f"Failed to write JSON file {filepath}: {str(e)}")
            raise
    
    def _generate_summary(self, consolidation_result: ConsolidationResult, data_type: str) -> Dict[str, Any]:
        """
        Generate summary statistics.
        
        Args:
            consolidation_result: Consolidation result
            data_type: Type of data
            
        Returns:
            Dictionary with summary statistics
        """
        data = consolidation_result.consolidated_data.get(data_type, [])
        
        summary = {
            'overview': {
                'data_type': data_type,
                'total_records': len(data),
                'total_sources': consolidation_result.total_sources,
                'records_processed': consolidation_result.total_records_processed,
                'records_consolidated': consolidation_result.total_records_consolidated,
                'consolidation_rate': (
                    consolidation_result.total_records_consolidated / 
                    consolidation_result.total_records_processed 
                    if consolidation_result.total_records_processed > 0 else 0
                )
            },
            'quality_metrics': {
                'duplicates_removed': consolidation_result.duplicates_removed,
                'conflicts_resolved': consolidation_result.conflicts_resolved,
                'validation_errors': len(consolidation_result.validation_errors),
                'warnings': len(consolidation_result.warnings),
                'quality_score': self._calculate_quality_score(consolidation_result)
            },
            'data_statistics': self._calculate_data_statistics(data, data_type),
            'sources': consolidation_result.metadata.get('sources', []),
            'processing_timestamp': consolidation_result.metadata.get('processing_timestamp')
        }
        
        return summary
    
    def _calculate_data_statistics(self, data: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """
        Calculate statistics for the data.
        
        Args:
            data: List of data records
            data_type: Type of data
            
        Returns:
            Dictionary with data statistics
        """
        if not data:
            return {'record_count': 0}
        
        stats = {
            'record_count': len(data),
            'field_completeness': {},
            'data_types': {}
        }
        
        # Calculate field completeness
        if data:
            sample_record = data[0]
            for field in sample_record.keys():
                if not field.startswith('_'):  # Skip metadata fields
                    non_null_count = sum(1 for record in data if record.get(field) is not None)
                    completeness = non_null_count / len(data)
                    stats['field_completeness'][field] = {
                        'complete_records': non_null_count,
                        'total_records': len(data),
                        'completeness_ratio': completeness
                    }
        
        # Add data type specific statistics
        if data_type == 'drugs':
            stats['drug_statistics'] = self._calculate_drug_statistics(data)
        elif data_type == 'diseases':
            stats['disease_statistics'] = self._calculate_disease_statistics(data)
        elif data_type == 'mappings':
            stats['mapping_statistics'] = self._calculate_mapping_statistics(data)
        
        return stats
    
    def _calculate_drug_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate drug-specific statistics."""
        stats = {
            'unique_drug_ids': len(set(r.get('DrugID', '') for r in data)),
            'drugs_with_targets': sum(1 for r in data if r.get('processed_targets', {}).get('target_count', 0) > 0),
            'drugs_with_mechanisms': sum(1 for r in data if r.get('processed_mechanisms', {}).get('mechanism_type', '') != 'unknown'),
            'drugs_with_categories': sum(1 for r in data if r.get('processed_categories', {}).get('category_count', 0) > 0)
        }
        
        return stats
    
    def _calculate_disease_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate disease-specific statistics."""
        stats = {
            'unique_disease_ids': len(set(r.get('DiseaseID', '') for r in data)),
            'diseases_with_descriptions': sum(1 for r in data if r.get('processed_description', {}).get('word_count', 0) > 0),
            'diseases_with_categories': sum(1 for r in data if r.get('processed_categories', {}).get('category_count', 0) > 0),
            'diseases_with_pathways': sum(1 for r in data if r.get('processed_pathways', {}).get('pathway_count', 0) > 0)
        }
        
        return stats
    
    def _calculate_mapping_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate mapping-specific statistics."""
        stats = {
            'unique_mappings': len(set(r.get('mapping_key', '') for r in data)),
            'unique_drugs': len(set(r.get('DrugID', '') for r in data)),
            'unique_diseases': len(set(r.get('DiseaseID', '') for r in data)),
            'valid_mappings': sum(1 for r in data if r.get('validation_info', {}).get('is_valid', False))
        }
        
        return stats
    
    def _calculate_quality_score(self, consolidation_result: ConsolidationResult) -> float:
        """
        Calculate overall data quality score.
        
        Args:
            consolidation_result: Consolidation result
            
        Returns:
            Quality score between 0 and 1
        """
        total_records = consolidation_result.total_records_processed
        if total_records == 0:
            return 0.0
        
        # Base score from consolidation success rate
        consolidation_rate = consolidation_result.total_records_consolidated / total_records
        
        # Penalty for validation errors
        error_penalty = len(consolidation_result.validation_errors) / total_records
        error_penalty = min(error_penalty, 0.5)  # Cap penalty at 50%
        
        # Penalty for warnings (less severe)
        warning_penalty = len(consolidation_result.warnings) / total_records * 0.1
        warning_penalty = min(warning_penalty, 0.1)  # Cap penalty at 10%
        
        # Calculate final score
        quality_score = consolidation_rate - error_penalty - warning_penalty
        quality_score = max(0, min(1, quality_score))  # Ensure between 0 and 1
        
        return quality_score
    
    def _get_schema_for_type(self, data_type: str) -> Dict[str, Any]:
        """
        Get schema definition for data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            Schema definition
        """
        schemas = {
            'drugs': {
                'type': 'object',
                'properties': {
                    'DrugID': {'type': 'string', 'description': 'Unique drug identifier'},
                    'DrugName': {'type': 'string', 'description': 'Drug name'},
                    'standardized_drug_id': {'type': 'object', 'description': 'Standardized drug ID information'},
                    'standardized_drug_name': {'type': 'object', 'description': 'Standardized drug name information'},
                    'processed_targets': {'type': 'object', 'description': 'Processed target information'},
                    'processed_mechanisms': {'type': 'object', 'description': 'Processed mechanism information'},
                    'processed_categories': {'type': 'object', 'description': 'Processed category information'},
                    'processed_conditions': {'type': 'object', 'description': 'Processed condition information'}
                },
                'required': ['DrugID', 'DrugName']
            },
            'diseases': {
                'type': 'object',
                'properties': {
                    'DiseaseID': {'type': 'string', 'description': 'Unique disease identifier'},
                    'DiseaseName': {'type': 'string', 'description': 'Disease name'},
                    'standardized_disease_id': {'type': 'object', 'description': 'Standardized disease ID information'},
                    'standardized_disease_name': {'type': 'object', 'description': 'Standardized disease name information'},
                    'processed_description': {'type': 'object', 'description': 'Processed description information'},
                    'processed_categories': {'type': 'object', 'description': 'Processed category information'},
                    'processed_pathways': {'type': 'object', 'description': 'Processed pathway information'}
                },
                'required': ['DiseaseID', 'DiseaseName']
            },
            'mappings': {
                'type': 'object',
                'properties': {
                    'DrugID': {'type': 'string', 'description': 'Drug identifier'},
                    'DiseaseID': {'type': 'string', 'description': 'Disease identifier'},
                    'standardized_drug_id': {'type': 'object', 'description': 'Standardized drug ID information'},
                    'standardized_disease_id': {'type': 'object', 'description': 'Standardized disease ID information'},
                    'mapping_key': {'type': 'string', 'description': 'Unique mapping key'},
                    'relationship_info': {'type': 'object', 'description': 'Relationship information'},
                    'validation_info': {'type': 'object', 'description': 'Validation information'}
                },
                'required': ['DrugID', 'DiseaseID', 'mapping_key']
            }
        }
        
        return schemas.get(data_type, {})
