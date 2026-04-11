"""
Main data ingestion orchestration script.

This module provides the main entry point for the data ingestion
system, orchestrating the entire pipeline from ingestion to export.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from data.ingestion import CSVIngestor, ConsolidationEngine, ConsolidationConfig
from data.processors import DrugProcessor, DiseaseProcessor, MappingProcessor
from data.exporters import JSONExporter
from data.validators import SchemaValidator, QualityValidator
from data.config import ConfigManager, DataSourceConfig, DataFormat, CSVOptions


class DataIngestionOrchestrator:
    """
    Main orchestrator for the data ingestion pipeline.
    
    Coordinates all components of the ingestion system to
    process data from multiple sources and export consolidated results.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config_file: Path to configuration file
        """
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration
        self.config_manager = ConfigManager(config_file)
        
        # Initialize components
        self.consolidation_engine = ConsolidationEngine(
            ConsolidationConfig(
                merge_strategy="merge",
                conflict_resolution="highest_priority",
                deduplication=True,
                validation_enabled=True
            )
        )
        
        # Initialize processors
        self.drug_processor = DrugProcessor()
        self.disease_processor = DiseaseProcessor()
        self.mapping_processor = MappingProcessor()
        
        # Register processors with consolidation engine
        self.consolidation_engine.register_processor('drugs', self.drug_processor)
        self.consolidation_engine.register_processor('diseases', self.disease_processor)
        self.consolidation_engine.register_processor('mappings', self.mapping_processor)
        
        # Initialize validators
        self.schema_validator = SchemaValidator()
        self.quality_validator = QualityValidator()
        
        # Initialize exporter
        output_config = self.config_manager.get_output_config()
        self.exporter = JSONExporter(
            output_dir=output_config.output_dir,
            compress=output_config.compress
        )
        
        self.logger.info("Data ingestion orchestrator initialized")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('data_ingestion.log')
            ]
        )
    
    def run_ingestion_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete data ingestion pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting data ingestion pipeline")
        
        pipeline_results = {
            'success': True,
            'data_types_processed': [],
            'ingestion_results': {},
            'consolidation_results': {},
            'export_results': {},
            'validation_results': {},
            'quality_results': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Get data source configurations
            data_sources_config = self.config_manager.get_data_sources_config()
            
            # Process each data type
            for data_type in ['drugs', 'diseases', 'mappings']:
                try:
                    self.logger.info(f"Processing {data_type} data")
                    
                    # Get sources for this data type
                    sources = self._get_sources_for_type(data_sources_config, data_type)
                    
                    if not sources:
                        self.logger.warning(f"No data sources found for {data_type}")
                        continue
                    
                    # Ingest data from all sources
                    ingestion_results = self._ingest_from_sources(sources, data_type)
                    pipeline_results['ingestion_results'][data_type] = ingestion_results
                    
                    # Consolidate data
                    consolidation_result = self.consolidation_engine.consolidate_data(
                        ingestion_results, data_type
                    )
                    pipeline_results['consolidation_results'][data_type] = consolidation_result
                    
                    # Validate consolidated data
                    validation_result = self._validate_consolidated_data(
                        consolidation_result, data_type
                    )
                    pipeline_results['validation_results'][data_type] = validation_result
                    
                    # Assess data quality
                    quality_result = self.quality_validator.assess_data_quality(
                        consolidation_result.consolidated_data.get(data_type, []),
                        data_type
                    )
                    pipeline_results['quality_results'][data_type] = quality_result
                    
                    # Export data
                    export_result = self.exporter.export_consolidated_data(
                        consolidation_result, data_type
                    )
                    pipeline_results['export_results'][data_type] = export_result
                    
                    pipeline_results['data_types_processed'].append(data_type)
                    
                    self.logger.info(f"Successfully processed {data_type} data")
                    
                except Exception as e:
                    error_msg = f"Failed to process {data_type} data: {str(e)}"
                    self.logger.error(error_msg)
                    pipeline_results['errors'].append(error_msg)
                    pipeline_results['success'] = False
            
            # Generate summary
            pipeline_results['summary'] = self._generate_pipeline_summary(pipeline_results)
            
            self.logger.info("Data ingestion pipeline completed")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            pipeline_results['errors'].append(error_msg)
            pipeline_results['success'] = False
        
        return pipeline_results
    
    def _get_sources_for_type(self, data_sources_config: Dict[str, Any], data_type: str) -> List[Dict[str, Any]]:
        """Get data source configurations for a specific data type."""
        all_sources = data_sources_config.get('data_sources', [])
        return [source for source in all_sources if source.get('data_type') == data_type and source.get('enabled', True)]
    
    def _ingest_from_sources(self, sources: List[Dict[str, Any]], data_type: str) -> List[Any]:
        """Ingest data from multiple sources."""
        ingestion_results = []
        
        for source_config in sources:
            try:
                self.logger.info(f"Ingesting from source: {source_config['name']}")
                
                # Create data source object
                from data.ingestion.base_ingestor import DataSource, DataSourceType
                
                data_source = DataSource(
                    name=source_config['name'],
                    source_type=DataSourceType.CSV,
                    location=source_config['location'],
                    options=source_config.get('options', {}),
                    priority=source_config.get('priority', 1),
                    enabled=source_config.get('enabled', True)
                )
                
                # Create ingestor and ingest data
                ingestor = CSVIngestor(data_source)
                result = ingestor.ingest_data()
                
                # Set source priority for consolidation
                self.consolidation_engine.set_source_priority(
                    source_config['name'], 
                    source_config.get('priority', 1)
                )
                
                ingestion_results.append(result)
                
                self.logger.info(f"Ingested {result.records_valid} records from {source_config['name']}")
                
            except Exception as e:
                error_msg = f"Failed to ingest from {source_config['name']}: {str(e)}"
                self.logger.error(error_msg)
                # Continue with other sources
        
        return ingestion_results
    
    def _validate_consolidated_data(self, consolidation_result: Any, data_type: str) -> Dict[str, Any]:
        """Validate consolidated data."""
        try:
            data = consolidation_result.consolidated_data.get(data_type, [])
            validation_result = self.schema_validator.validate_dataset(data, data_type)
            
            self.logger.info(f"Validation completed for {data_type}: {validation_result['valid_records']}/{validation_result['total_records']} valid")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation failed for {data_type}: {str(e)}")
            return {'total_records': 0, 'valid_records': 0, 'errors': [str(e)]}
    
    def _generate_pipeline_summary(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of pipeline results."""
        summary = {
            'pipeline_success': pipeline_results['success'],
            'data_types_processed': pipeline_results['data_types_processed'],
            'total_errors': len(pipeline_results['errors']),
            'total_warnings': len(pipeline_results['warnings']),
            'data_summary': {}
        }
        
        for data_type in pipeline_results['data_types_processed']:
            # Ingestion summary
            ingestion_results = pipeline_results['ingestion_results'].get(data_type, [])
            total_processed = sum(r.records_processed for r in ingestion_results)
            total_valid = sum(r.records_valid for r in ingestion_results)
            
            # Consolidation summary
            consolidation_result = pipeline_results['consolidation_results'].get(data_type)
            consolidated_count = consolidation_result.total_records_consolidated if consolidation_result else 0
            
            # Quality summary
            quality_result = pipeline_results['quality_results'].get(data_type, {})
            quality_score = quality_result.get('overall_score', 0.0) if isinstance(quality_result, dict) else 0.0
            quality_grade = quality_result.get('overall_grade', 'unknown') if isinstance(quality_result, dict) else 'unknown'
            
            summary['data_summary'][data_type] = {
                'sources_processed': len(ingestion_results),
                'total_records_processed': total_processed,
                'total_records_valid': total_valid,
                'records_consolidated': consolidated_count,
                'quality_score': quality_score,
                'quality_grade': quality_grade
            }
        
        return summary
    
    def run_test_with_current_datasets(self) -> Dict[str, Any]:
        """
        Run a test with the current datasets from the Downloads folder.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("Running test with current datasets")
        
        # Create test configuration for current datasets
        test_sources = {
            'data_sources': [
                {
                    'name': 'test_drugs',
                    'data_type': 'drugs',
                    'format': 'csv',
                    'location': 'C:/Users/sparta/Downloads/archive (1)/drugsInfo.csv',
                    'priority': 1,
                    'enabled': True,
                    'options': {
                        'encoding': 'utf-8',
                        'delimiter': ',',
                        'header_row': 0
                    }
                },
                {
                    'name': 'test_diseases',
                    'data_type': 'diseases',
                    'format': 'csv',
                    'location': 'C:/Users/sparta/Downloads/archive (1)/diseasesInfo.csv',
                    'priority': 1,
                    'enabled': True,
                    'options': {
                        'encoding': 'utf-8',
                        'delimiter': ',',
                        'header_row': 0
                    }
                },
                {
                    'name': 'test_mappings',
                    'data_type': 'mappings',
                    'format': 'csv',
                    'location': 'C:/Users/sparta/Downloads/archive (1)/mapping.csv',
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
        
        # Temporarily update configuration
        original_sources = self.config_manager.get_data_sources_config()
        
        try:
            # Override the data sources configuration
            self.config_manager.data_sources = test_sources
            
            # Run the pipeline
            results = self.run_ingestion_pipeline()
            
            self.logger.info("Test completed successfully")
            return results
            
        finally:
            # Restore original configuration
            self.config_manager.data_sources = original_sources


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PK-PD Data Ingestion System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Run with test datasets')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = DataIngestionOrchestrator(args.config)
    
    # Update output directory if specified
    if args.output_dir:
        orchestrator.config_manager.config.output.output_dir = args.output_dir
        orchestrator.exporter.output_dir = Path(args.output_dir)
    
    # Run pipeline
    if args.test:
        results = orchestrator.run_test_with_current_datasets()
    else:
        results = orchestrator.run_ingestion_pipeline()
    
    # Print summary
    print("\n" + "="*50)
    print("DATA INGESTION PIPELINE SUMMARY")
    print("="*50)
    
    if results['success']:
        print("✅ Pipeline completed successfully")
    else:
        print("❌ Pipeline completed with errors")
    
    print(f"Data types processed: {', '.join(results['data_types_processed'])}")
    print(f"Total errors: {len(results['errors'])}")
    print(f"Total warnings: {len(results['warnings'])}")
    
    if 'summary' in results:
        print("\nData Summary:")
        for data_type, summary in results['summary']['data_summary'].items():
            print(f"  {data_type}:")
            print(f"    Records processed: {summary['total_records_processed']}")
            print(f"    Records consolidated: {summary['records_consolidated']}")
            print(f"    Quality grade: {summary['quality_grade']} ({summary['quality_score']:.2f})")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  ❌ {error}")
        if len(results['errors']) > 5:
            print(f"  ... and {len(results['errors']) - 5} more errors")
    
    print("\nOutput files created in:", args.output_dir)
    
    return results


if __name__ == "__main__":
    main()
