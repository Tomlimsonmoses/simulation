# PK-PD Data Ingestion System

A comprehensive, extensible data ingestion system for consolidating drug and biological system interaction data from multiple sources.

## Overview

This system provides a robust framework for:
- **Multi-source data ingestion** (CSV, JSON, APIs, databases)
- **Data processing and standardization** 
- **Consolidation and deduplication**
- **Quality validation and assessment**
- **Export to JSON format for model integration**

## Architecture

### Core Components

1. **Ingestion Framework** (`src/data/ingestion/`)
   - `BaseIngestor`: Abstract base class for all ingestors
   - `CSVIngestor`: Specialized CSV file ingestor
   - `ConsolidationEngine`: Merges data from multiple sources

2. **Data Processors** (`src/data/processors/`)
   - `DrugProcessor`: Standardizes drug data (IDs, targets, mechanisms)
   - `DiseaseProcessor`: Standardizes disease data (IDs, classifications, pathways)
   - `MappingProcessor`: Processes drug-disease relationships

3. **Validation System** (`src/data/validators/`)
   - `SchemaValidator`: Validates data structure and types
   - `QualityValidator`: Assesses data quality metrics

4. **Export System** (`src/data/exporters/`)
   - `JSONExporter`: Exports consolidated data with proper schema

5. **Configuration** (`src/data/config/`)
   - `ConfigManager`: Manages system configuration
   - `DataSourceConfig`: Manages data source configurations

## Current Datasets

The system is configured to work with your current datasets:

- **Drugs**: `C:/Users/sparta/Downloads/archive (1)/drugsInfo.csv`
  - 1,487 drugs with IDs, names, descriptions, targets, mechanisms, SMILES
- **Diseases**: `C:/Users/sparta/Downloads/archive (1)/diseasesInfo.csv`
  - 1,575 diseases with IDs, names, descriptions, categories, pathways
- **Mappings**: `C:/Users/sparta/Downloads/archive (1)/mapping.csv`
  - 42,202 drug-disease relationships

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy chardet pyyaml
```

### 2. Run the Test

```bash
cd c:/Users/sparta/WORK/projects/MEDAI/PK-PD
python test_data_ingestion.py
```

This will:
- Test all dependencies
- Check dataset files
- Run the complete ingestion pipeline
- Generate output files in `test_output/`

### 3. Review Results

The test will show:
- Number of records processed from each source
- Data quality assessment
- Validation results
- Exported file locations

## Output Structure

The system generates structured JSON output:

```
output/
├── drugs_consolidated.json          # Main drug dataset
├── drugs_summary.json               # Statistics and metrics
├── drugs_metadata.json              # Processing metadata
├── drugs_schema.json                # Data schema
├── diseases_consolidated.json       # Main disease dataset
├── diseases_summary.json            # Statistics and metrics
├── diseases_metadata.json           # Processing metadata
├── diseases_schema.json             # Data schema
├── mappings_consolidated.json       # Main mapping dataset
├── mappings_summary.json            # Statistics and metrics
├── mappings_metadata.json           # Processing metadata
└── mappings_schema.json             # Data schema
```

## Data Processing Features

### Drug Data Standardization

- **ID Standardization**: Recognizes DrugBank, ChEMBL, PubChem formats
- **Target Processing**: Identifies UniProt, gene symbol, Ensembl IDs
- **Mechanism Classification**: Categorizes mechanisms (inhibitor, activator, etc.)
- **Category Normalization**: Cleans and deduplicates drug categories
- **SMILES Validation**: Validates chemical structure notation

### Disease Data Standardization

- **ID Standardization**: Recognizes MeSH, OMIM, ICD formats
- **Name Processing**: Extracts synonyms and search terms
- **Description Analysis**: Summarizes and extracts medical terms
- **Category Mapping**: Maps to broad disease categories
- **Pathway Processing**: Identifies key biological pathways

### Relationship Processing

- **Mapping Validation**: Ensures valid drug-disease pairs
- **Relationship Classification**: Identifies treatment, prevention, etc.
- **Deduplication**: Removes duplicate mappings
- **Quality Scoring**: Assesses relationship confidence

## Adding New Data Sources

### 1. CSV Files

```python
from data.config import DataSourceConfig, DataFormat, CSVOptions

source_config = DataSourceConfig(
    name="new_drug_data",
    data_type="drugs",
    format=DataFormat.CSV,
    location="path/to/new_drugs.csv",
    priority=2,
    options=CSVOptions(
        encoding="utf-8",
        delimiter=",",
        header_row=0
    )
)
```

### 2. Custom Ingestors

Create a new ingestor by extending `BaseIngestor`:

```python
from data.ingestion.base_ingestor import BaseIngestor

class CustomIngestor(BaseIngestor):
    def connect(self):
        # Implement connection logic
        pass
    
    def ingest_data(self):
        # Implement ingestion logic
        pass
    
    def validate_record(self, record):
        # Implement validation logic
        pass
```

### 3. Custom Processors

Create a new processor by extending `BaseProcessor`:

```python
from data.ingestion.base_ingestor import BaseProcessor

class CustomProcessor(BaseProcessor):
    def process(self, data):
        # Implement processing logic
        return processed_data
    
    def get_output_schema(self):
        # Define output schema
        return schema_dict
```

## Configuration

### System Configuration

Create `config.yaml`:

```yaml
processing:
  enable_validation: true
  enable_quality_check: true
  enable_deduplication: true
  chunk_size: 1000
  max_workers: 4

output:
  output_dir: "output"
  format: "json"
  compress: false
  include_metadata: true

logging:
  level: "INFO"
  file_path: "data_ingestion.log"
```

### Data Sources Configuration

Create `data_sources.yaml`:

```yaml
data_sources:
  - name: primary_drugs
    data_type: drugs
    format: csv
    location: "data/drugs.csv"
    priority: 1
    enabled: true
    options:
      encoding: utf-8
      delimiter: ","
      header_row: 0
```

## Quality Assessment

The system provides comprehensive quality metrics:

- **Completeness**: Field coverage and missing data analysis
- **Consistency**: Format consistency and value standardization
- **Uniqueness**: Duplicate detection and ID validation
- **Validity**: Schema compliance and data format validation

Quality grades: `excellent` (>0.95), `good` (>0.85), `acceptable` (>0.70), `poor` (>0.50)

## Integration with PK-PD Models

The exported JSON files are structured for easy integration with your existing PK-PD models:

```json
{
  "metadata": {
    "data_type": "drugs",
    "total_records": 1487,
    "quality_score": 0.92
  },
  "data": [
    {
      "DrugID": "DB14509",
      "standardized_drug_id": {
        "original_id": "DB14509",
        "standardized_id": "DB14509",
        "source": "drugbank",
        "is_valid": true
      },
      "processed_targets": {
        "uniprot_targets": ["O14732", "P29218"],
        "target_count": 4
      }
    }
  ]
}
```

## Extending the System

### Adding New Data Types

1. Create a processor for the new data type
2. Register it with the consolidation engine
3. Add validation schemas
4. Update configuration

### Adding New Validation Rules

```python
# Custom validation rule
def validate_custom_field(record):
    if 'custom_field' in record:
        return len(record['custom_field']) > 5
    return True

# Add to schema validator
schema_validator.add_custom_rule('custom_field', validate_custom_field)
```

### Adding New Export Formats

```python
from data.exporters.json_exporter import JSONExporter

class CustomExporter(JSONExporter):
    def export_consolidated_data(self, result, data_type):
        # Implement custom export logic
        pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **File Not Found**: Check file paths and permissions
3. **Encoding Issues**: Specify correct encoding in CSV options
4. **Memory Issues**: Reduce chunk_size for large files
5. **Validation Failures**: Check data format and required fields

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

- Increase `max_workers` for parallel processing
- Adjust `chunk_size` based on available memory
- Enable compression for large output files
- Use SSD storage for better I/O performance

## Next Steps

1. **Test with Current Data**: Run `python test_data_ingestion.py`
2. **Review Output**: Check generated JSON files
3. **Integrate with Models**: Use exported data in PK-PD models
4. **Add More Sources**: Configure additional data sources
5. **Customize Processing**: Add domain-specific processing rules

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the log files generated during processing
3. Examine the validation results for data quality issues
4. Test with smaller datasets first

---

This system provides a solid foundation for consolidating drug and biological system interaction data from multiple sources, ensuring high-quality, standardized data for your PK-PD modeling pipeline.
