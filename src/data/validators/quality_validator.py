"""
Quality validator for data quality assessment.

This module provides data quality checking functionality to assess
the overall quality, completeness, and consistency of datasets.
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)


class QualityValidator:
    """
    Validator for data quality assessment.
    
    Assesses data quality metrics including completeness,
    consistency, accuracy, and reliability.
    """
    
    def __init__(self):
        """Initialize quality validator."""
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.quality_thresholds = {
            'completeness': {
                'excellent': 0.95,
                'good': 0.85,
                'acceptable': 0.70,
                'poor': 0.50
            },
            'consistency': {
                'excellent': 0.95,
                'good': 0.85,
                'acceptable': 0.70,
                'poor': 0.50
            },
            'uniqueness': {
                'excellent': 0.95,
                'good': 0.85,
                'acceptable': 0.70,
                'poor': 0.50
            }
        }
    
    def assess_data_quality(self, data: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """
        Assess overall data quality.
        
        Args:
            data: List of data records
            data_type: Type of data (drugs, diseases, mappings)
            
        Returns:
            Dictionary with quality assessment results
        """
        self.logger.info(f"Assessing quality for {len(data)} {data_type} records")
        
        quality_assessment = {
            'data_type': data_type,
            'total_records': len(data),
            'overall_score': 0.0,
            'overall_grade': 'unknown',
            'dimensions': {},
            'issues': [],
            'recommendations': []
        }
        
        if not data:
            quality_assessment['overall_score'] = 0.0
            quality_assessment['overall_grade'] = 'poor'
            quality_assessment['issues'].append('No data to assess')
            return quality_assessment
        
        # Assess different quality dimensions
        completeness_score = self._assess_completeness(data, data_type)
        consistency_score = self._assess_consistency(data, data_type)
        uniqueness_score = self._assess_uniqueness(data, data_type)
        validity_score = self._assess_validity(data, data_type)
        
        quality_assessment['dimensions'] = {
            'completeness': completeness_score,
            'consistency': consistency_score,
            'uniqueness': uniqueness_score,
            'validity': validity_score
        }
        
        # Calculate overall score (weighted average)
        weights = {
            'completeness': 0.3,
            'consistency': 0.25,
            'uniqueness': 0.25,
            'validity': 0.2
        }
        
        overall_score = sum(
            (score['score'] if isinstance(score, dict) else float(score)) * weights[dim] 
            for dim, score in quality_assessment['dimensions'].items()
        )
        
        quality_assessment['overall_score'] = overall_score
        try:
            quality_grade = self._get_quality_grade(overall_score)
        except (TypeError, KeyError, AttributeError) as e:
            self.logger.warning(f"Quality assessment error for {data_type}: {str(e)}")
            quality_grade = 'error'
        
        quality_assessment['overall_grade'] = quality_grade
        
        # Generate issues and recommendations
        quality_assessment['issues'] = self._identify_quality_issues(
            quality_assessment['dimensions'], data_type
        )
        quality_assessment['recommendations'] = self._generate_recommendations(
            quality_assessment['dimensions'], data_type
        )
        
        self.logger.info(f"Quality assessment complete: {quality_assessment['overall_grade']} ({overall_score:.2f})")
        return quality_assessment
    
    def _assess_completeness(self, data: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """Assess data completeness."""
        if not data:
            return {'score': 0.0, 'field_completeness': {}}
        
        # Analyze field completeness
        field_stats = {}
        all_fields = set()
        
        for record in data:
            all_fields.update(record.keys())
        
        for field in all_fields:
            non_null_count = sum(1 for record in data if record.get(field) is not None and str(record.get(field)).strip())
            non_empty_count = sum(1 for record in data if record.get(field) is not None and str(record.get(field)).strip() != '')
            
            field_stats[field] = {
                'total_records': len(data),
                'non_null_count': non_null_count,
                'non_empty_count': non_empty_count,
                'null_ratio': (len(data) - non_null_count) / len(data),
                'empty_ratio': (len(data) - non_empty_count) / len(data),
                'completeness_score': non_empty_count / len(data)
            }
        
        # Calculate overall completeness score
        if data_type == 'drugs':
            required_fields = ['DrugID', 'DrugName']
            important_fields = ['DrugDescription', 'DrugTarget', 'DrugMechanism']
        elif data_type == 'diseases':
            required_fields = ['DiseaseID', 'DiseaseName']
            important_fields = ['DiseaseDescription', 'SlimMapping']
        elif data_type == 'mappings':
            required_fields = ['DrugID', 'DiseaseID']
            important_fields = []
        else:
            required_fields = []
            important_fields = []
        
        # Weight required fields more heavily
        required_score = 0.0
        if required_fields:
            required_scores = [
                field_stats[field]['completeness_score'] 
                for field in required_fields if field in field_stats
            ]
            required_score = statistics.mean(required_scores) if required_scores else 0.0
        
        # Weight important fields moderately
        important_score = 0.0
        if important_fields:
            important_scores = [
                field_stats[field]['completeness_score'] 
                for field in important_fields if field in field_stats
            ]
            important_score = statistics.mean(important_scores) if important_scores else 0.0
        
        # Weight other fields lightly
        other_fields = [f for f in all_fields if f not in required_fields and f not in important_fields]
        other_score = 0.0
        if other_fields:
            other_scores = [
                field_stats[field]['completeness_score'] 
                for field in other_fields if field in field_stats
            ]
            other_score = statistics.mean(other_scores) if other_scores else 0.0
        
        # Calculate weighted completeness score
        overall_score = (
            float(required_score) * 0.5 +
            float(important_score) * 0.3 +
            float(other_score) * 0.2
        )
        
        return {
            'score': overall_score,
            'field_completeness': field_stats,
            'required_fields_completeness': required_score,
            'important_fields_completeness': important_score,
            'other_fields_completeness': other_score
        }
    
    def _assess_consistency(self, data: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """Assess data consistency."""
        if not data:
            return {'score': 0.0, 'consistency_issues': []}
        
        consistency_issues = []
        consistency_scores = []
        
        # Check ID format consistency
        if data_type == 'drugs':
            id_field = 'DrugID'
            id_pattern = r'^[A-Z]{2,}\d+$'
        elif data_type == 'diseases':
            id_field = 'DiseaseID'
            id_pattern = r'^[A-Z]+:[A-Z0-9]+$'
        elif data_type == 'mappings':
            id_field = 'DrugID'  # Check DrugID format
            id_pattern = r'^[A-Z]{2,}\d+$'
        else:
            id_field = None
            id_pattern = None
        
        if id_field and id_pattern:
            id_consistency = self._check_id_consistency(data, id_field, id_pattern)
            consistency_scores.append(id_consistency['score'])
            consistency_issues.extend(id_consistency['issues'])
        
        # Check data type consistency
        type_consistency = self._check_type_consistency(data)
        consistency_scores.append(type_consistency['score'])
        consistency_issues.extend(type_consistency['issues'])
        
        # Check value consistency for categorical fields
        if data_type in ['drugs', 'diseases']:
            categorical_consistency = self._check_categorical_consistency(data, data_type)
            consistency_scores.append(categorical_consistency['score'])
            consistency_issues.extend(categorical_consistency['issues'])
        
        # Calculate overall consistency score
        overall_score = statistics.mean(consistency_scores) if consistency_scores else 0.0
        
        return {
            'score': overall_score,
            'consistency_issues': consistency_issues[:20],  # Limit to top 20 issues
            'id_consistency': id_consistency if id_field else None,
            'type_consistency': type_consistency,
            'categorical_consistency': categorical_consistency if data_type in ['drugs', 'diseases'] else None
        }
    
    def _assess_uniqueness(self, data: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """Assess data uniqueness."""
        if not data:
            return {'score': 0.0, 'duplicate_analysis': {}}
        
        duplicate_analysis = {}
        
        # Check for duplicate IDs
        if data_type == 'drugs':
            id_field = 'DrugID'
        elif data_type == 'diseases':
            id_field = 'DiseaseID'
        elif data_type == 'mappings':
            id_field = 'mapping_key'  # Use mapping key for uniqueness
        else:
            id_field = None
        
        if id_field:
            id_duplicates = self._check_duplicate_ids(data, id_field)
            duplicate_analysis['id_duplicates'] = id_duplicates
        
        # Check for duplicate names (for drugs and diseases)
        if data_type in ['drugs', 'diseases']:
            name_field = 'DrugName' if data_type == 'drugs' else 'DiseaseName'
            name_duplicates = self._check_duplicate_names(data, name_field)
            duplicate_analysis['name_duplicates'] = name_duplicates
        
        # Calculate uniqueness score
        total_records = len(data)
        unique_records = total_records
        
        if id_field and id_field in duplicate_analysis:
            unique_records -= duplicate_analysis['id_duplicates']['duplicate_count']
        
        uniqueness_score = unique_records / total_records if total_records > 0 else 0.0
        
        return {
            'score': uniqueness_score,
            'duplicate_analysis': duplicate_analysis,
            'unique_records': unique_records,
            'total_records': total_records
        }
    
    def _assess_validity(self, data: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """Assess data validity."""
        if not data:
            return {'score': 0.0, 'validity_issues': []}
        
        validity_issues = []
        valid_records = 0
        
        for i, record in enumerate(data):
            record_issues = []
            
            # Check basic validity
            if data_type == 'drugs':
                issues = self._validate_drug_record(record)
            elif data_type == 'diseases':
                issues = self._validate_disease_record(record)
            elif data_type == 'mappings':
                issues = self._validate_mapping_record(record)
            else:
                issues = []
            
            record_issues.extend(issues)
            
            if not record_issues:
                valid_records += 1
            else:
                validity_issues.extend([f"Record {i}: {issue}" for issue in record_issues])
        
        validity_score = valid_records / len(data) if data else 0.0
        
        return {
            'score': validity_score,
            'validity_issues': validity_issues[:50],  # Limit to first 50 issues
            'valid_records': valid_records,
            'invalid_records': len(data) - valid_records
        }
    
    def _check_id_consistency(self, data: List[Dict[str, Any]], id_field: str, pattern: str) -> Dict[str, Any]:
        """Check ID format consistency."""
        issues = []
        matching_ids = 0
        
        for record in data:
            if id_field in record:
                id_value = str(record[id_field])
                if re.match(pattern, id_value):
                    matching_ids += 1
                else:
                    issues.append(f"Invalid {id_field} format: {id_value}")
        
        consistency_score = matching_ids / len(data) if data else 0.0
        
        return {
            'score': consistency_score,
            'matching_ids': matching_ids,
            'total_ids': len(data),
            'issues': issues[:10]  # Limit to first 10 issues
        }
    
    def _check_type_consistency(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check data type consistency across records."""
        issues = []
        type_consistency_scores = []
        
        # Get all fields
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())
        
        for field in all_fields:
            types_found = set()
            for record in data:
                if field in record and record[field] is not None:
                    types_found.add(type(record[field]).__name__)
            
            if len(types_found) > 1:
                issues.append(f"Field {field} has inconsistent types: {types_found}")
                type_consistency_scores.append(0.5)  # Partial penalty
            else:
                type_consistency_scores.append(1.0)
        
        overall_score = statistics.mean(type_consistency_scores) if type_consistency_scores else 1.0
        
        return {
            'score': overall_score,
            'inconsistent_fields': len([i for i in issues if 'inconsistent types' in i]),
            'issues': issues[:10]
        }
    
    def _check_categorical_consistency(self, data: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """Check consistency of categorical fields."""
        issues = []
        
        if data_type == 'drugs':
            categorical_fields = ['DrugCategories']
        elif data_type == 'diseases':
            categorical_fields = ['SlimMapping']
        else:
            return {'score': 1.0, 'issues': []}
        
        for field in categorical_fields:
            # Collect all unique values
            all_values = []
            for record in data:
                if field in record:
                    value = record[field]
                    if isinstance(value, list):
                        all_values.extend(value)
                    elif isinstance(value, str):
                        # Try to parse as list if it looks like one
                        try:
                            import ast
                            parsed = ast.literal_eval(value)
                            if isinstance(parsed, list):
                                all_values.extend(parsed)
                            else:
                                all_values.append(value)
                        except:
                            all_values.append(value)
                    else:
                        all_values.append(str(value))
            
            # Check for similar values that might be duplicates
            normalized_values = [v.lower().strip() for v in all_values if v]
            value_counts = Counter(normalized_values)
            
            # Look for potential duplicates with slight variations
            for value, count in value_counts.items():
                if count > 1 and len(value) > 10:
                    # Check for very similar values
                    similar_values = [v for v in value_counts.keys() 
                                    if v != value and (value in v or v in value) and abs(len(v) - len(value)) < 5]
                    if similar_values:
                        issues.append(f"Field {field} has potentially duplicate values: {value} vs {similar_values}")
        
        consistency_score = 1.0 - (len(issues) * 0.1)  # Penalize for each issue
        consistency_score = max(0.0, consistency_score)
        
        return {
            'score': consistency_score,
            'issues': issues[:10]
        }
    
    def _check_duplicate_ids(self, data: List[Dict[str, Any]], id_field: str) -> Dict[str, Any]:
        """Check for duplicate IDs."""
        id_counts = Counter()
        for record in data:
            if id_field in record:
                id_counts[record[id_field]] += 1
        
        duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}
        duplicate_count = sum(count - 1 for count in duplicates.values())
        
        return {
            'duplicate_count': duplicate_count,
            'unique_ids': len(id_counts),
            'total_records': len(data),
            'duplicate_ids': dict(list(duplicates.items())[:10])  # Limit to first 10
        }
    
    def _check_duplicate_names(self, data: List[Dict[str, Any]], name_field: str) -> Dict[str, Any]:
        """Check for duplicate names."""
        name_counts = Counter()
        for record in data:
            if name_field in record:
                name = str(record[name_field]).strip().lower()
                if name:
                    name_counts[name] += 1
        
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        duplicate_count = sum(count - 1 for count in duplicates.values())
        
        return {
            'duplicate_count': duplicate_count,
            'unique_names': len(name_counts),
            'total_records': len(data),
            'duplicate_names': dict(list(duplicates.items())[:10])  # Limit to first 10
        }
    
    def _validate_drug_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate a drug record."""
        issues = []
        
        # Check required fields
        if not record.get('DrugID'):
            issues.append("Missing DrugID")
        if not record.get('DrugName'):
            issues.append("Missing DrugName")
        
        # Check SMILES format if present
        if 'DrugSmile' in record and record['DrugSmile']:
            smiles = str(record['DrugSmile']).strip()
            if smiles and not re.match(r'^[BCNOPSFI\[\]()=#0123456789+.\-@]+$', smiles):
                issues.append("Invalid SMILES format")
        
        return issues
    
    def _validate_disease_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate a disease record."""
        issues = []
        
        # Check required fields
        if not record.get('DiseaseID'):
            issues.append("Missing DiseaseID")
        if not record.get('DiseaseName'):
            issues.append("Missing DiseaseName")
        
        return issues
    
    def _validate_mapping_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate a mapping record."""
        issues = []
        
        # Check required fields
        if not record.get('DrugID'):
            issues.append("Missing DrugID")
        if not record.get('DiseaseID'):
            issues.append("Missing DiseaseID")
        
        return issues
    
    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade from score."""
        thresholds = self.quality_thresholds['completeness']  # Use completeness thresholds
        
        if score >= thresholds['excellent']:
            return 'excellent'
        elif score >= thresholds['good']:
            return 'good'
        elif score >= thresholds['acceptable']:
            return 'acceptable'
        elif score >= thresholds['poor']:
            return 'poor'
        else:
            return 'very_poor'
    
    def _identify_quality_issues(self, dimensions: Dict[str, Any], data_type: str) -> List[str]:
        """Identify quality issues from dimension scores."""
        issues = []
        
        for dimension, result in dimensions.items():
            if isinstance(result, dict) and 'score' in result:
                score = result['score']
                if score < 0.7:
                    issues.append(f"Low {dimension} score: {score:.2f}")
            elif isinstance(result, (int, float)):
                score = float(result)
                if score < 0.7:
                    issues.append(f"Low {dimension} score: {score:.2f}")
        
        return issues
    
    def _generate_recommendations(self, dimensions: Dict[str, Any], data_type: str) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Check completeness
        completeness = dimensions.get('completeness', {})
        if isinstance(completeness, dict) and completeness.get('score', 0) < 0.8:
            recommendations.append("Improve data completeness by filling missing fields")
            
            field_completeness = completeness.get('field_completeness', {})
            low_completeness_fields = [
                field for field, stats in field_completeness.items()
                if isinstance(stats, dict) and stats.get('completeness_score', 0) < 0.5
            ]
            
            if low_completeness_fields:
                recommendations.append(f"Focus on improving: {', '.join(low_completeness_fields[:3])}")
        
        # Check consistency
        consistency = dimensions.get('consistency', {})
        if isinstance(consistency, dict) and consistency.get('score', 0) < 0.8:
            recommendations.append("Standardize data formats and values")
        
        # Check uniqueness
        uniqueness = dimensions.get('uniqueness', {})
        if isinstance(uniqueness, dict) and uniqueness.get('score', 0) < 0.9:
            recommendations.append("Remove duplicate records and standardize identifiers")
        
        # Check validity
        validity = dimensions.get('validity', {})
        if isinstance(validity, dict) and validity.get('score', 0) < 0.9:
            recommendations.append("Validate and correct data format errors")
        
        return recommendations
