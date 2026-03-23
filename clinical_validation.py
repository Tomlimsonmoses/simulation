"""
Clinical Trial Validation Framework for PK-PD Model
===============================================

This module provides comprehensive validation of the PK-PD model against clinical trial data.
Includes statistical validation, cross-validation, and clinical endpoint analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ClinicalTrialValidator:
    """
    Validates PK-PD model predictions against clinical trial data
    """
    
    def __init__(self, model, clinical_data_path=None):
        """
        Initialize validator with trained model and clinical data
        
        Args:
            model: Trained PK-PD model
            clinical_data_path: Path to clinical trial dataset
        """
        self.model = model
        self.clinical_data = None
        self.validation_results = {}
        
        if clinical_data_path:
            self.load_clinical_data(clinical_data_path)
        else:
            self.generate_synthetic_clinical_data()
    
    def load_clinical_data(self, data_path):
        """
        Load real clinical trial data from CSV file
        
        Expected format:
        - patient_id: Unique patient identifier
        - drug_name: Name of the drug
        - dose: Administered dose
        - age: Patient age
        - weight: Patient weight (kg)
        - gender: Patient gender (M/F)
        - disease_state: Disease condition
        - pk_observed: Observed PK concentration
        - pd_observed: Observed PD response
        - time_point: Measurement time point
        """
        try:
            self.clinical_data = pd.read_csv(data_path)
            print(f"✅ Loaded clinical data: {len(self.clinical_data)} records")
            self.validate_data_format()
        except FileNotFoundError:
            print(f"⚠️ Clinical data file not found: {data_path}")
            print("Generating synthetic clinical data for demonstration...")
            self.generate_synthetic_clinical_data()
        except Exception as e:
            print(f"❌ Error loading clinical data: {e}")
            self.generate_synthetic_clinical_data()
    
    def generate_synthetic_clinical_data(self, n_patients=200):
        """
        Generate synthetic clinical trial data for validation
        Simulates realistic clinical trial scenarios
        """
        print("📊 Generating synthetic clinical trial data...")
        
        np.random.seed(42)
        
        # Patient demographics
        patient_ids = [f"PCT_{i:04d}" for i in range(n_patients)]
        ages = np.random.normal(55, 15, n_patients).astype(int)
        ages = np.clip(ages, 18, 85)
        weights = np.random.normal(75, 20, n_patients)
        weights = np.clip(weights, 40, 150)
        genders = np.random.choice(['M', 'F'], n_patients, p=[0.52, 0.48])
        
        # Drug assignments (more realistic distribution)
        drugs = ['Acetaminophen', 'Ibuprofen', 'Metformin', 'Lisinopril', 'Atorvastatin']
        drug_choices = np.random.choice(drugs, n_patients, p=[0.25, 0.20, 0.20, 0.15, 0.20])
        
        # Disease states
        diseases = ['Hypertension', 'Diabetes', 'Hyperlipidemia', 'Arthritis', 'Healthy']
        disease_choices = np.random.choice(diseases, n_patients, p=[0.25, 0.20, 0.20, 0.15, 0.20])
        
        # Dosing regimens (realistic clinical doses)
        dose_mapping = {
            'Acetaminophen': np.random.uniform(500, 1000, n_patients),  # mg
            'Ibuprofen': np.random.uniform(200, 800, n_patients),      # mg
            'Metformin': np.random.uniform(500, 2000, n_patients),      # mg
            'Lisinopril': np.random.uniform(5, 40, n_patients),        # mg
            'Atorvastatin': np.random.uniform(10, 80, n_patients)       # mg
        }
        
        # Time points for measurements
        time_points = np.random.choice([0.5, 1, 2, 4, 6, 8, 12, 24], n_patients)
        
        # Generate realistic PK/PD values with biological variability
        pk_base = {
            'Acetaminophen': 100, 'Ibuprofen': 80, 'Metformin': 60,
            'Lisinopril': 40, 'Atorvastatin': 30
        }
        pd_base = {
            'Acetaminophen': 70, 'Ibuprofen': 60, 'Metformin': 50,
            'Lisinopril': 35, 'Atorvastatin': 45
        }
        
        pk_observed = []
        pd_observed = []
        
        for i in range(n_patients):
            drug = drug_choices[i]
            dose = dose_mapping[drug][i]
            time = time_points[i]
            
            # PK calculation with dose and time dependency
            pk_peak = pk_base[drug] * (dose / np.mean([dose_mapping[drug]]))
            pk_time_factor = np.exp(-time / 4)  # Exponential decay
            pk_variability = np.random.normal(1.0, 0.15)  # 15% CV
            pk = pk_peak * pk_time_factor * pk_variability
            
            # PD calculation with delayed response
            pd_peak = pd_base[drug] * (dose / np.mean([dose_mapping[drug]]))
            pd_time_factor = 1 - np.exp(-time / 2)  # Delayed onset
            pd_variability = np.random.normal(1.0, 0.12)  # 12% CV
            pd = pd_peak * pd_time_factor * pd_variability
            
            # Disease state modifications
            if disease_choices[i] == 'Hypertension':
                pk *= 0.9  # Reduced clearance
                pd *= 1.1  # Enhanced response
            elif disease_choices[i] == 'Diabetes':
                pk *= 0.85  # Altered metabolism
                pd *= 0.9   # Reduced response
            elif disease_choices[i] == 'Hyperlipidemia':
                pk *= 0.95
                pd *= 1.05
            
            pk_observed.append(max(0, pk))
            pd_observed.append(max(0, pd))
        
        # Create DataFrame
        self.clinical_data = pd.DataFrame({
            'patient_id': patient_ids,
            'drug_name': drug_choices,
            'dose': [dose_mapping[drug][i] for i, drug in enumerate(drug_choices)],
            'age': ages,
            'weight': weights,
            'gender': genders,
            'disease_state': disease_choices,
            'pk_observed': pk_observed,
            'pd_observed': pd_observed,
            'time_point': time_points
        })
        
        print(f"✅ Generated synthetic clinical data: {len(self.clinical_data)} records")
        print(f"   Drugs: {', '.join(drugs)}")
        print(f"   Diseases: {', '.join(diseases)}")
    
    def validate_data_format(self):
        """Validate that clinical data has required columns"""
        required_columns = [
            'patient_id', 'drug_name', 'dose', 'age', 'weight', 
            'gender', 'disease_state', 'pk_observed', 'pd_observed', 'time_point'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.clinical_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"✅ Clinical data format validation passed")
    
    def prepare_features(self, data):
        """
        Prepare features for model prediction
        
        Args:
            data: Clinical data DataFrame
            
        Returns:
            Feature matrix for model input
        """
        # One-hot encode categorical variables
        drugs_encoded = pd.get_dummies(data['drug_name'], prefix='drug')
        genders_encoded = pd.get_dummies(data['gender'], prefix='gender')
        diseases_encoded = pd.get_dummies(data['disease_state'], prefix='disease')
        
        # Combine features
        features = pd.concat([
            data[['dose', 'age', 'weight', 'time_point']],
            drugs_encoded,
            genders_encoded,
            diseases_encoded
        ], axis=1)
        
        # Ensure consistent feature ordering
        if hasattr(self.model, 'feature_names_'):
            for col in self.model.feature_names_:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.model.feature_names_]
        
        return features.values
    
    def predict_clinical_outcomes(self):
        """
        Generate model predictions for clinical data
        
        Returns:
            Tuple of (pk_predictions, pd_predictions)
        """
        if self.clinical_data is None:
            raise ValueError("No clinical data available for prediction")
        
        features = self.prepare_features(self.clinical_data)
        
        # Generate predictions
        predictions = self.model.predict(features)
        pk_pred = predictions[:, 0]
        pd_pred = predictions[:, 1]
        
        return pk_pred, pd_pred
    
    def calculate_validation_metrics(self, y_true, y_pred, metric_name):
        """
        Calculate comprehensive validation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric_name: Name of the metric (PK or PD)
            
        Returns:
            Dictionary of validation metrics
        """
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'bias': np.mean(y_pred - y_true),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1]
        }
        
        # Calculate prediction intervals
        residuals = y_true - y_pred
        metrics['residual_std'] = np.std(residuals)
        metrics['prediction_95ci'] = 1.96 * metrics['residual_std']
        
        # Clinical validation metrics
        metrics['within_20_percent'] = np.mean(np.abs(y_pred - y_true) / y_true < 0.20) * 100
        metrics['within_50_percent'] = np.mean(np.abs(y_pred - y_true) / y_true < 0.50) * 100
        
        return {f"{metric_name}_{k}": v for k, v in metrics.items()}
    
    def perform_cross_validation(self, n_folds=5):
        """
        Perform k-fold cross-validation on clinical data
        
        Args:
            n_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        if self.clinical_data is None:
            raise ValueError("No clinical data available for cross-validation")
        
        features = self.prepare_features(self.clinical_data)
        pk_true = self.clinical_data['pk_observed'].values
        pd_true = self.clinical_data['pd_observed'].values
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        pk_scores = []
        pd_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
            X_train, X_val = features[train_idx], features[val_idx]
            pk_train, pk_val = pk_true[train_idx], pk_true[val_idx]
            pd_train, pd_val = pd_true[train_idx], pd_true[val_idx]
            
            # Train model on fold
            self.model.fit(X_train, np.column_stack([pk_train, pd_train]))
            
            # Predict on validation set
            predictions = self.model.predict(X_val)
            pk_pred = predictions[:, 0]
            pd_pred = predictions[:, 1]
            
            # Calculate scores
            pk_r2 = r2_score(pk_val, pk_pred)
            pd_r2 = r2_score(pd_val, pd_pred)
            
            pk_scores.append(pk_r2)
            pd_scores.append(pd_r2)
            
            print(f"   Fold {fold+1}: PK R² = {pk_r2:.3f}, PD R² = {pd_r2:.3f}")
        
        cv_results = {
            'pk_cv_mean': np.mean(pk_scores),
            'pk_cv_std': np.std(pk_scores),
            'pd_cv_mean': np.mean(pd_scores),
            'pd_cv_std': np.std(pd_scores),
            'pk_scores': pk_scores,
            'pd_scores': pd_scores
        }
        
        return cv_results
    
    def run_comprehensive_validation(self):
        """
        Run comprehensive validation analysis
        
        Returns:
            Dictionary of all validation results
        """
        print("🔬 Starting Comprehensive Clinical Validation...")
        print("=" * 60)
        
        # Generate predictions
        pk_pred, pd_pred = self.predict_clinical_outcomes()
        pk_true = self.clinical_data['pk_observed'].values
        pd_true = self.clinical_data['pd_observed'].values
        
        # Calculate metrics
        pk_metrics = self.calculate_validation_metrics(pk_true, pk_pred, "pk")
        pd_metrics = self.calculate_validation_metrics(pd_true, pd_pred, "pd")
        
        # Cross-validation
        print("\n📊 Performing Cross-Validation...")
        cv_results = self.perform_cross_validation()
        
        # Statistical significance tests
        print("\n🧪 Performing Statistical Tests...")
        statistical_results = self.perform_statistical_tests(pk_true, pk_pred, pd_true, pd_pred)
        
        # Clinical endpoint analysis
        print("\n🏥 Analyzing Clinical Endpoints...")
        clinical_endpoints = self.analyze_clinical_endpoints(pk_true, pk_pred, pd_true, pd_pred)
        
        # Compile all results
        self.validation_results = {
            'pk_metrics': pk_metrics,
            'pd_metrics': pd_metrics,
            'cross_validation': cv_results,
            'statistical_tests': statistical_results,
            'clinical_endpoints': clinical_endpoints,
            'sample_size': len(self.clinical_data),
            'validation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return self.validation_results
    
    def perform_statistical_tests(self, pk_true, pk_pred, pd_true, pd_pred):
        """
        Perform statistical significance tests
        
        Returns:
            Dictionary of statistical test results
        """
        results = {}
        
        # Paired t-tests
        pk_t_stat, pk_p_value = stats.ttest_rel(pk_true, pk_pred)
        pd_t_stat, pd_p_value = stats.ttest_rel(pd_true, pd_pred)
        
        results['pk_paired_ttest'] = {'t_statistic': pk_t_stat, 'p_value': pk_p_value}
        results['pd_paired_ttest'] = {'t_statistic': pd_t_stat, 'p_value': pd_p_value}
        
        # Wilcoxon signed-rank tests (non-parametric)
        pk_wilcoxon, pk_w_p = stats.wilcoxon(pk_true, pk_pred)
        pd_wilcoxon, pd_w_p = stats.wilcoxon(pd_true, pd_pred)
        
        results['pk_wilcoxon'] = {'statistic': pk_wilcoxon, 'p_value': pk_w_p}
        results['pd_wilcoxon'] = {'statistic': pd_wilcoxon, 'p_value': pk_w_p}
        
        # Correlation tests
        pk_corr, pk_corr_p = stats.pearsonr(pk_true, pk_pred)
        pd_corr, pd_corr_p = stats.pearsonr(pd_true, pd_pred)
        
        results['pk_correlation'] = {'correlation': pk_corr, 'p_value': pk_corr_p}
        results['pd_correlation'] = {'correlation': pd_corr, 'p_value': pd_corr_p}
        
        return results
    
    def analyze_clinical_endpoints(self, pk_true, pk_pred, pd_true, pd_pred):
        """
        Analyze clinical endpoints and therapeutic ranges
        
        Returns:
            Dictionary of clinical endpoint analysis
        """
        results = {}
        
        # Therapeutic range analysis
        pk_therapeutic_min, pk_therapeutic_max = 50, 150  # Example therapeutic range
        pd_therapeutic_min, pd_therapeutic_max = 30, 80
        
        # Calculate percentage within therapeutic range
        pk_true_therapeutic = np.mean((pk_true >= pk_therapeutic_min) & (pk_true <= pk_therapeutic_max)) * 100
        pk_pred_therapeutic = np.mean((pk_pred >= pk_therapeutic_min) & (pk_pred <= pk_therapeutic_max)) * 100
        
        pd_true_therapeutic = np.mean((pd_true >= pd_therapeutic_min) & (pd_true <= pd_therapeutic_max)) * 100
        pd_pred_therapeutic = np.mean((pd_pred >= pd_therapeutic_min) & (pd_pred <= pd_therapeutic_max)) * 100
        
        results['pk_therapeutic_range'] = {
            'observed_percent': pk_true_therapeutic,
            'predicted_percent': pk_pred_therapeutic,
            'agreement': np.abs(pk_true_therapeutic - pk_pred_therapeutic)
        }
        
        results['pd_therapeutic_range'] = {
            'observed_percent': pd_true_therapeutic,
            'predicted_percent': pd_pred_therapeutic,
            'agreement': np.abs(pd_true_therapeutic - pd_pred_therapeutic)
        }
        
        # Safety analysis (toxicity thresholds)
        pk_toxicity_threshold = 200  # Example toxicity threshold
        pk_true_toxic = np.mean(pk_true > pk_toxicity_threshold) * 100
        pk_pred_toxic = np.mean(pk_pred > pk_toxicity_threshold) * 100
        
        results['safety_analysis'] = {
            'toxicity_threshold': pk_toxicity_threshold,
            'observed_toxicity_percent': pk_true_toxic,
            'predicted_toxicity_percent': pk_pred_toxic,
            'false_positive_rate': np.mean((pk_pred > pk_toxicity_threshold) & (pk_true <= pk_toxicity_threshold)) * 100,
            'false_negative_rate': np.mean((pk_pred <= pk_toxicity_threshold) & (pk_true > pk_toxicity_threshold)) * 100
        }
        
        return results
    
    def generate_validation_report(self):
        """
        Generate comprehensive validation report
        
        Returns:
            Formatted string with validation results
        """
        if not self.validation_results:
            return "No validation results available. Run run_comprehensive_validation() first."
        
        report = []
        report.append("📋 CLINICAL VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Validation Date: {self.validation_results['validation_date']}")
        report.append(f"Sample Size: {self.validation_results['sample_size']} patients")
        report.append("")
        
        # PK Metrics
        pk_metrics = self.validation_results['pk_metrics']
        report.append("🩸 PHARMACOKINETIC (PK) VALIDATION")
        report.append("-" * 40)
        report.append(f"R² Score: {pk_metrics['pk_r2_score']:.3f}")
        report.append(f"RMSE: {pk_metrics['pk_rmse']:.3f}")
        report.append(f"MAE: {pk_metrics['pk_mae']:.3f}")
        report.append(f"MAPE: {pk_metrics['pk_mape']:.1f}%")
        report.append(f"Correlation: {pk_metrics['pk_correlation']:.3f}")
        report.append(f"Within 20%: {pk_metrics['pk_within_20_percent']:.1f}%")
        report.append("")
        
        # PD Metrics
        pd_metrics = self.validation_results['pd_metrics']
        report.append("💊 PHARMACODYNAMIC (PD) VALIDATION")
        report.append("-" * 40)
        report.append(f"R² Score: {pd_metrics['pd_r2_score']:.3f}")
        report.append(f"RMSE: {pd_metrics['pd_rmse']:.3f}")
        report.append(f"MAE: {pd_metrics['pd_mae']:.3f}")
        report.append(f"MAPE: {pd_metrics['pd_mape']:.1f}%")
        report.append(f"Correlation: {pd_metrics['pd_correlation']:.3f}")
        report.append(f"Within 20%: {pd_metrics['pd_within_20_percent']:.1f}%")
        report.append("")
        
        # Cross-validation
        cv = self.validation_results['cross_validation']
        report.append("🔄 CROSS-VALIDATION RESULTS")
        report.append("-" * 40)
        report.append(f"PK R²: {cv['pk_cv_mean']:.3f} ± {cv['pk_cv_std']:.3f}")
        report.append(f"PD R²: {cv['pd_cv_mean']:.3f} ± {cv['pd_cv_std']:.3f}")
        report.append("")
        
        # Clinical Endpoints
        endpoints = self.validation_results['clinical_endpoints']
        report.append("🏥 CLINICAL ENDPOINT ANALYSIS")
        report.append("-" * 40)
        report.append(f"PK Therapeutic Range Agreement: {endpoints['pk_therapeutic_range']['agreement']:.1f}%")
        report.append(f"PD Therapeutic Range Agreement: {endpoints['pd_therapeutic_range']['agreement']:.1f}%")
        report.append(f"Safety False Negative Rate: {endpoints['safety_analysis']['false_negative_rate']:.1f}%")
        report.append("")
        
        # Statistical Significance
        stats_results = self.validation_results['statistical_tests']
        report.append("🧪 STATISTICAL SIGNIFICANCE")
        report.append("-" * 40)
        report.append(f"PK Paired t-test p-value: {stats_results['pk_paired_ttest']['p_value']:.4f}")
        report.append(f"PD Paired t-test p-value: {stats_results['pd_paired_ttest']['p_value']:.4f}")
        report.append("")
        
        # Overall Assessment
        report.append("📊 OVERALL VALIDATION ASSESSMENT")
        report.append("-" * 40)
        
        pk_r2 = pk_metrics['pk_r2_score']
        pd_r2 = pd_metrics['pd_r2_score']
        
        if pk_r2 >= 0.7 and pd_r2 >= 0.7:
            assessment = "EXCELLENT - Model shows strong clinical validity"
        elif pk_r2 >= 0.5 and pd_r2 >= 0.5:
            assessment = "GOOD - Model demonstrates acceptable clinical performance"
        elif pk_r2 >= 0.3 and pd_r2 >= 0.3:
            assessment = "MODERATE - Model shows potential but needs improvement"
        else:
            assessment = "NEEDS IMPROVEMENT - Model requires refinement before clinical use"
        
        report.append(f"Assessment: {assessment}")
        report.append("")
        
        return "\n".join(report)
    
    def plot_validation_results(self, save_path=None):
        """
        Create comprehensive validation plots
        
        Args:
            save_path: Path to save plots (optional)
        """
        if not self.validation_results:
            print("No validation results to plot. Run validation first.")
            return
        
        # Generate predictions for plotting
        pk_pred, pd_pred = self.predict_clinical_outcomes()
        pk_true = self.clinical_data['pk_observed'].values
        pd_true = self.clinical_data['pd_observed'].values
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clinical Validation Results', fontsize=16, fontweight='bold')
        
        # PK Observed vs Predicted
        axes[0, 0].scatter(pk_true, pk_pred, alpha=0.6, s=50)
        axes[0, 0].plot([pk_true.min(), pk_true.max()], [pk_true.min(), pk_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Observed PK')
        axes[0, 0].set_ylabel('Predicted PK')
        axes[0, 0].set_title(f'PK Validation (R² = {self.validation_results["pk_metrics"]["pk_r2_score"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # PD Observed vs Predicted
        axes[0, 1].scatter(pd_true, pd_pred, alpha=0.6, s=50)
        axes[0, 1].plot([pd_true.min(), pd_true.max()], [pd_true.min(), pd_true.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Observed PD')
        axes[0, 1].set_ylabel('Predicted PD')
        axes[0, 1].set_title(f'PD Validation (R² = {self.validation_results["pd_metrics"]["pd_r2_score"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals Plot
        pk_residuals = pk_true - pk_pred
        pd_residuals = pd_true - pd_pred
        axes[0, 2].scatter(pk_pred, pk_residuals, alpha=0.6, s=50, label='PK')
        axes[0, 2].scatter(pd_pred, pd_residuals, alpha=0.6, s=50, label='PD')
        axes[0, 2].axhline(y=0, color='r', linestyle='--')
        axes[0, 2].set_xlabel('Predicted Values')
        axes[0, 2].set_ylabel('Residuals')
        axes[0, 2].set_title('Residuals Plot')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Bland-Altman Plot for PK
        pk_mean = (pk_true + pk_pred) / 2
        pk_diff = pk_true - pk_pred
        axes[1, 0].scatter(pk_mean, pk_diff, alpha=0.6, s=50)
        axes[1, 0].axhline(y=np.mean(pk_diff), color='r', linestyle='--')
        axes[1, 0].axhline(y=np.mean(pk_diff) + 1.96*np.std(pk_diff), color='r', linestyle=':', alpha=0.7)
        axes[1, 0].axhline(y=np.mean(pk_diff) - 1.96*np.std(pk_diff), color='r', linestyle=':', alpha=0.7)
        axes[1, 0].set_xlabel('Mean of Observed and Predicted PK')
        axes[1, 0].set_ylabel('Difference (Observed - Predicted)')
        axes[1, 0].set_title('Bland-Altman Plot - PK')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bland-Altman Plot for PD
        pd_mean = (pd_true + pd_pred) / 2
        pd_diff = pd_true - pd_pred
        axes[1, 1].scatter(pd_mean, pd_diff, alpha=0.6, s=50)
        axes[1, 1].axhline(y=np.mean(pd_diff), color='r', linestyle='--')
        axes[1, 1].axhline(y=np.mean(pd_diff) + 1.96*np.std(pd_diff), color='r', linestyle=':', alpha=0.7)
        axes[1, 1].axhline(y=np.mean(pd_diff) - 1.96*np.std(pd_diff), color='r', linestyle=':', alpha=0.7)
        axes[1, 1].set_xlabel('Mean of Observed and Predicted PD')
        axes[1, 1].set_ylabel('Difference (Observed - Predicted)')
        axes[1, 1].set_title('Bland-Altman Plot - PD')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Cross-validation Results
        cv_results = self.validation_results['cross_validation']
        fold_numbers = range(1, len(cv_results['pk_scores']) + 1)
        axes[1, 2].plot(fold_numbers, cv_results['pk_scores'], 'o-', label='PK R²', linewidth=2, markersize=8)
        axes[1, 2].plot(fold_numbers, cv_results['pd_scores'], 's-', label='PD R²', linewidth=2, markersize=8)
        axes[1, 2].set_xlabel('Fold Number')
        axes[1, 2].set_ylabel('R² Score')
        axes[1, 2].set_title('Cross-Validation Results')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Validation plots saved to: {save_path}")
        
        plt.show()
    
    def save_validation_results(self, filepath):
        """
        Save validation results to file
        
        Args:
            filepath: Path to save results
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for key, value in self.validation_results.items():
            if isinstance(value, dict):
                results_to_save[key] = {}
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'tolist'):
                        results_to_save[key][subkey] = subvalue.tolist()
                    else:
                        results_to_save[key][subkey] = subvalue
            else:
                results_to_save[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        print(f"💾 Validation results saved to: {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # This would be used with the actual trained model
    print("🔬 Clinical Trial Validation Framework Ready")
    print("Use with: validator = ClinicalTrialValidator(model, 'clinical_data.csv')")
    print("Then: results = validator.run_comprehensive_validation()")
