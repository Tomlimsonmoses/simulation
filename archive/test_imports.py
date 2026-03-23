"""
Test script to check module imports
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing module imports...")

try:
    from pk_pd_model import PKPDModel
    print("✅ pk_pd_model imported successfully")
except ImportError as e:
    print(f"❌ pk_pd_model import failed: {e}")

try:
    from expanded_drug_database import ExpandedDrugDatabase
    print("✅ expanded_drug_database imported successfully")
except ImportError as e:
    print(f"❌ expanded_drug_database import failed: {e}")

try:
    from disease_specific_models import DiseaseSpecificModeler
    print("✅ disease_specific_models imported successfully")
except ImportError as e:
    print(f"❌ disease_specific_models import failed: {e}")

try:
    from clinical_validation import ClinicalTrialValidator
    print("✅ clinical_validation imported successfully")
except ImportError as e:
    print(f"❌ clinical_validation import failed: {e}")

print("\nTesting basic functionality...")

try:
    from pk_pd_model import PKPDModel
    model = PKPDModel()
    print(f"✅ PKPDModel created: {model.get_model_info()}")
except Exception as e:
    print(f"❌ PKPDModel creation failed: {e}")

try:
    from expanded_drug_database import ExpandedDrugDatabase
    db = ExpandedDrugDatabase()
    print(f"✅ DrugDB created with {len(db.get_all_drug_names())} drugs")
except Exception as e:
    print(f"❌ DrugDB creation failed: {e}")

print("\nDone testing!")
