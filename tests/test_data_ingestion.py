"""
Test script for the data ingestion system.

This script tests the complete data ingestion pipeline with the
current datasets from the Downloads folder.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_data_ingestion():
    """Test the data ingestion system with current datasets."""
    
    print("🚀 Starting PK-PD Data Ingestion Test")
    print("=" * 50)
    
    try:
        # Import the orchestrator
        from data.main_data_ingestion import DataIngestionOrchestrator
        
        # Create output directory
        output_dir = project_root / "results" / "simulations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize orchestrator
        orchestrator = DataIngestionOrchestrator()
        orchestrator.exporter.output_dir = output_dir
        
        print("✅ Orchestrator initialized successfully")
        
        # Test with current datasets
        print("\n📊 Testing with current datasets...")
        results = orchestrator.run_test_with_current_datasets()
        
        # Display results
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        
        if results['success']:
            print("✅ Test completed successfully!")
        else:
            print("❌ Test completed with errors")
        
        print(f"\nData types processed: {', '.join(results['data_types_processed'])}")
        print(f"Total errors: {len(results['errors'])}")
        print(f"Total warnings: {len(results['warnings'])}")
        
        # Show detailed results for each data type
        if 'summary' in results and 'data_summary' in results['summary']:
            print("\n📈 Detailed Results:")
            for data_type, summary in results['summary']['data_summary'].items():
                print(f"\n🔬 {data_type.upper()}:")
                print(f"  Sources processed: {summary['sources_processed']}")
                print(f"  Records processed: {summary['total_records_processed']}")
                print(f"  Records valid: {summary['total_records_valid']}")
                print(f"  Records consolidated: {summary['records_consolidated']}")
                print(f"  Quality grade: {summary['quality_grade']} ({summary['quality_score']:.3f})")
        
        # Show errors if any
        if results['errors']:
            print("\n❌ Errors encountered:")
            for i, error in enumerate(results['errors'][:5], 1):
                print(f"  {i}. {error}")
            if len(results['errors']) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
        
        # Show warnings if any
        if results['warnings']:
            print("\n⚠️  Warnings:")
            for i, warning in enumerate(results['warnings'][:3], 1):
                print(f"  {i}. {warning}")
            if len(results['warnings']) > 3:
                print(f"  ... and {len(results['warnings']) - 3} more warnings")
        
        # Show output files
        print(f"\n📁 Output files created in: {output_dir}")
        if output_dir.exists():
            output_files = list(output_dir.glob("*"))
            if output_files:
                print("Files created:")
                for file in sorted(output_files):
                    size_kb = file.stat().st_size / 1024
                    print(f"  📄 {file.name} ({size_kb:.1f} KB)")
            else:
                print("No output files found")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all modules are properly installed")
        return None
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas',
        'numpy',
        'chardet',
        'yaml',
        'pathlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All dependencies found!")
        return True


def check_dataset_files():
    """Check if the dataset files exist."""
    print("\n📁 Checking dataset files...")
    
    dataset_files = [
        "C:/Users/sparta/Downloads/archive (1)/drugsInfo.csv",
        "C:/Users/sparta/Downloads/archive (1)/diseasesInfo.csv", 
        "C:/Users/sparta/Downloads/archive (1)/mapping.csv"
    ]
    
    all_exist = True
    
    for file_path in dataset_files:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


if __name__ == "__main__":
    print("PK-PD Data Ingestion System Test")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check dataset files
    if not check_dataset_files():
        print("\n⚠️  Some dataset files are missing.")
        print("Please ensure the files are in the correct location:")
        print("C:/Users/sparta/Downloads/archive (1)/")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run the test
    print("\n" + "=" * 50)
    results = test_data_ingestion()
    
    if results and results['success']:
        print("\n🎉 Test completed successfully!")
        print("The data ingestion system is ready to use!")
    else:
        print("\n💥 Test failed!")
        print("Please check the errors above and fix any issues.")
        sys.exit(1)
