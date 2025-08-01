#!/usr/bin/env python3
"""
Test script to verify workspace organization and configuration
"""

import os
from pathlib import Path
from config import (
    PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    RESULTS_DIR, EXPERIMENTS_DIR, MODEL_COMPARISONS_DIR, LOGS_DIR,
    BCI_DATA_DIR, MNE_DATA_DIR,
    get_experiment_results_path, get_model_comparison_path, get_log_path
)

def test_directory_structure():
    """Test that all directories exist and are properly organized"""
    
    print("=== Testing Directory Structure ===")
    
    # Test base directories
    directories = [
        ("Project Root", PROJECT_ROOT),
        ("Data Directory", DATA_DIR),
        ("Raw Data Directory", RAW_DATA_DIR),
        ("Processed Data Directory", PROCESSED_DATA_DIR),
        ("Results Directory", RESULTS_DIR),
        ("Experiments Directory", EXPERIMENTS_DIR),
        ("Model Comparisons Directory", MODEL_COMPARISONS_DIR),
        ("Logs Directory", LOGS_DIR),
        ("BCI Data Directory", BCI_DATA_DIR),
        ("MNE Data Directory", MNE_DATA_DIR)
    ]
    
    all_good = True
    for name, directory in directories:
        if directory.exists():
            print(f"✅ {name}: {directory}")
        else:
            print(f"❌ {name}: {directory} (MISSING)")
            all_good = False
    
    return all_good

def test_file_paths():
    """Test that file path generation works correctly"""
    
    print("\n=== Testing File Path Generation ===")
    
    # Test path generation functions
    experiment_path = get_experiment_results_path("test_experiment")
    comparison_path = get_model_comparison_path("test_comparison")
    log_path = get_log_path("test_log")
    
    print(f"✅ Experiment results path: {experiment_path}")
    print(f"✅ Model comparison path: {comparison_path}")
    print(f"✅ Log path: {log_path}")
    
    # Check that paths are in correct directories
    assert experiment_path.parent == EXPERIMENTS_DIR, "Experiment path not in experiments directory"
    assert comparison_path.parent == MODEL_COMPARISONS_DIR, "Comparison path not in model comparisons directory"
    assert log_path.parent == LOGS_DIR, "Log path not in logs directory"
    
    print("✅ All file paths correctly organized")

def test_data_migration():
    """Test that data files are in the correct locations"""
    
    print("\n=== Testing Data Migration ===")
    
    # Check if BCI data exists in new location
    bci_files = list(BCI_DATA_DIR.glob("*.gdf")) if BCI_DATA_DIR.exists() else []
    print(f"BCI files found: {len(bci_files)}")
    
    # Check if MNE data exists
    mne_exists = MNE_DATA_DIR.exists()
    print(f"MNE data directory exists: {mne_exists}")
    
    # Check for any old files in root
    old_files = []
    for pattern in ["*.json", "bci_competition_data", "mne_data"]:
        old_files.extend(Path(".").glob(pattern))
    
    if old_files:
        print(f"⚠️  Found old files in root: {[f.name for f in old_files]}")
    else:
        print("✅ No old files found in root directory")

def test_config_import():
    """Test that config can be imported and used"""
    
    print("\n=== Testing Config Import ===")
    
    try:
        from config import MODEL_CONFIG, TRAINING_CONFIG, BCI_CONFIG, FEATURE_CONFIG
        print("✅ Config imports successful")
        print(f"✅ Model config: {len(MODEL_CONFIG)} sections")
        print(f"✅ Training config: {len(TRAINING_CONFIG)} parameters")
        print(f"✅ BCI config: {len(BCI_CONFIG)} parameters")
        print(f"✅ Feature config: {len(FEATURE_CONFIG)} parameters")
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

def main():
    """Run all organization tests"""
    
    print("🧹 Testing Workspace Organization")
    print("=" * 50)
    
    # Run tests
    dir_test = test_directory_structure()
    test_file_paths()
    test_data_migration()
    config_test = test_config_import()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Organization Test Summary")
    print("=" * 50)
    
    if dir_test and config_test:
        print("✅ All tests passed! Workspace is properly organized.")
        print("\n📁 Directory Structure:")
        print(f"   Data: {DATA_DIR}")
        print(f"   Results: {RESULTS_DIR}")
        print(f"   Logs: {LOGS_DIR}")
        print("\n🎯 Next steps:")
        print("   1. Run experiments with: python run_bci_experiment.py")
        print("   2. Check results in: {RESULTS_DIR}")
        print("   3. View logs in: {LOGS_DIR}")
    else:
        print("❌ Some tests failed. Please check the output above.")
    
    return dir_test and config_test

if __name__ == "__main__":
    main() 