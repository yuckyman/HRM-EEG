#!/usr/bin/env python3
"""
Download all BCI Competition IV datasets
This script downloads all available datasets for comprehensive testing
"""

import sys
from pathlib import Path
from .download_bci_dataset import BCI_DATASETS, download_bci_dataset

def download_all_datasets():
    """Download all available BCI Competition IV datasets."""
    
    print("=== Downloading All BCI Competition IV Datasets ===")
    print()
    
    results = {}
    
    for dataset_key, dataset_info in BCI_DATASETS.items():
        print(f"📥 Downloading {dataset_key}: {dataset_info['name']}")
        print(f"   Description: {dataset_info['description']}")
        
        # Download first available data type for each dataset
        data_types = [list(dataset_info['urls'].keys())[0]]
        
        try:
            data_dir, downloaded_files = download_bci_dataset(dataset_key, data_types)
            results[dataset_key] = {
                'success': len(downloaded_files) > 0,
                'files': len(downloaded_files),
                'directory': data_dir
            }
            print(f"✅ {dataset_key}: Downloaded {len(downloaded_files)} files")
        except Exception as e:
            print(f"❌ {dataset_key}: Failed - {e}")
            results[dataset_key] = {
                'success': False,
                'error': str(e)
            }
        
        print()
    
    # Summary
    print("=== Download Summary ===")
    for dataset_key, result in results.items():
        if result['success']:
            print(f"✅ {dataset_key}: {result['files']} files in {result['directory']}")
        else:
            print(f"❌ {dataset_key}: Failed - {result.get('error', 'Unknown error')}")
    
    return results

def download_specific_dataset(dataset_key):
    """Download a specific dataset."""
    
    if dataset_key not in BCI_DATASETS:
        print(f"❌ Unknown dataset: {dataset_key}")
        print(f"Available datasets: {list(BCI_DATASETS.keys())}")
        return None
    
    print(f"📥 Downloading {dataset_key}: {BCI_DATASETS[dataset_key]['name']}")
    
    # Download first available data type
    data_types = [list(BCI_DATASETS[dataset_key]['urls'].keys())[0]]
    
    try:
        data_dir, downloaded_files = download_bci_dataset(dataset_key, data_types)
        print(f"✅ Downloaded {len(downloaded_files)} files to {data_dir}")
        return data_dir, downloaded_files
    except Exception as e:
        print(f"❌ Failed to download {dataset_key}: {e}")
        return None, []

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Download specific dataset
        dataset_key = sys.argv[1]
        download_specific_dataset(dataset_key)
    else:
        # Download all datasets
        download_all_datasets() 