#!/usr/bin/env python3
"""
Download BCI Competition IV datasets
Supports multiple datasets from the official BCI Competition IV website
"""

import os
import numpy as np
import scipy.io as sio
import requests
from pathlib import Path
import zipfile
import tempfile
import shutil
import mne
from tqdm import tqdm
import time

# BCI Competition IV dataset URLs
BCI_DATASETS = {
    'dataset1': {
        'name': 'BCI Competition IV Dataset 1 (Berlin)',
        'urls': {
            '100hz_mat': 'https://www.bbci.de/competition/download/competition_iv/BCICIV_1_mat.zip',
            '100hz_asc': 'https://www.bbci.de/competition/download/competition_iv/BCICIV_1_asc.zip',
            '1000hz_calib': 'https://www.bbci.de/competition/download/competition_iv/BCICIV_1calib_1000Hz_mat.zip',
            '1000hz_eval': 'https://www.bbci.de/competition/download/competition_iv/BCICIV_1eval_1000Hz_mat.zip'
        },
        'description': 'Motor imagery, 100Hz and 1000Hz data'
    },
    'dataset2a': {
        'name': 'BCI Competition IV Dataset 2a (Graz)',
        'urls': {
            'gdf': 'https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip'
        },
        'description': '4-class motor imagery, 22 EEG channels, 9 subjects'
    },
    'dataset2b': {
        'name': 'BCI Competition IV Dataset 2b (Graz)',
        'urls': {
            'gdf': 'https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip'
        },
        'description': '3-class motor imagery, 3 EEG channels, 9 subjects'
    },
    'dataset3': {
        'name': 'BCI Competition IV Dataset 3 (Freiburg/Tübingen)',
        'urls': {
            'mat': 'https://www.bbci.de/competition/download/competition_iv/BCICIV_3_mat.zip'
        },
        'description': 'Motor imagery with feedback'
    },
    'dataset4': {
        'name': 'BCI Competition IV Dataset 4 (Washington/Albany)',
        'urls': {
            'mat': 'https://www.bbci.de/competition/download/competition_iv/BCICIV_4_mat.zip'
        },
        'description': 'Motor imagery with feedback'
    }
}

def download_file(url, output_path, chunk_size=8192):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def download_bci_dataset(dataset_key='dataset2a', data_types=None):
    """Download BCI Competition IV dataset."""
    
    if dataset_key not in BCI_DATASETS:
        print(f"❌ Unknown dataset: {dataset_key}")
        print(f"Available datasets: {list(BCI_DATASETS.keys())}")
        return None, []
    
    dataset_info = BCI_DATASETS[dataset_key]
    
    # Create data directory
    data_dir = Path("data/raw/bci_competition_data") / dataset_key
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {dataset_info['name']}...")
    print(f"Description: {dataset_info['description']}")
    print("=" * 60)
    
    downloaded_files = []
    
    # Determine which data types to download
    if data_types is None:
        # Default to first available type
        data_types = [list(dataset_info['urls'].keys())[0]]
    
    for data_type in data_types:
        if data_type not in dataset_info['urls']:
            print(f"❌ Unknown data type: {data_type}")
            continue
            
        url = dataset_info['urls'][data_type]
        filename = f"{dataset_key}_{data_type}.zip"
        zip_path = data_dir / filename
        
        print(f"\n📥 Downloading {data_type} data...")
        print(f"URL: {url}")
        
        if download_file(url, zip_path):
            print(f"✅ Downloaded {zip_path}")
            
            # Extract zip file
            print(f"📦 Extracting {filename}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                
                # List extracted files
                extracted_files = []
                for ext in ['.gdf', '.mat', '.asc']:
                    extracted_files.extend(data_dir.glob(f"*{ext}"))
                
                print(f"✅ Extracted {len(extracted_files)} files:")
                for f in extracted_files:
                    print(f"  - {f.name}")
                    downloaded_files.append(f)
                    
            except Exception as e:
                print(f"❌ Failed to extract {filename}: {e}")
    
    return data_dir, downloaded_files

def load_gdf_data(filepath):
    """Load a single BCI Competition IV-2a .gdf file using MNE."""
    
    print(f"Loading {filepath}...")
    
    try:
        # Load the GDF file using MNE
        raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
        
        # Get events (trial markers)
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        
        print(f"  Raw data shape: {raw.get_data().shape}")
        print(f"  Number of events: {len(events)}")
        print(f"  Event types: {event_dict}")
        
        # Extract data and labels
        data = raw.get_data()  # (channels, timepoints)
        
        # Get trial information
        trial_data = []
        trial_labels = []
        
        # Process events to extract trials
        for event in events:
            if event[2] in [1, 2, 3, 4]:  # Motor imagery events
                # Extract trial window (4 seconds = 1000 samples at 250Hz)
                start_sample = event[0]
                end_sample = start_sample + 1000  # 4 seconds
                
                if end_sample <= data.shape[1]:
                    trial = data[:, start_sample:end_sample]  # (channels, timepoints)
                    trial_data.append(trial)
                    trial_labels.append(event[2])  # Class label
        
        trial_data = np.array(trial_data)  # (trials, channels, timepoints)
        trial_labels = np.array(trial_labels)
        
        print(f"  Trial data shape: {trial_data.shape}")
        print(f"  Trial labels shape: {trial_labels.shape}")
        print(f"  Label distribution: {np.bincount(trial_labels)}")
        
        return {
            'data': trial_data,
            'labels': trial_labels,
            'raw': raw,
            'events': events,
            'event_dict': event_dict,
            'filename': filepath.name
        }
        
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def load_mat_data(filepath):
    """Load a .mat file using scipy."""
    
    print(f"Loading {filepath}...")
    
    try:
        # Load the MAT file
        mat_data = sio.loadmat(filepath)
        
        print(f"  Available variables: {list(mat_data.keys())}")
        
        # Extract data based on common BCI dataset structure
        data = None
        labels = None
        
        # Look for common variable names
        for key in mat_data.keys():
            if key.startswith('__'):  # Skip metadata
                continue
            if 'data' in key.lower() or 'eeg' in key.lower():
                data = mat_data[key]
                print(f"  Found data in '{key}': {data.shape}")
            elif 'label' in key.lower() or 'class' in key.lower():
                labels = mat_data[key]
                print(f"  Found labels in '{key}': {labels.shape}")
        
        if data is None:
            print(f"  Could not find data in {filepath}")
            return None
            
        return {
            'data': data,
            'labels': labels,
            'mat_data': mat_data,
            'filename': filepath.name
        }
        
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def preprocess_bci_data(data_dict):
    """Preprocess BCI data for our hierarchical RNN models."""
    
    data = data_dict['data']
    labels = data_dict['labels']
    
    print(f"Preprocessing data...")
    print(f"  Original data shape: {data.shape}")
    
    # Handle different data formats
    if len(data.shape) == 3:  # (trials, channels, timepoints)
        # Already in trial format
        pass
    elif len(data.shape) == 2:  # (channels, timepoints)
        # Single continuous recording, need to segment
        print("  Converting continuous data to trials...")
        # This would need trial markers from events
        data = data.reshape(1, data.shape[0], data.shape[1])  # Single trial for now
    
    # Convert labels to 0-based indexing if needed
    if labels is not None and np.min(labels) > 0:
        labels = labels - 1  # Convert from 1-4 to 0-3
    
    # Normalize data per trial
    data_normalized = np.zeros_like(data)
    for i in range(data.shape[0]):
        trial = data[i]
        # Normalize each trial
        trial_norm = (trial - np.mean(trial, axis=1, keepdims=True)) / (np.std(trial, axis=1, keepdims=True) + 1e-8)
        data_normalized[i] = trial_norm
    
    print(f"  Normalized data shape: {data_normalized.shape}")
    if labels is not None:
        print(f"  Label distribution: {np.bincount(labels)}")
    
    return {
        'data': data_normalized,
        'labels': labels,
        'original_data': data,
        'raw': data_dict.get('raw')
    }

def extract_features_for_hierarchical_rnn(data, fast_window_size=10, slow_context_size=50):
    """Extract fast and slow features for our hierarchical RNN models."""
    
    print(f"Extracting hierarchical features...")
    print(f"  Fast window size: {fast_window_size} samples")
    print(f"  Slow context size: {slow_context_size} samples")
    
    num_trials, num_channels, num_timepoints = data.shape
    
    # Convert window sizes to samples (250Hz sampling rate)
    fast_samples = int(fast_window_size * 250 / 1000)  # ms to samples
    slow_samples = int(slow_context_size * 250 / 1000)
    
    print(f"  Fast samples: {fast_samples}")
    print(f"  Slow samples: {slow_samples}")
    
    # Calculate valid time points
    start_idx = max(slow_samples, fast_samples)
    valid_length = num_timepoints - start_idx
    
    # Initialize feature arrays
    X_fast = []
    X_slow = []
    y = []
    
    for trial_idx in range(num_trials):
        trial_data = data[trial_idx]  # (channels, timepoints)
        
        # Process each time point
        for t in range(start_idx, num_timepoints):
            # Fast features (immediate dynamics)
            fast_window = trial_data[:, t-fast_samples:t]
            fast_mean = np.mean(fast_window, axis=1)  # (channels,)
            fast_std = np.std(fast_window, axis=1)
            fast_diff = fast_window[:, -1] - fast_window[:, -2]  # last - second_last
            
            # Slow features (contextual information)
            slow_window = trial_data[:, t-slow_samples:t]
            slow_mean = np.mean(slow_window, axis=1)
            slow_std = np.std(slow_window, axis=1)
            
            # Flatten features across channels
            fast_features = np.concatenate([fast_mean, fast_std, fast_diff])
            slow_features = np.concatenate([slow_mean, slow_std])
            
            X_fast.append(fast_features)
            X_slow.append(slow_features)
            y.append(trial_idx)  # Use trial index as label for now
    
    X_fast = np.array(X_fast)
    X_slow = np.array(X_slow)
    y = np.array(y)
    
    print(f"  Fast features shape: {X_fast.shape}")
    print(f"  Slow features shape: {X_slow.shape}")
    print(f"  Labels shape: {y.shape}")
    
    return X_fast, X_slow, y

def main():
    """Main function to download and preprocess BCI dataset."""
    
    print("=== BCI Competition IV Dataset Downloader ===")
    print()
    
    # Show available datasets
    print("Available datasets:")
    for key, info in BCI_DATASETS.items():
        print(f"  {key}: {info['name']}")
        print(f"    {info['description']}")
        print(f"    Data types: {list(info['urls'].keys())}")
        print()
    
    # Download dataset 2a by default (most commonly used)
    dataset_key = 'dataset2a'
    data_types = ['gdf']  # Download GDF files for dataset 2a
    
    print(f"Downloading {dataset_key}...")
    data_dir, downloaded_files = download_bci_dataset(dataset_key, data_types)
    
    if not downloaded_files:
        print("❌ No files downloaded. Exiting.")
        return
    
    # Load and preprocess first file as example
    first_file = downloaded_files[0]
    
    if first_file.suffix == '.gdf':
        data_dict = load_gdf_data(first_file)
    elif first_file.suffix == '.mat':
        data_dict = load_mat_data(first_file)
    else:
        print(f"❌ Unsupported file type: {first_file.suffix}")
        return
    
    if data_dict is None:
        print("❌ Could not load data. Exiting.")
        return
    
    # Preprocess data
    processed_data = preprocess_bci_data(data_dict)
    
    # Extract features for hierarchical RNN
    X_fast, X_slow, y = extract_features_for_hierarchical_rnn(
        processed_data['data'],
        fast_window_size=10,
        slow_context_size=50
    )
    
    # Save processed data
    output_file = data_dir / "processed_bci_data.npz"
    np.savez(
        output_file,
        X_fast=X_fast,
        X_slow=X_slow,
        y=y,
        labels=processed_data['labels'],
        info={'filename': data_dict['filename']}
    )
    
    print(f"\n✅ Processed data saved to {output_file}")
    print(f"   Fast features: {X_fast.shape}")
    print(f"   Slow features: {X_slow.shape}")
    print(f"   Labels: {y.shape}")
    
    return output_file

if __name__ == "__main__":
    main() 