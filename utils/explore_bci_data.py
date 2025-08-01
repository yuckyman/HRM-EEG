#!/usr/bin/env python3
"""
Explore and visualize BCI Competition IV data
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import scipy.io as sio

def explore_gdf_file(filepath):
    """Explore a single GDF file."""
    
    print(f"\n=== Exploring {filepath.name} ===")
    
    try:
        # Load the GDF file
        raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
        
        # Basic info
        print(f"Data shape: {raw.get_data().shape}")
        print(f"Sampling rate: {raw.info['sfreq']} Hz")
        print(f"Duration: {raw.times[-1]:.1f} seconds")
        print(f"Channels: {len(raw.ch_names)}")
        
        # Get events
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        print(f"Number of events: {len(events)}")
        print(f"Event types: {event_dict}")
        
        # Plot raw data (first 10 seconds)
        data = raw.get_data()
        time_points = min(10 * int(raw.info['sfreq']), data.shape[1])
        
        plt.figure(figsize=(15, 8))
        plt.plot(raw.times[:time_points], data[:5, :time_points].T)
        plt.title(f'Raw EEG Data - First 10 seconds ({filepath.name})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend(raw.ch_names[:5])
        plt.tight_layout()
        plt.savefig(f'explore_{filepath.stem}_raw.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot event markers
        if len(events) > 0:
            plt.figure(figsize=(15, 6))
            event_times = events[:, 0] / raw.info['sfreq']
            event_types = events[:, 2]
            
            plt.scatter(event_times, event_types, alpha=0.7)
            plt.title(f'Event Markers ({filepath.name})')
            plt.xlabel('Time (s)')
            plt.ylabel('Event Type')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'explore_{filepath.stem}_events.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        return {
            'raw': raw,
            'events': events,
            'event_dict': event_dict,
            'data_shape': data.shape,
            'duration': raw.times[-1]
        }
        
    except Exception as e:
        print(f"❌ Error exploring {filepath}: {e}")
        return None

def explore_processed_data(filepath):
    """Explore the processed data."""
    
    print(f"\n=== Exploring Processed Data ===")
    
    try:
        data = np.load(filepath)
        
        print("Available arrays:")
        for key in data.keys():
            if key != 'info':
                print(f"  {key}: {data[key].shape}")
        
        if 'X_fast' in data and 'X_slow' in data:
            print(f"\nFast features: {data['X_fast'].shape}")
            print(f"Slow features: {data['X_slow'].shape}")
            print(f"Labels: {data['y'].shape}")
            
            # Plot feature distributions
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.hist(data['X_fast'].flatten(), bins=50, alpha=0.7)
            plt.title('Fast Features Distribution')
            plt.xlabel('Feature Value')
            plt.ylabel('Count')
            
            plt.subplot(1, 3, 2)
            plt.hist(data['X_slow'].flatten(), bins=50, alpha=0.7)
            plt.title('Slow Features Distribution')
            plt.xlabel('Feature Value')
            plt.ylabel('Count')
            
            plt.subplot(1, 3, 3)
            if 'labels' in data:
                plt.hist(data['labels'], bins=range(min(data['labels']), max(data['labels'])+2), alpha=0.7)
                plt.title('Label Distribution')
                plt.xlabel('Class Label')
                plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig('explore_processed_features.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        return data
        
    except Exception as e:
        print(f"❌ Error exploring processed data: {e}")
        return None

def main():
    """Main exploration function."""
    
    print("=== BCI Data Explorer ===")
    
    # Explore dataset2a (most common)
    dataset_dir = Path("data/raw/bci_competition_data/dataset2a")
    
    if dataset_dir.exists():
        print(f"\n📁 Exploring dataset in {dataset_dir}")
        
        # Find GDF files
        gdf_files = list(dataset_dir.glob("*.gdf"))
        print(f"Found {len(gdf_files)} GDF files")
        
        # Explore first few files
        for i, gdf_file in enumerate(gdf_files[:3]):  # First 3 files
            explore_gdf_file(gdf_file)
        
        # Explore processed data
        processed_file = dataset_dir / "processed_bci_data.npz"
        if processed_file.exists():
            explore_processed_data(processed_file)
    
    # Explore other datasets if they exist
    for dataset_key in ['dataset1', 'dataset2b', 'dataset3', 'dataset4']:
        dataset_dir = Path(f"data/raw/bci_competition_data/{dataset_key}")
        if dataset_dir.exists():
            print(f"\n📁 Exploring {dataset_key}")
            
            # Find data files
            data_files = []
            for ext in ['.gdf', '.mat', '.asc']:
                data_files.extend(dataset_dir.glob(f"*{ext}"))
            
            if data_files:
                print(f"Found {len(data_files)} data files")
                # Explore first file
                explore_gdf_file(data_files[0]) if data_files[0].suffix == '.gdf' else None
    
    print("\n✅ Exploration complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main() 