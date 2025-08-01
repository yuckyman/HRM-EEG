#!/usr/bin/env python3
"""
Attention-based BCI experiment with real BCI Competition IV data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import mne

from attention_modules import (
    AttentionFastRNN, AttentionSlowRNN, AttentionIntegrationNet,
    AttentionVisualizer, analyze_brain_region_attention
)
from attention_hierarchical_processor import AttentionHierarchicalProcessor
from config import (
    get_experiment_results_path, get_model_comparison_path, get_log_path,
    MODEL_CONFIG, TRAINING_CONFIG, BCI_CONFIG, FEATURE_CONFIG,
    BCI_DATA_DIR
)

def load_bci_data(filepath):
    """Load BCI Competition data and extract features."""
    
    print(f"Loading BCI data from {filepath}...")
    
    try:
        # Load the GDF file
        raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
        
        # Get events (trial markers)
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        
        print(f"  Raw data shape: {raw.get_data().shape}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Number of events: {len(events)}")
        print(f"  Event types: {event_dict}")
        
        # Extract data and labels
        data = raw.get_data()  # (channels, timepoints)
        
        # Check data quality
        print(f"  Data quality check:")
        print(f"    Mean: {data.mean():.6f}")
        print(f"    Std: {data.std():.6f}")
        print(f"    Min: {data.min():.6f}")
        print(f"    Max: {data.max():.6f}")
        
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
                    
                    # Check trial quality
                    if trial.std() > 1e-6:  # Only keep trials with some variance
                        trial_data.append(trial)
                        # Keep all 4 classes (0-based indexing)
                        trial_labels.append(event[2] - 1)  # Convert 1-4 to 0-3
        
        trial_data = np.array(trial_data)  # (trials, channels, timepoints)
        trial_labels = np.array(trial_labels)
        
        print(f"  Trial data shape: {trial_data.shape}")
        print(f"  Trial labels shape: {trial_labels.shape}")
        print(f"  Label distribution: {np.bincount(trial_labels)}")
        
        # Check if we have enough trials
        if len(trial_data) < 10:
            print(f"  ⚠️  Warning: Only {len(trial_data)} trials found. This file may have issues.")
            return None
        
        return {
            'data': trial_data,
            'labels': trial_labels,
            'raw': raw,
            'events': events,
            'event_dict': event_dict,
            'filename': filepath.name,
            'sampling_rate': raw.info['sfreq']
        }
        
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def process_bci_trials_for_attention(trial_data, labels, processor):
    """Process BCI trials through the attention processor."""
    
    print(f"Processing {len(trial_data)} trials for attention...")
    
    all_X_fast = []
    all_X_slow = []
    all_y = []
    
    # Find the best channel (highest variance)
    print(f"  Finding best channel...")
    channel_variances = []
    for ch in range(trial_data.shape[1]):  # For each channel
        channel_data = trial_data[:, ch, :]  # (trials, timepoints)
        variance = channel_data.var()
        channel_variances.append(variance)
        print(f"    Channel {ch}: variance = {variance:.8f}")
    
    best_channel = np.argmax(channel_variances)
    print(f"  Best channel: {best_channel} (variance = {channel_variances[best_channel]:.8f})")
    
    for i, (trial, label) in enumerate(zip(trial_data, labels)):
        # trial shape: (channels, timepoints)
        # Use the best channel instead of first channel
        single_channel_data = trial[best_channel, :]  # (timepoints,)
        
        # Check channel data quality
        if single_channel_data.std() < 1e-6:
            print(f"  ⚠️  Skipping trial {i} - low variance channel data")
            continue
        
        # Extract features using the attention processor
        fast_features, slow_features = processor.extract_features(single_channel_data)
        
        if len(fast_features) > 0 and len(slow_features) > 0:
            # Use the trial label for all features from this trial
            all_X_fast.extend(fast_features)
            all_X_slow.extend(slow_features)
            all_labels = [label] * len(fast_features)
            all_y.extend(all_labels)
    
    X_fast = np.array(all_X_fast)
    X_slow = np.array(all_X_slow)
    y = np.array(all_y)
    
    print(f"  Fast features shape: {X_fast.shape}")
    print(f"  Slow features shape: {X_slow.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Label distribution: {np.bincount(y)}")
    
    # Check feature quality
    print(f"  Feature quality check:")
    print(f"    Fast features - mean: {X_fast.mean():.6f}, std: {X_fast.std():.6f}")
    print(f"    Slow features - mean: {X_slow.mean():.6f}, std: {X_slow.std():.6f}")
    
    return X_fast, X_slow, y

def run_attention_bci_experiment():
    """Run attention-based BCI experiment with real data."""
    
    print("=== Attention-Based BCI Experiment (Real Data) ===")
    
    # Initialize attention processor
    processor = AttentionHierarchicalProcessor(
        fast_window_size=FEATURE_CONFIG['fast_window_ms'],
        slow_context_size=FEATURE_CONFIG['slow_window_ms'],
        num_classes=BCI_CONFIG['num_classes'],
        num_heads=4,
        dropout=0.2
    )
    
    # Load BCI Competition data
    print("Loading BCI Competition IV data...")
    bci_dir = BCI_DATA_DIR / "dataset2a"
    
    if not bci_dir.exists():
        print("❌ BCI data not found. Please run utils/download_bci_dataset.py first.")
        return
    
    # Find GDF files
    gdf_files = list(bci_dir.glob("*.gdf"))
    print(f"Found {len(gdf_files)} GDF files")
    
    # Process first few files for demonstration
    results_summary = {}
    
    for i, gdf_file in enumerate(gdf_files[:3]):  # Process first 3 files
        print(f"\n--- Processing {gdf_file.name} ---")
        
        # Load data
        data_dict = load_bci_data(gdf_file)
        if data_dict is None:
            continue
        
        # Process trials for attention
        X_fast, X_slow, y = process_bci_trials_for_attention(
            data_dict['data'], 
            data_dict['labels'], 
            processor
        )
        
        if len(X_fast) == 0:
            print(f"  No valid trials found in {gdf_file.name}")
            continue
        
        # Train attention models
        print(f"\n  Training attention models for {gdf_file.stem}...")
        (fast_test, slow_test, labels_test, 
         train_losses, test_accuracies, attention_weights) = processor.train_attention_models(
            X_fast, X_slow, y, test_size=0.2, num_epochs=5
        )
        
        # Evaluate models
        print(f"\n  Evaluating attention models for {gdf_file.stem}...")
        results = processor.evaluate_attention_models(
            fast_test, slow_test, labels_test, attention_weights
        )
        
        # Visualize attention
        print(f"\n  Visualizing attention patterns for {gdf_file.stem}...")
        processor.visualize_attention(gdf_file.stem)
        
        # Save results
        processor.save_attention_results(results, gdf_file.stem)
        
        # Store results summary
        results_summary[gdf_file.stem] = {
            'accuracy': results['accuracy'],
            'num_trials': len(data_dict['data']),
            'label_distribution': np.bincount(data_dict['labels']).tolist(),
            'classes_present': np.unique(data_dict['labels']).tolist(),
            'attention_analysis': results['attention_analysis']
        }
        
        print(f"  ✅ Completed processing {gdf_file.name}")
    
    # Generate comprehensive report
    print(f"\n=== Attention BCI Experiment Summary ===")
    print(f"Processed {len(results_summary)} BCI files")
    
    # Print detailed results
    print(f"\nDetailed Results:")
    for file_name, result in results_summary.items():
        print(f"\n{file_name}:")
        print(f"  Trials: {result['num_trials']}")
        print(f"  Classes present: {result['classes_present']}")
        print(f"  Label distribution: {result['label_distribution']}")
        print(f"  Attention model accuracy: {result['accuracy']:.3f}")
    
    # Save comprehensive results
    comprehensive_results = {
        'experiment_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'experiment_type': 'attention_bci_real_data',
        'processor_config': {
            'fast_window_size': processor.fast_window_size,
            'slow_context_size': processor.slow_context_size,
            'num_classes': processor.num_classes,
            'num_heads': processor.num_heads,
            'dropout': processor.dropout
        },
        'results_summary': results_summary
    }
    
    # Save to results directory
    results_file = get_experiment_results_path("attention_bci_real_data")
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    print(f"✅ Attention BCI experiment with real data complete!")
    
    return results_summary

def quick_attention_test():
    """Quick test with a single BCI file."""
    
    print("=== Quick Attention BCI Test ===")
    
    # Initialize attention processor
    processor = AttentionHierarchicalProcessor(
        fast_window_size=FEATURE_CONFIG['fast_window_ms'],
        slow_context_size=FEATURE_CONFIG['slow_window_ms'],
        num_classes=BCI_CONFIG['num_classes'],
        num_heads=4,
        dropout=0.2
    )
    
    # Load first BCI file
    bci_dir = BCI_DATA_DIR / "dataset2a"
    gdf_files = list(bci_dir.glob("*.gdf"))
    
    if not gdf_files:
        print("❌ No BCI files found. Please run utils/download_bci_dataset.py first.")
        return
    
    print(f"Testing with {gdf_files[0].name}...")
    
    # Load data
    data_dict = load_bci_data(gdf_files[0])
    if data_dict is None:
        return
    
    # Process trials
    X_fast, X_slow, y = process_bci_trials_for_attention(
        data_dict['data'], 
        data_dict['labels'], 
        processor
    )
    
    if len(X_fast) == 0:
        print("❌ No valid trials found.")
        return
    
    print(f"Processed {len(X_fast)} samples")
    print(f"Feature shapes: X_fast {X_fast.shape}, X_slow {X_slow.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Quick training test
    print("\nRunning quick attention training test...")
    (fast_test, slow_test, labels_test, 
     train_losses, test_accuracies, attention_weights) = processor.train_attention_models(
        X_fast, X_slow, y, test_size=0.2, num_epochs=3
    )
    
    results = processor.evaluate_attention_models(
        fast_test, slow_test, labels_test, attention_weights
    )
    
    print(f"\n✅ Quick attention test complete!")
    print(f"Attention model accuracy: {results['accuracy']:.3f}")
    
    # Visualize attention
    processor.visualize_attention(gdf_files[0].stem)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_attention_test()
    else:
        run_attention_bci_experiment() 