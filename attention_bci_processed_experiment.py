#!/usr/bin/env python3
"""
Attention-based BCI experiment with preprocessed BCI Competition data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from einops import rearrange, reduce

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

def load_processed_bci_data():
    """Load preprocessed BCI Competition data."""
    
    print("Loading preprocessed BCI data...")
    
    processed_file = BCI_DATA_DIR / "dataset2a" / "processed_bci_data.npz"
    
    if not processed_file.exists():
        print("❌ Processed BCI data not found.")
        return None
    
    try:
        # Load the processed data with allow_pickle=True
        data = np.load(processed_file, allow_pickle=True)
        
        # Print available keys
        print(f"Available data keys: {list(data.keys())}")
        
        # Load the data - check for different possible key combinations
        if 'X_fast' in data and 'X_slow' in data and 'y' in data:
            X_fast = data['X_fast']
            X_slow = data['X_slow']
            y = data['y']
            
            print(f"  Fast data shape: {X_fast.shape}")
            print(f"  Slow data shape: {X_slow.shape}")
            print(f"  Labels shape: {y.shape}")
            print(f"  Label distribution: {np.bincount(y)}")
            print(f"  Fast data stats - mean: {X_fast.mean():.6f}, std: {X_fast.std():.6f}")
            print(f"  Slow data stats - mean: {X_slow.mean():.6f}, std: {X_slow.std():.6f}")
            
            return {
                'X_fast': X_fast,
                'X_slow': X_slow,
                'y': y,
                'labels': data.get('labels', None),
                'info': data.get('info', None)
            }
        elif 'X' in data and 'y' in data:
            X = data['X']
            y = data['y']
            
            print(f"  Data shape: {X.shape}")
            print(f"  Labels shape: {y.shape}")
            print(f"  Label distribution: {np.bincount(y)}")
            print(f"  Data stats - mean: {X.mean():.6f}, std: {X.std():.6f}")
            
            return {
                'X': X,
                'y': y,
                'feature_names': data.get('feature_names', None),
                'subject_ids': data.get('subject_ids', None)
            }
        else:
            print("❌ Expected data keys not found in processed data.")
            return None
            
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        return None

def process_processed_data_for_attention(data_dict, processor):
    """Process preprocessed BCI data for attention."""
    
    print(f"Processing preprocessed data for attention...")
    
    # Check if we have pre-extracted features
    if 'X_fast' in data_dict and 'X_slow' in data_dict:
        print(f"  Using pre-extracted features")
        X_fast = data_dict['X_fast']
        X_slow = data_dict['X_slow']
        y = data_dict['y']
        
        print(f"  Fast features shape: {X_fast.shape}")
        print(f"  Slow features shape: {X_slow.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Label distribution: {np.bincount(y)}")
        
        # Check feature quality
        print(f"  Feature quality check:")
        print(f"    Fast features - mean: {X_fast.mean():.6f}, std: {X_fast.std():.6f}")
        print(f"    Slow features - mean: {X_slow.mean():.6f}, std: {X_slow.std():.6f}")
        
        return X_fast, X_slow, y
    
    # Otherwise, process raw data (only if processor is provided)
    if processor is None:
        print("❌ Processor is required for raw data processing")
        return np.array([]), np.array([]), np.array([])
    
    X = data_dict['X']
    y = data_dict['y']
    
    print(f"  Original data shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    
    # If X is 3D (samples, channels, timepoints), use first channel
    if len(X.shape) == 3:
        print(f"  Using first channel from {X.shape[1]} channels")
        single_channel_data = X[:, 0, :]  # (samples, timepoints)
    else:
        single_channel_data = X  # Assume it's already 2D
    
    print(f"  Single channel data shape: {single_channel_data.shape}")
    print(f"  Single channel stats - mean: {single_channel_data.mean():.6f}, std: {single_channel_data.std():.6f}")
    
    # Extract features for each sample
    all_X_fast = []
    all_X_slow = []
    all_y = []
    
    for i, (sample, label) in enumerate(zip(single_channel_data, y)):
        # Extract features using the attention processor
        fast_features, slow_features = processor.extract_features(sample)
        
        if len(fast_features) > 0 and len(slow_features) > 0:
            # Use the sample label for all features from this sample
            all_X_fast.extend(fast_features)
            all_X_slow.extend(slow_features)
            all_labels = [label] * len(fast_features)
            all_y.extend(all_labels)
    
    X_fast = np.array(all_X_fast)
    X_slow = np.array(all_X_slow)
    y_processed = np.array(all_y)
    
    print(f"  Fast features shape: {X_fast.shape}")
    print(f"  Slow features shape: {X_slow.shape}")
    print(f"  Labels shape: {y_processed.shape}")
    print(f"  Label distribution: {np.bincount(y_processed)}")
    
    # Check feature quality
    print(f"  Feature quality check:")
    print(f"    Fast features - mean: {X_fast.mean():.6f}, std: {X_fast.std():.6f}")
    print(f"    Slow features - mean: {X_slow.mean():.6f}, std: {X_slow.std():.6f}")
    
    return X_fast, X_slow, y_processed

def create_sequences_for_rnn(X, y, seq_length=10):
    """Create sequences for RNN training from feature data."""
    
    print(f"Creating sequences for RNN...")
    print(f"  Input shape: {X.shape}")
    print(f"  Sequence length: {seq_length}")
    
    sequences = []
    sequence_labels = []
    
    for i in range(0, len(X) - seq_length + 1, seq_length // 2):  # 50% overlap
        seq = X[i:i + seq_length]
        # Use the label of the middle sample in the sequence
        label = y[i + seq_length // 2]
        
        sequences.append(seq)
        sequence_labels.append(label)
    
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    
    print(f"  Output shapes:")
    print(f"    Sequences: {sequences.shape}")
    print(f"    Labels: {sequence_labels.shape}")
    print(f"    Label distribution: {np.bincount(sequence_labels)}")
    
    return sequences, sequence_labels

def run_attention_processed_experiment():
    """Run attention-based BCI experiment with preprocessed data."""
    
    print("=== Attention-Based BCI Experiment (Preprocessed Data) ===")
    
    # Initialize attention processor
    processor = AttentionHierarchicalProcessor(
        fast_window_size=FEATURE_CONFIG['fast_window_ms'],
        slow_context_size=FEATURE_CONFIG['slow_window_ms'],
        num_classes=BCI_CONFIG['num_classes'],
        num_heads=4,
        dropout=0.2
    )
    
    # Load preprocessed BCI data
    data_dict = load_processed_bci_data()
    if data_dict is None:
        return
    
    # Process data for attention
    X_fast, X_slow, y = process_processed_data_for_attention(data_dict, processor)
    
    if len(X_fast) == 0:
        print("❌ No valid features extracted.")
        return
    
    print(f"Processed {len(X_fast)} samples")
    print(f"Feature shapes: X_fast {X_fast.shape}, X_slow {X_slow.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Train attention models
    print("\nTraining attention models...")
    (fast_test, slow_test, labels_test, 
     train_losses, test_accuracies, attention_weights) = processor.train_attention_models(
        X_fast, X_slow, y, test_size=0.2, num_epochs=5
    )
    
    # Evaluate models
    print("\nEvaluating attention models...")
    results = processor.evaluate_attention_models(
        fast_test, slow_test, labels_test, attention_weights
    )
    
    # Visualize attention
    print("\nVisualizing attention patterns...")
    processor.visualize_attention("processed_data")
    
    # Save results
    processor.save_attention_results(results, "processed_data")
    
    print(f"\n✅ Attention experiment with preprocessed data complete!")
    print(f"Attention model accuracy: {results['accuracy']:.3f}")
    
    return results

def quick_attention_processed_test():
    """Quick test with preprocessed BCI data."""
    
    print("=== Quick Attention BCI Test (Preprocessed Data) ===")
    
    # Load preprocessed BCI data
    data_dict = load_processed_bci_data()
    if data_dict is None:
        return
    
    # Process data for attention
    X_fast, X_slow, y = process_processed_data_for_attention(data_dict, None)  # Pass None for now
    
    if len(X_fast) == 0:
        print("❌ No valid features extracted.")
        return
    
    print(f"Processed {len(X_fast)} samples")
    print(f"Feature shapes: X_fast {X_fast.shape}, X_slow {X_slow.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Create attention models with correct dimensions
    num_classes = len(np.unique(y))  # This should be 10 (0-9)
    fast_input_size = X_fast.shape[1]
    slow_input_size = X_slow.shape[1]
    
    print(f"\nCreating attention models:")
    print(f"  Fast input size: {fast_input_size}")
    print(f"  Slow input size: {slow_input_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Class range: {y.min()} to {y.max()}")
    
    # Initialize attention processor with correct dimensions
    processor = AttentionHierarchicalProcessor(
        fast_window_size=FEATURE_CONFIG['fast_window_ms'],
        slow_context_size=FEATURE_CONFIG['slow_window_ms'],
        num_classes=num_classes,  # Use actual number of classes
        num_heads=4,
        dropout=0.2
    )
    
    # Initialize models with correct dimensions
    processor.initialize_models(fast_input_size, slow_input_size)
    
    # Quick training test with feature data directly (no double sequencing)
    print("\nRunning quick attention training test with feature data...")
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
    processor.visualize_attention("processed_data_quick")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_attention_processed_test()
    else:
        run_attention_processed_experiment() 