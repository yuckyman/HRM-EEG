#!/usr/bin/env python3
"""
Debug script for attention model issues with real BCI data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne

from attention_modules import AttentionFastRNN, AttentionSlowRNN
from config import BCI_DATA_DIR, FEATURE_CONFIG, BCI_CONFIG

def debug_bci_data_loading():
    """Debug BCI data loading and processing."""
    
    print("=== Debugging BCI Data Loading ===")
    
    # Load first BCI file
    bci_dir = BCI_DATA_DIR / "dataset2a"
    gdf_files = list(bci_dir.glob("*.gdf"))
    
    if not gdf_files:
        print("❌ No BCI files found.")
        return
    
    gdf_file = gdf_files[0]
    print(f"Testing with {gdf_file.name}...")
    
    # Load data
    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    print(f"Raw data shape: {raw.get_data().shape}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Number of events: {len(events)}")
    print(f"Event types: {event_dict}")
    
    # Analyze events
    event_types = [event[2] for event in events]
    unique_events, counts = np.unique(event_types, return_counts=True)
    
    print(f"\nEvent analysis:")
    for event_type, count in zip(unique_events, counts):
        print(f"  Event {event_type}: {count} occurrences")
    
    # Extract motor imagery trials
    data = raw.get_data()
    trial_data = []
    trial_labels = []
    
    for event in events:
        if event[2] in [1, 2, 3, 4]:  # Motor imagery events
            start_sample = event[0]
            end_sample = start_sample + 1000  # 4 seconds
            
            if end_sample <= data.shape[1]:
                trial = data[:, start_sample:end_sample]
                trial_data.append(trial)
                trial_labels.append(event[2] - 1)  # Convert 1-4 to 0-3
    
    trial_data = np.array(trial_data)
    trial_labels = np.array(trial_labels)
    
    print(f"\nTrial analysis:")
    print(f"  Trial data shape: {trial_data.shape}")
    print(f"  Trial labels shape: {trial_labels.shape}")
    print(f"  Label distribution: {np.bincount(trial_labels)}")
    print(f"  Unique labels: {np.unique(trial_labels)}")
    
    return trial_data, trial_labels

def debug_feature_extraction(trial_data, trial_labels):
    """Debug feature extraction process."""
    
    print("\n=== Debugging Feature Extraction ===")
    
    # Test with first trial
    first_trial = trial_data[0]  # (channels, timepoints)
    first_label = trial_labels[0]
    
    print(f"First trial shape: {first_trial.shape}")
    print(f"First trial label: {first_label}")
    
    # Use first channel for testing
    single_channel_data = first_trial[0, :]  # (timepoints,)
    print(f"Single channel data shape: {single_channel_data.shape}")
    print(f"Single channel data stats: mean={single_channel_data.mean():.3f}, std={single_channel_data.std():.3f}")
    
    # Manual feature extraction
    fast_window_size = FEATURE_CONFIG['fast_window_ms']
    slow_context_size = FEATURE_CONFIG['slow_window_ms']
    
    print(f"\nFeature extraction parameters:")
    print(f"  Fast window size: {fast_window_size}ms")
    print(f"  Slow context size: {slow_context_size}ms")
    
    # Extract fast features manually
    fast_features = []
    for i in range(0, len(single_channel_data) - fast_window_size, fast_window_size // 2):
        window = single_channel_data[i:i + fast_window_size]
        if len(window) == fast_window_size:
            fast_features.append([
                np.mean(window),
                np.std(window),
                np.max(window) - np.min(window)  # Delta
            ])
    
    # Extract slow features manually
    slow_features = []
    for i in range(0, len(single_channel_data) - slow_context_size, slow_context_size // 2):
        window = single_channel_data[i:i + slow_context_size]
        if len(window) == slow_context_size:
            slow_features.append([
                np.mean(window),
                np.std(window)
            ])
    
    fast_features = np.array(fast_features)
    slow_features = np.array(slow_features)
    
    print(f"\nManual feature extraction:")
    print(f"  Fast features shape: {fast_features.shape}")
    print(f"  Slow features shape: {slow_features.shape}")
    print(f"  Fast features stats: mean={fast_features.mean():.3f}, std={fast_features.std():.3f}")
    print(f"  Slow features stats: mean={slow_features.mean():.3f}, std={slow_features.std():.3f}")
    
    return fast_features, slow_features

def debug_attention_model(fast_features, slow_features, labels):
    """Debug attention model with real data."""
    
    print("\n=== Debugging Attention Model ===")
    
    # Create attention models
    fast_rnn = AttentionFastRNN(input_size=3, hidden_size=32, num_classes=4)
    slow_rnn = AttentionSlowRNN(input_size=2, hidden_size=16, num_classes=4)
    
    print(f"Fast RNN parameters: {sum(p.numel() for p in fast_rnn.parameters())}")
    print(f"Slow RNN parameters: {sum(p.numel() for p in slow_rnn.parameters())}")
    
    # Test with a small batch
    batch_size = 8
    seq_len = 10
    
    # Create sequences
    if len(fast_features) >= seq_len:
        fast_seq = fast_features[:seq_len]  # (seq_len, 3)
        slow_seq = slow_features[:seq_len]  # (seq_len, 2)
        
        # Repeat for batch
        fast_batch = torch.FloatTensor(fast_seq).unsqueeze(0).repeat(batch_size, 1, 1)
        slow_batch = torch.FloatTensor(slow_seq).unsqueeze(0).repeat(batch_size, 1, 1)
        labels_batch = torch.LongTensor([labels[0]] * batch_size)
        
        print(f"Batch shapes:")
        print(f"  Fast batch: {fast_batch.shape}")
        print(f"  Slow batch: {slow_batch.shape}")
        print(f"  Labels batch: {labels_batch.shape}")
        
        # Test forward pass
        with torch.no_grad():
            fast_output, fast_attention = fast_rnn(fast_batch)
            slow_output, slow_attention = slow_rnn(slow_batch)
            
            print(f"\nForward pass results:")
            print(f"  Fast output shape: {fast_output.shape}")
            print(f"  Slow output shape: {slow_output.shape}")
            print(f"  Fast attention shape: {fast_attention.shape}")
            print(f"  Slow attention shape: {slow_attention.shape}")
            
            print(f"  Fast output stats: mean={fast_output.mean():.3f}, std={fast_output.std():.3f}")
            print(f"  Slow output stats: mean={slow_output.mean():.3f}, std={slow_output.std():.3f}")
            
            # Test predictions
            fast_probs = torch.softmax(fast_output, dim=1)
            slow_probs = torch.softmax(slow_output, dim=1)
            
            print(f"  Fast probabilities: {fast_probs[0]}")
            print(f"  Slow probabilities: {slow_probs[0]}")
            
            fast_pred = torch.argmax(fast_output, dim=1)
            slow_pred = torch.argmax(slow_output, dim=1)
            
            print(f"  Fast predictions: {fast_pred}")
            print(f"  Slow predictions: {slow_pred}")
            print(f"  True labels: {labels_batch}")
            
            # Calculate accuracy
            fast_acc = (fast_pred == labels_batch).float().mean()
            slow_acc = (slow_pred == labels_batch).float().mean()
            
            print(f"  Fast accuracy: {fast_acc:.3f}")
            print(f"  Slow accuracy: {slow_acc:.3f}")

def main():
    """Run all debug tests."""
    
    print("🔍 Debugging Attention Model with Real BCI Data")
    print("=" * 60)
    
    try:
        # Debug data loading
        trial_data, trial_labels = debug_bci_data_loading()
        
        # Debug feature extraction
        fast_features, slow_features = debug_feature_extraction(trial_data, trial_labels)
        
        # Debug attention model
        debug_attention_model(fast_features, slow_features, trial_labels)
        
        print("\n" + "=" * 60)
        print("✅ Debug complete!")
        
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 