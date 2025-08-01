#!/usr/bin/env python3
"""
Debug script for attention model with processed BCI data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from attention_modules import AttentionFastRNN, AttentionSlowRNN
from config import BCI_DATA_DIR, FEATURE_CONFIG, BCI_CONFIG

def debug_processed_data():
    """Debug the processed BCI data."""
    
    print("=== Debugging Processed BCI Data ===")
    
    # Load processed data
    processed_file = BCI_DATA_DIR / "dataset2a" / "processed_bci_data.npz"
    data = np.load(processed_file, allow_pickle=True)
    
    X_fast = data['X_fast']
    X_slow = data['X_slow']
    y = data['y']
    
    print(f"Data shapes:")
    print(f"  X_fast: {X_fast.shape}")
    print(f"  X_slow: {X_slow.shape}")
    print(f"  y: {y.shape}")
    
    print(f"\nData stats:")
    print(f"  X_fast - mean: {X_fast.mean():.6f}, std: {X_fast.std():.6f}")
    print(f"  X_slow - mean: {X_slow.mean():.6f}, std: {X_slow.std():.6f}")
    
    print(f"\nLabel analysis:")
    print(f"  Unique labels: {np.unique(y)}")
    print(f"  Label distribution: {np.bincount(y)}")
    
    # Check if data is already in sequence format
    print(f"\nData format analysis:")
    print(f"  X_fast shape: {X_fast.shape}")
    print(f"  Expected for RNN: (batch_size, seq_len, features)")
    print(f"  Current format: (samples, features)")
    
    return X_fast, X_slow, y

def debug_attention_model_with_processed_data():
    """Debug attention model with processed data."""
    
    print("\n=== Debugging Attention Model with Processed Data ===")
    
    # Load data
    X_fast, X_slow, y = debug_processed_data()
    
    # Create models
    num_classes = len(np.unique(y))
    fast_input_size = X_fast.shape[1]
    slow_input_size = X_slow.shape[1]
    
    print(f"\nModel parameters:")
    print(f"  Fast input size: {fast_input_size}")
    print(f"  Slow input size: {slow_input_size}")
    print(f"  Num classes: {num_classes}")
    
    # Create attention models
    fast_rnn = AttentionFastRNN(
        input_size=fast_input_size, 
        hidden_size=32, 
        num_classes=num_classes,
        num_heads=4, 
        dropout=0.2
    )
    
    slow_rnn = AttentionSlowRNN(
        input_size=slow_input_size, 
        hidden_size=16, 
        num_classes=num_classes,
        num_heads=2, 
        dropout=0.2
    )
    
    print(f"\nModel architectures:")
    print(f"  Fast RNN parameters: {sum(p.numel() for p in fast_rnn.parameters())}")
    print(f"  Slow RNN parameters: {sum(p.numel() for p in slow_rnn.parameters())}")
    
    # Test with a small batch
    batch_size = 8
    seq_len = 10
    
    # Reshape data for RNN (create sequences)
    if len(X_fast) >= seq_len:
        # Create sequences by taking consecutive samples
        fast_seq = X_fast[:seq_len]  # (seq_len, features)
        slow_seq = X_slow[:seq_len]  # (seq_len, features)
        
        # Repeat for batch
        fast_batch = torch.FloatTensor(fast_seq).unsqueeze(0).repeat(batch_size, 1, 1)
        slow_batch = torch.FloatTensor(slow_seq).unsqueeze(0).repeat(batch_size, 1, 1)
        labels_batch = torch.LongTensor([y[0]] * batch_size)
        
        print(f"\nBatch shapes:")
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
    
    print("🔍 Debugging Attention Model with Processed BCI Data")
    print("=" * 60)
    
    try:
        debug_attention_model_with_processed_data()
        
        print("\n" + "=" * 60)
        print("✅ Debug complete!")
        
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 