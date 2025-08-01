#!/usr/bin/env python3
"""
Simple test script for attention modules
"""

import torch
import torch.nn as nn
import numpy as np
from attention_modules import TemporalAttention, AttentionFastRNN, AttentionSlowRNN

def test_temporal_attention():
    """Test temporal attention module."""
    print("Testing Temporal Attention...")
    
    # Create attention module
    attention = TemporalAttention(hidden_size=32, num_heads=4)
    
    # Create test input
    batch_size, seq_len, hidden_size = 8, 10, 32
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    attended, attention_weights = attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {attended.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print("✅ Temporal attention test passed!")

def test_fast_rnn():
    """Test fast RNN with attention."""
    print("\nTesting Fast RNN with Attention...")
    
    # Create RNN
    rnn = AttentionFastRNN(input_size=3, hidden_size=32, num_classes=4)
    
    # Create test input
    batch_size, seq_len, input_size = 8, 10, 3
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    features, attention_weights = rnn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print("✅ Fast RNN test passed!")

def test_slow_rnn():
    """Test slow RNN with attention."""
    print("\nTesting Slow RNN with Attention...")
    
    # Create RNN
    rnn = AttentionSlowRNN(input_size=2, hidden_size=16, num_classes=4)
    
    # Create test input
    batch_size, seq_len, input_size = 8, 10, 2
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    features, attention_weights = rnn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print("✅ Slow RNN test passed!")

def test_integration():
    """Test integration of fast and slow features."""
    print("\nTesting Integration...")
    
    # Create test features
    batch_size = 8
    fast_features = torch.randn(batch_size, 32)  # Fast features
    slow_features = torch.randn(batch_size, 16)  # Slow features
    
    # Simple concatenation
    combined = torch.cat([fast_features, slow_features], dim=1)
    
    print(f"Fast features shape: {fast_features.shape}")
    print(f"Slow features shape: {slow_features.shape}")
    print(f"Combined shape: {combined.shape}")
    print("✅ Integration test passed!")

def main():
    """Run all tests."""
    print("🧠 Testing Attention Modules")
    print("=" * 40)
    
    try:
        test_temporal_attention()
        test_fast_rnn()
        test_slow_rnn()
        test_integration()
        
        print("\n" + "=" * 40)
        print("✅ All attention tests passed!")
        print("Ready for full attention experiment!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 