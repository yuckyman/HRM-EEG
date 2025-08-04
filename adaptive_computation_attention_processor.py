#!/usr/bin/env python3
"""
Adaptive Computation Time (ACT) Attention Processor for EEG
Dynamically adapts computation based on input complexity
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime
from einops import rearrange, reduce, repeat

from multi_channel_attention_processor import (
    MultiChannelSpatialAttention, MultiChannelFastRNN, MultiChannelSlowRNN,
    MultiChannelIntegrationNet, MultiChannelAttentionProcessor
)
from attention_modules import (
    TemporalAttention, CrossModalAttention, AttentionVisualizer,
    analyze_brain_region_attention
)
from config import (
    get_experiment_results_path, get_model_comparison_path, get_log_path,
    MODEL_CONFIG, TRAINING_CONFIG, BCI_CONFIG, FEATURE_CONFIG
)

class HaltingNetwork(nn.Module):
    """Network that decides when to stop computation."""
    
    def __init__(self, input_size: int, hidden_size: int = 16):
        super().__init__()
        self.halting_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute halting probability.
        
        Args:
            x: Input features of shape (batch_size, input_size)
            
        Returns:
            halt_prob: Halting probability of shape (batch_size, 1)
        """
        return self.halting_network(x)

class AdaptiveComputationFastRNN(nn.Module):
    """Fast RNN with adaptive computation time."""
    
    def __init__(self, num_channels=25, hidden_size=32, num_classes=4,
                 num_heads=4, dropout=0.2, max_steps=10, halting_threshold=0.01):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.halting_threshold = halting_threshold
        
        # Channel-wise feature extraction
        self.channel_encoder = nn.Linear(1, hidden_size // 2)
        
        # Multi-channel spatial attention
        self.spatial_attention = MultiChannelSpatialAttention(
            num_channels=num_channels,
            hidden_size=hidden_size // 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Temporal processing
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Halting network
        self.halting_network = HaltingNetwork(hidden_size)
        
        # Feature output
        self.feature_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive computation time.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_channels)
            
        Returns:
            features: Feature vectors of shape (batch_size, hidden_size)
            attention_weights: Dictionary containing all attention weights
        """
        batch_size, seq_len, num_channels = x.shape
        
        # 1. Channel-wise feature encoding
        x_reshaped = x.unsqueeze(-1)  # (batch, seq, channels, 1)
        channel_features = self.channel_encoder(x_reshaped)  # (batch, seq, channels, hidden//2)
        
        # 2. Multi-channel spatial attention
        spatial_attended, channel_weights, spatial_attention = self.spatial_attention(channel_features)
        
        # 3. Global pooling across channels
        pooled_features = spatial_attended.mean(dim=2)  # (batch, seq, hidden//2)
        
        # 4. Adaptive computation time
        accumulated_output = torch.zeros(batch_size, self.hidden_size, device=x.device)
        accumulated_halting = torch.zeros(batch_size, 1, device=x.device)
        step_count = 0
        
        # Initialize LSTM state
        lstm_state = None
        
        for step in range(self.max_steps):
            # Process one timestep at a time
            if step == 0:
                lstm_out, lstm_state = self.temporal_lstm(pooled_features)
            else:
                # Use the last output as input for next step
                lstm_out, lstm_state = self.temporal_lstm(
                    pooled_features, lstm_state
                )
            
            # Apply temporal attention
            temporal_attended, temporal_attention = self.temporal_attention(lstm_out)
            
            # Global average pooling over time
            pooled = reduce(temporal_attended, 'b s h -> b h', 'mean')
            
            # Compute halting probability
            halt_prob = self.halting_network(pooled)
            
            # Accumulate weighted output
            accumulated_output += halt_prob * pooled
            accumulated_halting += halt_prob
            step_count += 1
            
            # Check if we should stop
            if accumulated_halting.mean() > (1.0 - self.halting_threshold):
                break
        
        # 5. Feature output
        features = self.feature_output(accumulated_output)
        
        # Return attention weights for analysis
        attention_weights = {
            'channel_weights': channel_weights,
            'spatial_attention': spatial_attention,
            'temporal_attention': temporal_attention,
            'halting_probabilities': accumulated_halting,
            'computation_steps': step_count
        }
        
        return features, attention_weights

class AdaptiveComputationSlowRNN(nn.Module):
    """Slow RNN with adaptive computation time."""
    
    def __init__(self, num_channels=25, hidden_size=16, num_classes=4,
                 num_heads=2, dropout=0.2, max_steps=8, halting_threshold=0.01):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.halting_threshold = halting_threshold
        
        # Channel-wise feature extraction
        self.channel_encoder = nn.Linear(1, hidden_size // 2)
        
        # Multi-channel spatial attention
        self.spatial_attention = MultiChannelSpatialAttention(
            num_channels=num_channels,
            hidden_size=hidden_size // 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Temporal processing
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Halting network
        self.halting_network = HaltingNetwork(hidden_size)
        
        # Feature output
        self.feature_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive computation time.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_channels)
            
        Returns:
            features: Feature vectors of shape (batch_size, hidden_size)
            attention_weights: Dictionary containing all attention weights
        """
        batch_size, seq_len, num_channels = x.shape
        
        # 1. Channel-wise feature encoding
        x_reshaped = x.unsqueeze(-1)  # (batch, seq, channels, 1)
        channel_features = self.channel_encoder(x_reshaped)  # (batch, seq, channels, hidden//2)
        
        # 2. Multi-channel spatial attention
        spatial_attended, channel_weights, spatial_attention = self.spatial_attention(channel_features)
        
        # 3. Global pooling across channels
        pooled_features = spatial_attended.mean(dim=2)  # (batch, seq, hidden//2)
        
        # 4. Adaptive computation time
        accumulated_output = torch.zeros(batch_size, self.hidden_size, device=x.device)
        accumulated_halting = torch.zeros(batch_size, 1, device=x.device)
        step_count = 0
        
        # Initialize LSTM state
        lstm_state = None
        
        for step in range(self.max_steps):
            # Process one timestep at a time
            if step == 0:
                lstm_out, lstm_state = self.temporal_lstm(pooled_features)
            else:
                # Use the last output as input for next step
                lstm_out, lstm_state = self.temporal_lstm(
                    pooled_features, lstm_state
                )
            
            # Apply temporal attention
            temporal_attended, temporal_attention = self.temporal_attention(lstm_out)
            
            # Global average pooling over time
            pooled = reduce(temporal_attended, 'b s h -> b h', 'mean')
            
            # Compute halting probability
            halt_prob = self.halting_network(pooled)
            
            # Accumulate weighted output
            accumulated_output += halt_prob * pooled
            accumulated_halting += halt_prob
            step_count += 1
            
            # Check if we should stop
            if accumulated_halting.mean() > (1.0 - self.halting_threshold):
                break
        
        # 5. Feature output
        features = self.feature_output(accumulated_output)
        
        # Return attention weights for analysis
        attention_weights = {
            'channel_weights': channel_weights,
            'spatial_attention': spatial_attention,
            'temporal_attention': temporal_attention,
            'halting_probabilities': accumulated_halting,
            'computation_steps': step_count
        }
        
        return features, attention_weights

class AdaptiveComputationIntegrationNet(nn.Module):
    """Integration network with adaptive computation time."""
    
    def __init__(self, fast_size=32, slow_size=16, hidden_size=32, 
                 num_classes=4, num_heads=4, dropout=0.2, max_steps=6):
        super().__init__()
        self.fast_size = fast_size
        self.slow_size = slow_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_steps = max_steps
        
        # Cross-modal attention for feature-level integration
        self.cross_attention = CrossModalAttention(
            fast_size=fast_size,
            slow_size=slow_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Halting network for integration
        self.integration_halting = HaltingNetwork(fast_size + slow_size)
        
        # Integration layers
        self.integration = nn.Sequential(
            nn.Linear(fast_size + slow_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, fast_features: torch.Tensor, slow_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate fast and slow features with adaptive computation time.
        
        Args:
            fast_features: Fast features of shape (batch_size, fast_size)
            slow_features: Slow features of shape (batch_size, slow_size)
            
        Returns:
            output: Classification logits of shape (batch_size, num_classes)
            attention_weights: Cross-modal attention weights
        """
        # Apply cross-modal attention between features
        attended_fast, attention_weights = self.cross_attention(fast_features, slow_features)
        
        # Remove sequence dimension if it was added
        if attended_fast.dim() == 3:
            attended_fast = attended_fast.squeeze(1)  # (batch, fast_size)
        
        # Concatenate attended fast features with slow features
        combined = torch.cat([attended_fast, slow_features], dim=1)
        
        # Adaptive computation for integration
        accumulated_output = torch.zeros(combined.shape[0], self.hidden_size, device=combined.device)
        accumulated_halting = torch.zeros(combined.shape[0], 1, device=combined.device)
        
        for step in range(self.max_steps):
            # Process integration with halting
            halt_prob = self.integration_halting(combined)
            
            # Simple integration step - use first layer only
            integration_step = self.integration[0](combined)  # First layer only
            
            accumulated_output += halt_prob * integration_step
            accumulated_halting += halt_prob
            
            if accumulated_halting.mean() > 0.99:
                break
        
        # Final classification with remaining layers
        output = self.integration[1:](accumulated_output)  # Remaining layers
        
        return output, attention_weights

class AdaptiveComputationAttentionProcessor(MultiChannelAttentionProcessor):
    """Multi-channel attention processor with adaptive computation time."""
    
    def __init__(self, num_channels=25, fast_window_size=10, slow_context_size=50,
                 num_classes=4, num_heads=4, dropout=0.2, learning_rate=0.001,
                 fast_max_steps=10, slow_max_steps=8, integration_max_steps=6,
                 halting_threshold=0.01):
        super().__init__(num_channels, fast_window_size, slow_context_size,
                        num_classes, num_heads, dropout, learning_rate)
        
        # Override with adaptive computation models
        self.fast_rnn = AdaptiveComputationFastRNN(
            num_channels=num_channels,
            hidden_size=32,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout,
            max_steps=fast_max_steps,
            halting_threshold=halting_threshold
        )
        
        self.slow_rnn = AdaptiveComputationSlowRNN(
            num_channels=num_channels,
            hidden_size=16,
            num_classes=num_classes,
            num_heads=num_heads // 2,
            dropout=dropout,
            max_steps=slow_max_steps,
            halting_threshold=halting_threshold
        )
        
        self.integration_net = AdaptiveComputationIntegrationNet(
            fast_size=32,
            slow_size=16,
            hidden_size=32,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout,
            max_steps=integration_max_steps
        )
        
        # Reinitialize optimizer with new models
        self.optimizer = optim.Adam([
            {'params': self.fast_rnn.parameters()},
            {'params': self.slow_rnn.parameters()},
            {'params': self.integration_net.parameters()}
        ], lr=learning_rate)
        
        # Store ACT parameters
        self.fast_max_steps = fast_max_steps
        self.slow_max_steps = slow_max_steps
        self.integration_max_steps = integration_max_steps
        self.halting_threshold = halting_threshold
        
    def analyze_adaptive_computation(self) -> Dict:
        """
        Analyze adaptive computation patterns.
        
        Returns:
            analysis: Dictionary with ACT analysis
        """
        fast_attention = self.attention_weights['fast']
        slow_attention = self.attention_weights['slow']
        
        fast_steps = fast_attention['computation_steps']
        slow_steps = slow_attention['computation_steps']
        fast_halting = fast_attention['halting_probabilities'].mean().item()
        slow_halting = slow_attention['halting_probabilities'].mean().item()
        
        analysis = {
            'fast_computation_steps': fast_steps,
            'slow_computation_steps': slow_steps,
            'fast_halting_probability': fast_halting,
            'slow_halting_probability': slow_halting,
            'computation_efficiency': {
                'fast_efficiency': fast_steps / self.fast_max_steps,
                'slow_efficiency': slow_steps / self.slow_max_steps
            },
            'adaptive_behavior': {
                'fast_early_stopping': fast_steps < self.fast_max_steps,
                'slow_early_stopping': slow_steps < self.slow_max_steps
            }
        }
        
        print(f"Adaptive Computation Analysis:")
        print(f"  Fast RNN Steps: {fast_steps}/{self.fast_max_steps} ({fast_steps/self.fast_max_steps:.1%} efficiency)")
        print(f"  Slow RNN Steps: {slow_steps}/{self.slow_max_steps} ({slow_steps/self.slow_max_steps:.1%} efficiency)")
        print(f"  Fast Halting Probability: {fast_halting:.3f}")
        print(f"  Slow Halting Probability: {slow_halting:.3f}")
        
        return analysis
    
    def save_adaptive_computation_results(self, results: Dict, subject_id: str) -> None:
        """
        Save adaptive computation results.
        
        Args:
            results: Results dictionary
            subject_id: Subject identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = get_experiment_results_path()
        results_path.mkdir(parents=True, exist_ok=True)
        results_path = results_path / f"adaptive_computation_attention_{subject_id}_{timestamp}.json"
        
        # Convert tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                serializable_results[key] = value.detach().cpu().numpy().tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        serializable_results[key][subkey] = subvalue.detach().cpu().numpy().tolist()
                    elif isinstance(subvalue, dict):
                        # Handle nested dictionaries
                        serializable_results[key][subkey] = {}
                        for nested_key, nested_value in subvalue.items():
                            if isinstance(nested_value, torch.Tensor):
                                serializable_results[key][subkey][nested_key] = nested_value.detach().cpu().numpy().tolist()
                            elif isinstance(nested_value, dict):
                                # Handle deeply nested dictionaries
                                serializable_results[key][subkey][nested_key] = {}
                                for deep_key, deep_value in nested_value.items():
                                    if isinstance(deep_value, torch.Tensor):
                                        serializable_results[key][subkey][nested_key][deep_key] = deep_value.detach().cpu().numpy().tolist()
                                    else:
                                        serializable_results[key][subkey][nested_key][deep_key] = deep_value
                            else:
                                serializable_results[key][subkey][nested_key] = nested_value
                    else:
                        serializable_results[key][subkey] = subvalue
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Adaptive computation results saved to {results_path}") 