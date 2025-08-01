#!/usr/bin/env python3
"""
Attention modules for hierarchical EEG processing
Implements temporal and spatial attention mechanisms for BCI classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from einops import rearrange, repeat, reduce

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for EEG sequences."""
    
    def __init__(self, hidden_size=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention to EEG sequence.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            attended: Attention-weighted output of same shape as input
            attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
        """
        # Apply self-attention: each timestep attends to all other timesteps
        attended, attention_weights = self.attention(x, x, x)
        
        # Residual connection + layer norm
        output = self.layer_norm(x + self.dropout(attended))
        
        return output, attention_weights

class ChannelAttention(nn.Module):
    """Spatial attention mechanism for multi-channel EEG."""
    
    def __init__(self, num_channels=22, hidden_size=32, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        
        # Channel attention network
        self.channel_attention = nn.Sequential(
            nn.Linear(num_channels, num_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels // 4, num_channels),
            nn.Sigmoid()
        )
        
        # Channel-wise feature processing
        self.channel_encoder = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial attention across EEG channels.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_channels, hidden_size)
            
        Returns:
            attended: Attention-weighted output of same shape as input
            channel_weights: Channel attention weights of shape (batch_size, num_channels)
        """
        batch_size, seq_len, num_channels, hidden_size = x.shape
        
        # Average over time to get channel importance
        channel_avg = x.mean(dim=1)  # (batch, num_channels, hidden_size)
        
        # Compute channel attention weights
        channel_importance = channel_avg.mean(dim=-1)  # (batch, num_channels)
        channel_weights = self.channel_attention(channel_importance)  # (batch, num_channels)
        
        # Apply attention across channels
        attended = x * channel_weights.unsqueeze(1).unsqueeze(-1)
        
        return attended, channel_weights

class CrossModalAttention(nn.Module):
    """Attention between fast and slow features."""
    
    def __init__(self, fast_size=32, slow_size=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.fast_size = fast_size
        self.slow_size = slow_size
        
        # Project to common space
        self.fast_projection = nn.Linear(fast_size, fast_size)
        self.slow_projection = nn.Linear(slow_size, fast_size)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fast_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(fast_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, fast_features: torch.Tensor, slow_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention between fast and slow features.
        
        Args:
            fast_features: Fast features of shape (batch_size, seq_len, fast_size) or (batch_size, fast_size)
            slow_features: Slow features of shape (batch_size, seq_len, slow_size) or (batch_size, slow_size)
            
        Returns:
            attended: Cross-modal attended features
            attention_weights: Cross-modal attention weights
        """
        # Handle both sequence and feature vector inputs
        if fast_features.dim() == 2:
            # Feature vectors: add sequence dimension
            fast_features = fast_features.unsqueeze(1)  # (batch, 1, fast_size)
            slow_features = slow_features.unsqueeze(1)  # (batch, 1, slow_size)
        
        # Project slow features to fast feature space
        slow_projected = self.slow_projection(slow_features)
        fast_projected = self.fast_projection(fast_features)
        
        # Apply cross-modal attention (fast attends to slow)
        attended, attention_weights = self.cross_attention(
            fast_projected, slow_projected, slow_projected
        )
        
        # Residual connection + layer norm
        output = self.layer_norm(fast_projected + self.dropout(attended))
        
        return output, attention_weights

class AttentionFastRNN(nn.Module):
    """Fast RNN with temporal attention."""
    
    def __init__(self, input_size=3, hidden_size=32, num_classes=4, 
                 num_heads=4, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=input_size,
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
        
        # Feature output (no classification head)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with temporal attention.
        
        Args:
            x: Input sequences of shape (batch_size, seq_len, input_size)
            
        Returns:
            features: Feature vectors of shape (batch_size, hidden_size)
            attention_weights: Temporal attention weights
        """
        # Use einops to handle dimensions properly
        # x: (batch_size, seq_len, input_size) or (batch_size, input_size)
        if x.dim() == 2:
            # Single feature vector per sample, add sequence dimension
            x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        batch_size, seq_len, input_size = x.shape
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Apply temporal attention
        attended, attention_weights = self.temporal_attention(lstm_out)
        
        # Global average pooling over time using einops
        # attended: (batch, seq_len, hidden_size) -> (batch, hidden_size)
        pooled = reduce(attended, 'b s h -> b h', 'mean')
        pooled = self.dropout(pooled)
        
        # Return features directly
        return pooled, attention_weights

class AttentionSlowRNN(nn.Module):
    """Slow RNN with temporal attention."""
    
    def __init__(self, input_size=2, hidden_size=16, num_classes=4,
                 num_heads=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # LSTM for slow temporal processing
        self.lstm = nn.LSTM(
            input_size=input_size,
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
        
        # Feature output (no classification head)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with temporal attention.
        
        Args:
            x: Input sequences of shape (batch_size, seq_len, input_size)
            
        Returns:
            features: Feature vectors of shape (batch_size, hidden_size)
            attention_weights: Temporal attention weights
        """
        # Handle both sequence and feature vector inputs
        if x.dim() == 2:
            # Single feature vector per sample, add sequence dimension
            x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Apply temporal attention
        attended, attention_weights = self.temporal_attention(lstm_out)
        
        # Global average pooling over time using einops
        pooled = reduce(attended, 'b s h -> b h', 'mean')
        pooled = self.dropout(pooled)
        
        # Return features directly
        return pooled, attention_weights

class AttentionIntegrationNet(nn.Module):
    """Integration network with cross-modal attention."""
    
    def __init__(self, fast_size=32, slow_size=16, hidden_size=32, 
                 num_classes=4, num_heads=4, dropout=0.2):
        super().__init__()
        self.fast_size = fast_size
        self.slow_size = slow_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
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
        
        # Cross-modal attention for feature-level integration
        self.cross_attention = CrossModalAttention(
            fast_size=fast_size,
            slow_size=slow_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, fast_features: torch.Tensor, slow_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate fast and slow features with cross-modal attention.
        
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
        
        # Final classification
        output = self.integration(combined)
        
        return output, attention_weights

class AttentionVisualizer:
    """Visualize attention weights for interpretability."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path("attention_plots")
        self.save_dir.mkdir(exist_ok=True)
        
    def visualize_temporal_attention(self, attention_weights: torch.Tensor, 
                                   timestamps: np.ndarray, subject_id: str,
                                   save_plot: bool = True) -> None:
        """
        Visualize temporal attention weights.
        
        Args:
            attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
            timestamps: Time points in milliseconds
            subject_id: Subject identifier
            save_plot: Whether to save the plot
        """
        # Average across batch
        avg_attention = attention_weights.mean(dim=0).detach().cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_attention, 
                    xticklabels=timestamps[::10],  # Every 10th timestep
                    yticklabels=timestamps[::10],
                    cmap='viridis',
                    cbar_kws={'label': 'Attention Weight'})
        plt.title(f'Temporal Attention Weights - Subject {subject_id}')
        plt.xlabel('Query Timepoint (ms)')
        plt.ylabel('Key Timepoint (ms)')
        
        if save_plot:
            plt.savefig(self.save_dir / f'temporal_attention_{subject_id}.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze attention patterns
        diagonal_attention = np.diag(avg_attention)  # Self-attention
        cross_attention = avg_attention - np.eye(avg_attention.shape[0])
        
        print(f"Self-attention strength: {diagonal_attention.mean():.3f}")
        print(f"Cross-attention strength: {cross_attention.mean():.3f}")
        
    def visualize_channel_attention(self, channel_weights: torch.Tensor,
                                  channel_names: List[str], subject_id: str,
                                  save_plot: bool = True) -> None:
        """
        Visualize channel attention weights.
        
        Args:
            channel_weights: Channel attention weights of shape (batch_size, num_channels)
            channel_names: List of channel names
            subject_id: Subject identifier
            save_plot: Whether to save the plot
        """
        # Average across batch
        avg_weights = channel_weights.mean(dim=0).detach().cpu().numpy()
        
        # Create bar plot
        plt.figure(figsize=(15, 6))
        bars = plt.bar(range(len(channel_names)), avg_weights)
        plt.xlabel('Channel')
        plt.ylabel('Attention Weight')
        plt.title(f'Channel Attention Weights - Subject {subject_id}')
        plt.xticks(range(len(channel_names)), channel_names, rotation=45)
        
        # Color bars by brain region
        brain_regions = {
            'motor': ['C3', 'C4', 'FC1', 'FC2', 'FC5', 'FC6'],
            'somatosensory': ['CP1', 'CP2', 'CP5', 'CP6'],
            'frontal': ['F3', 'F4', 'F7', 'F8'],
            'parietal': ['P3', 'P4', 'P7', 'P8'],
            'temporal': ['T7', 'T8'],
            'occipital': ['O1', 'O2']
        }
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (region, channels) in enumerate(brain_regions.items()):
            for j, ch in enumerate(channel_names):
                if ch in channels:
                    bars[j].set_color(colors[i % len(colors)])
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.save_dir / f'channel_attention_{subject_id}.png',
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top channels
        top_channels = np.argsort(avg_weights)[-5:]  # Top 5
        print(f"Top 5 attended channels for {subject_id}:")
        for i, idx in enumerate(reversed(top_channels)):
            print(f"  {i+1}. {channel_names[idx]}: {avg_weights[idx]:.3f}")

def analyze_brain_region_attention(attention_weights: torch.Tensor, 
                                 channel_names: List[str]) -> Dict[str, float]:
    """
    Analyze which brain regions the model focuses on.
    
    Args:
        attention_weights: Channel attention weights
        channel_names: List of channel names
        
    Returns:
        region_attention: Dictionary mapping brain regions to attention scores
    """
    # Group channels by brain region
    brain_regions = {
        'motor_cortex': ['C3', 'C4', 'FC1', 'FC2', 'FC5', 'FC6'],
        'somatosensory': ['CP1', 'CP2', 'CP5', 'CP6'],
        'frontal': ['F3', 'F4', 'F7', 'F8'],
        'parietal': ['P3', 'P4', 'P7', 'P8'],
        'temporal': ['T7', 'T8'],
        'occipital': ['O1', 'O2']
    }
    
    region_attention = {}
    avg_weights = attention_weights.mean(dim=0).detach().cpu().numpy()
    
    for region, channels in brain_regions.items():
        channel_indices = [i for i, ch in enumerate(channel_names) if ch in channels]
        if channel_indices:
            region_attention[region] = avg_weights[channel_indices].mean()
    
    return region_attention 