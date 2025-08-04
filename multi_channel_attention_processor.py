#!/usr/bin/env python3
"""
Multi-Channel Spatial Attention Processor for EEG
Extends attention mechanisms to handle all 22 EEG channels with spatial attention
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

from attention_modules import (
    TemporalAttention, CrossModalAttention, AttentionVisualizer,
    analyze_brain_region_attention
)
from config import (
    get_experiment_results_path, get_model_comparison_path, get_log_path,
    MODEL_CONFIG, TRAINING_CONFIG, BCI_CONFIG, FEATURE_CONFIG
)

class MultiChannelSpatialAttention(nn.Module):
    """Spatial attention mechanism for multi-channel EEG."""
    
    def __init__(self, num_channels=22, hidden_size=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Channel attention network
        self.channel_attention = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels // 2, num_channels),
            nn.Sigmoid()
        )
        
        # Multi-head spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply spatial attention across EEG channels.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_channels, hidden_size)
            
        Returns:
            attended: Attention-weighted output of shape (batch_size, seq_len, num_channels, hidden_size)
            channel_weights: Channel attention weights of shape (batch_size, num_channels)
            spatial_attention: Spatial attention weights of shape (batch_size, seq_len, num_channels, num_channels)
        """
        batch_size, seq_len, num_channels, hidden_size = x.shape
        
        # 1. Channel-level attention: learn which channels are important
        # Average over time and features to get channel importance
        channel_avg = x.mean(dim=(1, -1))  # (batch, num_channels)
        channel_weights = self.channel_attention(channel_avg)  # (batch, num_channels)
        
        # 2. Apply channel attention
        channel_attended = x * channel_weights.unsqueeze(1).unsqueeze(-1)
        
        # 3. Spatial attention: learn relationships between channels
        # Reshape for spatial attention: (batch*seq, num_channels, hidden_size)
        spatial_input = channel_attended.view(batch_size * seq_len, num_channels, hidden_size)
        
        # Apply spatial attention (channels attend to each other)
        spatial_attended, spatial_attention = self.spatial_attention(
            spatial_input, spatial_input, spatial_input
        )
        
        # Reshape back: (batch, seq_len, num_channels, hidden_size)
        spatial_attended = spatial_attended.view(batch_size, seq_len, num_channels, hidden_size)
        spatial_attention = spatial_attention.view(batch_size, seq_len, num_channels, num_channels)
        
        # 4. Residual connection + layer norm
        output = self.layer_norm(channel_attended + self.dropout(spatial_attended))
        
        return output, channel_weights, spatial_attention

class MultiChannelFastRNN(nn.Module):
    """Fast RNN with multi-channel spatial attention."""
    
    def __init__(self, num_channels=22, hidden_size=32, num_classes=10, 
                 num_heads=4, dropout=0.2):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Channel-wise feature extraction
        self.channel_encoder = nn.Linear(1, hidden_size // 2)  # Each channel gets encoded
        
        # Multi-channel spatial attention
        self.spatial_attention = MultiChannelSpatialAttention(
            num_channels=num_channels,
            hidden_size=hidden_size // 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Temporal processing after spatial attention
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
        
        # Feature output (no classification head for integration)
        self.feature_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multi-channel spatial attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_channels)
            
        Returns:
            output: Classification logits of shape (batch_size, num_classes)
            attention_weights: Dictionary containing all attention weights
        """
        batch_size, seq_len, num_channels = x.shape
        
        # 1. Channel-wise feature encoding
        # Reshape: (batch, seq, channels) -> (batch, seq, channels, 1)
        x_reshaped = x.unsqueeze(-1)  # (batch, seq, channels, 1)
        
        # Encode each channel
        channel_features = self.channel_encoder(x_reshaped)  # (batch, seq, channels, hidden//2)
        
        # 2. Multi-channel spatial attention
        spatial_attended, channel_weights, spatial_attention = self.spatial_attention(channel_features)
        
        # 3. Global pooling across channels
        # Average across channels: (batch, seq, channels, hidden//2) -> (batch, seq, hidden//2)
        pooled_features = spatial_attended.mean(dim=2)  # (batch, seq, hidden//2)
        
        # 4. Temporal processing
        lstm_out, _ = self.temporal_lstm(pooled_features)  # (batch, seq, hidden)
        
        # 5. Temporal attention
        temporal_attended, temporal_attention = self.temporal_attention(lstm_out)
        
        # 6. Global average pooling over time
        pooled = reduce(temporal_attended, 'b s h -> b h', 'mean')
        
        # 7. Feature output
        features = self.feature_output(pooled)
        
        # Return attention weights for analysis
        attention_weights = {
            'channel_weights': channel_weights,
            'spatial_attention': spatial_attention,
            'temporal_attention': temporal_attention
        }
        
        return features, attention_weights

class MultiChannelSlowRNN(nn.Module):
    """Slow RNN with multi-channel spatial attention."""
    
    def __init__(self, num_channels=22, hidden_size=16, num_classes=10,
                 num_heads=2, dropout=0.2):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Channel-wise feature extraction (smaller for slow features)
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
        
        # Feature output (no classification head for integration)
        self.feature_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multi-channel spatial attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_channels)
            
        Returns:
            output: Classification logits of shape (batch_size, num_classes)
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
        
        # 4. Temporal processing
        lstm_out, _ = self.temporal_lstm(pooled_features)  # (batch, seq, hidden)
        
        # 5. Temporal attention
        temporal_attended, temporal_attention = self.temporal_attention(lstm_out)
        
        # 6. Global average pooling over time
        pooled = reduce(temporal_attended, 'b s h -> b h', 'mean')
        
        # 7. Feature output
        features = self.feature_output(pooled)
        
        # Return attention weights for analysis
        attention_weights = {
            'channel_weights': channel_weights,
            'spatial_attention': spatial_attention,
            'temporal_attention': temporal_attention
        }
        
        return features, attention_weights

class MultiChannelIntegrationNet(nn.Module):
    """Integration network with multi-channel cross-modal attention."""
    
    def __init__(self, fast_size=32, slow_size=16, hidden_size=32, 
                 num_classes=10, num_heads=4, dropout=0.2):
        super().__init__()
        self.fast_size = fast_size
        self.slow_size = slow_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Cross-modal attention for feature-level integration
        self.cross_attention = CrossModalAttention(
            fast_size=fast_size,
            slow_size=slow_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
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

class MultiChannelAttentionProcessor:
    """Multi-channel attention processor for EEG classification."""
    
    def __init__(self, num_channels=25, fast_window_size=10, slow_context_size=50,
                 num_classes=10, num_heads=4, dropout=0.2, learning_rate=0.001):
        self.num_channels = num_channels
        self.fast_window_size = fast_window_size
        self.slow_context_size = slow_context_size
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Initialize models
        self.fast_rnn = MultiChannelFastRNN(
            num_channels=num_channels,
            hidden_size=32,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.slow_rnn = MultiChannelSlowRNN(
            num_channels=num_channels,
            hidden_size=16,
            num_classes=num_classes,
            num_heads=num_heads // 2,
            dropout=dropout
        )
        
        self.integration_net = MultiChannelIntegrationNet(
            fast_size=32,
            slow_size=16,
            hidden_size=32,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Optimizer and loss function
        self.optimizer = optim.Adam([
            {'params': self.fast_rnn.parameters()},
            {'params': self.slow_rnn.parameters()},
            {'params': self.integration_net.parameters()}
        ], lr=learning_rate)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Attention visualizer
        self.visualizer = AttentionVisualizer()
        
        # Store attention weights for analysis
        self.attention_weights = {}
        
    def extract_multi_channel_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fast and slow features from multi-channel EEG data.
        
        Args:
            eeg_data: EEG data of shape (channels, timepoints)
            
        Returns:
            fast_features: Fast features of shape (timepoints, channels)
            slow_features: Slow features of shape (timepoints, channels)
        """
        channels, timepoints = eeg_data.shape
        
        print(f"    Extracting features from {channels} channels, {timepoints} timepoints")
        
        # Fast features: short windows (10ms = 2.5 samples at 250Hz)
        fast_features = []
        for t in range(0, timepoints - self.fast_window_size, self.fast_window_size):
            window = eeg_data[:, t:t + self.fast_window_size]
            # Average over window to get (channels,) for each timepoint
            fast_features.append(window.mean(axis=1))  # (channels,)
        
        # Slow features: longer windows (50ms = 12.5 samples at 250Hz)
        slow_features = []
        for t in range(0, timepoints - self.slow_context_size, self.slow_context_size):
            window = eeg_data[:, t:t + self.slow_context_size]
            # Average over window to get (channels,) for each timepoint
            slow_features.append(window.mean(axis=1))  # (channels,)
        
        fast_array = np.array(fast_features)  # (timepoints, channels)
        slow_array = np.array(slow_features)  # (timepoints, channels)
        
        print(f"    Fast features: {fast_array.shape}")
        print(f"    Slow features: {slow_array.shape}")
        
        return fast_array, slow_array
    
    def prepare_multi_channel_sequences(self, trial_data: np.ndarray, labels: np.ndarray,
                                       fast_sequence_length: int = 10, slow_sequence_length: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare multi-channel sequences for training.
        
        Args:
            trial_data: Trial data of shape (trials, channels, timepoints)
            labels: Trial labels
            sequence_length: Length of sequences to create
            
        Returns:
            fast_sequences: Fast sequences of shape (total_sequences, seq_len, channels)
            slow_sequences: Slow sequences of shape (total_sequences, seq_len, channels)
            sequence_labels: Labels for sequences
        """
        all_fast_sequences = []
        all_slow_sequences = []
        all_labels = []
        
        for trial_idx, trial in enumerate(trial_data):
            # Extract features for this trial
            fast_features, slow_features = self.extract_multi_channel_features(trial)
            
            # Only create sequences if we have enough data
            if len(fast_features) >= fast_sequence_length and len(slow_features) >= slow_sequence_length:
                # Create sequences
                for i in range(len(fast_features) - fast_sequence_length + 1):
                    fast_seq = fast_features[i:i + fast_sequence_length]  # (seq_len, channels)
                    
                    # For slow features, use a sliding window approach
                    slow_start = min(i, len(slow_features) - slow_sequence_length)
                    slow_seq = slow_features[slow_start:slow_start + slow_sequence_length]  # (seq_len, channels)
                    
                    # Ensure sequences have correct shape
                    if fast_seq.shape == (fast_sequence_length, self.num_channels) and slow_seq.shape == (slow_sequence_length, self.num_channels):
                        all_fast_sequences.append(fast_seq)
                        all_slow_sequences.append(slow_seq)
                        all_labels.append(labels[trial_idx])
        
        if not all_fast_sequences:
            raise ValueError("No valid sequences could be created. Check data dimensions and sequence_length.")
        
        # Convert to tensors
        fast_sequences = torch.FloatTensor(np.array(all_fast_sequences))
        slow_sequences = torch.FloatTensor(np.array(all_slow_sequences))
        sequence_labels = torch.LongTensor(np.array(all_labels))
        
        print(f"  Created {len(all_fast_sequences)} sequences")
        print(f"  Fast sequences shape: {fast_sequences.shape}")
        print(f"  Slow sequences shape: {slow_sequences.shape}")
        
        return fast_sequences, slow_sequences, sequence_labels
    
    def train_multi_channel_models(self, trial_data: np.ndarray, labels: np.ndarray,
                                  test_size: float = 0.2, num_epochs: int = 10) -> Tuple:
        """
        Train multi-channel attention models.
        
        Args:
            trial_data: Trial data of shape (trials, channels, timepoints)
            labels: Trial labels
            test_size: Fraction of data for testing
            num_epochs: Number of training epochs
            
        Returns:
            training_results: Dictionary with training results
        """
        print(f"Training multi-channel attention models on {len(trial_data)} trials...")
        
        # Prepare sequences
        fast_sequences, slow_sequences, sequence_labels = self.prepare_multi_channel_sequences(
            trial_data, labels
        )
        
        print(f"  Fast sequences shape: {fast_sequences.shape}")
        print(f"  Slow sequences shape: {slow_sequences.shape}")
        print(f"  Labels shape: {sequence_labels.shape}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(fast_sequences))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=sequence_labels.numpy(), random_state=42
        )
        
        fast_train = fast_sequences[train_idx]
        slow_train = slow_sequences[train_idx]
        labels_train = sequence_labels[train_idx]
        
        fast_test = fast_sequences[test_idx]
        slow_test = slow_sequences[test_idx]
        labels_test = sequence_labels[test_idx]
        
        print(f"  Training set: {len(train_idx)} sequences")
        print(f"  Test set: {len(test_idx)} sequences")
        
        # Training loop
        training_losses = []
        test_accuracies = []
        
        for epoch in range(num_epochs):
            # Training
            self.fast_rnn.train()
            self.slow_rnn.train()
            self.integration_net.train()
            
            # Forward pass
            fast_output, fast_attention = self.fast_rnn(fast_train)
            slow_output, slow_attention = self.slow_rnn(slow_train)
            integration_output, integration_attention = self.integration_net(fast_output, slow_output)
            
            # Loss
            loss = self.criterion(integration_output, labels_train)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Evaluation
            self.fast_rnn.eval()
            self.slow_rnn.eval()
            self.integration_net.eval()
            
            with torch.no_grad():
                fast_test_output, _ = self.fast_rnn(fast_test)
                slow_test_output, _ = self.slow_rnn(slow_test)
                integration_test_output, _ = self.integration_net(fast_test_output, slow_test_output)
                
                test_pred = torch.argmax(integration_test_output, dim=1)
                test_accuracy = (test_pred == labels_test).float().mean().item()
            
            training_losses.append(loss.item())
            test_accuracies.append(test_accuracy)
            
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={loss.item():.3f}, Test Acc={test_accuracy:.3f}")
        
        # Store attention weights for analysis
        self.attention_weights = {
            'fast': fast_attention,
            'slow': slow_attention,
            'integration': integration_attention
        }
        
        # Final evaluation
        final_results = self.evaluate_multi_channel_models(
            fast_test, slow_test, labels_test
        )
        
        return {
            'training_losses': training_losses,
            'test_accuracies': test_accuracies,
            'final_results': final_results,
            'attention_weights': self.attention_weights
        }
    
    def evaluate_multi_channel_models(self, fast_test: torch.Tensor, slow_test: torch.Tensor,
                                    labels_test: torch.Tensor) -> Dict:
        """
        Evaluate multi-channel attention models.
        
        Args:
            fast_test: Fast test sequences
            slow_test: Slow test sequences
            labels_test: Test labels
            
        Returns:
            results: Dictionary with evaluation results
        """
        self.fast_rnn.eval()
        self.slow_rnn.eval()
        self.integration_net.eval()
        
        with torch.no_grad():
            # Individual model evaluation
            fast_output, _ = self.fast_rnn(fast_test)
            slow_output, _ = self.slow_rnn(slow_test)
            integration_output, _ = self.integration_net(fast_output, slow_output)
            
            # Predictions
            fast_pred = torch.argmax(fast_output, dim=1)
            slow_pred = torch.argmax(slow_output, dim=1)
            integration_pred = torch.argmax(integration_output, dim=1)
            
            # Accuracies
            fast_acc = (fast_pred == labels_test).float().mean().item()
            slow_acc = (slow_pred == labels_test).float().mean().item()
            integration_acc = (integration_pred == labels_test).float().mean().item()
            
            # Confusion matrices
            from sklearn.metrics import confusion_matrix, classification_report
            
            fast_cm = confusion_matrix(labels_test.numpy(), fast_pred.numpy())
            slow_cm = confusion_matrix(labels_test.numpy(), slow_pred.numpy())
            integration_cm = confusion_matrix(labels_test.numpy(), integration_pred.numpy())
            
            results = {
                'fast_accuracy': fast_acc,
                'slow_accuracy': slow_acc,
                'integration_accuracy': integration_acc,
                'fast_confusion_matrix': fast_cm.tolist(),
                'slow_confusion_matrix': slow_cm.tolist(),
                'integration_confusion_matrix': integration_cm.tolist(),
                'fast_classification_report': classification_report(
                    labels_test.numpy(), fast_pred.numpy(), output_dict=True
                ),
                'slow_classification_report': classification_report(
                    labels_test.numpy(), slow_pred.numpy(), output_dict=True
                ),
                'integration_classification_report': classification_report(
                    labels_test.numpy(), integration_pred.numpy(), output_dict=True
                )
            }
            
            print(f"Multi-Channel Attention Results:")
            print(f"  Fast RNN Accuracy: {fast_acc:.3f}")
            print(f"  Slow RNN Accuracy: {slow_acc:.3f}")
            print(f"  Integration Accuracy: {integration_acc:.3f}")
            
            return results
    
    def analyze_channel_attention(self, channel_names: List[str] = None) -> Dict:
        """
        Analyze channel attention patterns.
        
        Args:
            channel_names: List of channel names (default: C1-C22)
            
        Returns:
            analysis: Dictionary with channel attention analysis
        """
        if channel_names is None:
            channel_names = [f'C{i+1}' for i in range(self.num_channels)]
        
        # Get channel attention weights
        fast_channel_weights = self.attention_weights['fast']['channel_weights']
        slow_channel_weights = self.attention_weights['slow']['channel_weights']
        
        # Average across batch
        fast_avg_weights = fast_channel_weights.mean(dim=0).detach().cpu().numpy()
        slow_avg_weights = slow_channel_weights.mean(dim=0).detach().cpu().numpy()
        
        # Analyze brain region attention
        fast_region_attention = analyze_brain_region_attention(
            fast_channel_weights, channel_names
        )
        slow_region_attention = analyze_brain_region_attention(
            slow_channel_weights, channel_names
        )
        
        # Top channels
        fast_top_channels = np.argsort(fast_avg_weights)[-5:]
        slow_top_channels = np.argsort(slow_avg_weights)[-5:]
        
        analysis = {
            'fast_channel_weights': fast_avg_weights.tolist(),
            'slow_channel_weights': slow_avg_weights.tolist(),
            'fast_region_attention': fast_region_attention,
            'slow_region_attention': slow_region_attention,
            'fast_top_channels': [channel_names[i] for i in fast_top_channels],
            'slow_top_channels': [channel_names[i] for i in slow_top_channels],
            'channel_names': channel_names
        }
        
        print(f"Channel Attention Analysis:")
        print(f"  Fast RNN Top Channels: {analysis['fast_top_channels']}")
        print(f"  Slow RNN Top Channels: {analysis['slow_top_channels']}")
        print(f"  Fast RNN Region Attention: {fast_region_attention}")
        print(f"  Slow RNN Region Attention: {slow_region_attention}")
        
        return analysis
    
    def visualize_multi_channel_attention(self, subject_id: str, 
                                        channel_names: List[str] = None) -> None:
        """
        Visualize multi-channel attention patterns.
        
        Args:
            subject_id: Subject identifier
            channel_names: List of channel names
        """
        if channel_names is None:
            channel_names = [f'C{i+1}' for i in range(self.num_channels)]
        
        # Visualize channel attention
        fast_channel_weights = self.attention_weights['fast']['channel_weights']
        slow_channel_weights = self.attention_weights['slow']['channel_weights']
        
        self.visualizer.visualize_channel_attention(
            fast_channel_weights, channel_names, f"{subject_id}_fast"
        )
        self.visualizer.visualize_channel_attention(
            slow_channel_weights, channel_names, f"{subject_id}_slow"
        )
        
        # Visualize temporal attention
        fast_temporal_attention = self.attention_weights['fast']['temporal_attention']
        slow_temporal_attention = self.attention_weights['slow']['temporal_attention']
        
        # Create timestamps for visualization
        timestamps = np.arange(fast_temporal_attention.shape[1]) * 10  # 10ms windows
        
        self.visualizer.visualize_temporal_attention(
            fast_temporal_attention, timestamps, f"{subject_id}_fast"
        )
        self.visualizer.visualize_temporal_attention(
            slow_temporal_attention, timestamps, f"{subject_id}_slow"
        )
    
    def save_multi_channel_results(self, results: Dict, subject_id: str) -> None:
        """
        Save multi-channel attention results.
        
        Args:
            results: Results dictionary
            subject_id: Subject identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = get_experiment_results_path()
        results_path.mkdir(parents=True, exist_ok=True)
        results_path = results_path / f"multi_channel_attention_{subject_id}_{timestamp}.json"
        
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
                            else:
                                serializable_results[key][subkey][nested_key] = nested_value
                    else:
                        serializable_results[key][subkey] = subvalue
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Multi-channel attention results saved to {results_path}") 