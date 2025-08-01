#!/usr/bin/env python3
"""
Attention-aware hierarchical EEG processor
Integrates temporal and spatial attention mechanisms for BCI classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime

from attention_modules import (
    AttentionFastRNN, AttentionSlowRNN, AttentionIntegrationNet,
    AttentionVisualizer, analyze_brain_region_attention
)
from config import (
    get_experiment_results_path, get_model_comparison_path, get_log_path,
    MODEL_CONFIG, TRAINING_CONFIG, BCI_CONFIG, FEATURE_CONFIG
)

class AttentionHierarchicalProcessor:
    """Hierarchical EEG processor with attention mechanisms."""
    
    def __init__(self, fast_window_size=10, slow_context_size=50, 
                 num_classes=4, num_heads=4, dropout=0.2,
                 learning_rate=0.001, batch_size=32):
        self.fast_window_size = fast_window_size
        self.slow_context_size = slow_context_size
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Initialize attention models
        self.fast_rnn = AttentionFastRNN(
            input_size=3,
            hidden_size=MODEL_CONFIG['fast_rnn']['hidden_size'],
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.slow_rnn = AttentionSlowRNN(
            input_size=2,
            hidden_size=MODEL_CONFIG['slow_rnn']['hidden_size'],
            num_classes=num_classes,
            num_heads=num_heads // 2,  # Fewer heads for slow features
            dropout=dropout
        )
        
        self.integration_net = AttentionIntegrationNet(
            fast_size=MODEL_CONFIG['fast_rnn']['hidden_size'],
            slow_size=MODEL_CONFIG['slow_rnn']['hidden_size'],
            hidden_size=MODEL_CONFIG['integration']['hidden_size'],
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
        
    def extract_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fast and slow features from EEG data.
        
        Args:
            eeg_data: EEG data of shape (timepoints,)
            
        Returns:
            fast_features: Fast features of shape (n_windows, 3)
            slow_features: Slow features of shape (n_windows, 2)
        """
        fast_features = []
        slow_features = []
        
        # Fast features (10ms windows)
        for i in range(0, len(eeg_data) - self.fast_window_size, self.fast_window_size // 2):
            window = eeg_data[i:i + self.fast_window_size]
            if len(window) == self.fast_window_size:
                fast_features.append([
                    np.mean(window),
                    np.std(window),
                    np.max(window) - np.min(window)  # Delta
                ])
        
        # Slow features (50ms windows)
        for i in range(0, len(eeg_data) - self.slow_context_size, self.slow_context_size // 2):
            window = eeg_data[i:i + self.slow_context_size]
            if len(window) == self.slow_context_size:
                slow_features.append([
                    np.mean(window),
                    np.std(window)
                ])
        
        return np.array(fast_features), np.array(slow_features)
    
    def prepare_sequences(self, X_fast: np.ndarray, X_slow: np.ndarray, 
                         y: np.ndarray, sequence_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for attention-based training.
        
        Args:
            X_fast: Fast features
            X_slow: Slow features
            y: Labels
            sequence_length: Length of sequences
            
        Returns:
            fast_sequences: Fast feature sequences
            slow_sequences: Slow feature sequences
            labels: Labels for sequences
        """
        fast_sequences = []
        slow_sequences = []
        labels = []
        
        for i in range(len(X_fast) - sequence_length + 1):
            fast_seq = X_fast[i:i + sequence_length]
            slow_seq = X_slow[i:i + sequence_length]
            
            if len(fast_seq) == sequence_length and len(slow_seq) == sequence_length:
                fast_sequences.append(fast_seq)
                slow_sequences.append(slow_seq)
                labels.append(y[i + sequence_length - 1])  # Label for last timestep
        
        return (torch.FloatTensor(np.array(fast_sequences)),
                torch.FloatTensor(np.array(slow_sequences)),
                torch.LongTensor(np.array(labels)))
    
    def train_attention_models(self, X_fast: np.ndarray, X_slow: np.ndarray, 
                              y: np.ndarray, test_size: float = 0.2, 
                              num_epochs: int = 10) -> Tuple:
        """
        Train attention-based hierarchical models.
        
        Args:
            X_fast: Fast features
            X_slow: Slow features
            y: Labels
            test_size: Fraction of data for testing
            num_epochs: Number of training epochs
            
        Returns:
            Training results and attention weights
        """
        print(f"Training attention models with {len(X_fast)} samples...")
        
        # Prepare sequences
        fast_sequences, slow_sequences, labels = self.prepare_sequences(X_fast, X_slow, y)
        
        # Split data
        split_idx = int(len(fast_sequences) * (1 - test_size))
        fast_train, fast_test = fast_sequences[:split_idx], fast_sequences[split_idx:]
        slow_train, slow_test = slow_sequences[:split_idx], slow_sequences[split_idx:]
        labels_train, labels_test = labels[:split_idx], labels[split_idx:]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(fast_train, slow_train, labels_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        test_dataset = torch.utils.data.TensorDataset(fast_test, slow_test, labels_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        train_losses = []
        test_accuracies = []
        
        for epoch in range(num_epochs):
            # Training
            self.fast_rnn.train()
            self.slow_rnn.train()
            self.integration_net.train()
            
            epoch_loss = 0.0
            for batch_fast, batch_slow, batch_labels in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass with attention
                fast_output, fast_attention = self.fast_rnn(batch_fast)
                slow_output, slow_attention = self.slow_rnn(batch_slow)
                
                # Integration with cross-modal attention
                integration_output, integration_attention = self.integration_net(fast_output, slow_output)
                
                # Loss calculation
                loss = self.criterion(integration_output, batch_labels)
                
                # Optional: Attention regularization
                attention_sparsity = self.compute_attention_sparsity(
                    fast_attention, slow_attention, integration_attention
                )
                loss += 0.01 * attention_sparsity  # Encourage focused attention
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Evaluation
            self.fast_rnn.eval()
            self.slow_rnn.eval()
            self.integration_net.eval()
            
            correct = 0
            total = 0
            test_attention_weights = {'fast': [], 'slow': [], 'integration': []}
            
            with torch.no_grad():
                for batch_fast, batch_slow, batch_labels in test_loader:
                    fast_output, fast_attention = self.fast_rnn(batch_fast)
                    slow_output, slow_attention = self.slow_rnn(batch_slow)
                    integration_output, integration_attention = self.integration_net(fast_output, slow_output)
                    
                    # Store attention weights for analysis
                    test_attention_weights['fast'].append(fast_attention)
                    test_attention_weights['slow'].append(slow_attention)
                    test_attention_weights['integration'].append(integration_attention)
                    
                    # Accuracy calculation
                    _, predicted = torch.max(integration_output.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            accuracy = 100 * correct / total
            train_losses.append(epoch_loss / len(train_loader))
            test_accuracies.append(accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_losses[-1]:.4f}, Accuracy={accuracy:.2f}%")
        
        # Store attention weights for visualization
        self.attention_weights = {
            'fast': torch.cat(test_attention_weights['fast'], dim=0),
            'slow': torch.cat(test_attention_weights['slow'], dim=0),
            'integration': torch.cat(test_attention_weights['integration'], dim=0)
        }
        
        return (fast_test, slow_test, labels_test, 
                train_losses, test_accuracies, self.attention_weights)
    
    def compute_attention_sparsity(self, fast_attention: torch.Tensor, 
                                 slow_attention: torch.Tensor, 
                                 integration_attention: torch.Tensor) -> torch.Tensor:
        """
        Compute attention sparsity to encourage focused attention.
        
        Args:
            fast_attention: Fast attention weights
            slow_attention: Slow attention weights
            integration_attention: Integration attention weights
            
        Returns:
            sparsity_loss: Sparsity regularization loss
        """
        # Encourage sparse attention (fewer high values, more zeros)
        sparsity_loss = 0.0
        
        for attention in [fast_attention, slow_attention, integration_attention]:
            # L1 regularization on attention weights
            sparsity_loss += torch.mean(torch.abs(attention))
            
            # Entropy regularization to encourage focused attention
            attention_probs = torch.softmax(attention, dim=-1)
            entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
            sparsity_loss += torch.mean(entropy)
        
        return sparsity_loss
    
    def evaluate_attention_models(self, fast_test: torch.Tensor, slow_test: torch.Tensor,
                                labels_test: torch.Tensor, attention_weights: Dict) -> Dict:
        """
        Evaluate attention models and analyze attention patterns.
        
        Args:
            fast_test: Test fast features
            slow_test: Test slow features
            labels_test: Test labels
            attention_weights: Attention weights from training
            
        Returns:
            Evaluation results and attention analysis
        """
        self.fast_rnn.eval()
        self.slow_rnn.eval()
        self.integration_net.eval()
        
        with torch.no_grad():
            # Forward pass
            fast_output, _ = self.fast_rnn(fast_test)
            slow_output, _ = self.slow_rnn(slow_test)
            integration_output, _ = self.integration_net(fast_output, slow_output)
            
            # Calculate metrics
            _, predicted = torch.max(integration_output.data, 1)
            accuracy = 100 * (predicted == labels_test).sum().item() / labels_test.size(0)
            
            # Attention analysis
            attention_analysis = self.analyze_attention_patterns(attention_weights, labels_test)
            
            results = {
                'accuracy': accuracy,
                'attention_analysis': attention_analysis,
                'attention_weights': attention_weights
            }
            
            print(f"Attention Model Accuracy: {accuracy:.2f}%")
            print(f"Attention Analysis: {attention_analysis}")
            
            return results
    
    def analyze_attention_patterns(self, attention_weights: Dict, labels: torch.Tensor) -> Dict:
        """
        Analyze attention patterns for different classes.
        
        Args:
            attention_weights: Attention weights from different components
            labels: Class labels
            
        Returns:
            attention_analysis: Analysis of attention patterns
        """
        analysis = {}
        
        # Analyze temporal attention patterns
        for component, weights in attention_weights.items():
            # Average attention weights per class
            class_attention = {}
            for class_idx in range(self.num_classes):
                class_mask = (labels == class_idx)
                if class_mask.sum() > 0:
                    class_weights = weights[class_mask].mean(dim=0)
                    class_attention[f'class_{class_idx}'] = class_weights.cpu().numpy()
            
            analysis[component] = class_attention
        
        return analysis
    
    def visualize_attention(self, subject_id: str, timestamps: Optional[np.ndarray] = None) -> None:
        """
        Visualize attention patterns for a subject.
        
        Args:
            subject_id: Subject identifier
            timestamps: Time points for temporal visualization
        """
        if not self.attention_weights:
            print("No attention weights available. Run training first.")
            return
        
        if timestamps is None:
            timestamps = np.arange(self.attention_weights['fast'].shape[1]) * 10  # 10ms intervals
        
        # Visualize temporal attention
        for component, weights in self.attention_weights.items():
            self.visualizer.visualize_temporal_attention(
                weights, timestamps, f"{subject_id}_{component}"
            )
    
    def save_attention_results(self, results: Dict, subject_id: str) -> None:
        """
        Save attention results and analysis.
        
        Args:
            results: Evaluation results
            subject_id: Subject identifier
        """
        # Prepare results for JSON serialization
        serializable_results = {
            'subject_id': subject_id,
            'accuracy': results['accuracy'],
            'attention_analysis': {
                component: {
                    class_name: weights.tolist() if isinstance(weights, np.ndarray) else weights
                    for class_name, weights in analysis.items()
                }
                for component, analysis in results['attention_analysis'].items()
            },
            'model_config': {
                'fast_window_size': self.fast_window_size,
                'slow_context_size': self.slow_context_size,
                'num_classes': self.num_classes,
                'num_heads': self.num_heads,
                'dropout': self.dropout
            }
        }
        
        # Save to results directory
        results_file = get_experiment_results_path(f"attention_bci_{subject_id}")
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Attention results saved to: {results_file}")

def run_attention_experiment():
    """Run attention-based BCI experiment."""
    
    print("=== Attention-Based BCI Experiment ===")
    
    # Initialize attention processor
    processor = AttentionHierarchicalProcessor(
        fast_window_size=FEATURE_CONFIG['fast_window_ms'],
        slow_context_size=FEATURE_CONFIG['slow_window_ms'],
        num_classes=BCI_CONFIG['num_classes'],
        num_heads=4,
        dropout=0.2
    )
    
    # Load BCI data (you'll need to implement data loading)
    # For now, we'll create synthetic data for demonstration
    print("Loading BCI data...")
    
    # Create synthetic EEG data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_channels = 22
    sampling_rate = BCI_CONFIG['sampling_rate']
    trial_duration = BCI_CONFIG['trial_duration']
    
    # Generate synthetic EEG data with motor imagery patterns
    eeg_data = np.random.randn(n_samples, int(sampling_rate * trial_duration))
    
    # Add motor imagery patterns (simplified)
    for i in range(n_samples):
        # Add alpha rhythm (8-13 Hz) for motor imagery
        t = np.linspace(0, trial_duration, int(sampling_rate * trial_duration))
        alpha_rhythm = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        eeg_data[i] += alpha_rhythm
    
    # Generate labels (4 classes)
    labels = np.random.randint(0, 4, n_samples)
    
    print(f"Generated {n_samples} synthetic EEG trials")
    print(f"Data shape: {eeg_data.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Extract features
    print("Extracting features...")
    all_fast_features = []
    all_slow_features = []
    all_labels = []
    
    for i in range(n_samples):
        fast_features, slow_features = processor.extract_features(eeg_data[i])
        
        if len(fast_features) > 0 and len(slow_features) > 0:
            # Use the label for all features from this trial
            all_fast_features.extend(fast_features)
            all_slow_features.extend(slow_features)
            all_labels.extend([labels[i]] * len(fast_features))
    
    X_fast = np.array(all_fast_features)
    X_slow = np.array(all_slow_features)
    y = np.array(all_labels)
    
    print(f"Feature shapes: X_fast {X_fast.shape}, X_slow {X_slow.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Train attention models
    print("Training attention models...")
    (fast_test, slow_test, labels_test, 
     train_losses, test_accuracies, attention_weights) = processor.train_attention_models(
        X_fast, X_slow, y, test_size=0.2, num_epochs=5
    )
    
    # Evaluate models
    print("Evaluating attention models...")
    results = processor.evaluate_attention_models(
        fast_test, slow_test, labels_test, attention_weights
    )
    
    # Visualize attention
    print("Visualizing attention patterns...")
    processor.visualize_attention("synthetic_subject")
    
    # Save results
    processor.save_attention_results(results, "synthetic_subject")
    
    print("✅ Attention experiment complete!")
    return results

if __name__ == "__main__":
    run_attention_experiment() 