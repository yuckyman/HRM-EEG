#!/usr/bin/env python3
"""
Run BCI Competition IV data through the hierarchical EEG processor
Extended for multi-class motor imagery classification
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from eeg_hierarchical_processor import HierarchicalEEGProcessor, HRMEEGProcessor, ModelComparator, ModelResults
import json
from datetime import datetime
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import time
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import configuration
from config import (
    BCI_DATA_DIR, MNE_DATA_DIR, EXPERIMENTS_DIR, MODEL_COMPARISONS_DIR,
    get_experiment_results_path, get_model_comparison_path, get_timestamp,
    MODEL_CONFIG, TRAINING_CONFIG, BCI_CONFIG, FEATURE_CONFIG
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
                    trial_data.append(trial)
                    # Keep all 4 classes (0-based indexing)
                    trial_labels.append(event[2] - 1)  # Convert 1-4 to 0-3
        
        trial_data = np.array(trial_data)  # (trials, channels, timepoints)
        trial_labels = np.array(trial_labels)
        
        print(f"  Trial data shape: {trial_data.shape}")
        print(f"  Trial labels shape: {trial_labels.shape}")
        print(f"  Label distribution: {np.bincount(trial_labels)}")
        
        return {
            'data': trial_data,
            'labels': trial_labels,
            'raw': raw,
            'events': events,
            'event_dict': event_dict,
            'filename': Path(filepath).name,
            'sampling_rate': raw.info['sfreq']
        }
        
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def process_bci_trials(trial_data, labels, processor):
    """Process BCI trials through the hierarchical processor."""
    
    print(f"Processing {len(trial_data)} trials...")
    
    all_X_fast = []
    all_X_slow = []
    all_y = []
    
    for i, (trial, label) in enumerate(zip(trial_data, labels)):
        # trial shape: (channels, timepoints)
        # Use first channel for now (can be extended to multi-channel)
        single_channel_data = trial[0, :]  # (timepoints,)
        
        # Create labels for the trial (since we want to predict the class)
        # We'll use the trial label as the target
        trial_labels = np.full(len(single_channel_data), label)
        
        # Extract features using the processor
        X_fast, X_slow, y = processor.extract_features_from_eeg(
            single_channel_data.reshape(1, -1),  # (1, timepoints)
            trial_labels,
            sampling_rate=250.0  # BCI data is 250Hz
        )
        
        all_X_fast.append(X_fast)
        all_X_slow.append(X_slow)
        all_y.append(y)
    
    # Concatenate all trials
    X_fast = np.concatenate(all_X_fast, axis=0)
    X_slow = np.concatenate(all_X_slow, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"  Final feature shapes:")
    print(f"    X_fast: {X_fast.shape}")
    print(f"    X_slow: {X_slow.shape}")
    print(f"    y: {y.shape}")
    print(f"    Classes: {np.unique(y)}")
    print(f"    Label distribution: {np.bincount(y.astype(int))}")
    
    return X_fast, X_slow, y

class MultiClassFastRNN(nn.Module):
    """Fast RNN for processing immediate dynamics (gamma oscillations) - Multi-class version."""
    
    def __init__(self, input_size=3, hidden_size=32, num_layers=2, num_classes=4, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output for classification
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # Classification head
        output = self.fc(last_output)
        output = self.softmax(output)
        
        return output

class MultiClassSlowRNN(nn.Module):
    """Slow RNN for processing contextual information (alpha/beta rhythms) - Multi-class version."""
    
    def __init__(self, input_size=2, hidden_size=16, num_layers=1, num_classes=4, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output for classification
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # Classification head
        output = self.fc(last_output)
        output = self.softmax(output)
        
        return output

class MultiClassIntegrationNet(nn.Module):
    """Integration network that combines fast and slow RNN outputs - Multi-class version."""
    
    def __init__(self, fast_size=32, slow_size=16, hidden_size=32, num_classes=4, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(fast_size + slow_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, fast_features, slow_features):
        # Concatenate fast and slow features
        combined = torch.cat([fast_features, slow_features], dim=1)
        
        # Integration layers
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        output = self.fc3(x)
        output = self.softmax(output)
        
        return output

class MultiClassHierarchicalProcessor(HierarchicalEEGProcessor):
    """Extended processor for multi-class classification."""
    
    def __init__(self, fast_window_size=10, slow_context_size=50, num_classes=4):
        super().__init__(fast_window_size, slow_context_size)
        self.num_classes = num_classes
    
    def train_multiclass_models(self, X_fast: np.ndarray, X_slow: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """Train multi-class models using One-vs-Rest approach."""
        # Split data
        perm = np.random.permutation(len(y))
        split = int((1 - test_size) * len(y))
        train_idx, test_idx = perm[:split], perm[split:]
        
        Xf_train, Xf_test = X_fast[train_idx], X_fast[test_idx]
        Xs_train, Xs_test = X_slow[train_idx], X_slow[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train fast-only model (One-vs-Rest)
        self.fast_model = make_pipeline(
            StandardScaler(), 
            OneVsRestClassifier(LogisticRegression(max_iter=500))
        )
        start_time = time.time()
        self.fast_model.fit(Xf_train, y_train)
        training_time = time.time() - start_time
        
        # Compute fast predictions for hierarchical model
        fast_pred_train = self.fast_model.predict(Xf_train)
        fast_pred_test = self.fast_model.predict(Xf_test)
        
        # Build hierarchical features: [fast_pred, slow_mean, slow_std]
        H_train = np.column_stack([fast_pred_train, Xs_train])
        H_test = np.column_stack([fast_pred_test, Xs_test])
        
        # Train hierarchical model (One-vs-Rest)
        self.hierarchical_model = make_pipeline(
            StandardScaler(), 
            OneVsRestClassifier(LogisticRegression(max_iter=500))
        )
        start_time = time.time()
        self.hierarchical_model.fit(H_train, y_train)
        training_time += time.time() - start_time
        
        return (Xf_train, Xf_test, Xs_train, Xs_test, 
                y_train, y_test, H_train, H_test, training_time)
    
    def evaluate_multiclass_models(self, Xf_test: np.ndarray, Xs_test: np.ndarray, y_test: np.ndarray, H_test: np.ndarray, training_time: float) -> Dict[str, Dict[str, float]]:
        """Evaluate multi-class models."""
        # Fast-only model
        fast_pred = self.fast_model.predict(Xf_test)
        fast_pred_proba = self.fast_model.predict_proba(Xf_test)
        
        print("=== Fast-only model (Multi-class) ===")
        print("Accuracy:", accuracy_score(y_test, fast_pred))
        print("AUC (macro):", roc_auc_score(y_test, fast_pred_proba, multi_class='ovr', average='macro'))
        print(classification_report(y_test, fast_pred, digits=3))
        
        # Hierarchical model
        hier_pred = self.hierarchical_model.predict(H_test)
        hier_pred_proba = self.hierarchical_model.predict_proba(H_test)
        
        print("\n=== Hierarchical (fast + slow) model (Multi-class) ===")
        print("Accuracy:", accuracy_score(y_test, hier_pred))
        print("AUC (macro):", roc_auc_score(y_test, hier_pred_proba, multi_class='ovr', average='macro'))
        print(classification_report(y_test, hier_pred, digits=3))
        
        # Evaluate performance using ModelResults
        fast_results = self.evaluate_model_performance(self.fast_model, Xf_test, y_test, "Fast-only Multi-class Logistic Regression")
        hier_results = self.evaluate_model_performance(self.hierarchical_model, H_test, y_test, "Hierarchical Multi-class Logistic Regression")
        
        self.comparator.add_result(fast_results)
        self.comparator.add_result(hier_results)
        
        return {
            'fast': {'accuracy': accuracy_score(y_test, fast_pred), 
                     'auc': roc_auc_score(y_test, fast_pred_proba, multi_class='ovr', average='macro')},
            'hierarchical': {'accuracy': accuracy_score(y_test, hier_pred), 
                           'auc': roc_auc_score(y_test, hier_pred_proba, multi_class='ovr', average='macro')}
        }
    
    def evaluate_model_performance(self, model, X_test, y_test, model_name: str, 
                                 training_time: float = 0.0) -> ModelResults:
        """Comprehensive model evaluation with timing and memory tracking."""
        import psutil
        import os
        
        # Memory tracking
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Inference timing
        start_time = time.time()
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = model.predict(X_test)
        inference_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        
        # Memory after inference
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        # Handle multi-class AUC
        if len(np.unique(y_test)) > 2:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1]) if y_pred_proba.shape[1] > 1 else 0.5
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Extract precision, recall, f1 from classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        
        # Count parameters (estimate for sklearn models)
        num_parameters = 0
        if hasattr(model, 'named_steps'):
            for step_name, step in model.named_steps.items():
                if hasattr(step, 'estimators_'):
                    # One-vs-Rest classifier
                    for estimator in step.estimators_:
                        if hasattr(estimator, 'coef_'):
                            num_parameters += estimator.coef_.size + estimator.intercept_.size
                elif hasattr(step, 'coef_'):
                    num_parameters += step.coef_.size + step.intercept_.size
        
        return ModelResults(
            model_name=model_name,
            accuracy=accuracy,
            auc=auc,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            training_time=training_time,
            inference_time=inference_time,
            memory_usage=memory_usage,
            num_parameters=num_parameters,
            confusion_matrix=conf_matrix,
            classification_report=classification_report(y_test, y_pred)
        )

class MultiClassHRMEEGProcessor(HRMEEGProcessor):
    """
    Multi-class HRM-inspired EEG processor with multi-timescale RNN networks.
    Extends the base processor with LSTM-based hierarchical processing for multi-class classification.
    """
    
    def __init__(self, fast_window_size=10, slow_context_size=50, 
                 fast_hidden_size=32, slow_hidden_size=16, 
                 integration_hidden_size=32, num_classes=4, dropout=0.2):
        super().__init__(fast_window_size, slow_context_size, 
                        fast_hidden_size, slow_hidden_size, 
                        integration_hidden_size, dropout)
        
        # Replace with multi-class versions
        self.fast_rnn = MultiClassFastRNN(
            input_size=3,  # mean, std, delta
            hidden_size=fast_hidden_size,
            num_layers=2,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.slow_rnn = MultiClassSlowRNN(
            input_size=2,  # mean, std
            hidden_size=slow_hidden_size,
            num_layers=1,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.integration_net = MultiClassIntegrationNet(
            fast_size=fast_hidden_size,
            slow_size=slow_hidden_size,
            hidden_size=integration_hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Move models to device
        self.fast_rnn.to(self.device)
        self.slow_rnn.to(self.device)
        self.integration_net.to(self.device)
        
        # Optimizers
        self.fast_optimizer = optim.Adam(self.fast_rnn.parameters(), lr=self.learning_rate)
        self.slow_optimizer = optim.Adam(self.slow_rnn.parameters(), lr=self.learning_rate)
        self.integration_optimizer = optim.Adam(self.integration_net.parameters(), lr=self.learning_rate)
        
        # Loss function for multi-class
        self.criterion = nn.CrossEntropyLoss()
    
    def prepare_sequences(self, X_fast: np.ndarray, X_slow: np.ndarray, 
                         y: np.ndarray, sequence_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for RNN training by creating sequences."""
        # Reshape features into sequences
        # For now, we'll use sliding windows of the features
        fast_sequences = []
        slow_sequences = []
        labels = []
        
        for i in range(len(X_fast) - sequence_length + 1):
            fast_seq = X_fast[i:i+sequence_length]
            slow_seq = X_slow[i:i+sequence_length]
            label = int(y[i+sequence_length-1])  # Label for the last timestep
            
            fast_sequences.append(fast_seq)
            slow_sequences.append(slow_seq)
            labels.append(label)
        
        # Convert to tensors
        fast_tensor = torch.FloatTensor(np.array(fast_sequences))
        slow_tensor = torch.FloatTensor(np.array(slow_sequences))
        labels_tensor = torch.LongTensor(np.array(labels))  # Use LongTensor for multi-class
        
        return fast_tensor, slow_tensor, labels_tensor
    
    def train_rnn_models(self, X_fast: np.ndarray, X_slow: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2) -> Tuple:
        """Train the multi-class RNN models with proper train/test split."""
        # Prepare sequences
        fast_tensor, slow_tensor, labels_tensor = self.prepare_sequences(X_fast, X_slow, y)
        
        # Split data
        total_samples = len(fast_tensor)
        train_size = int((1 - test_size) * total_samples)
        
        # Random split
        indices = torch.randperm(total_samples)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Split tensors
        fast_train = fast_tensor[train_indices]
        slow_train = slow_tensor[train_indices]
        labels_train = labels_tensor[train_indices]
        
        fast_test = fast_tensor[test_indices]
        slow_test = slow_tensor[test_indices]
        labels_test = labels_tensor[test_indices]
        
        # Create data loaders
        train_dataset = TensorDataset(fast_train, slow_train, labels_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        test_dataset = TensorDataset(fast_test, slow_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        start_time = time.time()
        print(f"\nStarting Multi-class RNN training on {self.device}...")
        print(f"Training data: {len(train_loader)} batches, {self.batch_size} samples per batch")
        print(f"Model parameters: Fast RNN: {sum(p.numel() for p in self.fast_rnn.parameters()):,}, "
              f"Slow RNN: {sum(p.numel() for p in self.slow_rnn.parameters()):,}, "
              f"Integration: {sum(p.numel() for p in self.integration_net.parameters()):,}")
        print("-" * 60)
        
        best_loss = float('inf')
        print(f"Training for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            # Training phase
            self.fast_rnn.train()
            self.slow_rnn.train()
            self.integration_net.train()
            
            total_loss = 0
            fast_loss_sum = 0
            slow_loss_sum = 0
            integration_loss_sum = 0
            num_batches = 0
            
            for batch_idx, (batch_fast, batch_slow, batch_labels) in enumerate(train_loader):
                batch_fast = batch_fast.to(self.device)
                batch_slow = batch_slow.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass through fast RNN
                fast_output = self.fast_rnn(batch_fast)
                fast_loss = self.criterion(fast_output, batch_labels)
                
                # Forward pass through slow RNN
                slow_output = self.slow_rnn(batch_slow)
                slow_loss = self.criterion(slow_output, batch_labels)
                
                # Forward pass through integration network
                fast_features = self.fast_rnn.lstm(batch_fast)[0][:, -1, :]
                slow_features = self.slow_rnn.lstm(batch_slow)[0][:, -1, :]
                integration_output = self.integration_net(fast_features, slow_features)
                integration_loss = self.criterion(integration_output, batch_labels)
                
                # Total loss (weighted combination)
                total_batch_loss = fast_loss + slow_loss + integration_loss
                
                # Backward pass
                self.fast_optimizer.zero_grad()
                self.slow_optimizer.zero_grad()
                self.integration_optimizer.zero_grad()
                
                total_batch_loss.backward()
                
                self.fast_optimizer.step()
                self.slow_optimizer.step()
                self.integration_optimizer.step()
                
                total_loss += total_batch_loss.item()
                fast_loss_sum += fast_loss.item()
                slow_loss_sum += slow_loss.item()
                integration_loss_sum += integration_loss.item()
                num_batches += 1
                
                # Print detailed progress every 5 epochs
                if (epoch + 1) % 5 == 0 and batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                          f"Total Loss: {total_batch_loss.item():.4f} "
                          f"(Fast: {fast_loss.item():.4f}, Slow: {slow_loss.item():.4f}, Int: {integration_loss.item():.4f})")
            
            # Epoch summary
            avg_loss = total_loss / len(train_loader)
            avg_fast_loss = fast_loss_sum / num_batches
            avg_slow_loss = slow_loss_sum / num_batches
            avg_integration_loss = integration_loss_sum / num_batches
            
            # Track best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                improvement = "✨"
            else:
                improvement = ""
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch+1}/{self.num_epochs} Summary:")
                print(f"  Total Loss: {avg_loss:.4f} {improvement}")
                print(f"  Fast RNN Loss: {avg_fast_loss:.4f}")
                print(f"  Slow RNN Loss: {avg_slow_loss:.4f}")
                print(f"  Integration Loss: {avg_integration_loss:.4f}")
                print(f"  Best Loss: {best_loss:.4f}")
                print("-" * 40)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final average loss: {avg_loss:.4f}")
        print(f"Best loss achieved: {best_loss:.4f}")
        
        return (fast_train, fast_test, slow_train, slow_test, 
                labels_train, labels_test, train_loader, test_loader, training_time)
    
    def evaluate_rnn_models(self, fast_test: torch.Tensor, slow_test: torch.Tensor, 
                           labels_test: torch.Tensor, training_time: float) -> Dict[str, ModelResults]:
        """Evaluate multi-class RNN models and return structured results."""
        print(f"\nEvaluating Multi-class RNN models on test set...")
        print(f"Test samples: {len(fast_test)}")
        
        self.fast_rnn.eval()
        self.slow_rnn.eval()
        self.integration_net.eval()
        
        results = {}
        
        with torch.no_grad():
            # Move to device
            fast_test = fast_test.to(self.device)
            slow_test = slow_test.to(self.device)
            labels_test = labels_test.to(self.device)
            
            print("Evaluating Fast RNN...")
            # Fast RNN evaluation
            fast_output = self.fast_rnn(fast_test)
            fast_pred = torch.argmax(fast_output, dim=1)
            fast_results = self._create_multiclass_rnn_results(
                fast_pred, fast_output, labels_test, 
                "Fast RNN (Multi-class)", training_time, self.fast_rnn
            )
            results['fast_rnn'] = fast_results
            print(f"  Fast RNN - Accuracy: {fast_results.accuracy:.3f}, AUC: {fast_results.auc:.3f}")
            
            print("Evaluating Slow RNN...")
            # Slow RNN evaluation
            slow_output = self.slow_rnn(slow_test)
            slow_pred = torch.argmax(slow_output, dim=1)
            slow_results = self._create_multiclass_rnn_results(
                slow_pred, slow_output, labels_test,
                "Slow RNN (Multi-class)", training_time, self.slow_rnn
            )
            results['slow_rnn'] = slow_results
            print(f"  Slow RNN - Accuracy: {slow_results.accuracy:.3f}, AUC: {slow_results.auc:.3f}")
            
            print("Evaluating Hierarchical Integration Network...")
            # Integration network evaluation
            fast_features = self.fast_rnn.lstm(fast_test)[0][:, -1, :]
            slow_features = self.slow_rnn.lstm(slow_test)[0][:, -1, :]
            integration_output = self.integration_net(fast_features, slow_features)
            integration_pred = torch.argmax(integration_output, dim=1)
            integration_results = self._create_multiclass_rnn_results(
                integration_pred, integration_output, labels_test,
                "Hierarchical RNN (Multi-class)", training_time, self.integration_net
            )
            results['integration_rnn'] = integration_results
            print(f"  Hierarchical RNN - Accuracy: {integration_results.accuracy:.3f}, AUC: {integration_results.auc:.3f}")
        
        print(f"\nMulti-class RNN Evaluation Complete!")
        return results
    
    def _create_multiclass_rnn_results(self, pred: torch.Tensor, output: torch.Tensor, 
                                      labels: torch.Tensor, model_name: str, 
                                      training_time: float, model: nn.Module) -> ModelResults:
        """Create ModelResults for multi-class RNN models."""
        import psutil
        import os
        
        # Memory tracking
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Inference timing
        start_time = time.time()
        with torch.no_grad():
            # Use correct input size based on model type
            if hasattr(model, 'lstm') and hasattr(model.lstm, 'input_size'):
                input_size = model.lstm.input_size
            elif isinstance(model, MultiClassSlowRNN):
                input_size = 2  # Slow RNN expects 2 features
            elif isinstance(model, MultiClassIntegrationNet):
                # IntegrationNet expects two separate inputs
                fast_dummy = torch.randn(1, 32).to(self.device)  # Fast RNN output size
                slow_dummy = torch.randn(1, 16).to(self.device)  # Slow RNN output size
                _ = model(fast_dummy, slow_dummy)  # Dummy inference
                inference_time = time.time() - start_time
            else:
                input_size = 3  # Default for Fast RNN
                _ = model(torch.randn(1, 10, input_size).to(self.device))  # Dummy inference
                inference_time = time.time() - start_time
        inference_time = time.time() - start_time
        
        # Memory after inference
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # Convert to numpy for sklearn metrics
        y_pred = pred.cpu().numpy()
        y_pred_proba = output.cpu().numpy()
        y_true = labels.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        # Handle multi-class AUC
        if len(np.unique(y_true)) > 2:
            auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1]) if y_pred_proba.shape[1] > 1 else 0.5
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Extract precision, recall, f1 from classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        
        # Count parameters
        num_parameters = sum(p.numel() for p in model.parameters())
        
        return ModelResults(
            model_name=model_name,
            accuracy=accuracy,
            auc=auc,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            training_time=training_time,
            inference_time=inference_time,
            memory_usage=memory_usage,
            num_parameters=num_parameters,
            confusion_matrix=conf_matrix,
            classification_report=classification_report(y_true, y_pred)
        )

def run_multiclass_bci_experiment():
    """Run the complete multi-class BCI experiment."""
    
    print("=== BCI Competition IV Multi-Class Hierarchical Processing Experiment ===")
    print()
    
    # Initialize processors
    print("1. Initializing multi-class processors...")
    processor = MultiClassHierarchicalProcessor(fast_window_size=10, slow_context_size=50, num_classes=4)
    rnn_processor = MultiClassHRMEEGProcessor(fast_window_size=10, slow_context_size=50, num_classes=4)
    
    # Load BCI data
    print("\n2. Loading BCI Competition data...")
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
        
        # Process trials
        X_fast, X_slow, y = process_bci_trials(
            data_dict['data'], 
            data_dict['labels'], 
            processor
        )
        
        if len(X_fast) == 0:
            print(f"  No valid trials found in {gdf_file.name}")
            continue
        
        # Test 1: Multi-class Logistic Regression Models
        print(f"\n  Testing Multi-class Logistic Regression models...")
        (Xf_train, Xf_test, Xs_train, Xs_test, 
         y_train, y_test, H_train, H_test, training_time) = processor.train_multiclass_models(X_fast, X_slow, y)
        
        results = processor.evaluate_multiclass_models(Xf_test, Xs_test, y_test, H_test, training_time)
        
        # Test 2: RNN Models (if enough data)
        if len(X_fast) > 100:  # Need enough data for RNN training
            print(f"\n  Testing RNN models...")
            try:
                (fast_train, fast_test, slow_train, slow_test, 
                 labels_train, labels_test, train_loader, test_loader, rnn_training_time) = rnn_processor.train_rnn_models(X_fast, X_slow, y)
                
                rnn_results = rnn_processor.evaluate_rnn_models(fast_test, slow_test, labels_test, rnn_training_time)
                
                # Add RNN results to comparison
                for model_name, result in rnn_results.items():
                    processor.comparator.add_result(result)
                    print(f"    Added {result.model_name} to comparison")
                    
            except Exception as e:
                print(f"    RNN training failed: {e}")
        
        # Store results
        results_summary[gdf_file.stem] = {
            'logistic_results': results,
            'data_shape': X_fast.shape,
            'num_trials': len(data_dict['data']),
            'label_distribution': np.bincount(data_dict['labels']).tolist(),
            'classes_present': np.unique(data_dict['labels']).tolist()
        }
        
        print(f"  ✅ Completed processing {gdf_file.name}")
    
    # Generate comprehensive report
    print(f"\n3. Generating comprehensive report...")
    processor.comparator.compare_models()
    report = processor.comparator.generate_report()
    
    # Save results
    timestamp = get_timestamp()
    results_file = get_experiment_results_path("multiclass_bci_experiment")
    
    # Prepare data for JSON serialization
    serializable_results = {}
    for key, value in results_summary.items():
        serializable_results[key] = {
            'logistic_results': {
                'fast': {k: float(v) for k, v in value['logistic_results']['fast'].items()},
                'hierarchical': {k: float(v) for k, v in value['logistic_results']['hierarchical'].items()}
            },
            'data_shape': list(value['data_shape']),
            'num_trials': value['num_trials'],
            'label_distribution': value['label_distribution'],
            'classes_present': value['classes_present']
        }
    
    # Save detailed results
    detailed_results = {
        'experiment_timestamp': timestamp,
        'processor_config': {
            'fast_window_size': processor.fast_window_size,
            'slow_context_size': processor.slow_context_size,
            'num_classes': processor.num_classes
        },
        'results_summary': serializable_results,
        'model_comparison': processor.comparator.comparison_stats
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    # Save model comparison results
    comparison_file = get_model_comparison_path("multiclass_bci_model_comparison")
    processor.comparator.save_results(str(comparison_file))
    
    # Print summary
    print(f"\n=== Multi-Class Experiment Summary ===")
    print(f"Processed {len(results_summary)} BCI files")
    print(f"Results saved to: {results_file}")
    print(f"Model comparison saved to: {comparison_file}")
    
    # Print detailed results
    print(f"\nDetailed Results:")
    for file_name, result in results_summary.items():
        print(f"\n{file_name}:")
        print(f"  Trials: {result['num_trials']}")
        print(f"  Features: {result['data_shape']}")
        print(f"  Classes present: {result['classes_present']}")
        print(f"  Label distribution: {result['label_distribution']}")
        print(f"  Fast model accuracy: {result['logistic_results']['fast']['accuracy']:.3f}")
        print(f"  Hierarchical model accuracy: {result['logistic_results']['hierarchical']['accuracy']:.3f}")
    
    # Generate comparison plot
    print(f"\n4. Generating comparison plots...")
    processor.comparator.plot_comparison()
    
    print(f"\n✅ Multi-class BCI experiment complete!")
    return results_summary

def quick_multiclass_test():
    """Quick test with a single BCI file for multi-class classification."""
    
    print("=== Quick Multi-Class BCI Test ===")
    
    # Initialize processor
    processor = MultiClassHierarchicalProcessor(fast_window_size=10, slow_context_size=50, num_classes=4)
    
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
    X_fast, X_slow, y = process_bci_trials(
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
    print(f"Label distribution: {np.bincount(y.astype(int))}")
    
    # Quick training test
    print("\nRunning quick multi-class training test...")
    (Xf_train, Xf_test, Xs_train, Xs_test, 
     y_train, y_test, H_train, H_test, training_time) = processor.train_multiclass_models(X_fast, X_slow, y)
    
    results = processor.evaluate_multiclass_models(Xf_test, Xs_test, y_test, H_test, training_time)
    
    print(f"\n✅ Quick multi-class test complete!")
    print(f"Fast model accuracy: {results['fast']['accuracy']:.3f}")
    print(f"Hierarchical model accuracy: {results['hierarchical']['accuracy']:.3f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_multiclass_test()
    else:
        run_multiclass_bci_experiment() 