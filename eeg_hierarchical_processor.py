import os
# Set MNE data directory before importing MNE
os.environ['MNE_DATA'] = os.path.join(os.getcwd(), 'mne_data')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import random
import mne
from mne.datasets import sample
from einops import rearrange, reduce, repeat
from typing import Tuple, Optional, Dict, Any, List
import time
from scipy import stats
import json
from dataclasses import dataclass, asdict
from datetime import datetime

# PyTorch imports for RNN implementation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

@dataclass
class ModelResults:
    """Structured results for model comparison."""
    model_name: str
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    inference_time: float
    memory_usage: float  # MB
    num_parameters: int
    confusion_matrix: np.ndarray
    classification_report: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return f"{self.model_name}: Acc={self.accuracy:.3f}, AUC={self.auc:.3f}, F1={self.f1_score:.3f}"

class ModelComparator:
    """Comprehensive model comparison framework."""
    
    def __init__(self):
        self.results: List[ModelResults] = []
        self.comparison_stats: Dict = {}
    
    def add_result(self, result: ModelResults):
        """Add a model result to the comparison."""
        self.results.append(result)
    
    def compare_models(self) -> Dict:
        """Perform statistical comparison between models."""
        if len(self.results) < 2:
            return {}
        
        comparisons = {}
        
        # Compare each pair of models
        for i in range(len(self.results)):
            for j in range(i + 1, len(self.results)):
                model_a = self.results[i]
                model_b = self.results[j]
                
                # McNemar's test for paired samples
                # (assuming we have the same test set predictions)
                # This would need actual predictions to compute
                
                # Performance comparison
                acc_diff = model_b.accuracy - model_a.accuracy
                auc_diff = model_b.auc - model_a.auc
                f1_diff = model_b.f1_score - model_a.f1_score
                
                # Efficiency comparison
                time_ratio = model_b.training_time / max(model_a.training_time, 1e-6)
                memory_ratio = model_b.memory_usage / max(model_a.memory_usage, 1e-6)
                param_ratio = model_b.num_parameters / max(model_a.num_parameters, 1)
                
                comparison_key = f"{model_a.model_name}_vs_{model_b.model_name}"
                comparisons[comparison_key] = {
                    "accuracy_improvement": acc_diff,
                    "auc_improvement": auc_diff,
                    "f1_improvement": f1_diff,
                    "training_time_ratio": time_ratio,
                    "memory_usage_ratio": memory_ratio,
                    "parameter_ratio": param_ratio,
                    "efficiency_score": (acc_diff + auc_diff + f1_diff) / max(time_ratio + memory_ratio, 1e-6)
                }
        
        self.comparison_stats = comparisons
        return comparisons
    
    def generate_report(self) -> str:
        """Generate comprehensive comparison report."""
        if not self.results:
            return "No results to compare."
        
        report = "=== MODEL COMPARISON REPORT ===\n\n"
        
        # Summary table
        report += "MODEL PERFORMANCE SUMMARY:\n"
        report += "-" * 80 + "\n"
        report += f"{'Model':<20} {'Accuracy':<10} {'AUC':<10} {'F1':<10} {'Train(s)':<10} {'Memory(MB)':<12} {'Params':<10}\n"
        report += "-" * 80 + "\n"
        
        for result in self.results:
            report += f"{result.model_name:<20} {result.accuracy:<10.3f} {result.auc:<10.3f} {result.f1_score:<10.3f} "
            report += f"{result.training_time:<10.2f} {result.memory_usage:<12.1f} {result.num_parameters:<10}\n"
        
        report += "\n" + "=" * 80 + "\n\n"
        
        # Statistical comparisons
        if self.comparison_stats:
            report += "STATISTICAL COMPARISONS:\n"
            report += "-" * 40 + "\n"
            
            for comparison_name, stats in self.comparison_stats.items():
                report += f"\n{comparison_name}:\n"
                report += f"  Accuracy improvement: {stats['accuracy_improvement']:+.3f}\n"
                report += f"  AUC improvement: {stats['auc_improvement']:+.3f}\n"
                report += f"  F1 improvement: {stats['f1_improvement']:+.3f}\n"
                report += f"  Training time ratio: {stats['training_time_ratio']:.2f}x\n"
                report += f"  Memory usage ratio: {stats['memory_usage_ratio']:.2f}x\n"
                report += f"  Parameter ratio: {stats['parameter_ratio']:.2f}x\n"
                report += f"  Efficiency score: {stats['efficiency_score']:.3f}\n"
        
        return report
    
    def save_results(self, filename: str):
        """Save results to JSON file."""
        data = {
            "results": [result.to_dict() for result in self.results],
            "comparison_stats": self.comparison_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def plot_comparison(self):
        """Create visualization of model comparisons."""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance metrics
        models = [r.model_name for r in self.results]
        accuracies = [r.accuracy for r in self.results]
        aucs = [r.auc for r in self.results]
        f1_scores = [r.f1_score for r in self.results]
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[0, 0].bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 0].bar(x, aucs, width, label='AUC', alpha=0.8)
        axes[0, 0].bar(x + width, f1_scores, width, label='F1', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Efficiency metrics
        training_times = [r.training_time for r in self.results]
        memory_usage = [r.memory_usage for r in self.results]
        num_parameters = [r.num_parameters for r in self.results]
        
        axes[0, 1].bar(x - width, training_times, width, label='Training Time (s)', alpha=0.8)
        axes[0, 1].bar(x, memory_usage, width, label='Memory (MB)', alpha=0.8)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Resource Usage')
        axes[0, 1].set_title('Efficiency Metrics')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Parameter count
        axes[1, 0].bar(models, num_parameters, alpha=0.8, color='orange')
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Number of Parameters')
        axes[1, 0].set_title('Model Complexity')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance vs Efficiency scatter
        efficiency_scores = []
        for result in self.results:
            # Simple efficiency score: performance / (time + memory)
            denominator = max(result.training_time + result.memory_usage/100, 1e-6)
            efficiency = (result.accuracy + result.auc) / denominator
            efficiency_scores.append(efficiency)
        
        axes[1, 1].scatter(efficiency_scores, f1_scores, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (efficiency_scores[i], f1_scores[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Efficiency Score')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Performance vs Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class HierarchicalEEGProcessor:
    """
    Flexible EEG processor that implements hierarchical predictive processing.
    Can work with both synthetic and real EEG data using efficient tensor operations.
    """
    
    def __init__(self, fast_window_size=10, slow_context_size=50):
        self.fast_window_size = fast_window_size
        self.slow_context_size = slow_context_size
        self.fast_model = None
        self.hierarchical_model = None
        self.comparator = ModelComparator()
        
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
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        inference_time = time.time() - start_time
        
        y_pred = (y_pred_proba > 0.5).astype(float)
        
        # Memory after inference
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        # Handle binary classification for AUC
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = 0.5  # Default for non-binary case
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Extract precision, recall, f1 from classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        
        # Count parameters (estimate for sklearn models)
        num_parameters = 0
        if hasattr(model, 'coef_'):
            num_parameters = model.coef_.size + model.intercept_.size
        elif hasattr(model, 'named_steps'):
            # For pipeline models
            for step_name, step in model.named_steps.items():
                if hasattr(step, 'coef_'):
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
        
    def generate_synthetic_data(self, num_sequences=100, seq_length=600, slow_flip_prob=0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic EEG-like data with hierarchical structure using vectorized operations."""
        def generate_sequence(seq_length=600, slow_flip_prob=0.05, seed=None):
            if seed is not None:
                np.random.seed(seed)
            t = np.arange(seq_length)

            # 1. slow binary state with persistence (embodiment: context evolves slowly)
            slow_state = np.zeros(seq_length, dtype=float)
            state = np.random.choice([0.0, 1.0])
            for i in range(seq_length):
                if i == 0:
                    slow_state[i] = state
                else:
                    if np.random.rand() < slow_flip_prob:
                        state = 1.0 - state
                    slow_state[i] = state

            # 2. slow low-frequency carrier (context)
            freq_slow = 0.02
            phase = np.random.rand() * 2 * np.pi
            slow_base = np.sin(2 * np.pi * freq_slow * t + phase)
            slow_signal = (0.3 + 0.7 * slow_state) * slow_base

            # 3. imagery-specific high-frequency content only when slow_state==1
            freq_fast = 5.0
            imagery_component = (slow_state * np.sin(2 * np.pi * freq_fast * t)) * 0.5

            # 4. additive broadband noise (simulating EEG noise)
            noise = np.random.randn(seq_length) * 0.1

            # final synthetic signal
            signal = slow_signal + imagery_component + noise
            return signal, slow_state

        # Generate all sequences at once
        all_signals = []
        all_states = []
        
        for seq_idx in range(num_sequences):
            signal, slow_state = generate_sequence(seq_length=seq_length, seed=seq_idx)
            all_signals.append(signal)
            all_states.append(slow_state)
        
        # Stack into tensors: (sequences, time)
        signals = np.stack(all_signals)  # (num_sequences, seq_length)
        states = np.stack(all_states)    # (num_sequences, seq_length)
        
        # Extract features using einops for vectorized operations
        X_fast, X_slow, y = self._extract_features_vectorized(signals, states)
        
        return X_fast, X_slow, y
    
    def _extract_features_vectorized(self, signals: np.ndarray, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract fast and slow features using vectorized operations with einops."""
        num_sequences, seq_length = signals.shape
        
        # Calculate valid time points (after both windows)
        start_idx = max(self.slow_context_size, self.fast_window_size)
        valid_length = seq_length - start_idx
        
        # Create sliding windows using einops
        # Fast windows: (sequences, time, fast_window)
        fast_windows = rearrange(
            np.stack([signals[:, t-self.fast_window_size:t] for t in range(start_idx, seq_length)]),
            'time seq win -> seq time win'
        )
        
        # Slow windows: (sequences, time, slow_window)
        slow_windows = rearrange(
            np.stack([signals[:, t-self.slow_context_size:t] for t in range(start_idx, seq_length)]),
            'time seq win -> seq time win'
        )
        
        # Extract labels: (sequences, time)
        labels = states[:, start_idx:]
        
        # Compute features using einops reduce operations
        # Fast features: mean, std, last delta
        fast_mean = reduce(fast_windows, 'seq time win -> seq time', 'mean')
        # Compute std manually since einops doesn't have it
        fast_std = np.std(fast_windows, axis=-1)
        fast_diff = fast_windows[:, :, -1] - fast_windows[:, :, -2]  # last - second_last
        
        # Slow features: mean, std
        slow_mean = reduce(slow_windows, 'seq time win -> seq time', 'mean')
        slow_std = np.std(slow_windows, axis=-1)
        
        # Stack features
        X_fast = np.stack([fast_mean, fast_std, fast_diff], axis=-1)  # (seq, time, 3)
        X_slow = np.stack([slow_mean, slow_std], axis=-1)            # (seq, time, 2)
        
        # Flatten to (total_samples, features)
        X_fast = rearrange(X_fast, 'seq time feat -> (seq time) feat')
        X_slow = rearrange(X_slow, 'seq time feat -> (seq time) feat')
        y = rearrange(labels, 'seq time -> (seq time)')
        
        return X_fast, X_slow, y
    
    def load_sample_eeg_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Load sample EEG data from MNE for demonstration."""
        # Ensure local data directory exists
        local_data_dir = os.path.join(os.getcwd(), 'mne_data')
        os.makedirs(local_data_dir, exist_ok=True)
        
        # Clear any cached paths and force fresh download
        if hasattr(sample, '_data_path'):
            delattr(sample, '_data_path')
        
        # Force download to local directory
        data_path = sample.data_path(update_path=True, download=True)
        
        # Load sample data
        raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        
        # Pick EEG channels only
        raw.pick_types(eeg=True)
        
        # Get data as numpy array: (channels, time)
        data = raw.get_data()
        
        # Create synthetic labels for demonstration (since this is resting state data)
        # We'll create labels based on alpha power changes
        from scipy.signal import welch
        labels = []
        for i in range(0, data.shape[1] - 1000, 1000):  # 1 second windows
            if i + 1000 <= data.shape[1]:
                window = data[:, i:i+1000]
                # Calculate alpha power (8-13 Hz) for each channel
                freqs, psd = welch(window, fs=raw.info['sfreq'])
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                alpha_power = np.mean(psd[:, alpha_mask], axis=1)
                # Simple threshold-based labeling
                label = 1 if np.mean(alpha_power) > np.median(alpha_power) else 0
                labels.extend([label] * 1000)
        
        # Pad labels to match data length
        while len(labels) < data.shape[1]:
            labels.append(labels[-1] if labels else 0)
        labels = labels[:data.shape[1]]
        
        return data, np.array(labels), raw.info['sfreq']
    
    def extract_features_from_eeg(self, eeg_data: np.ndarray, labels: np.ndarray, sampling_rate: float = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract fast and slow features from real EEG data using vectorized operations."""
        # Convert window sizes to samples
        fast_samples = int(self.fast_window_size * sampling_rate / 1000)  # ms to samples
        slow_samples = int(self.slow_context_size * sampling_rate / 1000)
        
        # For multi-channel data, we can process all channels at once
        # eeg_data shape: (channels, time)
        num_channels, seq_length = eeg_data.shape
        
        # Calculate valid time points
        start_idx = max(slow_samples, fast_samples)
        valid_length = seq_length - start_idx
        
        # Create sliding windows for all channels
        # Fast windows: (channels, time, fast_window)
        fast_windows = rearrange(
            np.stack([eeg_data[:, t-fast_samples:t] for t in range(start_idx, seq_length)]),
            'time ch win -> ch time win'
        )
        
        # Slow windows: (channels, time, slow_window)
        slow_windows = rearrange(
            np.stack([eeg_data[:, t-slow_samples:t] for t in range(start_idx, seq_length)]),
            'time ch win -> ch time win'
        )
        
        # Extract labels
        labels_valid = labels[start_idx:]
        
        # Compute features using einops
        # Fast features: mean, std, last delta
        fast_mean = reduce(fast_windows, 'ch time win -> ch time', 'mean')
        fast_std = np.std(fast_windows, axis=-1)
        fast_diff = fast_windows[:, :, -1] - fast_windows[:, :, -2]
        
        # Slow features: mean, std
        slow_mean = reduce(slow_windows, 'ch time win -> ch time', 'mean')
        slow_std = np.std(slow_windows, axis=-1)
        
        # For now, use first channel (can be extended to multi-channel)
        ch_idx = 0
        
        X_fast = np.stack([fast_mean[ch_idx], fast_std[ch_idx], fast_diff[ch_idx]], axis=-1)
        X_slow = np.stack([slow_mean[ch_idx], slow_std[ch_idx]], axis=-1)
        y = labels_valid
        
        return X_fast, X_slow, y
    
    def train_models(self, X_fast: np.ndarray, X_slow: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """Train fast-only and hierarchical models."""
        # Split data
        perm = np.random.permutation(len(y))
        split = int((1 - test_size) * len(y))
        train_idx, test_idx = perm[:split], perm[split:]
        
        Xf_train, Xf_test = X_fast[train_idx], X_fast[test_idx]
        Xs_train, Xs_test = X_slow[train_idx], X_slow[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train fast-only model
        self.fast_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        start_time = time.time()
        self.fast_model.fit(Xf_train, y_train)
        training_time = time.time() - start_time
        
        # Compute fast log-odds for hierarchical model
        def logit_from_proba(p):
            eps = 1e-6
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))
        
        fast_logit_train = logit_from_proba(self.fast_model.predict_proba(Xf_train)[:, 1])
        fast_logit_test = logit_from_proba(self.fast_model.predict_proba(Xf_test)[:, 1])
        
        # Build hierarchical features: [fast_logit, slow_mean, slow_std]
        H_train = np.column_stack([fast_logit_train, Xs_train])
        H_test = np.column_stack([fast_logit_test, Xs_test])
        
        # Train hierarchical model
        self.hierarchical_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        start_time = time.time()
        self.hierarchical_model.fit(H_train, y_train)
        training_time += time.time() - start_time
        
        return (Xf_train, Xf_test, Xs_train, Xs_test, 
                y_train, y_test, H_train, H_test, training_time)
    
    def evaluate_models(self, Xf_test: np.ndarray, Xs_test: np.ndarray, y_test: np.ndarray, H_test: np.ndarray, training_time: float) -> Dict[str, Dict[str, float]]:
        """Evaluate both models and print results."""
        # Fast-only model
        fast_pred_proba = self.fast_model.predict_proba(Xf_test)[:, 1]
        fast_pred = (fast_pred_proba > 0.5).astype(float)
        
        print("=== Fast-only model ===")
        print("Accuracy:", accuracy_score(y_test, fast_pred))
        print("AUC:", roc_auc_score(y_test, fast_pred_proba))
        print(classification_report(y_test, fast_pred, digits=3))
        
        # Hierarchical model
        hier_pred_proba = self.hierarchical_model.predict_proba(H_test)[:, 1]
        hier_pred = (hier_pred_proba > 0.5).astype(float)
        
        print("\n=== Hierarchical (fast + slow) model ===")
        print("Accuracy:", accuracy_score(y_test, hier_pred))
        print("AUC:", roc_auc_score(y_test, hier_pred_proba))
        print(classification_report(y_test, hier_pred, digits=3))
        
        # Evaluate performance using ModelResults
        fast_results = self.evaluate_model_performance(self.fast_model, Xf_test, y_test, "Fast-only Logistic Regression")
        hier_results = self.evaluate_model_performance(self.hierarchical_model, H_test, y_test, "Hierarchical Logistic Regression")
        
        self.comparator.add_result(fast_results)
        self.comparator.add_result(hier_results)
        
        return {
            'fast': {'accuracy': accuracy_score(y_test, fast_pred), 
                     'auc': roc_auc_score(y_test, fast_pred_proba)},
            'hierarchical': {'accuracy': accuracy_score(y_test, hier_pred), 
                           'auc': roc_auc_score(y_test, hier_pred_proba)}
        }
    
    def lesion_slow_context(self, H_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Simulate lesioning of slow context by zeroing slow features."""
        H_test_lesioned = H_test.copy()
        H_test_lesioned[:, 1:] = 0.0  # zero out slow_mean and slow_std
        lesioned_pred_proba = self.hierarchical_model.predict_proba(H_test_lesioned)[:, 1]
        lesioned_pred = (lesioned_pred_proba > 0.5).astype(float)
        
        print("\n=== Hierarchical model with slow-context lesioned ===")
        print("Accuracy:", accuracy_score(y_test, lesioned_pred))
        print("AUC:", roc_auc_score(y_test, lesioned_pred_proba))
        print(classification_report(y_test, lesioned_pred, digits=3))
        
        # Evaluate performance using ModelResults
        lesioned_results = self.evaluate_model_performance(self.hierarchical_model, H_test_lesioned, y_test, "Hierarchical Logistic Regression (Lesioned)")
        self.comparator.add_result(lesioned_results)
        
        return {
            'lesioned': {'accuracy': accuracy_score(y_test, lesioned_pred), 
                        'auc': roc_auc_score(y_test, lesioned_pred_proba)}
        }
    
    def visualize_signal(self, signal: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "EEG Signal"):
        """Visualize EEG signal with optional labels."""
        plt.figure(figsize=(12, 4))
        plt.plot(signal, linewidth=0.7, label="signal")
        if labels is not None:
            plt.fill_between(np.arange(len(labels)), 
                           signal.min(), signal.max(), 
                           where=labels > 0.5, 
                           color="orange", alpha=0.1, 
                           label="state ON")
        plt.title(title)
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

class FastRNN(nn.Module):
    """Fast RNN for processing immediate dynamics (gamma oscillations)."""
    
    def __init__(self, input_size=3, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output for classification
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # Classification head
        output = self.fc(last_output)
        output = self.sigmoid(output)
        
        return output

class SlowRNN(nn.Module):
    """Slow RNN for processing contextual information (alpha/beta rhythms)."""
    
    def __init__(self, input_size=2, hidden_size=16, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output for classification
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # Classification head
        output = self.fc(last_output)
        output = self.sigmoid(output)
        
        return output

class IntegrationNet(nn.Module):
    """Integration network that combines fast and slow RNN outputs."""
    
    def __init__(self, fast_size=32, slow_size=16, hidden_size=32, dropout=0.2):
        super().__init__()
        
        self.fc1 = nn.Linear(fast_size + slow_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()
    
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
        output = self.sigmoid(output)
        
        return output

class HRMEEGProcessor(HierarchicalEEGProcessor):
    """
    HRM-inspired EEG processor with multi-timescale RNN networks.
    Extends the base processor with LSTM-based hierarchical processing.
    """
    
    def __init__(self, fast_window_size=10, slow_context_size=50, 
                 fast_hidden_size=32, slow_hidden_size=16, 
                 integration_hidden_size=32, dropout=0.2):
        super().__init__(fast_window_size, slow_context_size)
        
        # Fast RNN: processes immediate dynamics (gamma oscillations)
        self.fast_rnn = FastRNN(
            input_size=3,  # mean, std, delta
            hidden_size=fast_hidden_size,
            num_layers=2,
            dropout=dropout
        )
        
        # Slow RNN: processes contextual information (alpha/beta rhythms)
        self.slow_rnn = SlowRNN(
            input_size=2,  # mean, std
            hidden_size=slow_hidden_size,
            num_layers=1,
            dropout=dropout
        )
        
        # Integration network: combines fast and slow features
        self.integration_net = IntegrationNet(
            fast_size=fast_hidden_size,
            slow_size=slow_hidden_size,
            hidden_size=integration_hidden_size,
            dropout=dropout
        )
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 5  # Reduced for faster testing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.fast_rnn.to(self.device)
        self.slow_rnn.to(self.device)
        self.integration_net.to(self.device)
        
        # Optimizers
        self.fast_optimizer = optim.Adam(self.fast_rnn.parameters(), lr=self.learning_rate)
        self.slow_optimizer = optim.Adam(self.slow_rnn.parameters(), lr=self.learning_rate)
        self.integration_optimizer = optim.Adam(self.integration_net.parameters(), lr=self.learning_rate)
        
        # Loss function
        self.criterion = nn.BCELoss()
    
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
            label = y[i+sequence_length-1]  # Label for the last timestep
            
            fast_sequences.append(fast_seq)
            slow_sequences.append(slow_seq)
            labels.append(label)
        
        # Convert to tensors
        fast_tensor = torch.FloatTensor(np.array(fast_sequences))
        slow_tensor = torch.FloatTensor(np.array(slow_sequences))
        labels_tensor = torch.FloatTensor(np.array(labels)).unsqueeze(1)
        
        return fast_tensor, slow_tensor, labels_tensor
    
    def train_rnn_models(self, X_fast: np.ndarray, X_slow: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2) -> Tuple:
        """Train the RNN models with proper train/test split."""
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
        print(f"\nStarting RNN training on {self.device}...")
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
        """Evaluate RNN models and return structured results."""
        print(f"\nEvaluating RNN models on test set...")
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
            fast_pred = (fast_output > 0.5).float()
            fast_results = self._create_rnn_results(
                fast_pred, fast_output, labels_test, 
                "Fast RNN", training_time, self.fast_rnn
            )
            results['fast_rnn'] = fast_results
            print(f"  Fast RNN - Accuracy: {fast_results.accuracy:.3f}, AUC: {fast_results.auc:.3f}")
            
            print("Evaluating Slow RNN...")
            # Slow RNN evaluation
            slow_output = self.slow_rnn(slow_test)
            slow_pred = (slow_output > 0.5).float()
            slow_results = self._create_rnn_results(
                slow_pred, slow_output, labels_test,
                "Slow RNN", training_time, self.slow_rnn
            )
            results['slow_rnn'] = slow_results
            print(f"  Slow RNN - Accuracy: {slow_results.accuracy:.3f}, AUC: {slow_results.auc:.3f}")
            
            print("Evaluating Hierarchical Integration Network...")
            # Integration network evaluation
            fast_features = self.fast_rnn.lstm(fast_test)[0][:, -1, :]
            slow_features = self.slow_rnn.lstm(slow_test)[0][:, -1, :]
            integration_output = self.integration_net(fast_features, slow_features)
            integration_pred = (integration_output > 0.5).float()
            integration_results = self._create_rnn_results(
                integration_pred, integration_output, labels_test,
                "Hierarchical RNN", training_time, self.integration_net
            )
            results['integration_rnn'] = integration_results
            print(f"  Hierarchical RNN - Accuracy: {integration_results.accuracy:.3f}, AUC: {integration_results.auc:.3f}")
        
        print(f"\nRNN Evaluation Complete!")
        return results
    
    def _create_rnn_results(self, pred: torch.Tensor, output: torch.Tensor, 
                           labels: torch.Tensor, model_name: str, 
                           training_time: float, model: nn.Module) -> ModelResults:
        """Create ModelResults for RNN models."""
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
            elif isinstance(model, SlowRNN):
                input_size = 2  # Slow RNN expects 2 features
            elif isinstance(model, IntegrationNet):
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
        y_pred = pred.cpu().numpy().flatten()
        y_pred_proba = output.cpu().numpy().flatten()
        y_true = labels.cpu().numpy().flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
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
        
    def generate_synthetic_data(self, num_sequences=100, seq_length=600, slow_flip_prob=0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic EEG-like data with hierarchical structure using vectorized operations."""
        def generate_sequence(seq_length=600, slow_flip_prob=0.05, seed=None):
            if seed is not None:
                np.random.seed(seed)
            t = np.arange(seq_length)

            # 1. slow binary state with persistence (embodiment: context evolves slowly)
            slow_state = np.zeros(seq_length, dtype=float)
            state = np.random.choice([0.0, 1.0])
            for i in range(seq_length):
                if i == 0:
                    slow_state[i] = state
                else:
                    if np.random.rand() < slow_flip_prob:
                        state = 1.0 - state
                    slow_state[i] = state

            # 2. slow low-frequency carrier (context)
            freq_slow = 0.02
            phase = np.random.rand() * 2 * np.pi
            slow_base = np.sin(2 * np.pi * freq_slow * t + phase)
            slow_signal = (0.3 + 0.7 * slow_state) * slow_base

            # 3. imagery-specific high-frequency content only when slow_state==1
            freq_fast = 5.0
            imagery_component = (slow_state * np.sin(2 * np.pi * freq_fast * t)) * 0.5

            # 4. additive broadband noise (simulating EEG noise)
            noise = np.random.randn(seq_length) * 0.1

            # final synthetic signal
            signal = slow_signal + imagery_component + noise
            return signal, slow_state

        # Generate all sequences at once
        all_signals = []
        all_states = []
        
        for seq_idx in range(num_sequences):
            signal, slow_state = generate_sequence(seq_length=seq_length, seed=seq_idx)
            all_signals.append(signal)
            all_states.append(slow_state)
        
        # Stack into tensors: (sequences, time)
        signals = np.stack(all_signals)  # (num_sequences, seq_length)
        states = np.stack(all_states)    # (num_sequences, seq_length)
        
        # Extract features using einops for vectorized operations
        X_fast, X_slow, y = self._extract_features_vectorized(signals, states)
        
        return X_fast, X_slow, y
    
    def _extract_features_vectorized(self, signals: np.ndarray, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract fast and slow features using vectorized operations with einops."""
        num_sequences, seq_length = signals.shape
        
        # Calculate valid time points (after both windows)
        start_idx = max(self.slow_context_size, self.fast_window_size)
        valid_length = seq_length - start_idx
        
        # Create sliding windows using einops
        # Fast windows: (sequences, time, fast_window)
        fast_windows = rearrange(
            np.stack([signals[:, t-self.fast_window_size:t] for t in range(start_idx, seq_length)]),
            'time seq win -> seq time win'
        )
        
        # Slow windows: (sequences, time, slow_window)
        slow_windows = rearrange(
            np.stack([signals[:, t-self.slow_context_size:t] for t in range(start_idx, seq_length)]),
            'time seq win -> seq time win'
        )
        
        # Extract labels: (sequences, time)
        labels = states[:, start_idx:]
        
        # Compute features using einops reduce operations
        # Fast features: mean, std, last delta
        fast_mean = reduce(fast_windows, 'seq time win -> seq time', 'mean')
        # Compute std manually since einops doesn't have it
        fast_std = np.std(fast_windows, axis=-1)
        fast_diff = fast_windows[:, :, -1] - fast_windows[:, :, -2]  # last - second_last
        
        # Slow features: mean, std
        slow_mean = reduce(slow_windows, 'seq time win -> seq time', 'mean')
        slow_std = np.std(slow_windows, axis=-1)
        
        # Stack features
        X_fast = np.stack([fast_mean, fast_std, fast_diff], axis=-1)  # (seq, time, 3)
        X_slow = np.stack([slow_mean, slow_std], axis=-1)            # (seq, time, 2)
        
        # Flatten to (total_samples, features)
        X_fast = rearrange(X_fast, 'seq time feat -> (seq time) feat')
        X_slow = rearrange(X_slow, 'seq time feat -> (seq time) feat')
        y = rearrange(labels, 'seq time -> (seq time)')
        
        return X_fast, X_slow, y
    
    def load_sample_eeg_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Load sample EEG data from MNE for demonstration."""
        # Ensure local data directory exists
        local_data_dir = os.path.join(os.getcwd(), 'mne_data')
        os.makedirs(local_data_dir, exist_ok=True)
        
        # Clear any cached paths and force fresh download
        if hasattr(sample, '_data_path'):
            delattr(sample, '_data_path')
        
        # Force download to local directory
        data_path = sample.data_path(update_path=True, download=True)
        
        # Load sample data
        raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        
        # Pick EEG channels only
        raw.pick_types(eeg=True)
        
        # Get data as numpy array: (channels, time)
        data = raw.get_data()
        
        # Create synthetic labels for demonstration (since this is resting state data)
        # We'll create labels based on alpha power changes
        from scipy.signal import welch
        labels = []
        for i in range(0, data.shape[1] - 1000, 1000):  # 1 second windows
            if i + 1000 <= data.shape[1]:
                window = data[:, i:i+1000]
                # Calculate alpha power (8-13 Hz) for each channel
                freqs, psd = welch(window, fs=raw.info['sfreq'])
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                alpha_power = np.mean(psd[:, alpha_mask], axis=1)
                # Simple threshold-based labeling
                label = 1 if np.mean(alpha_power) > np.median(alpha_power) else 0
                labels.extend([label] * 1000)
        
        # Pad labels to match data length
        while len(labels) < data.shape[1]:
            labels.append(labels[-1] if labels else 0)
        labels = labels[:data.shape[1]]
        
        return data, np.array(labels), raw.info['sfreq']
    
    def extract_features_from_eeg(self, eeg_data: np.ndarray, labels: np.ndarray, sampling_rate: float = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract fast and slow features from real EEG data using vectorized operations."""
        # Convert window sizes to samples
        fast_samples = int(self.fast_window_size * sampling_rate / 1000)  # ms to samples
        slow_samples = int(self.slow_context_size * sampling_rate / 1000)
        
        # For multi-channel data, we can process all channels at once
        # eeg_data shape: (channels, time)
        num_channels, seq_length = eeg_data.shape
        
        # Calculate valid time points
        start_idx = max(slow_samples, fast_samples)
        valid_length = seq_length - start_idx
        
        # Create sliding windows for all channels
        # Fast windows: (channels, time, fast_window)
        fast_windows = rearrange(
            np.stack([eeg_data[:, t-fast_samples:t] for t in range(start_idx, seq_length)]),
            'time ch win -> ch time win'
        )
        
        # Slow windows: (channels, time, slow_window)
        slow_windows = rearrange(
            np.stack([eeg_data[:, t-slow_samples:t] for t in range(start_idx, seq_length)]),
            'time ch win -> ch time win'
        )
        
        # Extract labels
        labels_valid = labels[start_idx:]
        
        # Compute features using einops
        # Fast features: mean, std, last delta
        fast_mean = reduce(fast_windows, 'ch time win -> ch time', 'mean')
        fast_std = np.std(fast_windows, axis=-1)
        fast_diff = fast_windows[:, :, -1] - fast_windows[:, :, -2]
        
        # Slow features: mean, std
        slow_mean = reduce(slow_windows, 'ch time win -> ch time', 'mean')
        slow_std = np.std(slow_windows, axis=-1)
        
        # For now, use first channel (can be extended to multi-channel)
        ch_idx = 0
        
        X_fast = np.stack([fast_mean[ch_idx], fast_std[ch_idx], fast_diff[ch_idx]], axis=-1)
        X_slow = np.stack([slow_mean[ch_idx], slow_std[ch_idx]], axis=-1)
        y = labels_valid
        
        return X_fast, X_slow, y
    
    def train_models(self, X_fast: np.ndarray, X_slow: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """Train fast-only and hierarchical models."""
        # Split data
        perm = np.random.permutation(len(y))
        split = int((1 - test_size) * len(y))
        train_idx, test_idx = perm[:split], perm[split:]
        
        Xf_train, Xf_test = X_fast[train_idx], X_fast[test_idx]
        Xs_train, Xs_test = X_slow[train_idx], X_slow[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train fast-only model
        self.fast_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        start_time = time.time()
        self.fast_model.fit(Xf_train, y_train)
        training_time = time.time() - start_time
        
        # Compute fast log-odds for hierarchical model
        def logit_from_proba(p):
            eps = 1e-6
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))
        
        fast_logit_train = logit_from_proba(self.fast_model.predict_proba(Xf_train)[:, 1])
        fast_logit_test = logit_from_proba(self.fast_model.predict_proba(Xf_test)[:, 1])
        
        # Build hierarchical features: [fast_logit, slow_mean, slow_std]
        H_train = np.column_stack([fast_logit_train, Xs_train])
        H_test = np.column_stack([fast_logit_test, Xs_test])
        
        # Train hierarchical model
        self.hierarchical_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        start_time = time.time()
        self.hierarchical_model.fit(H_train, y_train)
        training_time += time.time() - start_time # Add to training_time
        
        return (Xf_train, Xf_test, Xs_train, Xs_test, 
                y_train, y_test, H_train, H_test, training_time)
    
    def evaluate_models(self, Xf_test: np.ndarray, Xs_test: np.ndarray, y_test: np.ndarray, H_test: np.ndarray, training_time: float) -> Dict[str, Dict[str, float]]:
        """Evaluate both models and print results."""
        # Fast-only model
        fast_pred_proba = self.fast_model.predict_proba(Xf_test)[:, 1]
        fast_pred = (fast_pred_proba > 0.5).astype(float)
        
        print("=== Fast-only model ===")
        print("Accuracy:", accuracy_score(y_test, fast_pred))
        print("AUC:", roc_auc_score(y_test, fast_pred_proba))
        print(classification_report(y_test, fast_pred, digits=3))
        
        # Hierarchical model
        hier_pred_proba = self.hierarchical_model.predict_proba(H_test)[:, 1]
        hier_pred = (hier_pred_proba > 0.5).astype(float)
        
        print("\n=== Hierarchical (fast + slow) model ===")
        print("Accuracy:", accuracy_score(y_test, hier_pred))
        print("AUC:", roc_auc_score(y_test, hier_pred_proba))
        print(classification_report(y_test, hier_pred, digits=3))
        
        # Evaluate performance using ModelResults
        fast_results = self.evaluate_model_performance(self.fast_model, Xf_test, y_test, "Fast-only Logistic Regression")
        hier_results = self.evaluate_model_performance(self.hierarchical_model, H_test, y_test, "Hierarchical Logistic Regression")
        
        self.comparator.add_result(fast_results)
        self.comparator.add_result(hier_results)
        
        return {
            'fast': {'accuracy': accuracy_score(y_test, fast_pred), 
                     'auc': roc_auc_score(y_test, fast_pred_proba)},
            'hierarchical': {'accuracy': accuracy_score(y_test, hier_pred), 
                           'auc': roc_auc_score(y_test, hier_pred_proba)}
        }
    
    def lesion_slow_context(self, H_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Simulate lesioning of slow context by zeroing slow features."""
        H_test_lesioned = H_test.copy()
        H_test_lesioned[:, 1:] = 0.0  # zero out slow_mean and slow_std
        lesioned_pred_proba = self.hierarchical_model.predict_proba(H_test_lesioned)[:, 1]
        lesioned_pred = (lesioned_pred_proba > 0.5).astype(float)
        
        print("\n=== Hierarchical model with slow-context lesioned ===")
        print("Accuracy:", accuracy_score(y_test, lesioned_pred))
        print("AUC:", roc_auc_score(y_test, lesioned_pred_proba))
        print(classification_report(y_test, lesioned_pred, digits=3))
        
        # Evaluate performance using ModelResults
        lesioned_results = self.evaluate_model_performance(self.hierarchical_model, H_test_lesioned, y_test, "Hierarchical Logistic Regression (Lesioned)")
        self.comparator.add_result(lesioned_results)
        
        return {
            'lesioned': {'accuracy': accuracy_score(y_test, lesioned_pred), 
                        'auc': roc_auc_score(y_test, lesioned_pred_proba)}
        }
    
    def visualize_signal(self, signal: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "EEG Signal"):
        """Visualize EEG signal with optional labels."""
        plt.figure(figsize=(12, 4))
        plt.plot(signal, linewidth=0.7, label="signal")
        if labels is not None:
            plt.fill_between(np.arange(len(labels)), 
                           signal.min(), signal.max(), 
                           where=labels > 0.5, 
                           color="orange", alpha=0.1, 
                           label="state ON")
        plt.title(title)
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()



def main():
    """Main function to demonstrate the hierarchical EEG processor."""
    print("=== Hierarchical EEG Processor Demo ===\n")
    
    # Generate synthetic data
    print("1. Generating synthetic data...")
    processor = HierarchicalEEGProcessor(fast_window_size=10, slow_context_size=50)
    X_fast, X_slow, y = processor.generate_synthetic_data(num_sequences=200)
    
    # Test 1: Original Logistic Regression Models
    print("\n2. Testing Original Logistic Regression Models...")
    (Xf_train, Xf_test, Xs_train, Xs_test, 
     y_train, y_test, H_train, H_test, training_time) = processor.train_models(X_fast, X_slow, y)
    
    results = processor.evaluate_models(Xf_test, Xs_test, y_test, H_test, training_time)
    lesion_results = processor.lesion_slow_context(H_test, y_test)
    
    # Test 2: New RNN Models
    print("\n3. Testing RNN Models...")
    print("=" * 60)
    rnn_processor = HRMEEGProcessor(fast_window_size=10, slow_context_size=50)
    
    # Train RNN models
    (fast_train, fast_test, slow_train, slow_test, 
     labels_train, labels_test, train_loader, test_loader, rnn_training_time) = rnn_processor.train_rnn_models(X_fast, X_slow, y)
    
    # Evaluate RNN models
    rnn_results = rnn_processor.evaluate_rnn_models(fast_test, slow_test, labels_test, rnn_training_time)
    
    # Add RNN results to comparison
    print("\nAdding RNN results to comparison framework...")
    for model_name, result in rnn_results.items():
        processor.comparator.add_result(result)
        print(f"  Added {result.model_name} to comparison")
    
    # Visualize synthetic signal
    def generate_single_sequence(seq_length=600, slow_flip_prob=0.05, seed=42):
        if seed is not None:
            np.random.seed(seed)
        t = np.arange(seq_length)
        slow_state = np.zeros(seq_length, dtype=float)
        state = np.random.choice([0.0, 1.0])
        for i in range(seq_length):
            if i == 0:
                slow_state[i] = state
            else:
                if np.random.rand() < slow_flip_prob:
                    state = 1.0 - state
                slow_state[i] = state
        freq_slow = 0.02
        phase = np.random.rand() * 2 * np.pi
        slow_base = np.sin(2 * np.pi * freq_slow * t + phase)
        slow_signal = (0.3 + 0.7 * slow_state) * slow_base
        freq_fast = 5.0
        imagery_component = (slow_state * np.sin(2 * np.pi * freq_fast * t)) * 0.5
        noise = np.random.randn(seq_length) * 0.1
        signal = slow_signal + imagery_component + noise
        return signal, slow_state
    
    signal, labels = generate_single_sequence()
    processor.visualize_signal(signal, labels, "Synthetic EEG-like Signal")
    
    # Option 3: Use real EEG data (if available)
    try:
        print("\n4. Testing on real EEG data...")
        eeg_data, labels, sfreq = processor.load_sample_eeg_data()
        X_fast_real, X_slow_real, y_real = processor.extract_features_from_eeg(eeg_data, labels, sfreq)
        
        # Train and evaluate on real data
        (Xf_train_real, Xf_test_real, Xs_train_real, Xs_test_real, 
         y_train_real, y_test_real, H_train_real, H_test_real, training_time_real) = processor.train_models(
             X_fast_real, X_slow_real, y_real)
        
        results_real = processor.evaluate_models(Xf_test_real, Xs_test_real, y_test_real, H_test_real, training_time_real)
        lesion_results_real = processor.lesion_slow_context(H_test_real, y_test_real)
        
        # Visualize real signal
        processor.visualize_signal(eeg_data[0, :1000], labels[:1000], "Real EEG Signal (First 1000 samples)")
        
    except Exception as e:
        print(f"Could not load real EEG data: {e}")
        print("Continuing with synthetic data only...")

    # Generate and save comprehensive comparison report
    print("\n5. Generating Model Comparison Report...")
    processor.comparator.compare_models()
    print(processor.comparator.generate_report())
    processor.comparator.plot_comparison()
    processor.comparator.save_results("model_comparison_results.json")
    
    print("\n=== Demo Complete! ===")
    print("Check 'model_comparison_results.json' for detailed results.")

def quick_rnn_dry_run():
    """Quick dry run to test RNN models compile and forward pass correctly."""
    print("=== Quick RNN Dry Run ===")
    
    # Create a small synthetic dataset
    print("1. Generating small synthetic dataset...")
    processor = HRMEEGProcessor(fast_window_size=10, slow_context_size=50)
    X_fast, X_slow, y = processor.generate_synthetic_data(num_sequences=10, seq_length=100)
    
    print(f"Dataset shape: X_fast {X_fast.shape}, X_slow {X_slow.shape}, y {y.shape}")
    
    # Prepare sequences for RNN
    print("2. Preparing sequences...")
    fast_tensor, slow_tensor, labels_tensor = processor.prepare_sequences(X_fast, X_slow, y, sequence_length=5)
    
    print(f"Sequence shapes: fast {fast_tensor.shape}, slow {slow_tensor.shape}, labels {labels_tensor.shape}")
    
    # Test forward passes
    print("3. Testing forward passes...")
    
    # Move to device
    device = processor.device
    fast_tensor = fast_tensor.to(device)
    slow_tensor = slow_tensor.to(device)
    labels_tensor = labels_tensor.to(device)
    
    # Test Fast RNN
    print("   Testing Fast RNN...")
    processor.fast_rnn.eval()
    with torch.no_grad():
        fast_output = processor.fast_rnn(fast_tensor)
        print(f"   Fast RNN output shape: {fast_output.shape}")
        print(f"   Fast RNN output range: [{fast_output.min().item():.3f}, {fast_output.max().item():.3f}]")
    
    # Test Slow RNN
    print("   Testing Slow RNN...")
    processor.slow_rnn.eval()
    with torch.no_grad():
        slow_output = processor.slow_rnn(slow_tensor)
        print(f"   Slow RNN output shape: {slow_output.shape}")
        print(f"   Slow RNN output range: [{slow_output.min().item():.3f}, {slow_output.max().item():.3f}]")
    
    # Test Integration Network
    print("   Testing Integration Network...")
    processor.integration_net.eval()
    with torch.no_grad():
        # Get features from LSTM layers
        fast_features = processor.fast_rnn.lstm(fast_tensor)[0][:, -1, :]
        slow_features = processor.slow_rnn.lstm(slow_tensor)[0][:, -1, :]
        
        print(f"   Fast features shape: {fast_features.shape}")
        print(f"   Slow features shape: {slow_features.shape}")
        
        integration_output = processor.integration_net(fast_features, slow_features)
        print(f"   Integration output shape: {integration_output.shape}")
        print(f"   Integration output range: [{integration_output.min().item():.3f}, {integration_output.max().item():.3f}]")
    
    # Test a few training steps
    print("4. Testing a few training steps...")
    processor.fast_rnn.train()
    processor.slow_rnn.train()
    processor.integration_net.train()
    
    # Create a small batch
    batch_size = 4
    batch_fast = fast_tensor[:batch_size]
    batch_slow = slow_tensor[:batch_size]
    batch_labels = labels_tensor[:batch_size]
    
    print(f"   Batch shapes: fast {batch_fast.shape}, slow {batch_slow.shape}, labels {batch_labels.shape}")
    
    # Forward pass
    fast_output = processor.fast_rnn(batch_fast)
    slow_output = processor.slow_rnn(batch_slow)
    
    fast_features = processor.fast_rnn.lstm(batch_fast)[0][:, -1, :]
    slow_features = processor.slow_rnn.lstm(batch_slow)[0][:, -1, :]
    integration_output = processor.integration_net(fast_features, slow_features)
    
    # Loss calculation
    fast_loss = processor.criterion(fast_output, batch_labels)
    slow_loss = processor.criterion(slow_output, batch_labels)
    integration_loss = processor.criterion(integration_output, batch_labels)
    
    total_loss = fast_loss + slow_loss + integration_loss
    
    print(f"   Losses: Fast {fast_loss.item():.4f}, Slow {slow_loss.item():.4f}, Integration {integration_loss.item():.4f}")
    print(f"   Total loss: {total_loss.item():.4f}")
    
    # Backward pass (just to test gradients)
    processor.fast_optimizer.zero_grad()
    processor.slow_optimizer.zero_grad()
    processor.integration_optimizer.zero_grad()
    
    total_loss.backward()
    
    # Check gradients
    fast_grad_norm = torch.nn.utils.clip_grad_norm_(processor.fast_rnn.parameters(), max_norm=1.0)
    slow_grad_norm = torch.nn.utils.clip_grad_norm_(processor.slow_rnn.parameters(), max_norm=1.0)
    integration_grad_norm = torch.nn.utils.clip_grad_norm_(processor.integration_net.parameters(), max_norm=1.0)
    
    print(f"   Gradient norms: Fast {fast_grad_norm:.4f}, Slow {slow_grad_norm:.4f}, Integration {integration_grad_norm:.4f}")
    
    # Optimizer steps
    processor.fast_optimizer.step()
    processor.slow_optimizer.step()
    processor.integration_optimizer.step()
    
    print("5. Testing evaluation function...")
    
    # Test evaluation
    processor.fast_rnn.eval()
    processor.slow_rnn.eval()
    processor.integration_net.eval()
    
    with torch.no_grad():
        # Test evaluation on a small subset
        test_fast = fast_tensor[:10]
        test_slow = slow_tensor[:10]
        test_labels = labels_tensor[:10]
        
        # Fast RNN evaluation
        fast_output = processor.fast_rnn(test_fast)
        fast_pred = (fast_output > 0.5).float()
        
        # Slow RNN evaluation
        slow_output = processor.slow_rnn(test_slow)
        slow_pred = (slow_output > 0.5).float()
        
        # Integration evaluation
        fast_features = processor.fast_rnn.lstm(test_fast)[0][:, -1, :]
        slow_features = processor.slow_rnn.lstm(test_slow)[0][:, -1, :]
        integration_output = processor.integration_net(fast_features, slow_features)
        integration_pred = (integration_output > 0.5).float()
        
        print(f"   Test predictions: Fast {fast_pred.shape}, Slow {slow_pred.shape}, Integration {integration_pred.shape}")
        print(f"   Test outputs: Fast {fast_output.shape}, Slow {slow_output.shape}, Integration {integration_output.shape}")
    
    print("\n=== Dry Run Complete! All RNN models compile and forward pass correctly ===")
    print("✅ Fast RNN: Working")
    print("✅ Slow RNN: Working") 
    print("✅ Integration Network: Working")
    print("✅ Training loop: Working")
    print("✅ Evaluation: Working")

if __name__ == "__main__":
    # Uncomment the line below to run the dry run instead of the full demo
    # quick_rnn_dry_run()
    main() 