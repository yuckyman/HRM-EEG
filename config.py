"""
Configuration file for EEG Hierarchical Processor
Centralizes all paths and settings for easy configuration
"""

import os
from pathlib import Path
from datetime import datetime

# Base project directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
MODEL_COMPARISONS_DIR = RESULTS_DIR / "model_comparisons"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  RESULTS_DIR, EXPERIMENTS_DIR, MODEL_COMPARISONS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# BCI Competition data paths
BCI_DATA_DIR = RAW_DATA_DIR / "bci_competition_data"
MNE_DATA_DIR = RAW_DATA_DIR / "mne_data"

# Output file naming
def get_timestamp():
    """Get current timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_experiment_results_path(experiment_name: str = "bci_experiment") -> Path:
    """Get path for experiment results file"""
    timestamp = get_timestamp()
    return EXPERIMENTS_DIR / f"{experiment_name}_results_{timestamp}.json"

def get_model_comparison_path(comparison_name: str = "bci_model_comparison") -> Path:
    """Get path for model comparison results file"""
    timestamp = get_timestamp()
    return MODEL_COMPARISONS_DIR / f"{comparison_name}_{timestamp}.json"

def get_log_path(log_name: str = "eeg_processor") -> Path:
    """Get path for log file"""
    timestamp = get_timestamp()
    return LOGS_DIR / f"{log_name}_{timestamp}.log"

# Model configuration
MODEL_CONFIG = {
    "fast_rnn": {
        "input_size": 3,
        "hidden_size": 32,
        "num_layers": 2,
        "num_classes": 4,
        "dropout": 0.2
    },
    "slow_rnn": {
        "input_size": 2,
        "hidden_size": 16,
        "num_layers": 1,
        "num_classes": 4,
        "dropout": 0.2
    },
    "integration": {
        "fast_size": 32,
        "slow_size": 16,
        "hidden_size": 32,
        "num_classes": 4,
        "dropout": 0.2
    }
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 5,
    "test_size": 0.2,
    "random_state": 42
}

# BCI data configuration
BCI_CONFIG = {
    "sampling_rate": 250,
    "trial_duration": 4.0,  # seconds
    "num_classes": 4,
    "class_names": ["left_hand", "right_hand", "feet", "tongue"]
}

# Feature extraction configuration
FEATURE_CONFIG = {
    "fast_window_ms": 10,
    "slow_window_ms": 50,
    "overlap": 0.5
} 