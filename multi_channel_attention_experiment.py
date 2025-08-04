#!/usr/bin/env python3
"""
Multi-Channel Spatial Attention Experiment
Test the multi-channel attention processor on BCI Competition IV data
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne
import json
from datetime import datetime
from typing import Dict, List, Optional

from multi_channel_attention_processor import MultiChannelAttentionProcessor
from run_bci_experiment import load_bci_data
from config import (
    BCI_DATA_DIR, get_experiment_results_path, get_timestamp
)

def run_multi_channel_attention_experiment(subject_id: str = "A01E", 
                                         num_epochs: int = 5,
                                         test_size: float = 0.2) -> Dict:
    """
    Run multi-channel attention experiment on BCI data.
    
    Args:
        subject_id: Subject identifier (e.g., "A01E")
        num_epochs: Number of training epochs
        test_size: Fraction of data for testing
        
    Returns:
        results: Dictionary with experiment results
    """
    print(f"🚀 Running Multi-Channel Attention Experiment")
    print(f"  Subject: {subject_id}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Test size: {test_size}")
    
    # Load BCI data
    data_file = BCI_DATA_DIR / "dataset2a" / f"{subject_id}.gdf"
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return None
    
    bci_data = load_bci_data(str(data_file))
    if bci_data is None:
        print(f"❌ Failed to load BCI data for {subject_id}")
        return None
    
    trial_data = bci_data['data']  # (trials, channels, timepoints)
    labels = bci_data['labels']
    
    print(f"  Loaded {len(trial_data)} trials with {trial_data.shape[1]} channels")
    print(f"  Trial shape: {trial_data.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")
    
    # Initialize multi-channel attention processor
    processor = MultiChannelAttentionProcessor(
        num_channels=25,  # BCI data has 25 channels
        fast_window_size=10,  # 10ms windows
        slow_context_size=50,  # 50ms windows
        num_classes=4,  # 4 motor imagery classes
        num_heads=4,
        dropout=0.2,
        learning_rate=0.001
    )
    
    # Train models
    print(f"\n🧠 Training Multi-Channel Attention Models...")
    training_results = processor.train_multi_channel_models(
        trial_data=trial_data,
        labels=labels,
        test_size=test_size,
        num_epochs=num_epochs
    )
    
    # Analyze channel attention
    print(f"\n🔍 Analyzing Channel Attention Patterns...")
    channel_analysis = processor.analyze_channel_attention()
    
    # Visualize attention patterns
    print(f"\n📊 Visualizing Attention Patterns...")
    processor.visualize_multi_channel_attention(subject_id)
    
    # Compile results
    results = {
        'subject_id': subject_id,
        'experiment_timestamp': datetime.now().isoformat(),
        'data_info': {
            'num_trials': len(trial_data),
            'num_channels': trial_data.shape[1],
            'trial_length': trial_data.shape[2],
            'label_distribution': np.bincount(labels).tolist()
        },
        'training_results': training_results,
        'channel_analysis': channel_analysis,
        'model_config': {
            'num_channels': 25,
            'fast_window_size': 10,
            'slow_context_size': 50,
            'num_classes': 4,
            'num_heads': 4,
            'dropout': 0.2,
            'learning_rate': 0.001
        }
    }
    
    # Save results
    processor.save_multi_channel_results(results, subject_id)
    
    print(f"\n✅ Multi-Channel Attention Experiment Complete!")
    print(f"  Final Integration Accuracy: {training_results['final_results']['integration_accuracy']:.3f}")
    print(f"  Fast RNN Accuracy: {training_results['final_results']['fast_accuracy']:.3f}")
    print(f"  Slow RNN Accuracy: {training_results['final_results']['slow_accuracy']:.3f}")
    
    return results

def run_multi_subject_experiment(subjects: List[str] = None, 
                                num_epochs: int = 3) -> Dict:
    """
    Run multi-channel attention experiment on multiple subjects.
    
    Args:
        subjects: List of subject IDs (default: A01E, A02E, A03E)
        num_epochs: Number of training epochs per subject
        
    Returns:
        results: Dictionary with multi-subject results
    """
    if subjects is None:
        subjects = ["A01E", "A02E", "A03E"]
    
    print(f"🧠 Running Multi-Subject Multi-Channel Attention Experiment")
    print(f"  Subjects: {subjects}")
    print(f"  Epochs per subject: {num_epochs}")
    
    all_results = {}
    
    for subject_id in subjects:
        print(f"\n{'='*60}")
        print(f"Processing Subject: {subject_id}")
        print(f"{'='*60}")
        
        try:
            subject_results = run_multi_channel_attention_experiment(
                subject_id=subject_id,
                num_epochs=num_epochs
            )
            
            if subject_results is not None:
                all_results[subject_id] = subject_results
            else:
                print(f"❌ Failed to process subject {subject_id}")
                
        except Exception as e:
            print(f"❌ Error processing subject {subject_id}: {e}")
            continue
    
    # Compile multi-subject summary
    if all_results:
        accuracies = []
        for subject_id, results in all_results.items():
            acc = results['training_results']['final_results']['integration_accuracy']
            accuracies.append(acc)
            print(f"  {subject_id}: {acc:.3f}")
        
        print(f"\n📊 Multi-Subject Summary:")
        print(f"  Mean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"  Best Subject: {subjects[np.argmax(accuracies)]} ({max(accuracies):.3f})")
        print(f"  Worst Subject: {subjects[np.argmin(accuracies)]} ({min(accuracies):.3f})")
        
        # Save multi-subject results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = get_experiment_results_path() / f"multi_subject_multi_channel_attention_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for subject_id, results in all_results.items():
            serializable_results[subject_id] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[subject_id][key] = value.tolist()
                else:
                    serializable_results[subject_id][key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"  Multi-subject results saved to {results_path}")
    
    return all_results

def quick_multi_channel_test():
    """Quick test of multi-channel attention on a single subject."""
    print("🧪 Quick Multi-Channel Attention Test")
    
    results = run_multi_channel_attention_experiment(
        subject_id="A01E",
        num_epochs=2,  # Quick test with few epochs
        test_size=0.3
    )
    
    if results:
        print(f"\n✅ Quick Test Complete!")
        print(f"  Integration Accuracy: {results['training_results']['final_results']['integration_accuracy']:.3f}")
        
        # Show channel attention analysis
        channel_analysis = results['channel_analysis']
        print(f"\n🔍 Channel Attention Analysis:")
        print(f"  Fast RNN Top Channels: {channel_analysis['fast_top_channels']}")
        print(f"  Slow RNN Top Channels: {channel_analysis['slow_top_channels']}")
        print(f"  Fast RNN Region Attention: {channel_analysis['fast_region_attention']}")
        print(f"  Slow RNN Region Attention: {channel_analysis['slow_region_attention']}")
    
    return results

if __name__ == "__main__":
    # Run quick test first
    print("🧪 Starting Multi-Channel Attention Experiment...")
    
    # Uncomment to run full experiment
    # results = run_multi_channel_attention_experiment("A01E", num_epochs=5)
    
    # Run quick test
    results = quick_multi_channel_test()
    
    print("\n🎉 Multi-Channel Attention Experiment Complete!") 