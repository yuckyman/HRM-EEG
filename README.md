# Hierarchical EEG Processing with Embodied Priors

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

A research project comparing classical machine learning and deep learning approaches for hierarchical EEG signal processing. This project implements a novel framework for modeling fast and slow temporal dynamics in neural signals.

## Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd python

# Install dependencies (using uv for fast dependency resolution)
uv sync

# Run the main experiment
python eeg_hierarchical_processor.py
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- scikit-learn
- MNE-Python
- Other dependencies are listed in `pyproject.toml`

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [ModelResults](#modelresults-dataclass)
  - [ModelComparator](#modelcomparator)
  - [HierarchicalEEGProcessor](#hierarchicleeegprocessor)
  - [PyTorch RNN Models](#pytorch-rnn-models-fastrnn-slowrnn-integrationnet)
  - [HRMEEGProcessor](#hrmeegprocessor)
- [Workflow](#workflow-the-main-function)

## Overview

The script's main goal is to build and compare different machine learning models for a binary classification task on EEG-like data. It specifically compares two main approaches:

## Project Structure

```
python/
├── eeg_hierarchical_processor.py    # Main experiment script
├── eeg_sim.py                       # EEG simulation utilities
├── hrm_sim-demo.py                  # Demo script for HRM models
├── hierarchical_eeg_progress.md     # Development notes
├── public-eeg-datasets.md           # Dataset documentation
├── model_comparison_results.json    # Experiment results
├── pyproject.toml                   # Project configuration
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

1. A **classical machine learning approach** using Logistic Regression in a two-step hierarchical process.
2. A **deep learning approach** using Recurrent Neural Networks (RNNs) that explicitly model fast and slow dynamics with separate networks before integrating their outputs.

To make the comparison fair and comprehensive, the script includes a powerful `ModelComparator` framework that evaluates models on accuracy, efficiency (time, memory), and complexity (number of parameters).

---

## Architecture

### 1\. Imports and Initial Setup

The script begins by importing necessary libraries.

- **`os`, `datetime`, `time`, `json`, `psutil`**: Standard libraries for interacting with the operating system, handling time, saving data, and monitoring system resources (memory usage).
- **`numpy`, `matplotlib.pyplot`, `scipy.stats`**: The fundamental stack for scientific computing in Python (numerical operations, plotting, and statistics).
- **`sklearn`**: Scikit-learn is used for the classical machine learning models (`LogisticRegression`), data preprocessing (`StandardScaler`, `make_pipeline`), and evaluation metrics (`accuracy_score`, `roc_auc_score`, etc.).
- **`mne`**: A powerful open-source Python package for exploring, visualizing, and analyzing human neurophysiological data (EEG, MEG, etc.). It's used here to load a sample EEG dataset.
	- `os.environ['MNE_DATA'] = ...`: This line is important. It tells the `mne` library to download its sample datasets into a local folder named `mne_data` within the current working directory, rather than a system-wide location.
- **`einops`**: A library for flexible and powerful tensor operations. Its name stands for "Einstein Operations." It makes complex reshaping, rearranging, and reducing of multi-dimensional arrays (like EEG data) much more readable and less error-prone than using standard NumPy/PyTorch functions.
- **`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`**: PyTorch is the deep learning framework used to build and train the RNN models.

---

### 2\. Core Components (The Classes)

The script is organized into several classes, each with a distinct responsibility.

#### ModelResults (Dataclass)

This is a simple data container created using `@dataclass`. Its job is to hold all the evaluation metrics for a single model run in a structured way. This includes:

- Performance metrics: accuracy, AUC, precision, recall, F1-score.
- Efficiency metrics: training time, inference time, memory usage.
- Model complexity: number of parameters.
- Detailed results: confusion matrix and classification report.

#### ModelComparator

This is a powerful utility class for managing the entire comparison process.

- **`__init__`**: It initializes an empty list to store `ModelResults` objects.
- **`add_result()`**: Adds a completed `ModelResults` object to its list.
- **`compare_models()`**: Calculates relative performance and efficiency differences between pairs of models (e.g., "Model B has a +0.05 accuracy improvement over Model A and took 1.5x longer to train").
- **`generate_report()`**: Creates a clean, formatted text report summarizing the performance of all models in a table and showing the pairwise comparisons.
- **`plot_comparison()`**: Generates a set of plots to visually compare the models on performance, efficiency, and complexity. This is excellent for presentations and quick insights.
- **`save_results()`**: Saves all the collected results and comparisons to a JSON file for later analysis.

#### HierarchicalEEGProcessor

This class implements the **first modeling approach** using Scikit-learn's Logistic Regression. It serves as the baseline model.

- **`generate_synthetic_data()`**: This method is crucial. It creates artificial data that mimics the key properties of hierarchical EEG signals.
	- A **slow-changing state** (e.g., the overall context or task).
	- A **slow signal** (low-frequency wave) whose properties depend on this state.
	- A **fast signal** (high-frequency wave) that *only appears* when the slow state is "ON".
	- This setup directly tests a model's ability to use the slow context to correctly interpret the fast signals.
- **`_extract_features_vectorized()`**: This function uses `einops` to efficiently create sliding windows over the data and extract features.
	- **Fast Features**: Calculated over a short window (e.g., 10ms). Includes mean, standard deviation, and the change from the previous time step.
	- **Slow Features**: Calculated over a long window (e.g., 50ms). Includes mean and standard deviation.
- **`train_models()`**: This is the core of the hierarchical logic for this approach:
	1. **Train a "Fast-only" Model**: A `LogisticRegression` model is trained using *only the fast features*.
	2. **Get Log-Odds**: The output probabilities from the fast model are converted to log-odds. This represents the fast model's "belief" at each time point.
	3. **Create Hierarchical Features**: A new feature set is created by combining the fast model's log-odds with the original slow features. The feature vector becomes `[fast_model_belief, slow_feature_1, slow_feature_2, ...]`.
	4. **Train a "Hierarchical" Model**: A second `LogisticRegression` model is trained on this new, combined feature set. This second model learns to use the slow, contextual features to *correct or gate* the predictions of the first model.
- **`lesion_slow_context()`**: A "lesion study" is a technique from neuroscience. Here, it simulates brain damage by taking the fully trained hierarchical model and feeding it test data where the slow features have been set to zero. This shows how much the model's performance depends on that contextual information.

#### PyTorch RNN Models (FastRNN, SlowRNN, IntegrationNet)

These three `nn.Module` classes define the architecture for the **second, deep learning approach**.

- **`FastRNN`**: An LSTM network designed to process sequences of the "fast features". It learns the short-term temporal dynamics.
- **`SlowRNN`**: A separate, smaller LSTM network that processes sequences of the "slow features". It learns the long-term contextual dynamics.
- **`IntegrationNet`**: A simple feed-forward network (MLP). It takes the final hidden state (the "summary") from both the `FastRNN` and the `SlowRNN`, concatenates them, and makes the final prediction. This is where the fast and slow information streams are integrated.

#### HRMEEGProcessor

This class inherits from `HierarchicalEEGProcessor` but implements the deep learning workflow using the RNNs.

- **`__init__`**: It initializes the three PyTorch models, their optimizers (`Adam`), and the loss function (`BCELoss`, for binary classification).
- **`prepare_sequences()`**: RNNs require input data in the shape of `(batch, sequence_length, features)`. This method converts the flat feature arrays into overlapping sequences suitable for training LSTMs.
- **`train_rnn_models()`**: Implements a standard PyTorch training loop. In each step, it:
	1. Passes data through the `FastRNN` and `SlowRNN`.
	2. Passes their outputs to the `IntegrationNet`.
	3. Calculates a loss for each component and combines them.
	4. Performs backpropagation to update the weights of all three networks simultaneously.
- **`evaluate_rnn_models()`**: After training, this method evaluates the performance of the `FastRNN` alone, the `SlowRNN` alone, and the combined `IntegrationNet`. It uses the `_create_rnn_results` helper to package the metrics into the `ModelResults` structure for comparison.

---

### 3\. Workflow (The main function)

The `if __name__ == "__main__":` block executes the `main()` function, which orchestrates the entire experiment.

1. **Generate Data**: It starts by creating the synthetic dataset using `HierarchicalEEGProcessor`.
2. **Run Experiment 1 (Logistic Regression)**:
	- It trains the fast-only and hierarchical logistic regression models.
	- It evaluates both models and adds their `ModelResults` to the `ModelComparator`.
	- It runs the lesion study on the hierarchical model and also adds that result.
3. **Run Experiment 2 (RNNs)**:
	- It initializes the `HRMEEGProcessor`.
	- It trains the `FastRNN`, `SlowRNN`, and `IntegrationNet` models.
	- It evaluates all three RNN models (`fast`, `slow`, and `hierarchical/integrated`) and adds their results to the same `ModelComparator`.
4. **Visualize and Report**:
	- It visualizes a sample of the synthetic signal to show the underlying structure.
	- **(Optional)** It has a `try...except` block to attempt downloading a real EEG dataset from `mne`. If successful, it runs the entire comparison on this real data as well.
	- Finally, it calls the `ModelComparator` 's methods to:
		- Perform the final statistical comparisons.
		- Print the comprehensive report to the console.
		- Generate the comparison plots.
		- Save all results to `model_comparison_results.json`.

In summary, this script is a well-designed research tool. It sets up a clear problem (classifying a signal with hierarchical structure), implements two different solutions (one classical, one deep learning), and provides a robust framework for rigorously comparing them on multiple criteria.

## Contributing

This is a research project. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hierarchical_eeg_processor,
  title={Hierarchical EEG Processing with Embodied Priors},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/your-repo-name}
}
```