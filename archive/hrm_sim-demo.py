import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import random

# -------------------------
# 1. Synthetic dataset generation (fast + slow structure)
# -------------------------

def generate_sequence(seq_length=1000, slow_flip_prob=0.05, seed=None):
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
    # amplitude modulated by slow_state: when imagery is "on" amplitude increases
    slow_signal = (0.3 + 0.7 * slow_state) * slow_base  # amplitude larger for imagery

    # 3. imagery-specific high-frequency content only when slow_state==1
    freq_fast = 5.0  # higher frequency component representing “imagery detail”
    imagery_component = (slow_state * np.sin(2 * np.pi * freq_fast * t)) * 0.5

    # 4. additive broadband noise (simulating EEG noise)
    noise = np.random.randn(seq_length) * 0.1

    # final synthetic signal
    signal = slow_signal + imagery_component + noise

    return signal, slow_state

def build_dataset(num_sequences=100, seq_length=600, slow_context_size=50, fast_window_size=10):
    X_fast = []
    X_slow = []
    y = []
    for seq_idx in range(num_sequences):
        signal, slow_state = generate_sequence(seq_length=seq_length, seed=seq_idx)
        for t in range(max(slow_context_size, fast_window_size), seq_length):
            fast_win = signal[t - fast_window_size : t]
            slow_ctx = signal[t - slow_context_size : t]
            label = slow_state[t]  # whether imagery is active at time t

            # Feature engineering (simple): 
            # fast features: mean, std, last delta (local dynamics)
            fast_mean = fast_win.mean()
            fast_std = fast_win.std()
            fast_diff = fast_win[-1] - fast_win[-2]

            # slow features: mean and std over longer context (embodied prior summary)
            slow_mean = slow_ctx.mean()
            slow_std = slow_ctx.std()

            X_fast.append([fast_mean, fast_std, fast_diff])
            X_slow.append([slow_mean, slow_std])
            y.append(label)
    X_fast = np.array(X_fast)
    X_slow = np.array(X_slow)
    y = np.array(y)
    return X_fast, X_slow, y

# Build dataset
X_fast, X_slow, y = build_dataset(num_sequences=200)  # ~200* (600-50)= ~110k examples

# Split (simple random split)
perm = np.random.permutation(len(y))
split = int(0.8 * len(y))
train_idx, test_idx = perm[:split], perm[split:]
Xf_train, Xf_test = X_fast[train_idx], X_fast[test_idx]
Xs_train, Xs_test = X_slow[train_idx], X_slow[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# -------------------------
# 2. Tier 1: Fast-only model (baseline)
# -------------------------
fast_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
fast_model.fit(Xf_train, y_train)
fast_pred_proba = fast_model.predict_proba(Xf_test)[:, 1]
fast_pred = (fast_pred_proba > 0.5).astype(float)
print("=== Fast-only model ===")
print("Accuracy:", accuracy_score(y_test, fast_pred))
print("AUC:", roc_auc_score(y_test, fast_pred_proba))
print(classification_report(y_test, fast_pred, digits=3))

# -------------------------
# 3. Hierarchical integration (Tier 1 extended): adjust fast logit with slow context as a top-down prior
#    Simple implementation: take log-odds from fast model and learn a linear refinement with slow features
# -------------------------
# Compute fast log-odds (logit) on train set
def logit_from_proba(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

fast_logit_train = logit_from_proba(fast_model.predict_proba(Xf_train)[:, 1])
fast_logit_test = logit_from_proba(fast_model.predict_proba(Xf_test)[:, 1])

# Build hierarchical feature: [fast_logit, slow_mean, slow_std]
H_train = np.column_stack([fast_logit_train, Xs_train])  # shape (n, 3)
H_test = np.column_stack([fast_logit_test, Xs_test])

hier_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
hier_model.fit(H_train, y_train)
hier_pred_proba = hier_model.predict_proba(H_test)[:, 1]
hier_pred = (hier_pred_proba > 0.5).astype(float)

print("=== Hierarchical (fast + slow) model ===")
print("Accuracy:", accuracy_score(y_test, hier_pred))
print("AUC:", roc_auc_score(y_test, hier_pred_proba))
print(classification_report(y_test, hier_pred, digits=3))

# -------------------------
# 4. Tier 2 / lesion simulations
#    a) remove slow context (zero it out) -> test hierarchical model's degradation
#    b) degrade imagery signal in the generative process (simulate aphantasia) and re-evaluate
# -------------------------

# a) lesion slow context: zero slow features in hierarchical input (simulate removing top-down prior)
H_test_lesioned = H_test.copy()
H_test_lesioned[:, 1:] = 0.0  # zero out slow_mean and slow_std
lesioned_pred_proba = hier_model.predict_proba(H_test_lesioned)[:, 1]
lesioned_pred = (lesioned_pred_proba > 0.5).astype(float)

print("=== Hierarchical model with slow-context lesioned ===")
print("Accuracy:", accuracy_score(y_test, lesioned_pred))
print("AUC:", roc_auc_score(y_test, lesioned_pred_proba))
print(classification_report(y_test, lesioned_pred, digits=3))

# b) simulate aphantasia: regenerate a test sequence with suppressed imagery component
def generate_aphantasia_sequence(seq_length=600, slow_flip_prob=0.05, seed=None):
    # same as before but kill imagery-specific high-frequency content
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
    # imagery component zeroed out (aphantasia-like)
    imagery_component = np.zeros_like(slow_state)
    noise = np.random.randn(seq_length) * 0.1
    signal = slow_signal + imagery_component + noise
    return signal, slow_state

# build a small test dataset with aphantasia-like signal
def build_aphantasia_dataset(num_sequences=50, seq_length=600, slow_context_size=50, fast_window_size=10):
    Xf, Xs, y = [], [], []
    for seq_idx in range(num_sequences):
        signal, slow_state = generate_aphantasia_sequence(seq_length=seq_length, seed=1000 + seq_idx)
        for t in range(max(slow_context_size, fast_window_size), seq_length):
            fast_win = signal[t - fast_window_size : t]
            slow_ctx = signal[t - slow_context_size : t]
            label = slow_state[t]
            fast_mean = fast_win.mean()
            fast_std = fast_win.std()
            fast_diff = fast_win[-1] - fast_win[-2]
            slow_mean = slow_ctx.mean()
            slow_std = slow_ctx.std()
            Xf.append([fast_mean, fast_std, fast_diff])
            Xs.append([slow_mean, slow_std])
            y.append(label)
    return np.array(Xf), np.array(Xs), np.array(y)

Xf_aph, Xs_aph, y_aph = build_aphantasia_dataset()
# hierarchical input for aphantasia test
logit_fast_aph = logit_from_proba(fast_model.predict_proba(Xf_aph)[:, 1])
H_aph = np.column_stack([logit_fast_aph, Xs_aph])
hier_aph_proba = hier_model.predict_proba(H_aph)[:, 1]
hier_aph_pred = (hier_aph_proba > 0.5).astype(float)
print("=== Hierarchical model on aphantasia-like degraded imagery ===")
print("Accuracy:", accuracy_score(y_aph, hier_aph_pred))
print("AUC:", roc_auc_score(y_aph, hier_aph_proba))
print(classification_report(y_aph, hier_aph_pred, digits=3))

# -------------------------
# 5. quick visualization
# -------------------------
plt.figure(figsize=(10, 4))
plt.title("Example synthetic signal with slow imagery state")
signal, slow_state = generate_sequence(seq_length=600, seed=42)
plt.plot(signal, label="synthetic EEG-like signal", linewidth=0.7)
plt.fill_between(np.arange(len(slow_state)), -2, 2, where=slow_state > 0.5, color="orange", alpha=0.1, label="imagery ON")
plt.legend()
plt.xlabel("time")
plt.ylabel("signal")
plt.tight_layout()
plt.show()
