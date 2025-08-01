import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. generate synthetic eeg-like signals for three classes (alpha, beta, gamma)
np.random.seed(0)
samples_per_class = 1000
sampling_rate = 128
signal_length = 256  # number of samples (~2 sec)
class_freqs = {0: 10, 1: 20, 2: 40}
signals, labels = [], []

for label, freq in class_freqs.items():
    for _ in range(samples_per_class):
        t = np.linspace(0, signal_length / sampling_rate, signal_length, endpoint=False)
        amplitude = np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, 2 * np.pi)
        signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
        noise = 0.3 * np.random.randn(signal_length)
        signals.append(signal + noise)
        labels.append(label)

signals = np.array(signals)
labels = np.array(labels)

# 2. split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(signals, labels, test_size=0.2, random_state=42)

# 3. normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. build and train mlp
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                    max_iter=200, random_state=42)
mlp.fit(X_train, y_train)

# 5. evaluate
train_acc = accuracy_score(y_train, mlp.predict(X_train))
test_acc = accuracy_score(y_test, mlp.predict(X_test))
print(f"training accuracy: {train_acc:.2f}, test accuracy: {test_acc:.2f}")
print(classification_report(y_test, mlp.predict(X_test)))
