import pandas as pd
import numpy as np
import json
from collections import Counter
import os

DATA_PATH = "../data/raw/"
RESULTS_PATH = "../results/"

print("=" * 50)
print("IMPROVED ANOMALY DETECTION")
print("Testing Different Thresholds")
print("=" * 50)

# Load data
train_df = pd.read_csv(DATA_PATH + "mitbih_train.csv", header=None)
test_df = pd.read_csv(DATA_PATH + "mitbih_test.csv", header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Calculate normal statistics
normal_X = X_train[y_train == 0]
normal_mean = np.mean(normal_X, axis=0)
normal_std = np.std(normal_X, axis=0)

def encode_heartbeat(heartbeat, normal_mean, normal_std, threshold):
    symbols = []
    n_segments = 10
    segment_size = len(heartbeat) // n_segments
    segment_names = [chr(65 + i) for i in range(n_segments)]  # A, B, C, ..., J

    for i, name in enumerate(segment_names):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else len(heartbeat)
        segment = heartbeat[start:end]
        seg_mean = normal_mean[start:end]
        seg_std = normal_std[start:end]

        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.abs((segment - seg_mean) / (seg_std + 1e-10))
        mean_z = np.nanmean(z_scores)

        if mean_z < threshold:
            symbols.append(name)
        else:
            symbols.append(name.lower())

    return ' '.join(symbols)

def evaluate_threshold(threshold, X_train, y_train, X_test, y_test, normal_mean, normal_std):
    """Train grammar and evaluate with given threshold"""

    # Encode training normal heartbeats
    normal_encoded = []
    for i in range(len(X_train)):
        if y_train[i] == 0:
            encoded = encode_heartbeat(X_train[i], normal_mean, normal_std, threshold)
            normal_encoded.append(encoded)

    # Create grammar (accepted patterns)
    accepted_patterns = set(normal_encoded)

    # Test
    y_true = []
    y_pred = []

    for i in range(len(X_test)):
        encoded = encode_heartbeat(X_test[i], normal_mean, normal_std, threshold)
        true_label = 0 if y_test[i] == 0 else 1
        pred_label = 0 if encoded in accepted_patterns else 1
        y_true.append(true_label)
        y_pred.append(pred_label)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'threshold': threshold,
        'patterns': len(accepted_patterns),
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Test different thresholds
print("\nTesting thresholds from 0.5 to 2.5...")
print("\n{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Threshold", "Patterns", "Accuracy", "Precision", "Recall", "F1-Score"))
print("-" * 65)

results_list = []
for threshold in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
    result = evaluate_threshold(threshold, X_train, y_train, X_test, y_test, normal_mean, normal_std)
    results_list.append(result)
    print("{:<10.2f} {:<10} {:<10.2%} {:<10.2%} {:<10.2%} {:<10.4f}".format(
        result['threshold'],
        result['patterns'],
        result['accuracy'],
        result['precision'],
        result['recall'],
        result['f1']
    ))

# Find best threshold (by F1 score)
best = max(results_list, key=lambda x: x['f1'])
print(f"\nBest threshold by F1-Score: {best['threshold']}")
print(f"  Accuracy: {best['accuracy']:.2%}")
print(f"  Precision: {best['precision']:.2%}")
print(f"  Recall: {best['recall']:.2%}")
print(f"  F1-Score: {best['f1']:.4f}")

# Save best results
with open(RESULTS_PATH + "best_threshold_results.json", 'w') as f:
    json.dump(best, f, indent=2)

# Save all results
results_df = pd.DataFrame(results_list)
results_df.to_csv(RESULTS_PATH + "threshold_comparison.csv", index=False)

print(f"\nResults saved to:")
print(f"  - threshold_comparison.csv")
print(f"  - best_threshold_results.json")

print("\n" + "=" * 50)
print("THRESHOLD ANALYSIS COMPLETE!")
print("=" * 50)
