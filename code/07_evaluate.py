import pandas as pd
import numpy as np
import json
from collections import Counter
import os

DATA_PATH = "../data/raw/"
RESULTS_PATH = "../results/"

print("=" * 50)
print("FULL EVALUATION ON TEST DATA")
print("=" * 50)

# Load test data
print("\nLoading test data...")
test_df = pd.read_csv(DATA_PATH + "mitbih_test.csv", header=None)
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values
print(f"Loaded {len(X_test)} test heartbeats")

# Load normal statistics
normal_mean = np.load(RESULTS_PATH + "normal_mean.npy")
normal_std = np.load(RESULTS_PATH + "normal_std.npy")

# Load grammar
with open(RESULTS_PATH + "learned_grammar.json", 'r') as f:
    grammar = json.load(f)
accepted_patterns = set(grammar['accepted_patterns'])
print(f"Grammar has {len(accepted_patterns)} accepted patterns")

# Encoding function
def encode_heartbeat(heartbeat, normal_mean, normal_std, threshold=1.75):
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

# Run detection on all test data
print("\nRunning detection...")
y_true = []  # 0 = normal, 1 = abnormal
y_pred = []  # 0 = predicted normal, 1 = predicted abnormal

for i in range(len(X_test)):
    # Encode heartbeat
    encoded = encode_heartbeat(X_test[i], normal_mean, normal_std)

    # True label
    true_label = 0 if y_test[i] == 0 else 1
    y_true.append(true_label)

    # Prediction: if pattern is in grammar -> normal, else -> abnormal
    if encoded in accepted_patterns:
        y_pred.append(0)
    else:
        y_pred.append(1)

    if (i + 1) % 5000 == 0:
        print(f"  Processed {i+1}/{len(X_test)}")

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate metrics
print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

# Basic counts
true_normal = np.sum(y_true == 0)
true_abnormal = np.sum(y_true == 1)
pred_normal = np.sum(y_pred == 0)
pred_abnormal = np.sum(y_pred == 1)

print(f"\nDataset: {len(y_true)} heartbeats")
print(f"  Actually Normal: {true_normal}")
print(f"  Actually Abnormal: {true_abnormal}")

# Confusion matrix
TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negative
FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positive
FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negative

print("\nConfusion Matrix:")
print(f"                  Predicted Normal  Predicted Abnormal")
print(f"  Actually Normal      {TN:>6}            {FP:>6}")
print(f"  Actually Abnormal    {FN:>6}            {TP:>6}")

# Metrics
accuracy = (TP + TN) / len(y_true)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMetrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Save results
results = {
    'total_samples': len(y_true),
    'true_normal': int(true_normal),
    'true_abnormal': int(true_abnormal),
    'confusion_matrix': {
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN)
    },
    'accuracy': round(accuracy, 4),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1_score': round(f1, 4)
}

with open(RESULTS_PATH + "evaluation_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: evaluation_results.json")

print("\n" + "=" * 50)
print("EVALUATION COMPLETE!")
print("=" * 50)
