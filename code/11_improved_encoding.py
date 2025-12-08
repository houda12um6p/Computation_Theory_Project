import pandas as pd
import numpy as np
from collections import Counter
import json
import os

DATA_PATH = "../data/raw/"
RESULTS_PATH = "../results/"

print("=" * 60)
print("IMPROVED ENCODING WITH 10 SEGMENTS")
print("More granular analysis for better anomaly detection")
print("=" * 60)

# Load data
print("\nLoading data...")
train_df = pd.read_csv(DATA_PATH + "mitbih_train.csv", header=None)
test_df = pd.read_csv(DATA_PATH + "mitbih_test.csv", header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

print(f"Training: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")

# Calculate normal statistics
normal_X = X_train[y_train == 0]
normal_mean = np.mean(normal_X, axis=0)
normal_std = np.std(normal_X, axis=0)

# NEW: 10-segment encoding
def encode_10_segments(heartbeat, normal_mean, normal_std, threshold=1.75):
    """
    Encode heartbeat into 10 segments for finer granularity.
    Segments: P1, P2, Q1, Q2, R1, R2, S1, S2, T1, T2
    Uppercase = normal, lowercase = abnormal
    """
    symbols = []
    segment_size = len(heartbeat) // 10
    segment_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    for i, name in enumerate(segment_names):
        start = i * segment_size
        end = start + segment_size if i < 9 else len(heartbeat)

        segment = heartbeat[start:end]
        seg_mean = normal_mean[start:end]
        seg_std = normal_std[start:end]

        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.abs((segment - seg_mean) / (seg_std + 1e-10))
        mean_z = np.nanmean(z_scores)

        if mean_z < threshold:
            symbols.append(name)  # Normal
        else:
            symbols.append(name.lower())  # Abnormal

    return ' '.join(symbols)

# Encode all training data
print("\nEncoding training data with 10 segments...")
train_encoded = []
for i, heartbeat in enumerate(X_train):
    encoded = encode_10_segments(heartbeat, normal_mean, normal_std)
    train_encoded.append(encoded)
    if (i + 1) % 20000 == 0:
        print(f"  Processed {i+1}/{len(X_train)}")

# Build grammar from normal heartbeats only
normal_encoded = [train_encoded[i] for i in range(len(y_train)) if y_train[i] == 0]
accepted_patterns = set(normal_encoded)

print(f"\nGrammar learned: {len(accepted_patterns)} unique patterns from normal heartbeats")

# Show top patterns
pattern_counts = Counter(normal_encoded)
print("\nTop 10 Normal Patterns:")
for pattern, count in pattern_counts.most_common(10):
    pct = 100 * count / len(normal_encoded)
    print(f"  '{pattern}': {count} ({pct:.1f}%)")

# Evaluate on test data
print("\n" + "=" * 60)
print("EVALUATING ON TEST DATA")
print("=" * 60)

# Test multiple thresholds
results = []
for threshold in [1.0, 1.25, 1.5, 1.75, 2.0]:
    # Re-encode with this threshold
    test_encoded = [encode_10_segments(hb, normal_mean, normal_std, threshold) for hb in X_test]
    train_normal_encoded = [encode_10_segments(X_train[i], normal_mean, normal_std, threshold)
                           for i in range(len(X_train)) if y_train[i] == 0]
    grammar = set(train_normal_encoded)

    # Predict
    y_true = (y_test != 0).astype(int)  # 1 = abnormal
    y_pred = np.array([0 if enc in grammar else 1 for enc in test_encoded])

    # Metrics
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results.append({
        'threshold': threshold,
        'patterns': len(grammar),
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

    print(f"\nThreshold {threshold}:")
    print(f"  Grammar size: {len(grammar)} patterns")
    print(f"  Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.4f}")

# Find best
best = max(results, key=lambda x: x['f1'])
print("\n" + "=" * 60)
print("BEST RESULT (10 SEGMENTS)")
print("=" * 60)
print(f"Threshold: {best['threshold']}")
print(f"Grammar Size: {best['patterns']} patterns")
print(f"Accuracy: {best['accuracy']:.2%}")
print(f"Precision: {best['precision']:.2%}")
print(f"Recall: {best['recall']:.2%}")
print(f"F1-Score: {best['f1']:.4f}")

# Save results
with open(RESULTS_PATH + "improved_10seg_results.json", 'w') as f:
    json.dump({
        'best_threshold': best['threshold'],
        'best_results': best,
        'all_results': results
    }, f, indent=2)

# Compare with 5-segment results
print("\n" + "=" * 60)
print("COMPARISON: 5 Segments vs 10 Segments")
print("=" * 60)

with open(RESULTS_PATH + "best_threshold_results.json", 'r') as f:
    old_results = json.load(f)

print(f"\n{'Metric':<15} {'5 Segments':<15} {'10 Segments':<15} {'Improvement':<15}")
print("-" * 60)
print(f"{'Patterns':<15} {old_results['patterns']:<15} {best['patterns']:<15} {'More detail':<15}")
print(f"{'Accuracy':<15} {old_results['accuracy']:.2%}{'':8} {best['accuracy']:.2%}{'':8} {(best['accuracy']-old_results['accuracy'])*100:+.2f}%")
print(f"{'Precision':<15} {old_results['precision']:.2%}{'':8} {best['precision']:.2%}{'':8} {(best['precision']-old_results['precision'])*100:+.2f}%")
print(f"{'Recall':<15} {old_results['recall']:.2%}{'':8} {best['recall']:.2%}{'':8} {(best['recall']-old_results['recall'])*100:+.2f}%")
print(f"{'F1-Score':<15} {old_results['f1']:.4f}{'':9} {best['f1']:.4f}{'':9} {best['f1']-old_results['f1']:+.4f}")

print("\n" + "=" * 60)
print("IMPROVED ENCODING COMPLETE!")
print("=" * 60)
