import pandas as pd
import numpy as np
from collections import Counter
import os

# Paths
DATA_PATH = "../data/raw/"
RESULTS_PATH = "../results/"

print("=" * 50)
print("ENCODING HEARTBEATS TO SYMBOLS")
print("=" * 50)

# Load data
print("\nLoading data...")
train_df = pd.read_csv(DATA_PATH + "mitbih_train.csv", header=None)
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values

print(f"Loaded {len(X)} heartbeats")

# Get normal heartbeats to calculate statistics
normal_X = X[y == 0]
normal_mean = np.mean(normal_X, axis=0)
normal_std = np.std(normal_X, axis=0)

print(f"Calculated statistics from {len(normal_X)} normal heartbeats")

def encode_heartbeat(heartbeat, normal_mean, normal_std, threshold=1.75):
    """
    Convert heartbeat to symbols.
    A-J uppercase = normal segments
    a-j lowercase = abnormal segments
    """
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

        # Calculate z-score
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.abs((segment - seg_mean) / (seg_std + 1e-10))
        mean_z = np.nanmean(z_scores)

        if mean_z < threshold:
            symbols.append(name)  # Normal
        else:
            symbols.append(name.lower())  # Abnormal

    return ' '.join(symbols)

# Encode all heartbeats
print("\nEncoding all heartbeats...")
encoded_all = []
for i in range(len(X)):
    encoded = encode_heartbeat(X[i], normal_mean, normal_std)
    encoded_all.append(encoded)
    if (i + 1) % 20000 == 0:
        print(f"  Processed {i+1}/{len(X)}")

print(f"Done! Encoded {len(encoded_all)} heartbeats")

# Separate by class
normal_encoded = [encoded_all[i] for i in range(len(y)) if y[i] == 0]
abnormal_encoded = [encoded_all[i] for i in range(len(y)) if y[i] != 0]

print(f"\nNormal sequences: {len(normal_encoded)}")
print(f"Abnormal sequences: {len(abnormal_encoded)}")

# Count patterns
print("\n--- Most Common Patterns (Normal) ---")
normal_patterns = Counter(normal_encoded)
for pattern, count in normal_patterns.most_common(5):
    pct = 100 * count / len(normal_encoded)
    print(f"  '{pattern}': {count} ({pct:.1f}%)")

print("\n--- Most Common Patterns (Abnormal) ---")
abnormal_patterns = Counter(abnormal_encoded)
for pattern, count in abnormal_patterns.most_common(5):
    pct = 100 * count / len(abnormal_encoded)
    print(f"  '{pattern}': {count} ({pct:.1f}%)")

# Save to files
print("\nSaving files...")

with open(RESULTS_PATH + "normal_sequences.txt", 'w') as f:
    for seq in normal_encoded:
        f.write(seq + '\n')

with open(RESULTS_PATH + "abnormal_sequences.txt", 'w') as f:
    for seq in abnormal_encoded:
        f.write(seq + '\n')

# Save encoded data with labels
encoded_df = pd.DataFrame({
    'encoded_sequence': encoded_all,
    'label': y,
    'is_normal': (y == 0).astype(int)
})
encoded_df.to_csv(RESULTS_PATH + "encoded_heartbeats.csv", index=False)

# Save statistics for later
np.save(RESULTS_PATH + "normal_mean.npy", normal_mean)
np.save(RESULTS_PATH + "normal_std.npy", normal_std)

print("\n--- Files Saved ---")
print(f"  normal_sequences.txt ({len(normal_encoded)} sequences)")
print(f"  abnormal_sequences.txt ({len(abnormal_encoded)} sequences)")
print(f"  encoded_heartbeats.csv")
print(f"  normal_mean.npy")
print(f"  normal_std.npy")

print("\n" + "=" * 50)
print("ENCODING COMPLETE!")
print("=" * 50)
