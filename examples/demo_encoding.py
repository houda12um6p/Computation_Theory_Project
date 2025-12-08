#!/usr/bin/env python3
"""
Demo: Heartbeat Encoding
========================
This example demonstrates how to encode ECG heartbeats into symbolic sequences.

Course: Computational Theory - Fall 2025
Institution: Mohammed VI Polytechnic University (UM6P)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Note: pandas not available, using numpy for data loading")

from encoder import HeartbeatEncoder


def main():
    print("=" * 60)
    print("DEMO: ECG Heartbeat Encoding")
    print("=" * 60)

    # Data path
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

    # Load a small sample of data
    print("\n[1] Loading sample data...")

    if HAS_PANDAS:
        train_df = pd.read_csv(os.path.join(DATA_PATH, "mitbih_train.csv"), header=None)
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
    else:
        # Fallback to numpy
        data = np.loadtxt(os.path.join(DATA_PATH, "mitbih_train.csv"), delimiter=',')
        X_train = data[:, :-1]
        y_train = data[:, -1]

    print(f"    Loaded {len(X_train)} heartbeats")
    print(f"    Each heartbeat has {X_train.shape[1]} samples")

    # Create encoder with different configurations
    print("\n[2] Creating encoder...")
    print("    Configuration: 10 segments, threshold=1.75")

    encoder = HeartbeatEncoder(n_segments=10, threshold=1.75)

    # Fit on normal heartbeats only
    print("\n[3] Fitting encoder on normal heartbeats...")
    normal_mask = y_train == 0
    normal_heartbeats = X_train[normal_mask]
    print(f"    Using {len(normal_heartbeats)} normal heartbeats for fitting")

    encoder.fit(normal_heartbeats)

    # Show learned statistics
    print("\n[4] Learned segment statistics:")
    print("    " + "-" * 50)
    print(f"    {'Segment':<10} {'Sample Mean':>12} {'Sample Std':>12}")
    print("    " + "-" * 50)
    segment_size = len(encoder.normal_mean) // encoder.n_segments
    for i in range(encoder.n_segments):
        start = i * segment_size
        end = start + segment_size if i < encoder.n_segments - 1 else len(encoder.normal_mean)
        seg_mean = np.mean(encoder.normal_mean[start:end])
        seg_std = np.mean(encoder.normal_std[start:end])
        print(f"    {f'Segment {i+1}':<10} {seg_mean:>12.4f} {seg_std:>12.4f}")

    # Encode some examples
    print("\n[5] Encoding example heartbeats...")
    print("    " + "-" * 60)

    # Get examples of each class
    class_names = {0: "Normal", 1: "Supraventricular", 2: "Ventricular",
                   3: "Fusion", 4: "Unknown"}

    for class_id in range(5):
        class_mask = y_train == class_id
        if np.sum(class_mask) > 0:
            # Get first example of this class
            idx = np.where(class_mask)[0][0]
            heartbeat = X_train[idx]
            sequence = encoder.encode(heartbeat)

            print(f"\n    Class {class_id} ({class_names[class_id]}):")
            print(f"    Sequence: {sequence}")

            # Highlight abnormal segments
            symbols = sequence.split()
            abnormal = [i+1 for i, s in enumerate(symbols) if s.islower()]
            if abnormal:
                print(f"    Abnormal segments: {abnormal}")
            else:
                print(f"    All segments normal")

    # Batch encoding demonstration
    print("\n[6] Batch encoding demonstration...")
    sample_size = 100
    sample_heartbeats = X_train[:sample_size]
    sequences = encoder.encode_batch(sample_heartbeats)

    # Count unique patterns
    unique_patterns = set(sequences)
    print(f"    Encoded {sample_size} heartbeats")
    print(f"    Found {len(unique_patterns)} unique patterns")

    # Show pattern distribution
    pattern_counts = {}
    for seq in sequences:
        pattern_counts[seq] = pattern_counts.get(seq, 0) + 1

    print("\n    Top 5 most common patterns:")
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])[:5]
    for pattern, count in sorted_patterns:
        pct = count / sample_size * 100
        print(f"      {pattern}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("ENCODING DEMO COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  - Uppercase letters (A-J) indicate normal segment amplitudes")
    print("  - Lowercase letters (a-j) indicate abnormal deviations")
    print("  - The threshold (1.75) controls sensitivity")
    print("  - Normal heartbeats typically produce all-uppercase sequences")


if __name__ == "__main__":
    main()
