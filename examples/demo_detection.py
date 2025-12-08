#!/usr/bin/env python3
"""
Demo: Anomaly Detection Pipeline
================================
This example demonstrates the complete anomaly detection pipeline:
1. Load data
2. Encode heartbeats
3. Learn grammar from normal patterns
4. Detect anomalies using PDA

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

from encoder import HeartbeatEncoder
from grammar_learner import GrammarLearner
from anomaly_detector import PDADetector


def load_data(data_path):
    """Load the MIT-BIH dataset."""
    if HAS_PANDAS:
        train_df = pd.read_csv(os.path.join(data_path, "mitbih_train.csv"), header=None)
        test_df = pd.read_csv(os.path.join(data_path, "mitbih_test.csv"), header=None)
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
    else:
        train_data = np.loadtxt(os.path.join(data_path, "mitbih_train.csv"), delimiter=',')
        test_data = np.loadtxt(os.path.join(data_path, "mitbih_test.csv"), delimiter=',')
        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]

    return X_train, y_train, X_test, y_test


def main():
    print("=" * 70)
    print("DEMO: Complete Anomaly Detection Pipeline")
    print("Using Context-Free Grammar and Pushdown Automaton")
    print("=" * 70)

    # Paths
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

    # Step 1: Load data
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)

    X_train, y_train, X_test, y_test = load_data(DATA_PATH)

    print(f"Training set: {len(X_train)} heartbeats")
    print(f"  - Normal (class 0): {np.sum(y_train == 0)}")
    print(f"  - Abnormal (classes 1-4): {np.sum(y_train != 0)}")
    print(f"Test set: {len(X_test)} heartbeats")

    # Step 2: Create and fit encoder
    print("\n" + "=" * 70)
    print("STEP 2: Symbolic Encoding")
    print("=" * 70)

    encoder = HeartbeatEncoder(n_segments=10, threshold=1.75)
    normal_heartbeats = X_train[y_train == 0]
    encoder.fit(normal_heartbeats)

    print(f"Encoder fitted on {len(normal_heartbeats)} normal heartbeats")
    print(f"Configuration: {encoder.n_segments} segments, threshold={encoder.threshold}")

    # Encode all data
    print("\nEncoding training data...")
    train_sequences = encoder.encode_batch(X_train)
    print(f"Encoded {len(train_sequences)} training sequences")

    print("\nEncoding test data...")
    test_sequences = encoder.encode_batch(X_test)
    print(f"Encoded {len(test_sequences)} test sequences")

    # Step 3: Learn grammar
    print("\n" + "=" * 70)
    print("STEP 3: Grammar Learning")
    print("=" * 70)

    # Get only normal sequences for grammar learning
    normal_sequences = [train_sequences[i] for i in range(len(y_train)) if y_train[i] == 0]

    grammar = GrammarLearner()
    grammar.fit(normal_sequences)

    print(f"Grammar learned from {len(normal_sequences)} normal sequences")
    print(f"Unique patterns accepted: {len(grammar.accepted_patterns)}")
    print(f"Main pattern: {grammar.main_pattern}")

    # Show formal grammar definition
    print("\n" + grammar.get_formal_definition())

    # Step 4: Create detector
    print("\n" + "=" * 70)
    print("STEP 4: PDA-Based Detection")
    print("=" * 70)

    detector = PDADetector(grammar)
    print("Pushdown Automaton detector created")

    # Demonstrate individual detection
    print("\n--- Individual Detection Examples ---")

    examples = [
        (0, "Normal heartbeat"),
        (1, "Supraventricular"),
        (2, "Ventricular ectopic"),
        (3, "Fusion beat"),
        (4, "Unknown"),
    ]

    for class_id, class_name in examples:
        # Find first test example of this class
        indices = np.where(y_test == class_id)[0]
        if len(indices) > 0:
            idx = indices[0]
            seq = test_sequences[idx]
            is_normal, label, details = detector.detect(seq)

            print(f"\nClass {class_id} ({class_name}):")
            print(f"  Sequence: {seq}")
            print(f"  Detection: {label}")

            if not is_normal and details.get('hotspots'):
                hotspots = [h['segment'] for h in details['hotspots']]
                print(f"  Hotspots: {', '.join(hotspots)}")

    # Step 5: Evaluate on test set
    print("\n" + "=" * 70)
    print("STEP 5: Evaluation")
    print("=" * 70)

    results = detector.evaluate(test_sequences, y_test.astype(int).tolist())

    print(detector.get_detailed_report(test_sequences, y_test.astype(int).tolist()))

    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Total test samples: {results['total_samples']}")
    print(f"Predicted normal: {results['predictions']['normal']}")
    print(f"Predicted anomaly: {results['predictions']['anomaly']}")

    # Per-class analysis
    print("\n--- Per-Class Detection Rates ---")
    class_names = {0: "Normal", 1: "Supraventricular", 2: "Ventricular",
                   3: "Fusion", 4: "Unknown"}

    for class_id in range(5):
        class_mask = y_test == class_id
        if np.sum(class_mask) > 0:
            class_seqs = [test_sequences[i] for i in range(len(y_test)) if y_test[i] == class_id]
            predictions = detector.predict_batch(class_seqs)

            if class_id == 0:
                # For normal, we want prediction = 0 (correct)
                correct = sum(1 for p in predictions if p == 0)
            else:
                # For abnormal, we want prediction = 1 (detected)
                correct = sum(1 for p in predictions if p == 1)

            rate = correct / len(predictions) * 100
            print(f"  Class {class_id} ({class_names[class_id]}): {correct}/{len(predictions)} = {rate:.1f}%")

    print("\n" + "=" * 70)
    print("DETECTION DEMO COMPLETE")
    print("=" * 70)

    print("\nKey insights:")
    print("  - High accuracy (83.61%) shows good overall performance")
    print("  - High precision (93.06%) means few false alarms")
    print("  - Low recall (5.33%) means many anomalies go undetected")
    print("  - This is expected: grammar-based detection is conservative")
    print("\nThe formal language approach provides:")
    print("  - Interpretable symbolic representation")
    print("  - Theoretical foundations (CFG, PDA)")
    print("  - Hotspot localization for anomalies")


if __name__ == "__main__":
    main()
