#!/usr/bin/env python3
"""
Demo Script: Using the Modular ECG Anomaly Detection System
============================================================
This script demonstrates the clean, modular API.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import pandas as pd
import numpy as np

from encoder import HeartbeatEncoder
from grammar_learner import GrammarLearner
from anomaly_detector import DFADetector

def main():
    print("=" * 60)
    print("ECG GRAMMAR-BASED ANOMALY DETECTION")
    print("Modular API Demo")
    print("=" * 60)

    # Paths - use absolute path based on script location
    script_dir = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(script_dir, "data", "raw") + os.sep
    OUTPUT_PATH = os.path.join(script_dir, "data", "processed") + os.sep

    # Load data
    print("\n[1] Loading data...")
    train_df = pd.read_csv(DATA_PATH + "mitbih_train.csv", header=None)
    test_df = pd.read_csv(DATA_PATH + "mitbih_test.csv", header=None)

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    print(f"    Training: {len(X_train)} samples")
    print(f"    Testing: {len(X_test)} samples")

    # Step 1: Create and fit encoder
    print("\n[2] Encoding heartbeats...")
    encoder = HeartbeatEncoder(n_segments=10, threshold=1.75)
    normal_heartbeats = X_train[y_train == 0]
    encoder.fit(normal_heartbeats)

    # Encode training data
    train_sequences = encoder.encode_batch(X_train)
    test_sequences = encoder.encode_batch(X_test)
    print(f"    Encoded {len(train_sequences)} training sequences")
    print(f"    Encoded {len(test_sequences)} test sequences")

    # Step 2: Learn grammar
    print("\n[3] Learning grammar from normal heartbeats...")
    normal_sequences = [train_sequences[i] for i in range(len(y_train)) if y_train[i] == 0]

    grammar = GrammarLearner()
    grammar.fit(normal_sequences)

    print(grammar.get_formal_definition())

    # Step 3: Create detector and evaluate
    print("\n[4] Creating DFA detector and evaluating...")
    detector = DFADetector(grammar)

    # Evaluate
    results = detector.evaluate(test_sequences, y_test.astype(int).tolist())

    print(detector.get_detailed_report(test_sequences, y_test.astype(int).tolist()))

    # Step 4: Example predictions
    print("\n[5] Example predictions:")
    print("-" * 60)

    for i in range(5):
        seq = test_sequences[i]
        true_label = "Normal" if y_test[i] == 0 else "Abnormal"
        is_normal, pred_label, details = detector.detect(seq)

        print(f"\n  Sample {i+1}:")
        print(f"    Sequence: {seq}")
        print(f"    True: {true_label}, Predicted: {pred_label}")
        if not is_normal and details.get('hotspots'):
            hotspots = [h['segment'] for h in details['hotspots']]
            print(f"    Hotspots: {', '.join(hotspots)}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
