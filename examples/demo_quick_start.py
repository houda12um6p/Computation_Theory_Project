#!/usr/bin/env python3
"""
Quick Start Demo
================
Minimal example to get started with the ECG anomaly detection system.

Course: Computational Theory - Fall 2025
Institution: Mohammed VI Polytechnic University (UM6P)
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


# Configuration
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
N_SEGMENTS = 10
THRESHOLD = 1.75


def main():
    # Load data
    train_df = pd.read_csv(os.path.join(DATA_PATH, "mitbih_train.csv"), header=None)
    test_df = pd.read_csv(os.path.join(DATA_PATH, "mitbih_test.csv"), header=None)

    X_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
    X_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values

    # 1. Encode heartbeats
    encoder = HeartbeatEncoder(n_segments=N_SEGMENTS, threshold=THRESHOLD)
    encoder.fit(X_train[y_train == 0])  # Fit on normal only

    train_seqs = encoder.encode_batch(X_train)
    test_seqs = encoder.encode_batch(X_test)

    # 2. Learn grammar from normal patterns
    normal_seqs = [train_seqs[i] for i in range(len(y_train)) if y_train[i] == 0]
    grammar = GrammarLearner()
    grammar.fit(normal_seqs)

    # 3. Create detector and evaluate
    detector = DFADetector(grammar)
    results = detector.evaluate(test_seqs, y_test.astype(int).tolist())

    # Print results
    print("ECG Anomaly Detection Results")
    print("=" * 40)
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall:    {results['recall']:.2%}")
    print(f"F1-Score:  {results['f1_score']:.4f}")


if __name__ == "__main__":
    main()
