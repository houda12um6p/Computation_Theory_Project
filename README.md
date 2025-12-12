# ECG Anomaly Detection Using Grammar Inference

**Course:** Computational Theory - Fall 2025
**Institution:** Mohammed VI Polytechnic University (UM6P)

**Authors:** Houda TOUDALI, Aya BENJELLOUN, Nour El Houda El IAMANI

**Repository:** [https://github.com/houda12um6p/Computation_Theory_Project](https://github.com/houda12um6p/Computation_Theory_Project)

---

## Overview

We built a system that detects abnormal heartbeats by treating ECG signals as sequences of letters and checking if they match patterns learned from normal heartbeats.

This project was developed for the Computational Theory course and applies concepts from formal language theory to a real medical dataset.

Instead of using machine learning, we:

1. Encode ECG signals as symbolic sequences using z-score normalization
2. Learn a Context-Free Grammar (CFG) from normal heartbeats
3. Detect anomalies using a Deterministic Finite Automaton (DFA) recognizer

### Research Question

Our goal was to see if we could learn what normal heartbeats look like using a grammar, then flag anything that does not fit as potentially abnormal.

### Key Results

| Metric | 5-Segment | 10-Segment | Improvement |
|--------|-----------|------------|-------------|
| Accuracy | 82.79% | 83.61% | +0.82% |
| Precision | 81.82% | 93.06% | +11.24% |
| Recall | 0.24% | 5.33% | +5.09% |
| F1-Score | 0.0048 | 0.1008 | 21x |

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Dataset](#dataset)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)
- [Authors](#authors)

---

## Quick Start

```python
from src.encoder import HeartbeatEncoder
from src.grammar_learner import GrammarLearner
from src.anomaly_detector import DFADetector

# 1. Encode heartbeats
encoder = HeartbeatEncoder(n_segments=10, threshold=1.75)
encoder.fit(normal_heartbeats)
sequences = encoder.encode_batch(heartbeats)

# 2. Learn grammar from normal patterns
grammar = GrammarLearner()
grammar.fit(normal_sequences)

# 3. Detect anomalies
detector = DFADetector(grammar)
is_normal, label, details = detector.detect(sequence)
```

---

## Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Install Dependencies

```bash
cd ecg_project
pip install -r requirements.txt
```

### Verify Installation

```bash
python examples/demo_quick_start.py
```

---

## Project Structure

```
ecg_project/
|
|-- README.md                  # This file
|-- LICENSE                    # MIT License
|-- requirements.txt           # Python dependencies
|
|-- src/                       # Main source code
|   |-- encoder.py             # HeartbeatEncoder class
|   |-- grammar_learner.py     # GrammarLearner class
|   |-- anomaly_detector.py    # DFADetector class
|
|-- code/                      # Analysis scripts (run in order)
|   |-- 01_verify_data.py      # Load and verify data
|   |-- 02_visualize_data.py   # Create visualizations
|   |-- 03_encode_heartbeats.py # Encode to symbols
|   |-- 04_create_summary.py   # Excel summary
|   |-- 05_learn_grammar.py    # Learn CFG
|   |-- 06_anomaly_detector.py # Build detector
|   |-- 07_evaluate.py         # Evaluate on test set
|   |-- 08_improved_detector.py # Threshold optimization
|   |-- 09_create_visualizations.py # Result charts
|   |-- 10_hotspot_analysis.py # Anomaly locations
|   |-- 11_improved_encoding.py # 10-segment comparison
|
|-- scripts/                   # Utility scripts
|   |-- run_all.py             # Run complete pipeline
|   |-- demo_modular.py        # Demo of modular API
|
|-- examples/                  # Usage examples
|   |-- demo_encoding.py       # Encoding demonstration
|   |-- demo_detection.py      # Detection pipeline demo
|   |-- demo_quick_start.py    # Minimal example
|
|-- docs/                      # Documentation
|   |-- API.md                 # API reference
|   |-- ALGORITHMS.md          # Algorithm details
|
|-- data/
|   |-- raw/                   # Original CSV files
|   |-- processed/             # Encoded sequences, grammar
|
|-- results/
|   |-- figures/               # All visualizations (PNG)
|   |-- metrics/               # JSON evaluation results
|
|-- submissions/               # Submission-ready files
    |-- paper/                 # Academic paper (PDF)
    |-- code/                  # Code copy
    |-- data_sample/           # Sample data
    |-- supplementary/         # Additional materials
```

---

## Methodology

### Step 1: Symbolic Encoding

Each heartbeat (187 samples) is divided into 10 segments. For each segment:
- Compute z-score relative to normal heartbeat statistics
- Assign UPPERCASE (A-J) if z-score < θ (normal)
- Assign lowercase (a-j) if z-score ≥ θ (abnormal)

We use threshold θ = 1.75, selected for optimal precision in clinical screening.

```
Input:  [0.1, 0.3, 0.8, 1.0, 0.9, ...] (187 values)
Output: "A B C D E F G H I J"         (10 symbols)
```

### Step 2: Grammar Learning

From 72,471 normal heartbeats, we learn a Context-Free Grammar:

```
G = (V, Sigma, R, S) where:
  V = {S, T_1, T_2, ..., T_10}     (Variables)
  Sigma = {A-J, a-j}               (Terminals)
  R = Production rules             (181 patterns)
  S = Start symbol
```

The dominant pattern `ABCDEFGHIJ` (all segments normal) covers 86.7% of normal heartbeats.

### Step 3: DFA-Based Detection

A Deterministic Finite Automaton (847 states) recognizes sequences matching the grammar:
- ACCEPT (Normal): Sequence matches a learned pattern
- REJECT (Anomaly): Sequence not in grammar + hotspot identification

---

## Results

### Performance Summary (10-Segment Encoding)

| Metric | Value |
|--------|-------|
| Accuracy | 83.61% |
| Precision | 93.06% |
| Recall | 5.33% |
| F1-Score | 0.1008 |

### Confusion Matrix

|  | Predicted Normal | Predicted Abnormal |
|--|------------------|-------------------|
| Actually Normal | 18,103 | 15 |
| Actually Abnormal | 3,573 | 201 |

### Hotspot Analysis

Different arrhythmia types affect different cardiac segments:

| Arrhythmia Type | Primary Hotspots | Abnormality Rate |
|-----------------|------------------|------------------|
| Ventricular | P-wave (31.1%), Q-wave (21.6%) | High |
| Supraventricular | T-wave (14.0%), S-wave (11.7%) | Moderate |
| Unknown | Q-wave (38.6%), P-wave (17.7%) | Very High |
| Fusion | All segments | Low (2-4%) |

These patterns are clinically meaningful: ventricular arrhythmias affect early depolarization (P, Q, R), while supraventricular arrhythmias affect repolarization (T-wave).

### Clinical Validation

Expert cardiologist review of 20 ECG samples showed:
- **70% agreement** (partial or exact match) with system hotspots
- **Cohen's Kappa κ = 0.42** (moderate inter-rater agreement)
- System rated as useful for "secondary check / quality assurance"
- Successfully detected ST-elevation emergencies and ventricular ectopy

---

## API Documentation

See [docs/API.md](docs/API.md) for the full API reference.

### Core Classes

**HeartbeatEncoder** - Encode ECG signals to symbolic sequences
```python
encoder = HeartbeatEncoder(n_segments=10, threshold=1.75)
encoder.fit(normal_heartbeats)
sequence = encoder.encode(heartbeat)
```

**GrammarLearner** - Learn CFG from normal patterns
```python
grammar = GrammarLearner()
grammar.fit(normal_sequences)
print(grammar.get_formal_definition())
```

**DFADetector** - Detect anomalies using learned grammar
```python
detector = DFADetector(grammar)
is_normal, label, details = detector.detect(sequence)
results = detector.evaluate(sequences, labels)
```

---

## Examples

### Run Example Scripts

```bash
# Encoding demonstration
python examples/demo_encoding.py

# Complete detection pipeline
python examples/demo_detection.py

# Quick start (minimal example)
python examples/demo_quick_start.py
```

### Run Complete Pipeline

```bash
python scripts/run_all.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | MIT-BIH Arrhythmia Database (Kaggle) |
| Training | 87,554 heartbeats |
| Testing | 21,892 heartbeats |
| Features | 187 time points per heartbeat |
| Classes | 5 (Normal + 4 arrhythmia types) |

### Class Distribution

| Class | Name | Count | Percentage |
|-------|------|-------|------------|
| 0 | Normal | 72,471 | 82.8% |
| 1 | Supraventricular | 2,223 | 2.5% |
| 2 | Ventricular | 5,788 | 6.6% |
| 3 | Fusion | 641 | 0.7% |
| 4 | Unknown | 6,431 | 7.3% |

---

## Limitations and Future Work

### Current Limitations
1. Conservative detection: High precision but low recall
2. Pattern-based: Cannot detect anomalies that happen to match normal patterns
3. Fixed segmentation: Equal-width segments may not align with ECG components

### Future Improvements
1. Adaptive segmentation based on ECG morphology
2. Hierarchical grammar (Sequitur algorithm)
3. Hybrid approach combining grammar features with deep learning
4. Multi-resolution encoding for different detection sensitivities

---

## References

1. Moody, G.B., & Mark, R.G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.

2. Keogh, E., Lonardi, S., & Ratanamahatana, C.A. (2004). Towards parameter-free data mining. Proceedings of the 10th ACM SIGKDD, 206-215.

3. Nevill-Manning, C.G., & Witten, I.H. (1997). Identifying hierarchical structure in sequences. Journal of Artificial Intelligence Research, 7, 67-82.

4. Hopcroft, J.E., Motwani, R., & Ullman, J.D. (2006). Introduction to Automata Theory, Languages, and Computation.

---

## Authors

- Houda TOUDALI
- Aya BENJELLOUN
- Nour El Houda El IAMANI

College of Computing, Mohammed VI Polytechnic University (UM6P)
Computational Theory Course - Fall 2025

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
