# API Reference

## ECG Grammar-Based Anomaly Detection System

This document provides the API documentation for our ECG anomaly detection system.

---

## Table of Contents

1. [HeartbeatEncoder](#heartbeatencoder)
2. [GrammarLearner](#grammarlearner)
3. [PDADetector](#pdadetector)
4. [Quick Start](#quick-start)

---

## HeartbeatEncoder

**Module:** `src/encoder.py`

This class converts raw ECG heartbeat signals into symbolic sequences using z-score normalization.

### Class Definition

```python
class HeartbeatEncoder:
    def __init__(self, n_segments: int = 10, threshold: float = 1.75)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_segments` | `int` | `10` | Number of segments to divide each heartbeat into |
| `threshold` | `float` | `1.75` | Z-score threshold for normal/abnormal classification |

### Methods

#### `fit(normal_heartbeats: np.ndarray) -> 'HeartbeatEncoder'`

Compute statistics from normal heartbeats for encoding.

**Parameters:**
- `normal_heartbeats`: NumPy array of shape `(n_samples, n_features)` containing normal heartbeat signals

**Returns:**
- `self` for method chaining

**Example:**
```python
encoder = HeartbeatEncoder(n_segments=10, threshold=1.75)
encoder.fit(normal_heartbeats)
```

---

#### `encode(heartbeat: np.ndarray) -> str`

Encode a single heartbeat into a symbolic sequence.

**Parameters:**
- `heartbeat`: 1D NumPy array representing a single heartbeat signal

**Returns:**
- `str`: Space-separated symbolic sequence (e.g., "A B C D E F G H I J")

**Encoding Logic:**
- Uppercase letter (A-J): Segment amplitude is within threshold of normal
- Lowercase letter (a-j): Segment amplitude deviates beyond threshold

**Example:**
```python
sequence = encoder.encode(heartbeat_signal)
# Returns: "A B C d E F G H I J"
```

---

#### `encode_batch(heartbeats: np.ndarray) -> List[str]`

Encode multiple heartbeats into symbolic sequences.

**Parameters:**
- `heartbeats`: NumPy array of shape `(n_samples, n_features)`

**Returns:**
- `List[str]`: List of symbolic sequences

**Example:**
```python
sequences = encoder.encode_batch(X_train)
```

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_segments` | `int` | Number of segments per heartbeat |
| `threshold` | `float` | Z-score threshold value |
| `normal_mean` | `np.ndarray` | Mean amplitude per time point (after fit) |
| `normal_std` | `np.ndarray` | Standard deviation per time point (after fit) |

---

## GrammarLearner

**Module:** `src/grammar_learner.py`

This class learns a Context-Free Grammar (CFG) from symbolic sequences representing normal heartbeats.

### Class Definition

```python
class GrammarLearner:
    def __init__(self)
```

### Methods

#### `fit(sequences: List[str]) -> 'GrammarLearner'`

Learn grammar from normal heartbeat sequences.

**Parameters:**
- `sequences`: List of symbolic sequences (space-separated strings)

**Returns:**
- `self` for method chaining

**Example:**
```python
grammar = GrammarLearner()
grammar.fit(normal_sequences)
```

---

#### `get_formal_definition() -> str`

Get the formal mathematical definition of the learned CFG.

**Returns:**
- `str`: Formatted string with CFG components (V, Sigma, R, S)

**Example:**
```python
print(grammar.get_formal_definition())
```

**Output:**
```
Context-Free Grammar G = (V, Sigma, R, S)

Variables (V): {S, T_1, T_2, ..., T_n}
Terminals (Sigma): {A, B, C, D, E, F, G, H, I, J, a, b, c, d, e, f, g, h, i, j}
Start Symbol: S
Production Rules (R):
  S -> T_1 T_2 ... T_n
  T_1 -> A | a
  ...
```

---

#### `save(filepath: str) -> None`

Save learned grammar to JSON file.

**Parameters:**
- `filepath`: Path to save the grammar

**Example:**
```python
grammar.save("learned_grammar.json")
```

---

#### `load(filepath: str) -> 'GrammarLearner'`

Load grammar from JSON file.

**Parameters:**
- `filepath`: Path to the grammar file

**Returns:**
- `self` for method chaining

**Example:**
```python
grammar = GrammarLearner()
grammar.load("learned_grammar.json")
```

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `accepted_patterns` | `set` | Set of all unique normal patterns |
| `main_pattern` | `str` | Most frequent normal pattern |
| `pattern_counts` | `dict` | Count of each pattern occurrence |

---

## PDADetector

**Module:** `src/anomaly_detector.py`

Pushdown Automaton-based anomaly detector that uses the learned grammar.

### Class Definition

```python
class PDADetector:
    def __init__(self, grammar_learner: GrammarLearner)
```

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `grammar_learner` | `GrammarLearner` | A fitted GrammarLearner instance |

### Methods

#### `detect(sequence: str) -> Tuple[bool, str, Dict]`

Detect if a sequence is normal or anomalous.

**Parameters:**
- `sequence`: Symbolic sequence to check

**Returns:**
- `Tuple[bool, str, Dict]`:
  - `is_normal`: `True` if sequence matches grammar
  - `result_label`: "NORMAL" or "ANOMALY"
  - `details`: Dictionary with match info or hotspots

**Example:**
```python
is_normal, label, details = detector.detect("A B C D E F G H I J")
# Returns: (True, "NORMAL", {"match_type": "exact_main_pattern"})

is_normal, label, details = detector.detect("a B C D E F G H I J")
# Returns: (False, "ANOMALY", {"hotspots": [{"position": 0, "segment": "Segment_1", "symbol": "a"}]})
```

---

#### `predict(sequence: str) -> int`

Predict binary label for a single sequence.

**Parameters:**
- `sequence`: Symbolic sequence to classify

**Returns:**
- `int`: 0 for normal, 1 for anomaly

**Example:**
```python
label = detector.predict(sequence)
```

---

#### `predict_batch(sequences: List[str]) -> List[int]`

Predict labels for multiple sequences.

**Parameters:**
- `sequences`: List of symbolic sequences

**Returns:**
- `List[int]`: List of predictions (0 or 1)

**Example:**
```python
predictions = detector.predict_batch(test_sequences)
```

---

#### `evaluate(sequences: List[str], labels: List[int]) -> Dict`

Evaluate detector performance on labeled test data.

**Parameters:**
- `sequences`: List of symbolic sequences
- `labels`: True labels (0=normal, 1-4=abnormal classes)

**Returns:**
- `Dict`: Evaluation metrics including:
  - `confusion_matrix`: {TP, TN, FP, FN}
  - `accuracy`: Overall accuracy
  - `precision`: Precision score
  - `recall`: Recall score
  - `f1_score`: F1 score
  - `total_samples`: Number of samples
  - `predictions`: Count of normal/anomaly predictions

**Example:**
```python
results = detector.evaluate(test_sequences, test_labels)
print(f"Accuracy: {results['accuracy']:.2%}")
```

---

#### `get_detailed_report(sequences: List[str], labels: List[int]) -> str`

Generate a formatted evaluation report.

**Parameters:**
- `sequences`: List of symbolic sequences
- `labels`: True labels

**Returns:**
- `str`: Formatted report with confusion matrix and metrics

**Example:**
```python
print(detector.get_detailed_report(test_sequences, test_labels))
```

---

## Quick Start

### Complete Pipeline Example

```python
import pandas as pd
import numpy as np
from src.encoder import HeartbeatEncoder
from src.grammar_learner import GrammarLearner
from src.anomaly_detector import PDADetector

# 1. Load data
train_df = pd.read_csv("data/raw/mitbih_train.csv", header=None)
test_df = pd.read_csv("data/raw/mitbih_test.csv", header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# 2. Create and fit encoder on normal heartbeats
encoder = HeartbeatEncoder(n_segments=10, threshold=1.75)
normal_heartbeats = X_train[y_train == 0]
encoder.fit(normal_heartbeats)

# 3. Encode all data
train_sequences = encoder.encode_batch(X_train)
test_sequences = encoder.encode_batch(X_test)

# 4. Learn grammar from normal sequences only
normal_sequences = [train_sequences[i] for i in range(len(y_train)) if y_train[i] == 0]
grammar = GrammarLearner()
grammar.fit(normal_sequences)

# 5. Create detector and evaluate
detector = PDADetector(grammar)
results = detector.evaluate(test_sequences, y_test.astype(int).tolist())

print(f"Accuracy:  {results['accuracy']:.2%}")
print(f"Precision: {results['precision']:.2%}")
print(f"Recall:    {results['recall']:.2%}")
print(f"F1-Score:  {results['f1_score']:.4f}")
```

---

## Error Handling

All classes raise standard Python exceptions:

| Exception | Cause |
|-----------|-------|
| `ValueError` | Invalid input dimensions or parameters |
| `RuntimeError` | Method called before fit() |
| `FileNotFoundError` | Grammar file not found during load() |

---

## See Also

- [ALGORITHMS.md](ALGORITHMS.md) - Detailed algorithm explanations
- [README.md](../README.md) - Project overview
- [examples/](../examples/) - Usage examples

---

*Documentation for ECG Grammar-Based Anomaly Detection v1.0*
