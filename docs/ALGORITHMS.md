# Algorithm Documentation

## ECG Grammar-Based Anomaly Detection System

This document explains the algorithms we used in our ECG anomaly detection system.

## Table of Contents

1. [Overview]
2. [Symbolic Encoding Algorithm]
3. [Grammar Inference Algorithm]
4. [PDA-Based Detection Algorithm]
5. [Hotspot Analysis Algorithm]
6. [Complexity Analysis]


## Overview

The system uses formal language theory to detect cardiac anomalies:

```
Raw ECG Signal --> Symbolic Encoding --> Grammar Matching --> Anomaly Detection
     |                   |                     |                    |
  187 samples      10 symbols           CFG membership         Normal/Anomaly
```

**Key Insight:** Normal heartbeats follow predictable patterns that can be captured by a grammar. Anomalies deviate from these patterns.


## Symbolic Encoding Algorithm

### Purpose

Transform continuous ECG signals into discrete symbolic sequences suitable for grammatical analysis.

### Algorithm 1: Heartbeat Encoding

```
ALGORITHM: EncodeHeartbeat
INPUT:
  - heartbeat: array of 187 amplitude values
  - normal_mean: array of mean values (from training)
  - normal_std: array of std values (from training)
  - threshold: z-score threshold (default 1.75)
  - n_segments: number of segments (default 10)

OUTPUT: symbolic sequence (string)

1. DIVIDE heartbeat into n_segments equal parts
2. FOR each segment i = 1 to n_segments:
   a. COMPUTE segment_amplitude = mean(segment_values)
   b. COMPUTE z_score = |segment_amplitude - normal_mean[i]| / normal_std[i]
   c. IF z_score <= threshold:
        symbol = UPPERCASE_LETTER[i]  // A, B, C, ... (normal)
      ELSE:
        symbol = LOWERCASE_LETTER[i]  // a, b, c, ... (abnormal)
   d. APPEND symbol to sequence
3. RETURN sequence joined by spaces
```

### Mathematical Formulation

For segment i with amplitude x_i:

```
z_i = |x_i - mean_i| / std_i
```

Where:
- mean_i = mean amplitude of segment i across normal heartbeats
- std_i = standard deviation of segment i across normal heartbeats

Symbol assignment:
- If z_i <= threshold: symbol = UPPERCASE (normal)
- If z_i > threshold: symbol = lowercase (abnormal)

### Example

```
Input: [0.1, 0.3, 0.8, 1.0, 0.9, 0.5, 0.2, 0.1, 0.05, 0.02] (simplified)
Threshold: 1.75

Segment Analysis:
  Segment 1: amplitude=0.15, z-score=0.8  --> A (normal)
  Segment 2: amplitude=0.55, z-score=1.2  --> B (normal)
  Segment 3: amplitude=0.95, z-score=0.5  --> C (normal)
  Segment 4: amplitude=0.70, z-score=2.1  --> d (ABNORMAL)
  ...

Output: "A B C d E F G H I J"
```

### Why 10 Segments?

| Segments | Granularity | F1-Score | Recommendation |
|----------|-------------|----------|----------------|
| 5 | Low | 0.0048 | Too coarse |
| 10 | Medium | 0.1008 | Best balance |
| 20 | High | ~0.08 | Overfitting risk |

10 segments provides 21x better F1-score than 5 segments while still being easy to interpret.

---

## Grammar Inference Algorithm

### Purpose

Learn a Context-Free Grammar (CFG) that accepts all normal heartbeat patterns.

### Formal Definition

A CFG is defined as G = (V, Sigma, R, S) where:
- V = Variables (non-terminals)
- Sigma = Terminals (alphabet)
- R = Production rules
- S = Start symbol

### Algorithm 2: Grammar Learning

```
ALGORITHM: LearnGrammar
INPUT:
  - normal_sequences: list of symbolic sequences from normal heartbeats

OUTPUT: CFG G = (V, Sigma, R, S)

1. INITIALIZE:
   - accepted_patterns = empty set
   - pattern_counts = empty dictionary

2. FOR each sequence in normal_sequences:
   a. ADD sequence to accepted_patterns
   b. INCREMENT pattern_counts[sequence]

3. IDENTIFY main_pattern = most_common(pattern_counts)

4. CONSTRUCT formal grammar:
   a. V = {S, T_1, T_2, ..., T_n}  // n = number of segments
   b. Sigma = {A-J, a-j}  // uppercase and lowercase letters
   c. S = start symbol
   d. R = production rules:
      - S -> T_1 T_2 ... T_n
      - FOR each position i:
          T_i -> {all symbols seen at position i in normal patterns}

5. RETURN G = (V, Sigma, R, S), accepted_patterns
```

### Example Grammar Output

```
Context-Free Grammar G = (V, Sigma, R, S)

Variables (V): {S, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10}

Terminals (Sigma): {A, B, C, D, E, F, G, H, I, J}

Start Symbol: S

Production Rules (R):
  S -> T_1 T_2 T_3 T_4 T_5 T_6 T_7 T_8 T_9 T_10
  T_1 -> A
  T_2 -> B
  T_3 -> C
  T_4 -> D
  T_5 -> E
  T_6 -> F
  T_7 -> G
  T_8 -> H
  T_9 -> I
  T_10 -> J

Accepted Patterns: 181 unique sequences
Main Pattern: "A B C D E F G H I J" (86.7% of normal)
```

### Why CFG Over Regular Grammar?

While our current patterns could be recognized by a finite automaton, the CFG framework provides:

1. **Extensibility**: Future enhancements can capture nested patterns
2. **Formal Foundation**: Enables rigorous theoretical analysis
3. **Stack-Based Recognition**: PDA can track context across segments
4. **Academic Value**: Shows formal language theory application


## PDA-Based Detection Algorithm

### Purpose

Use a Pushdown Automaton to recognize whether a sequence belongs to the learned grammar.

### Formal PDA Definition

M = (Q, Sigma, Gamma, delta, q0, Z0, F) where:
- Q = {q0, q1, ..., qn, q_accept, q_reject}
- Sigma = Terminal alphabet
- Gamma = Stack alphabet
- delta = Transition function
- q0 = Initial state
- Z0 = Initial stack symbol
- F = Accepting states

### Algorithm 3: Anomaly Detection

```
ALGORITHM: DetectAnomaly
INPUT:
  - sequence: symbolic sequence to check
  - accepted_patterns: set of normal patterns
  - main_pattern: most common normal pattern

OUTPUT: (is_normal, label, details)

1. IF sequence IN accepted_patterns:
   a. IF sequence == main_pattern:
        RETURN (True, "NORMAL", {match_type: "exact_main_pattern"})
      ELSE:
        RETURN (True, "NORMAL", {match_type: "variant_pattern"})

2. ELSE:  // Anomaly detected
   a. hotspots = FindHotspots(sequence)
   b. RETURN (False, "ANOMALY", {hotspots: hotspots})
```

### PDA Transition Diagram

```
                    +-----------+
                    |   START   |
                    |    q_0    |
                    +-----+-----+
                          |
                          | push(Z_0)
                          v
                    +-----+-----+
                    |  PROCESS  |
              +---->|    q_1    |<----+
              |     +-----+-----+     |
              |           |           |
         read symbol, verify against grammar
              |           |           |
              |     +-----+-----+     |
              +-----|   q_i     |-----+
                    +-----+-----+
                          |
                    all symbols processed
                          |
              +-----------+-----------+
              |                       |
        pattern matched         pattern NOT matched
              |                       |
              v                       v
        +-----------+           +-----------+
        |  ACCEPT   |           |  REJECT   |
        | q_accept  |           | q_reject  |
        +-----------+           +-----------+
             |                       |
             v                       v
          NORMAL                  ANOMALY
```

### Decision Logic

```
Sequence: "A B C D E F G H I J"
Check: Is it in accepted_patterns?
  YES --> NORMAL (matches grammar)

Sequence: "a B C D E F G H I J"
Check: Is it in accepted_patterns?
  NO --> ANOMALY (lowercase 'a' indicates segment 1 deviation)
  --> Hotspot identified: Segment 1
```

## Hotspot Analysis Algorithm

### Purpose

Identify which specific segments of an anomalous heartbeat deviate from normal.

### Algorithm 4: Find Hotspots

ALGORITHM: FindHotspots
INPUT:
  - sequence: anomalous symbolic sequence

OUTPUT: list of hotspot dictionaries

1. SPLIT sequence into symbols
2. hotspots = empty list
3. FOR each symbol at position i:
   a. IF symbol is lowercase:
        hotspot = {
          position: i,
          segment: "Segment_{i+1}",
          symbol: symbol
        }
        APPEND hotspot to hotspots
4. RETURN hotspots
```

### Clinical Interpretation

| Segment | ECG Component | Clinical Meaning |
|---------|---------------|------------------|
| 1-2 | P-wave region | Atrial activity |
| 3-4 | QRS onset | Ventricular depolarization start |
| 5-6 | QRS peak (R) | Main ventricular activation |
| 7-8 | QRS end (S) | Ventricular depolarization end |
| 9-10 | T-wave region | Ventricular repolarization |

### Hotspot Patterns by Arrhythmia Type

```
Ventricular Ectopic (Class 2):
  Affects: P-wave and early QRS (segments 1-4)
  Pattern: "a b c D E F G H I J"

Supraventricular (Class 1):
  Affects: T-wave region (segments 9-10)
  Pattern: "A B C D E F G H i j"

Fusion Beat (Class 3):
  Affects: QRS complex (segments 4-7)
  Pattern: "A B C d e f g H I J"
```

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Encode single heartbeat | O(n) | n = number of segments |
| Encode batch of m heartbeats | O(m * n) | Linear in batch size |
| Learn grammar from k sequences | O(k * n) | Set operations |
| Detect single sequence | O(n) | Pattern lookup is O(1) amortized |
| Evaluate m sequences | O(m * n) | Linear in test size |

### Space Complexity

| Data Structure | Complexity | Notes |
|----------------|------------|-------|
| Encoder statistics | O(n) | Means and stds per segment |
| Accepted patterns | O(p * n) | p = unique patterns |
| Grammar rules | O(n * a) | a = alphabet size |


## Theoretical Foundations

### Theorem 1: Grammar Correctness

*The learned grammar G accepts all and only the patterns observed in normal training data.*

**Proof Sketch:**
1. Construction ensures every normal pattern is added to accepted_patterns
2. Detection accepts only patterns in accepted_patterns
3. Therefore, L(G) = {normal training patterns}

### Theorem 2: PDA Recognition

*The PDA M recognizes exactly the language L(G) defined by grammar G.*

**Proof Sketch:**
1. PDA transitions mirror grammar production rules
2. Stack tracks derivation progress
3. Acceptance occurs iff complete derivation exists
4. By CFG-PDA equivalence theorem, L(M) = L(G)

### Theorem 3: Detection Completeness

*Every input sequence receives exactly one classification: NORMAL or ANOMALY.*

**Proof:**
- Case 1: sequence in accepted_patterns -> NORMAL
- Case 2: sequence not in accepted_patterns -> ANOMALY
- Cases are mutually exclusive and exhaustive


## References

1. Chomsky, N. (1956). Three models for the description of language.
2. Hopcroft, J.E., Motwani, R., & Ullman, J.D. (2006). Introduction to Automata Theory.
3. Moody, G.B., & Mark, R.G. (2001). The impact of the MIT-BIH Arrhythmia Database.
