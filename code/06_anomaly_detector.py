import json
import os

RESULTS_PATH = "../results/"

print("=" * 50)
print("BUILDING ANOMALY DETECTOR")
print("Deterministic Finite Automaton for ECG Classification")
print("=" * 50)

class ECGAnomalyDetector:
    """
    Anomaly detector based on learned grammar.
    Works like a DFA: accepts sequences in the grammar, rejects others.
    """

    def __init__(self, grammar_path):
        # Load grammar
        with open(grammar_path, 'r') as f:
            self.grammar = json.load(f)

        self.accepted_patterns = set(self.grammar['accepted_patterns'])
        self.main_pattern = self.grammar['main_pattern']

        print(f"Loaded grammar: {self.grammar['name']}")
        print(f"Accepted patterns: {len(self.accepted_patterns)}")
        print(f"Main pattern: '{self.main_pattern}'")

    def is_accepted(self, sequence):
        """
        Check if sequence is accepted by the grammar.
        Returns: (accepted, reason)
        """
        if sequence in self.accepted_patterns:
            if sequence == self.main_pattern:
                return True, "PERFECT_MATCH"
            else:
                return True, "VARIANT_MATCH"
        else:
            return False, "REJECTED"

    def detect(self, sequence):
        """
        Detect if a heartbeat is normal or anomalous.
        Returns: (is_normal, confidence, details)
        """
        accepted, reason = self.is_accepted(sequence)

        if accepted:
            # Calculate confidence based on how common this pattern is
            pattern_counts = self.grammar['pattern_counts']
            if sequence in pattern_counts:
                count = pattern_counts[sequence]
                total = self.grammar['total_training_samples']
                confidence = count / total
            else:
                confidence = 0.01

            return True, confidence, reason
        else:
            # Find how different from main pattern
            similarity = self._similarity(sequence, self.main_pattern)
            return False, 1 - similarity, "ANOMALY_DETECTED"

    def _similarity(self, seq1, seq2):
        """Calculate similarity between two sequences"""
        symbols1 = seq1.split()
        symbols2 = seq2.split()

        if len(symbols1) != len(symbols2):
            return 0.0

        matches = sum(1 for a, b in zip(symbols1, symbols2) if a == b)
        return matches / len(symbols1)

    def find_anomaly_location(self, sequence):
        """Find which segments are abnormal (hotspots) using 10-segment encoding"""
        symbols = sequence.split()
        # 10 segments with clinical interpretation
        segment_names = [
            'Segment-1 (P-wave)', 'Segment-2 (P-wave)',
            'Segment-3 (QRS onset)', 'Segment-4 (QRS onset)',
            'Segment-5 (QRS peak)', 'Segment-6 (QRS peak)',
            'Segment-7 (QRS end)', 'Segment-8 (QRS end)',
            'Segment-9 (T-wave)', 'Segment-10 (T-wave)'
        ]

        anomalies = []
        for i, symbol in enumerate(symbols):
            if symbol.islower():  # Lowercase = abnormal
                anomalies.append({
                    'position': i,
                    'segment': segment_names[i] if i < len(segment_names) else f'Segment-{i+1}',
                    'symbol': symbol
                })

        return anomalies


# Create the detector
print("\n" + "-" * 50)
detector = ECGAnomalyDetector(RESULTS_PATH + "learned_grammar.json")

# Test with examples
print("\n" + "=" * 50)
print("TESTING THE DETECTOR")
print("=" * 50)

test_cases = [
    ("A B C D E F G H I J", "All normal segments"),
    ("a B C D E F G H I J", "Abnormal Segment 1"),
    ("A B C D E F G H I j", "Abnormal Segment 10"),
    ("a b c d e f g h i j", "All abnormal segments"),
    ("A b C d E f G h I j", "Mixed abnormalities"),
    ("A B C D E F G H I J", "Normal (duplicate test)"),
]

print("\n{:<25} {:<25} {:<10} {:<10}".format("Sequence", "Description", "Result", "Confidence"))
print("-" * 75)

for sequence, description in test_cases:
    is_normal, confidence, details = detector.detect(sequence)
    result = "NORMAL" if is_normal else "ANOMALY"
    print("{:<25} {:<25} {:<10} {:.2%}".format(sequence, description, result, confidence))

    if not is_normal:
        hotspots = detector.find_anomaly_location(sequence)
        if hotspots:
            spots = ", ".join([h['segment'] for h in hotspots])
            print(f"                           ^ Hotspots: {spots}")

print("\n" + "=" * 50)
print("DETECTOR READY!")
print("=" * 50)
