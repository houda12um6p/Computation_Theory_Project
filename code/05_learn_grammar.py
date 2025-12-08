import json
from collections import Counter
import os

RESULTS_PATH = "../results/"

print("=" * 50)
print("GRAMMAR INFERENCE")
print("Learning the Language of Normal Heartbeats")
print("=" * 50)

# Load normal sequences
with open(RESULTS_PATH + "normal_sequences.txt", 'r') as f:
    normal_sequences = [line.strip() for line in f.readlines()]

print(f"\nLoaded {len(normal_sequences)} normal sequences")

# Count all patterns
pattern_counts = Counter(normal_sequences)
print(f"Found {len(pattern_counts)} unique patterns")

# Find the dominant pattern (this becomes our grammar's main rule)
most_common = pattern_counts.most_common(1)[0]
main_pattern = most_common[0]
main_count = most_common[1]
main_pct = 100 * main_count / len(normal_sequences)

print(f"\nMain pattern: '{main_pattern}'")
print(f"Appears {main_count} times ({main_pct:.1f}%)")

# Build the grammar
print("\n" + "=" * 50)
print("LEARNED CONTEXT-FREE GRAMMAR")
print("=" * 50)

# Define the grammar formally with 10-segment encoding (A-J)
grammar = {
    "name": "ECG_Normal_Grammar",
    "description": "Grammar learned from normal ECG heartbeats using 10-segment encoding",

    # Formal definition: 10 segments mapped to letters A-J
    "terminals": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    "non_terminals": ["S", "T_1", "T_2", "T_3", "T_4", "T_5", "T_6", "T_7", "T_8", "T_9", "T_10"],
    "start_symbol": "S",

    # Production rules for 10-segment CFG
    "production_rules": {
        "S": ["T_1 T_2 T_3 T_4 T_5 T_6 T_7 T_8 T_9 T_10"],
        "T_1": ["A", "a"],
        "T_2": ["B", "b"],
        "T_3": ["C", "c"],
        "T_4": ["D", "d"],
        "T_5": ["E", "e"],
        "T_6": ["F", "f"],
        "T_7": ["G", "g"],
        "T_8": ["H", "h"],
        "T_9": ["I", "i"],
        "T_10": ["J", "j"]
    },

    # Learned patterns (what we accept as "normal")
    "accepted_patterns": list(pattern_counts.keys()),
    "main_pattern": main_pattern,
    "pattern_counts": dict(pattern_counts.most_common(20)),

    # Statistics
    "total_training_samples": len(normal_sequences),
    "unique_patterns": len(pattern_counts),
    "main_pattern_coverage": round(main_pct, 2)
}

# Print the formal grammar
print("\n1. ALPHABET (Terminal Symbols):")
print(f"   Sigma = {{{', '.join(grammar['terminals'])}}}")

print("\n2. NON-TERMINAL SYMBOLS:")
print(f"   V = {{{', '.join(grammar['non_terminals'])}}}")

print("\n3. START SYMBOL:")
print(f"   S = {grammar['start_symbol']}")

print("\n4. PRODUCTION RULES:")
for lhs, rhs_list in grammar['production_rules'].items():
    for rhs in rhs_list:
        print(f"   {lhs} -> {rhs}")

print("\n5. ACCEPTED PATTERNS (Top 10):")
for pattern, count in pattern_counts.most_common(10):
    pct = 100 * count / len(normal_sequences)
    print(f"   '{pattern}' : {count} ({pct:.1f}%)")

# Save grammar to JSON
grammar_file = RESULTS_PATH + "learned_grammar.json"
with open(grammar_file, 'w') as f:
    json.dump(grammar, f, indent=2)

print(f"\nGrammar saved to: {grammar_file}")

print("\n" + "=" * 50)
print("GRAMMAR LEARNING COMPLETE!")
print("=" * 50)

print("\nInterpretation:")
print(f"  - The grammar accepts {len(grammar['accepted_patterns'])} different patterns")
print(f"  - The main pattern '{main_pattern}' covers {main_pct:.1f}% of normal heartbeats")
print(f"  - Any sequence NOT in accepted_patterns is considered ANOMALOUS")
