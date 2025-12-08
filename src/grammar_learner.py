"""
Grammar Inference Module
Learns Context-Free Grammar from normal ECG sequences.
"""

from collections import Counter
from typing import Set, Dict, List
import json

class GrammarLearner:
    """
    This class builds a grammar from training data.

    It collects all the patterns seen in normal heartbeats and stores them.
    """

    def __init__(self):
        self.accepted_patterns: Set[str] = set()
        self.pattern_counts: Counter = Counter()
        self.main_pattern: str = ""
        self.grammar: Dict = {}

    def fit(self, sequences: List[str]) -> 'GrammarLearner':
        """
        Learn grammar from normal sequences.

        Args:
            sequences: List of symbolic sequences from normal heartbeats

        Returns:
            self
        """
        self.pattern_counts = Counter(sequences)
        self.accepted_patterns = set(sequences)
        self.main_pattern = self.pattern_counts.most_common(1)[0][0]

        # Build formal grammar representation
        self.grammar = {
            "name": "ECG_Normal_Grammar",
            "terminals": list(set(''.join(sequences).replace(' ', ''))),
            "non_terminals": ["S", "HEARTBEAT"],
            "start_symbol": "S",
            "production_rules": {
                "S": ["HEARTBEAT"],
                "HEARTBEAT": [self.main_pattern]
            },
            "accepted_patterns": list(self.accepted_patterns),
            "main_pattern": self.main_pattern,
            "pattern_counts": dict(self.pattern_counts.most_common(20)),
            "total_training_samples": len(sequences),
            "unique_patterns": len(self.accepted_patterns)
        }

        return self

    def save(self, filepath: str):
        """Save grammar to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.grammar, f, indent=2)

    def load(self, filepath: str) -> 'GrammarLearner':
        """Load grammar from JSON file."""
        with open(filepath, 'r') as f:
            self.grammar = json.load(f)
        self.accepted_patterns = set(self.grammar['accepted_patterns'])
        self.main_pattern = self.grammar['main_pattern']
        self.pattern_counts = Counter(self.grammar.get('pattern_counts', {}))
        return self

    def get_formal_definition(self) -> str:
        """Return formal grammar definition as string."""
        coverage = 100 * self.pattern_counts[self.main_pattern] / self.grammar['total_training_samples']
        return f"""
Context-Free Grammar: {self.grammar['name']}
============================================
1. Alphabet (Sigma): {{{', '.join(sorted(self.grammar['terminals']))}}}
2. Non-terminals (V): {{{', '.join(self.grammar['non_terminals'])}}}
3. Start Symbol: {self.grammar['start_symbol']}
4. Production Rules:
   S -> HEARTBEAT
   HEARTBEAT -> {self.main_pattern}
5. Accepted Patterns: {self.grammar['unique_patterns']} patterns
6. Main Pattern Coverage: {coverage:.1f}%
"""

    def get_pattern_statistics(self) -> Dict:
        """Get statistics about learned patterns."""
        total = self.grammar['total_training_samples']
        return {
            'total_samples': total,
            'unique_patterns': len(self.accepted_patterns),
            'main_pattern': self.main_pattern,
            'main_pattern_count': self.pattern_counts[self.main_pattern],
            'main_pattern_coverage': self.pattern_counts[self.main_pattern] / total,
            'top_10_patterns': self.pattern_counts.most_common(10)
        }

    def is_valid_sequence(self, sequence: str) -> bool:
        """Check if sequence is accepted by grammar."""
        return sequence in self.accepted_patterns
