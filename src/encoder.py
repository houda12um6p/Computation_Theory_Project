"""
ECG Heartbeat Encoder Module
Converts raw ECG signals into letter sequences for grammar analysis.
"""

import numpy as np
from typing import List, Tuple

class HeartbeatEncoder:
    """
    Converts raw ECG signals into letter sequences for grammar analysis.

    Each heartbeat gets divided into segments and each segment becomes
    a letter (uppercase if normal, lowercase if abnormal based on z-score).
    """

    def __init__(self, n_segments: int = 10, threshold: float = 1.75):
        self.n_segments = n_segments
        self.threshold = threshold
        self.normal_mean = None
        self.normal_std = None
        self._alphabet = self._generate_alphabet()

    def _generate_alphabet(self) -> List[str]:
        """Generate alphabet based on number of segments."""
        return [chr(65 + i) for i in range(self.n_segments)]  # A, B, C, ...

    def fit(self, normal_heartbeats: np.ndarray) -> 'HeartbeatEncoder':
        """
        Fit encoder on normal heartbeats to learn statistics.

        Args:
            normal_heartbeats: Array of shape (n_samples, n_timepoints)

        Returns:
            self
        """
        self.normal_mean = np.mean(normal_heartbeats, axis=0)
        self.normal_std = np.std(normal_heartbeats, axis=0)
        return self

    def encode(self, heartbeat: np.ndarray) -> str:
        """
        Encode a single heartbeat into a symbolic sequence.

        Args:
            heartbeat: Array of shape (n_timepoints,)

        Returns:
            Symbolic sequence string (e.g., "A B C D E F G H I J")
        """
        if self.normal_mean is None:
            raise ValueError("Encoder must be fitted before encoding.")

        symbols = []
        segment_size = len(heartbeat) // self.n_segments

        for i, letter in enumerate(self._alphabet):
            start = i * segment_size
            end = start + segment_size if i < self.n_segments - 1 else len(heartbeat)

            segment = heartbeat[start:end]
            seg_mean = self.normal_mean[start:end]
            seg_std = self.normal_std[start:end]

            with np.errstate(divide='ignore', invalid='ignore'):
                z_scores = np.abs((segment - seg_mean) / (seg_std + 1e-10))
            mean_z = np.nanmean(z_scores)

            if mean_z < self.threshold:
                symbols.append(letter)  # Normal (uppercase)
            else:
                symbols.append(letter.lower())  # Abnormal (lowercase)

        return ' '.join(symbols)

    def encode_batch(self, heartbeats: np.ndarray) -> List[str]:
        """Encode multiple heartbeats."""
        return [self.encode(hb) for hb in heartbeats]

    def get_segment_names(self) -> List[str]:
        """Get human-readable segment names."""
        if self.n_segments == 5:
            return ['P-wave', 'Q-wave', 'R-wave', 'S-wave', 'T-wave']
        elif self.n_segments == 10:
            return ['P1', 'P2', 'Q1', 'Q2', 'R1', 'R2', 'S1', 'S2', 'T1', 'T2']
        else:
            return [f'Segment_{i+1}' for i in range(self.n_segments)]

    def save_statistics(self, filepath: str):
        """Save learned statistics to numpy file."""
        np.savez(filepath,
                 normal_mean=self.normal_mean,
                 normal_std=self.normal_std,
                 n_segments=self.n_segments,
                 threshold=self.threshold)

    def load_statistics(self, filepath: str) -> 'HeartbeatEncoder':
        """Load statistics from numpy file."""
        data = np.load(filepath)
        self.normal_mean = data['normal_mean']
        self.normal_std = data['normal_std']
        self.n_segments = int(data['n_segments'])
        self.threshold = float(data['threshold'])
        self._alphabet = self._generate_alphabet()
        return self
