"""
Anomaly Detector Module
Detector that uses the learned grammar to find abnormal heartbeats.
"""

from typing import Tuple, List, Dict

class DFADetector:
    """
    Detector that checks if heartbeat patterns match the learned grammar.

    If a pattern was seen during training, it is normal.
    If not, it is flagged as an anomaly.
    """

    def __init__(self, grammar_learner):
        self.accepted_patterns = grammar_learner.accepted_patterns
        self.main_pattern = grammar_learner.main_pattern

    def detect(self, sequence: str) -> Tuple[bool, str, Dict]:
        """
        Detect if a sequence is normal or anomalous.

        Args:
            sequence: Symbolic sequence to check

        Returns:
            Tuple of (is_normal, result_label, details)
        """
        if sequence in self.accepted_patterns:
            if sequence == self.main_pattern:
                return True, "NORMAL", {"match_type": "exact_main_pattern"}
            else:
                return True, "NORMAL", {"match_type": "variant_pattern"}
        else:
            hotspots = self._find_hotspots(sequence)
            return False, "ANOMALY", {"hotspots": hotspots}

    def _find_hotspots(self, sequence: str) -> List[Dict]:
        """Find which segments are abnormal."""
        symbols = sequence.split()
        hotspots = []
        for i, symbol in enumerate(symbols):
            if symbol.islower():
                hotspots.append({
                    "position": i,
                    "segment": f"Segment_{i+1}",
                    "symbol": symbol
                })
        return hotspots

    def predict(self, sequence: str) -> int:
        """
        Predict label for a single sequence.

        Returns:
            0 for normal, 1 for anomaly
        """
        is_normal, _, _ = self.detect(sequence)
        return 0 if is_normal else 1

    def predict_batch(self, sequences: List[str]) -> List[int]:
        """Predict labels for multiple sequences."""
        return [self.predict(seq) for seq in sequences]

    def evaluate(self, sequences: List[str], labels: List[int]) -> Dict:
        """
        Evaluate detector on test data.

        Args:
            sequences: List of symbolic sequences
            labels: List of true labels (0=normal, 1-4=abnormal classes)

        Returns:
            Dictionary with evaluation metrics
        """
        y_true = [1 if l != 0 else 0 for l in labels]
        y_pred = [0 if self.detect(seq)[0] else 1 for seq in sequences]

        TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        TN = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "confusion_matrix": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_samples": len(y_true),
            "predictions": {"normal": y_pred.count(0), "anomaly": y_pred.count(1)}
        }

    def get_detailed_report(self, sequences: List[str], labels: List[int]) -> str:
        """Generate detailed evaluation report."""
        metrics = self.evaluate(sequences, labels)
        cm = metrics['confusion_matrix']

        report = f"""
================================================================
       DFA ANOMALY DETECTOR - EVALUATION REPORT
================================================================

Confusion Matrix:
-----------------------------------------------------------------
                  | Pred Normal  | Pred Anomaly |
-----------------------------------------------------------------
Actually Normal   | {cm['TN']:>12} | {cm['FP']:>12} |
Actually Abnormal | {cm['FN']:>12} | {cm['TP']:>12} |
-----------------------------------------------------------------

Metrics:
  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
  F1-Score:  {metrics['f1_score']:.4f}

Total Samples: {metrics['total_samples']}
================================================================
"""
        return report
