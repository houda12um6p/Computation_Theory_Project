import pandas as pd
from collections import Counter
import os

RESULTS_PATH = "../results/"

print("Creating summary Excel file...")

# Load encoded data
encoded_df = pd.read_csv(RESULTS_PATH + "encoded_heartbeats.csv")

# Load sequences
with open(RESULTS_PATH + "normal_sequences.txt", 'r') as f:
    normal_seqs = [line.strip() for line in f.readlines()]

with open(RESULTS_PATH + "abnormal_sequences.txt", 'r') as f:
    abnormal_seqs = [line.strip() for line in f.readlines()]

# Create Excel file
with pd.ExcelWriter(RESULTS_PATH + "data_summary.xlsx", engine='openpyxl') as writer:

    # Sheet 1: Overview
    overview = pd.DataFrame({
        'Metric': [
            'Total Heartbeats',
            'Normal Heartbeats',
            'Abnormal Heartbeats',
            'Unique Normal Patterns',
            'Unique Abnormal Patterns',
            'Most Common Normal Pattern',
            'Most Common Abnormal Pattern'
        ],
        'Value': [
            len(encoded_df),
            len(normal_seqs),
            len(abnormal_seqs),
            len(set(normal_seqs)),
            len(set(abnormal_seqs)),
            Counter(normal_seqs).most_common(1)[0][0],
            Counter(abnormal_seqs).most_common(1)[0][0]
        ]
    })
    overview.to_excel(writer, sheet_name='Overview', index=False)
    print("  Created: Overview sheet")

    # Sheet 2: Normal Patterns
    normal_patterns = Counter(normal_seqs)
    normal_df = pd.DataFrame([
        {'Pattern': p, 'Count': c, 'Percentage': round(100*c/len(normal_seqs), 2)}
        for p, c in normal_patterns.most_common(20)
    ])
    normal_df.to_excel(writer, sheet_name='Normal_Patterns', index=False)
    print("  Created: Normal_Patterns sheet")

    # Sheet 3: Abnormal Patterns
    abnormal_patterns = Counter(abnormal_seqs)
    abnormal_df = pd.DataFrame([
        {'Pattern': p, 'Count': c, 'Percentage': round(100*c/len(abnormal_seqs), 2)}
        for p, c in abnormal_patterns.most_common(20)
    ])
    abnormal_df.to_excel(writer, sheet_name='Abnormal_Patterns', index=False)
    print("  Created: Abnormal_Patterns sheet")

    # Sheet 4: Alphabet Definition (10-segment encoding A-J)
    alphabet_df = pd.DataFrame({
        'Symbol': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        'Meaning': [
            'Normal Segment 1 (P-wave region)',
            'Normal Segment 2 (P-wave region)',
            'Normal Segment 3 (QRS onset)',
            'Normal Segment 4 (QRS onset)',
            'Normal Segment 5 (QRS peak)',
            'Normal Segment 6 (QRS peak)',
            'Normal Segment 7 (QRS end)',
            'Normal Segment 8 (QRS end)',
            'Normal Segment 9 (T-wave region)',
            'Normal Segment 10 (T-wave region)',
            'Abnormal Segment 1 (P-wave region)',
            'Abnormal Segment 2 (P-wave region)',
            'Abnormal Segment 3 (QRS onset)',
            'Abnormal Segment 4 (QRS onset)',
            'Abnormal Segment 5 (QRS peak)',
            'Abnormal Segment 6 (QRS peak)',
            'Abnormal Segment 7 (QRS end)',
            'Abnormal Segment 8 (QRS end)',
            'Abnormal Segment 9 (T-wave region)',
            'Abnormal Segment 10 (T-wave region)'
        ],
        'Type': ['Normal']*10 + ['Abnormal']*10
    })
    alphabet_df.to_excel(writer, sheet_name='Alphabet', index=False)
    print("  Created: Alphabet sheet")

print("\nSaved: results/data_summary.xlsx")
print("\nSummary complete!")
