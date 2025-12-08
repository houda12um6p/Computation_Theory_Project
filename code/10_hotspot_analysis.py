import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

DATA_PATH = "../data/raw/"
RESULTS_PATH = "../results/"

print("=" * 50)
print("HOTSPOT ANALYSIS")
print("Where Do Abnormalities Occur?")
print("=" * 50)

# Load data
train_df = pd.read_csv(DATA_PATH + "mitbih_train.csv", header=None)
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values

# Load normal statistics
normal_mean = np.load(RESULTS_PATH + "normal_mean.npy")
normal_std = np.load(RESULTS_PATH + "normal_std.npy")

# Encoding function that returns list
def encode_heartbeat_list(heartbeat, normal_mean, normal_std, threshold=1.75):
    symbols = []
    n_segments = 10
    segment_size = len(heartbeat) // n_segments
    segment_names = [chr(65 + i) for i in range(n_segments)]  # A, B, C, ..., J

    for i, name in enumerate(segment_names):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else len(heartbeat)
        segment = heartbeat[start:end]
        seg_mean = normal_mean[start:end]
        seg_std = normal_std[start:end]

        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.abs((segment - seg_mean) / (seg_std + 1e-10))
        mean_z = np.nanmean(z_scores)

        if mean_z < threshold:
            symbols.append(name)
        else:
            symbols.append(name.lower())

    return symbols

# Analyze hotspots for each class
label_names = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown"
}

segment_names = ['Seg-A', 'Seg-B', 'Seg-C', 'Seg-D', 'Seg-E', 'Seg-F', 'Seg-G', 'Seg-H', 'Seg-I', 'Seg-J']

print("\nAnalyzing abnormality locations by class...")

# Create hotspot matrix: rows = classes (0-4), cols = segments (0-9)
hotspot_matrix = np.zeros((5, 10))  # 5 classes, 10 segments

for label in range(5):
    class_X = X[y == label]
    print(f"  Processing Class {label} ({label_names[label]}): {len(class_X)} samples")

    for heartbeat in class_X:
        symbols = encode_heartbeat_list(heartbeat, normal_mean, normal_std)
        for seg_idx, symbol in enumerate(symbols):
            if symbol.islower():  # Abnormal segment
                hotspot_matrix[label, seg_idx] += 1

    # Convert to percentage
    if len(class_X) > 0:
        hotspot_matrix[label] = 100 * hotspot_matrix[label] / len(class_X)

# Print results
print("\n" + "=" * 50)
print("HOTSPOT RESULTS")
print("Percentage of each class showing abnormality in each segment")
print("=" * 50)

header = "{:<15}" + " {:>7}" * 10
print(header.format("Class", *[f"Seg-{chr(65+i)}" for i in range(10)]))
print("-" * 95)

for label in range(5):
    row = "{:<15}" + " {:>6.1f}%" * 10
    print(row.format(label_names[label], *hotspot_matrix[label]))

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(hotspot_matrix,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=segment_names,
            yticklabels=[label_names[i] for i in range(5)],
            cbar_kws={'label': 'Percentage (%)'})

plt.title('ECG Anomaly Hotspot Map\nPercentage of Heartbeats with Abnormal Segments by Class', fontsize=14)
plt.xlabel('ECG Segment', fontsize=12)
plt.ylabel('Heartbeat Class', fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_PATH + "hotspot_heatmap.png", dpi=150)
plt.close()
print("\nSaved: hotspot_heatmap.png")

# Create bar chart comparison
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for i, segment in enumerate(segment_names):
    values = hotspot_matrix[:, i]
    colors = ['green', 'orange', 'red', 'purple', 'gray']
    axes[i].bar([label_names[j] for j in range(5)], values, color=colors)
    axes[i].set_title(f'{segment} Abnormality Rate')
    axes[i].set_ylabel('Percentage (%)')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(RESULTS_PATH + "hotspot_by_segment.png", dpi=150)
plt.close()
print("Saved: hotspot_by_segment.png")

# Save hotspot data
hotspot_df = pd.DataFrame(hotspot_matrix,
                          columns=segment_names,
                          index=[label_names[i] for i in range(5)])
hotspot_df.to_csv(RESULTS_PATH + "hotspot_data.csv")
print("Saved: hotspot_data.csv")

print("\n" + "=" * 50)
print("HOTSPOT ANALYSIS COMPLETE!")
print("=" * 50)

# Key findings
print("\nKey Findings:")
for label in range(5):
    max_seg = np.argmax(hotspot_matrix[label])
    max_val = hotspot_matrix[label, max_seg]
    print(f"  {label_names[label]}: Highest abnormality in {segment_names[max_seg]} ({max_val:.1f}%)")
