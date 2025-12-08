import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

RESULTS_PATH = "../results/"

print("=" * 50)
print("CREATING VISUALIZATIONS")
print("=" * 50)

# Load threshold comparison results
threshold_df = pd.read_csv(RESULTS_PATH + "threshold_comparison.csv")

# Plot 1: Metrics vs Threshold
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Accuracy
axes[0, 0].plot(threshold_df['threshold'], threshold_df['accuracy'] * 100, 'b-o', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Threshold')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].set_title('Accuracy vs Threshold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([80, 85])

# Precision
axes[0, 1].plot(threshold_df['threshold'], threshold_df['precision'] * 100, 'g-o', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Threshold')
axes[0, 1].set_ylabel('Precision (%)')
axes[0, 1].set_title('Precision vs Threshold')
axes[0, 1].grid(True, alpha=0.3)

# Recall
axes[1, 0].plot(threshold_df['threshold'], threshold_df['recall'] * 100, 'r-o', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Threshold')
axes[1, 0].set_ylabel('Recall (%)')
axes[1, 0].set_title('Recall vs Threshold')
axes[1, 0].grid(True, alpha=0.3)

# F1-Score
axes[1, 1].plot(threshold_df['threshold'], threshold_df['f1'], 'm-o', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].set_title('F1-Score vs Threshold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH + "metrics_vs_threshold.png", dpi=150)
plt.close()
print("Saved: metrics_vs_threshold.png")

# Plot 2: Number of Patterns vs Threshold
plt.figure(figsize=(10, 6))
plt.bar(threshold_df['threshold'].astype(str), threshold_df['patterns'], color='steelblue')
plt.xlabel('Threshold')
plt.ylabel('Number of Accepted Patterns')
plt.title('Grammar Size vs Threshold\n(Number of patterns accepted as "normal")')
plt.grid(True, alpha=0.3, axis='y')
plt.savefig(RESULTS_PATH + "patterns_vs_threshold.png", dpi=150)
plt.close()
print("Saved: patterns_vs_threshold.png")

# Plot 3: Confusion Matrix Heatmap (for best threshold)
with open(RESULTS_PATH + "best_threshold_results.json", 'r') as f:
    best = json.load(f)

cm = np.array([[best['TN'], best['FP']],
               [best['FN'], best['TP']]])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted\nNormal', 'Predicted\nAbnormal'])
ax.set_yticklabels(['Actually\nNormal', 'Actually\nAbnormal'])

for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                      fontsize=16, color='white' if cm[i, j] > cm.max()/2 else 'black')

ax.set_title(f'Confusion Matrix (Threshold = {best["threshold"]})\nGrammar-Based ECG Anomaly Detection')
plt.colorbar(im)
plt.tight_layout()
plt.savefig(RESULTS_PATH + "confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# Plot 4: Summary Bar Chart
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [best['accuracy']*100, best['precision']*100, best['recall']*100, best['f1']*100]
colors = ['blue', 'green', 'red', 'purple']

bars = plt.bar(metrics, values, color=colors)
plt.ylabel('Percentage / Score')
plt.title(f'Detection Performance Summary (Threshold = {best["threshold"]})')
plt.ylim([0, 110])

for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val:.2f}%', ha='center', fontsize=12)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(RESULTS_PATH + "performance_summary.png", dpi=150)
plt.close()
print("Saved: performance_summary.png")

print("\n" + "=" * 50)
print("ALL VISUALIZATIONS CREATED!")
print("=" * 50)
