import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
DATA_PATH = "../data/raw/"
RESULTS_PATH = "../results/"

print("Loading data...")
train_df = pd.read_csv(DATA_PATH + "mitbih_train.csv", header=None)
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values

label_names = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown"
}

print("Creating heartbeat visualization...")

# Plot one example of each class
fig, axes = plt.subplots(5, 1, figsize=(12, 10))

for label in range(5):
    idx = np.where(y == label)[0][0]
    heartbeat = X[idx]

    axes[label].plot(heartbeat, linewidth=0.8, color='blue')
    axes[label].set_title(f"Class {label}: {label_names[label]}", fontsize=12)
    axes[label].set_ylabel("Amplitude")
    axes[label].grid(True, alpha=0.3)

axes[4].set_xlabel("Sample Point")
plt.tight_layout()
plt.savefig(RESULTS_PATH + "sample_heartbeats.png", dpi=150)
plt.close()

print("Saved: results/sample_heartbeats.png")

# Plot label distribution
print("Creating label distribution chart...")

fig, ax = plt.subplots(figsize=(10, 6))
labels_unique, counts = np.unique(y, return_counts=True)
colors = ['green', 'orange', 'red', 'purple', 'gray']
bars = ax.bar([label_names[int(l)] for l in labels_unique], counts, color=colors)

ax.set_ylabel("Count")
ax.set_title("Distribution of Heartbeat Types in Training Data")

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{count}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(RESULTS_PATH + "label_distribution.png", dpi=150)
plt.close()

print("Saved: results/label_distribution.png")
print("\nVisualization complete!")
