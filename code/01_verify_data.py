import pandas as pd
import numpy as np
import os

# Paths
DATA_PATH = "../data/raw/"

# Load data
print("=" * 50)
print("LOADING DATA")
print("=" * 50)

train_df = pd.read_csv(DATA_PATH + "mitbih_train.csv", header=None)
test_df = pd.read_csv(DATA_PATH + "mitbih_test.csv", header=None)

print(f"Training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# Label names
label_names = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown"
}

# Count labels in training
print("\n" + "=" * 50)
print("LABEL DISTRIBUTION (Training)")
print("=" * 50)

labels = train_df.iloc[:, -1]
for label in sorted(labels.unique()):
    count = (labels == label).sum()
    pct = 100 * count / len(labels)
    name = label_names[int(label)]
    print(f"  Class {int(label)} ({name}): {count} ({pct:.1f}%)")

print("\n" + "=" * 50)
print("SUCCESS! Data verified.")
print("=" * 50)
