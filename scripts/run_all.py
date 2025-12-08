#!/usr/bin/env python3
"""
Master Script: Run Complete ECG Grammar-Based Anomaly Detection Pipeline
=========================================================================
Course: Computational Theory - Fall 2025
Institution: Mohammed VI Polytechnic University (UM6P)
"""

import os
import sys
import time

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_subheader(text):
    print(f"\n>>> {text}")
    print("-" * 50)

def main():
    print_header("ECG ANOMALY DETECTION USING GRAMMAR INFERENCE")
    print("Formal Model: Context-Free Grammar (CFG)")
    print("Recognition: Pushdown Automaton (PDA)")
    print("-" * 70)

    # Change to code directory
    code_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'code')
    os.chdir(code_dir)
    print(f"Working directory: {os.getcwd()}")

    scripts = [
        ("01_verify_data.py", "Verifying Dataset"),
        ("02_visualize_data.py", "Creating Visualizations"),
        ("03_encode_heartbeats.py", "Encoding Heartbeats to Symbols"),
        ("04_create_summary.py", "Creating Data Summary"),
        ("05_learn_grammar.py", "Learning Grammar from Normal Heartbeats"),
        ("06_anomaly_detector.py", "Building Anomaly Detector"),
        ("07_evaluate.py", "Evaluating on Test Data"),
        ("08_improved_detector.py", "Optimizing Threshold"),
        ("09_create_visualizations.py", "Creating Result Visualizations"),
        ("10_hotspot_analysis.py", "Analyzing Anomaly Hotspots"),
        ("11_improved_encoding.py", "Testing 10-Segment Encoding"),
    ]

    start_time = time.time()
    completed = 0

    for script, description in scripts:
        if not os.path.exists(script):
            print(f"  [SKIP] {script} not found")
            continue

        print_subheader(f"Running: {description}")
        result = os.system(f"python {script}")

        if result != 0:
            print(f"  [ERROR] {script} failed with code {result}")
            print("  Continuing with next script...")
        else:
            print(f"  [OK] {description} complete")
            completed += 1

    elapsed = time.time() - start_time

    print_header("PIPELINE COMPLETE")
    print(f"Scripts executed: {completed}/{len(scripts)}")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

    print("\n" + "=" * 70)
    print("  OUTPUT SUMMARY")
    print("=" * 70)
    print("""
    results/
    ├── figures/           - All visualizations (PNG)
    ├── reports/           - Excel reports
    ├── metrics/           - JSON evaluation results
    └── processed/         - Encoded sequences & grammar

    Key Files:
    - FINAL_REPORT.xlsx    - Comprehensive analysis report
    - learned_grammar.json - The learned CFG
    - hotspot_heatmap.png  - Anomaly location analysis
    """)

    print("\nReady for paper writing!")
    print("=" * 70)

if __name__ == "__main__":
    main()
