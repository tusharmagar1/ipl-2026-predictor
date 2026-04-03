"""
run_all.py — Run the Entire IPL 2026 Prediction Pipeline
==========================================================
This runs all 4 steps in sequence.

Usage:
    python run_all.py
"""

import subprocess
import sys
import os

steps = [
    ("step1_explore.py",  "Step 1: Explore Data"),
    ("step2_features.py", "Step 2: Feature Engineering"),
    ("step3_train.py",    "Step 3: Train Model"),
    ("step4_predict.py",  "Step 4: Predict 2026 Winner"),
]

print("\n" + "🏏 " * 20)
print("  IPL 2026 WINNER PREDICTION — Full Pipeline")
print("🏏 " * 20 + "\n")

for script, label in steps:
    print(f"\n{'─'*60}")
    print(f"  ▶  Running {label}...")
    print(f"{'─'*60}\n")

    result = subprocess.run([sys.executable, script], capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ {label} failed. Fix errors above and retry.")
        sys.exit(1)

print("\n" + "🎉 " * 20)
print("  ALL STEPS COMPLETE! Check the .png files for charts.")
print("🎉 " * 20 + "\n")
