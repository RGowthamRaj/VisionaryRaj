"""
Model Training and Cross-Session Generalization Benchmark
=========================================================
Trains and compares RandomForestClassifier on:
1. Raw Landmark Features
2. Invariant (Translation + Scale) Landmark Features

Evaluates:
- Same-Session Accuracy (held-out split from train_data.csv)
- Cross-Session Accuracy (test_data.csv collected under altered environment)

Prints 2x2 comparison table and saves final Invariant model to gesture_model.pkl.
"""

import csv
import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from features import GestureFeatureExtractor


def load_dataset(csv_path):
    """Loads CSV and returns feature matrix X and label vector y."""
    if not os.path.exists(csv_path):
        print(f"Error: Dataset '{csv_path}' not found.")
        sys.exit(1)

    labels = []
    features = []
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            labels.append(int(float(row[0])))
            features.append([float(val) for val in row[1:]])

    y = np.array(labels, dtype=np.int64)
    X = np.array(features, dtype=np.float32)
    return X, y


def main():
    train_path = "train_data.csv"
    test_path = "test_data.csv"
    output_model_path = "gesture_model.pkl"

    print("=" * 65)
    print(" GESTURE RECOGNITION: MODEL TRAINING & BENCHMARK")
    print("=" * 65)

    # 1. Load Datasets
    print(f"\n[1/5] Loading datasets from '{train_path}' and '{test_path}'...")
    X_train_raw_full, y_train_full = load_dataset(train_path)
    X_cross_raw, y_cross = load_dataset(test_path)

    print(f" - Train Dataset (Session A): {X_train_raw_full.shape[0]} samples across classes {sorted(set(y_train_full))}")
    print(f" - Test Dataset  (Session B): {X_cross_raw.shape[0]} samples across classes {sorted(set(y_cross))}")

    # 2. Extract Invariant Features
    print("\n[2/5] Extracting Translation & Scale Invariant Features...")
    extractor = GestureFeatureExtractor()

    X_train_inv_full = extractor.extract_invariant_batch(X_train_raw_full)
    X_cross_inv = extractor.extract_invariant_batch(X_cross_raw)

    # 3. Create Same-Session Train/Validation Split (80% Train, 20% Same-Session Test)
    # Using fixed seed for reproducible benchmark
    random_state = 42
    indices = np.arange(len(y_train_full))
    train_idx, same_test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_state,
        stratify=y_train_full
    )

    y_train_split = y_train_full[train_idx]
    y_same_test = y_train_full[same_test_idx]

    # Raw splits
    X_tr_raw = X_train_raw_full[train_idx]
    X_same_raw = X_train_raw_full[same_test_idx]

    # Invariant splits
    X_tr_inv = X_train_inv_full[train_idx]
    X_same_inv = X_train_inv_full[same_test_idx]

    print(f" - Same-Session Train Split: {len(train_idx)} samples")
    print(f" - Same-Session Test Split:  {len(same_test_idx)} samples")
    print(f" - Cross-Session Test Set:   {len(y_cross)} samples")

    # 4. Train & Evaluate Models
    print("\n[3/5] Training Random Forest Classifiers...")
    rf_params = {
        "n_estimators": 100,
        "max_depth": 15,
        "random_state": random_state,
        "n_jobs": -1
    }

    # Model A: Raw Features
    rf_raw = RandomForestClassifier(**rf_params)
    rf_raw.fit(X_tr_raw, y_train_split)
    acc_raw_same = accuracy_score(y_same_test, rf_raw.predict(X_same_raw))
    acc_raw_cross = accuracy_score(y_cross, rf_raw.predict(X_cross_raw))

    # Model B: Invariant Features
    rf_inv = RandomForestClassifier(**rf_params)
    rf_inv.fit(X_tr_inv, y_train_split)
    acc_inv_same = accuracy_score(y_same_test, rf_inv.predict(X_same_inv))
    acc_inv_cross = accuracy_score(y_cross, rf_inv.predict(X_cross_inv))

    # 5. Print 2x2 Accuracy Table
    print("\n" + "=" * 65)
    print("            CRITICAL: 2x2 ACCURACY BENCHMARK TABLE")
    print("=" * 65)
    header_str = f"{'Feature Set':<15} | {'[Same Session] Accuracy':<23} | {'[Cross Session] Accuracy':<24}"
    print(header_str)
    print("-" * len(header_str))
    print(f"{'[Raw]':<15} | {acc_raw_same * 100:>20.2f}% | {acc_raw_cross * 100:>21.2f}%")
    print(f"{'[Invariant]':<15} | {acc_inv_same * 100:>20.2f}% | {acc_inv_cross * 100:>21.2f}%")
    print("=" * 65)

    delta_cross = (acc_inv_cross - acc_raw_cross) * 100
    if delta_cross >= 0:
        print(f"\n[INSIGHT] Invariant features improved cross-session generalization by +{delta_cross:.2f}% points!")
    else:
        print(f"\n[INSIGHT] Cross-session accuracy difference: {delta_cross:.2f}% points.")

    # Detailed Cross-Session Classification Reports
    print("\n[4/5] Cross-Session Classification Breakdown:")
    print("--- [Raw Features] Cross-Session Report ---")
    print(classification_report(y_cross, rf_raw.predict(X_cross_raw), digits=4, zero_division=0))
    print("--- [Invariant Features] Cross-Session Report ---")
    print(classification_report(y_cross, rf_inv.predict(X_cross_inv), digits=4, zero_division=0))

    # 6. Train Final Invariant Model on FULL Training Set and Save
    print(f"\n[5/5] Retraining Invariant model on full training set ({len(y_train_full)} samples)...")
    final_inv_model = RandomForestClassifier(**rf_params)
    final_inv_model.fit(X_train_inv_full, y_train_full)

    # Save model
    joblib.dump(final_inv_model, output_model_path)
    print(f" -> Model successfully saved to '{output_model_path}'")
    print("=" * 65)


if __name__ == "__main__":
    main()
