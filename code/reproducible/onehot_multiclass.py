#!/usr/bin/env python3
"""One-hot encoding demo — multiclass UCI (5 EEG conditions).

EEG features are continuous, so one-hot encoding applies to the categorical TARGET,
not the inputs. The UCI Epileptic Seizure Recognition label has 5 classes:
  1 = seizure activity
  2 = tumour-area EEG          3 = healthy-area EEG
  4 = eyes-closed             5 = eyes-open    (all non-seizure)

This script one-hot encodes the 5-class label (sklearn OneHotEncoder), trains a
multiclass classifier under stratified 5-fold CV, and reports overall + per-class
accuracy. It documents where one-hot legitimately enters the EEG->AI pipeline.

No fabricated numbers (Sec. 57.7). Run:
    UCI_CSV="/path/Epileptic Seizure Recognition.csv" \
    python code/reproducible/onehot_multiclass.py

Output: accuracy/onehot_multiclass.json
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[2]
UCI_CSV = os.environ.get("UCI_CSV", sys.argv[1] if len(sys.argv) > 1 else "data/Epileptic Seizure Recognition.csv")
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {1: "seizure", 2: "tumour_area", 3: "healthy_area", 4: "eyes_closed", 5: "eyes_open"}


def main():
    import pandas as pd
    df = pd.read_csv(UCI_CSV)
    X = df.iloc[:, 1:-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)          # 5 classes

    # --- one-hot encode the categorical label (demonstration) ---
    enc = OneHotEncoder(sparse_output=False)
    Y_onehot = enc.fit_transform(y.reshape(-1, 1))
    onehot_info = {"categories": [int(c) for c in enc.categories_[0]],
                   "onehot_shape": list(Y_onehot.shape),
                   "example_label": int(y[0]),
                   "example_onehot": Y_onehot[0].astype(int).tolist()}

    # --- multiclass classification under stratified 5-fold ---
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(sc.transform(X[tr]), y[tr])
        y_pred_all.append(clf.predict(sc.transform(X[te])))
        y_true_all.append(y[te])
    yt = np.concatenate(y_true_all); yp = np.concatenate(y_pred_all)

    rep = classification_report(yt, yp, output_dict=True, zero_division=0)
    per_class = {CLASS_NAMES[int(c)]: round(rep[str(c)]["recall"], 4)
                 for c in sorted(CLASS_NAMES)}
    # seizure(1) vs rest collapse
    bin_acc = accuracy_score((yt == 1).astype(int), (yp == 1).astype(int))

    result = {
        "name": "onehot_multiclass", "dataset": "UCI-178raw (5-class)",
        "n_samples": int(len(y)), "n_classes": 5, "class_names": CLASS_NAMES,
        "one_hot_encoding": onehot_info,
        "validation": "stratified 5-fold, multiclass RandomForest",
        "overall_accuracy": round(float(accuracy_score(yt, yp)), 4),
        "per_class_recall": per_class,
        "seizure_vs_rest_accuracy": round(float(bin_acc), 4),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
    }
    (OUT / "onehot_multiclass.json").write_text(json.dumps(result, indent=2))

    print(f"[onehot] label one-hot: {y[0]} -> {onehot_info['example_onehot']} "
          f"(shape {onehot_info['onehot_shape']})")
    print(f"[onehot] multiclass 5-fold accuracy = {result['overall_accuracy']}")
    print(f"[onehot] per-class recall: {per_class}")
    print(f"[onehot] seizure-vs-rest accuracy = {result['seizure_vs_rest_accuracy']}")
    print(f"[onehot] -> {OUT/'onehot_multiclass.json'}")


if __name__ == "__main__":
    main()
