#!/usr/bin/env python3
"""Multi-pipeline benchmark — Statistical vs ML vs DL models, across preprocessing
variants and validation protocols, on MULTIPLE datasets, compared head-to-head.

Answers: "if you use only a statistical model / only ML / a DL model / proper CV,
what accuracy do you get?" — by running every (dataset x preprocessing x model x
validation) combination and reporting accuracy/sensitivity/specificity/F1/AUC.

Datasets (REAL):
  * CHB-MIT-20D : cached 20-D engineered features (feature_cache.npz), 24 subjects
                  -> evaluated under BOTH subject-level LOSO and epoch-level 5-fold
                     (the leakage contrast — Sec. 83)
  * UCI-178raw  : UCI Epileptic Seizure Recognition, 178 raw samples/segment,
                  seizure(class 1) vs rest -> stratified 5-fold

Model families (each with explicit parameters):
  Statistical : LogisticRegression, LinearDiscriminantAnalysis, GaussianNB
  ML          : RandomForest, SVM(RBF), HistGradientBoosting, KNN
  DL          : MLPClassifier (2 hidden layers)

Preprocessing: StandardScaler (standardize) and MinMaxScaler (normalize),
fit on the TRAIN fold only (no leakage). Feature evaluation/selection live in
step12/step13; EDA summary is emitted per dataset.

No fabricated numbers (Sec. 57.7). Run:
    UCI_CSV="/path/Epileptic Seizure Recognition.csv" \
    python code/reproducible/multi_pipeline_benchmark.py

Output: accuracy/multi_pipeline_comparison.json + MULTI_PIPELINE_COMPARISON.md
"""
from __future__ import annotations
import os, sys, json, warnings, time
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("FEATURE_CACHE", Path(__file__).resolve().parent / "feature_cache.npz"))
UCI_CSV = os.environ.get("UCI_CSV", sys.argv[1] if len(sys.argv) > 1 else "data/Epileptic Seizure Recognition.csv")
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)


def models():
    """Return {family: {name: (estimator, params_str)}} — fresh estimators each call."""
    return {
        "Statistical": {
            "LogisticRegression": (LogisticRegression(C=1.0, class_weight="balanced",
                                                      max_iter=2000, random_state=42),
                                   "C=1.0, class_weight=balanced"),
            "LDA": (LinearDiscriminantAnalysis(), "solver=svd"),
            "GaussianNB": (GaussianNB(), "default"),
        },
        "ML": {
            "RandomForest": (RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    random_state=42, n_jobs=-1),
                             "n_estimators=300, class_weight=balanced"),
            "SVM_RBF": (SVC(C=1.0, kernel="rbf", class_weight="balanced",
                            random_state=42), "C=1.0, kernel=rbf (train capped at 4000 for scalability)"),
            "HistGradientBoosting": (HistGradientBoostingClassifier(max_iter=200, random_state=42),
                                     "max_iter=200"),
            "KNN": (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "k=5"),
        },
        "DL": {
            "MLP": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                                  early_stopping=True, random_state=42),
                    "hidden=(64,32), early_stopping"),
        },
    }


def scaler(kind):
    return StandardScaler() if kind == "standardize" else MinMaxScaler()


def metrics_from(y_true, y_pred, proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    auc = None
    try:
        auc = round(float(roc_auc_score(y_true, proba)), 4)
    except ValueError:
        pass
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc": auc,
    }


def run_protocol(X, y, groups, protocol, est_factory, prep):
    """Aggregate out-of-fold predictions across a CV protocol, return pooled metrics."""
    if protocol == "LOSO":
        splitter = LeaveOneGroupOut().split(X, y, groups)
    else:  # stratified 5-fold (epoch-level)
        splitter = StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)
    yt, yp, pr = [], [], []
    for tr, te in splitter:
        if len(np.unique(y[tr])) < 2:
            continue
        sc = scaler(prep).fit(X[tr])
        clf = est_factory()
        # RBF-SVM is O(n^2-n^3); cap training fold to keep it scalable on large datasets
        from sklearn.svm import SVC as _SVC
        if isinstance(clf, _SVC) and len(tr) > 4000:
            rng = np.random.RandomState(42)
            sub = rng.choice(tr, 4000, replace=False)
            tr = sub
        clf.fit(sc.transform(X[tr]), y[tr])
        Xte = sc.transform(X[te])
        pred = clf.predict(Xte)
        try:
            proba = clf.predict_proba(Xte)[:, 1]
        except Exception:
            proba = clf.decision_function(Xte) if hasattr(clf, "decision_function") else pred
        yt.append(y[te]); yp.append(pred); pr.append(proba)
    return metrics_from(np.concatenate(yt), np.concatenate(yp), np.concatenate(pr))


def load_chbmit():
    d = np.load(CACHE, allow_pickle=True)
    subs = list(d["subjects"])
    X = np.vstack([d[f"X_{s}"] for s in subs])
    y = np.concatenate([d[f"y_{s}"] for s in subs]).astype(int)
    groups = np.concatenate([[s] * len(d[f"y_{s}"]) for s in subs])
    return X, y, groups


def load_uci():
    import pandas as pd
    df = pd.read_csv(UCI_CSV)
    X = df.iloc[:, 1:-1].values.astype(float)        # 178 raw samples
    y = (df.iloc[:, -1].values == 1).astype(int)     # class 1 = seizure
    return X, y


def eda(name, X, y):
    return {
        "dataset": name, "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_seizure": int(y.sum()), "seizure_rate": round(float(y.mean()), 4),
        "feature_mean_range": [round(float(X.mean(0).min()), 4), round(float(X.mean(0).max()), 4)],
    }


def main():
    t0 = time.time()
    datasets = []

    if CACHE.exists():
        Xc, yc, gc = load_chbmit()
        datasets.append(("CHB-MIT-20D", Xc, yc, gc, ["LOSO", "epoch_5fold"]))
    else:
        print(f"[warn] {CACHE} missing — skipping CHB-MIT")

    try:
        Xu, yu = load_uci()
        datasets.append(("UCI-178raw", Xu, yu, None, ["epoch_5fold"]))
    except Exception as e:
        print(f"[warn] UCI not loaded ({e}) — set UCI_CSV; skipping")

    if not datasets:
        raise SystemExit("No datasets available. Provide feature_cache.npz and/or UCI_CSV.")

    all_models = models()
    eda_report, rows = [], []

    for name, X, y, groups, protocols in datasets:
        eda_report.append(eda(name, X, y))
        print(f"\n=== {name}: {X.shape[0]}x{X.shape[1]}, seizure={y.sum()} ===")
        for prep in ["standardize", "normalize"]:
            for family, fam_models in all_models.items():
                for mname, (est, params) in fam_models.items():
                    for proto in protocols:
                        # rebuild a fresh estimator each run via factory
                        factory = (lambda e=type(est), p=est.get_params(): e(**p))
                        try:
                            m = run_protocol(X, y, groups, proto, factory, prep)
                            row = {"dataset": name, "family": family, "model": mname,
                                   "params": params, "preprocessing": prep,
                                   "validation": proto, **m}
                            rows.append(row)
                            print(f"  [{family:11s}] {mname:20s} {prep:11s} {proto:11s} "
                                  f"acc={m['accuracy']} sens={m['sensitivity']} "
                                  f"spec={m['specificity']} auc={m['auc']}", flush=True)
                        except Exception as e:
                            print(f"  [{family}] {mname} {prep} {proto} FAILED: {e}")

    result = {
        "benchmark": "multi_pipeline_comparison",
        "datasets": eda_report,
        "model_families": {k: list(v.keys()) for k, v in all_models.items()},
        "preprocessing": ["standardize (StandardScaler)", "normalize (MinMaxScaler)"],
        "validation_protocols": ["LOSO (subject-level, leakage-free)",
                                 "epoch_5fold (stratified, leaky)"],
        "n_pipelines": len(rows),
        "runtime_sec": round(time.time() - t0, 1),
        "results": rows,
    }
    (OUT / "multi_pipeline_comparison.json").write_text(json.dumps(result, indent=2))

    # markdown table
    lines = ["# Multi-Pipeline Benchmark — Statistical vs ML vs DL\n",
             f"{len(rows)} pipelines across {len(datasets)} datasets. "
             f"All numbers from real data (no fabrication, Sec. 57.7).\n",
             "| Dataset | Family | Model | Prep | Validation | Acc | Sens | Spec | F1 | AUC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['family']} | {r['model']} | {r['preprocessing']} | "
                     f"{r['validation']} | {r['accuracy']} | {r['sensitivity']} | "
                     f"{r['specificity']} | {r['f1']} | {r['auc']} |")
    (OUT / "MULTI_PIPELINE_COMPARISON.md").write_text("\n".join(lines) + "\n")

    print(f"\n[done] {len(rows)} pipelines in {result['runtime_sec']}s")
    print(f"[done] -> {OUT/'multi_pipeline_comparison.json'}")
    print(f"[done] -> {OUT/'MULTI_PIPELINE_COMPARISON.md'}")


if __name__ == "__main__":
    main()
