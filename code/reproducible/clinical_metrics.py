#!/usr/bin/env python3
"""Clinical metrics + calibration on the CHB-MIT LOSO out-of-fold probabilities.

Computes the clinically-relevant numbers the reviewer asked for, all under strict
LOSO on the identical cached 20-D features:
  * Brier score, Expected Calibration Error (ECE, 10 bins)
  * PR-AUC, ROC-AUC, sensitivity at 90/95/99% specificity
and renders: calibration (reliability) curve, PR curve, ROC curve, and the
threshold-tradeoff (sensitivity/specificity vs threshold).

No fabricated numbers (Sec. 57.7). Run:
    python code/reproducible/clinical_metrics.py
Output: accuracy/clinical_metrics.json + images/{calibration,pr_curve,roc_curve,threshold_tradeoff}.png
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, brier_score_loss, confusion_matrix)

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("FEATURE_CACHE", Path(__file__).resolve().parent / "feature_cache.npz"))
IMG = ROOT / "images"; IMG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)


def load():
    d = np.load(CACHE, allow_pickle=True)
    subs = list(d["subjects"])
    X = np.vstack([d[f"X_{s}"] for s in subs])
    y = np.concatenate([d[f"y_{s}"] for s in subs]).astype(int)
    g = np.concatenate([[s] * len(d[f"y_{s}"]) for s in subs])
    return X, y, g


def loso_oof(X, y, g):
    proba = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(300, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(sc.transform(X[tr]), y[tr])
        proba[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return proba


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(e)


def sens_at_spec(y, p, target):
    best = 0.0
    for thr in np.unique(p):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0
        sens = tp / (tp + fn) if (tp + fn) else 0
        if spec >= target:
            best = max(best, sens)
    return round(float(best), 4)


def main():
    if not CACHE.exists():
        raise SystemExit("feature_cache.npz missing; run chbmit_loso_pipeline.py first.")
    X, y, g = load()
    p = loso_oof(X, y, g)

    fpr, tpr, _ = roc_curve(y, p); roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, p); pr_auc = average_precision_score(y, p)
    brier = brier_score_loss(y, p); cal_ece = ece(y, p)

    # calibration curve
    bins = np.linspace(0, 1, 11); mids, fracs = [], []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum():
            mids.append(p[m].mean()); fracs.append(y[m].mean())
    plt.figure(figsize=(4.6, 4))
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    plt.plot(mids, fracs, "o-", color="#e45756", label=f"RF (ECE={cal_ece:.3f})")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Calibration — CHB-MIT LOSO (Brier={brier:.3f})"); plt.legend()
    plt.tight_layout(); plt.savefig(IMG / "calibration.png", dpi=120); plt.close()

    plt.figure(figsize=(4.6, 4)); plt.plot(rec, prec, color="#4c78a8")
    plt.xlabel("Recall (sensitivity)"); plt.ylabel("Precision"); plt.title(f"PR curve (PR-AUC={pr_auc:.3f})")
    plt.tight_layout(); plt.savefig(IMG / "pr_curve.png", dpi=120); plt.close()

    plt.figure(figsize=(4.6, 4)); plt.plot(fpr, tpr, color="#4c78a8"); plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — LOSO (AUC={roc_auc:.3f})")
    plt.tight_layout(); plt.savefig(IMG / "roc_curve_loso.png", dpi=120); plt.close()

    # threshold tradeoff
    ths = np.linspace(0.05, 0.95, 19); sens, spec = [], []
    for t in ths:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens.append(tp / (tp + fn) if (tp + fn) else 0); spec.append(tn / (tn + fp) if (tn + fp) else 0)
    plt.figure(figsize=(4.8, 4)); plt.plot(ths, sens, "o-", label="sensitivity", color="#e45756")
    plt.plot(ths, spec, "s-", label="specificity", color="#4c78a8")
    plt.xlabel("Decision threshold"); plt.ylabel("Rate"); plt.title("Threshold tradeoff (LOSO)"); plt.legend()
    plt.tight_layout(); plt.savefig(IMG / "threshold_tradeoff.png", dpi=120); plt.close()

    result = {"name": "clinical_metrics", "dataset": "CHB-MIT LOSO (20-D, balanced RF)",
              "brier_score": round(float(brier), 4), "ece_10bin": round(float(cal_ece), 4),
              "roc_auc": round(float(roc_auc), 4), "pr_auc": round(float(pr_auc), 4),
              "sensitivity_at_specificity": {"90": sens_at_spec(y, p, 0.90),
                                             "95": sens_at_spec(y, p, 0.95),
                                             "99": sens_at_spec(y, p, 0.99)},
              "note": "uncalibrated RF; figures = calibration, pr_curve, roc_curve_loso, threshold_tradeoff"}
    (OUT / "clinical_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[clin] Brier={result['brier_score']} ECE={result['ece_10bin']} "
          f"PR-AUC={result['pr_auc']} ROC-AUC={result['roc_auc']}")
    print(f"[clin] sens@spec: {result['sensitivity_at_specificity']}")
    print(f"[clin] figures -> calibration/pr_curve/roc_curve_loso/threshold_tradeoff.png")
    print(f"[clin] -> {OUT/'clinical_metrics.json'}")


if __name__ == "__main__":
    main()
