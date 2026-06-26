#!/usr/bin/env python3
"""Honest accuracy-improvement experiment: operating-point (threshold) tuning.

The published CHB-MIT LOSO result (sensitivity 35%) uses a fixed 0.5 decision
threshold. On imbalanced seizure data that threshold favours specificity. This
experiment re-runs the SAME features + SAME RandomForest under LOSO, but saves the
per-epoch seizure probabilities and sweeps the decision threshold to show the real
sensitivity/specificity trade-off. AUC is unchanged (same model) — only the
operating point moves. No epoch-level inflation, no new data (§57.7, §162 invariant).

Run: python code/reproducible/improve_threshold_tuning.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chbmit_loso_pipeline import discover, build_case  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "accuracy" / "improvement_threshold_tuning.json"
THRESHOLDS = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15]


def confusion_metrics(y_true, y_prob, thr):
    """Sensitivity/specificity/etc at a given decision threshold."""
    yp = (y_prob >= thr).astype(int)
    tp = int(((yp == 1) & (y_true == 1)).sum())
    tn = int(((yp == 0) & (y_true == 0)).sum())
    fp = int(((yp == 1) & (y_true == 0)).sum())
    fn = int(((yp == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    youden = sens + spec - 1
    return dict(threshold=thr, sensitivity=sens, specificity=spec, ppv=ppv,
                f1=f1, accuracy=acc, youden_j=youden, tp=tp, tn=tn, fp=fp, fn=fn)


def main():
    cases = discover()
    CX, CY = {}, {}
    for cn, (summ, edfs) in cases.items():
        X, y = build_case(cn, summ, edfs)
        if len(y) and y.sum() > 0:
            CX[cn], CY[cn] = X, y
    names = list(CX)
    print(f"Usable subjects: {len(names)} {names}")

    # LOSO: collect held-out probabilities across ALL subjects
    all_true, all_prob = [], []
    for test in names:
        Xtr = np.vstack([CX[n] for n in names if n != test])
        ytr = np.concatenate([CY[n] for n in names if n != test])
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     n_jobs=-1, random_state=42)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(CX[test])[:, 1]
        all_true.append(CY[test]); all_prob.append(prob)
        print(f"  LOSO test={test}: n={len(prob)} seizures={int(CY[test].sum())}")

    y_true = np.concatenate(all_true); y_prob = np.concatenate(all_prob)
    auc = float(roc_auc_score(y_true, y_prob))  # threshold-independent — unchanged

    sweep = [confusion_metrics(y_true, y_prob, t) for t in THRESHOLDS]
    best = max(sweep, key=lambda m: m["youden_j"])  # best sensitivity+specificity balance

    report = dict(
        experiment="operating-point (threshold) tuning",
        cv="Leave-One-Subject-Out", n_subjects=len(names),
        auc=round(auc, 4),
        baseline_0_5=next(m for m in sweep if m["threshold"] == 0.5),
        best_by_youden=best,
        full_sweep=sweep,
        note=("AUC is unchanged (same model); lowering the threshold trades specificity "
              "for sensitivity. Honest improvement: pick the clinically-appropriate "
              "operating point, NOT a higher accuracy."),
    )
    OUT.write_text(json.dumps(report, indent=2))
    b = report["baseline_0_5"]
    print(f"\nAUC={auc:.3f} (unchanged)")
    print(f"baseline @0.5 : sens={b['sensitivity']*100:.1f}% spec={b['specificity']*100:.1f}%")
    print(f"best @{best['threshold']}: sens={best['sensitivity']*100:.1f}% spec={best['specificity']*100:.1f}% "
          f"(Youden J={best['youden_j']:.3f})")
    print(f"✓ wrote {OUT.name}")


if __name__ == "__main__":
    main()
