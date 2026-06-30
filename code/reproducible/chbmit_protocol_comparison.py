#!/usr/bin/env python3
"""SAME-DATASET leakage proof — CHB-MIT epoch-5fold vs CHB-MIT LOSO.

Addresses the #1 reviewer objection: the leakage claim must compare protocols on the
SAME dataset, features, and model -- not Bonn/UCI-epoch vs CHB-MIT-LOSO (which confounds
protocol with dataset). Here BOTH protocols use the identical cached 20-D CHB-MIT
features and the same balanced RandomForest. We report the full mandatory metric set
(acc, sens, spec, PPV, NPV, F1, MCC, balanced-acc, ROC-AUC, PR-AUC), pooled AND macro
per-subject, with subject-level bootstrap CIs, plus a LOSO threshold sweep and
sensitivity at fixed specificity (90/95/99%).

No fabricated numbers (Sec. 57.7). Run:
    python code/reproducible/chbmit_protocol_comparison.py
Output: accuracy/chbmit_protocol_comparison.json
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score)

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("FEATURE_CACHE", Path(__file__).resolve().parent / "feature_cache.npz"))
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.RandomState(42)


def load():
    d = np.load(CACHE, allow_pickle=True)
    subs = list(d["subjects"])
    X = np.vstack([d[f"X_{s}"] for s in subs])
    y = np.concatenate([d[f"y_{s}"] for s in subs]).astype(int)
    groups = np.concatenate([[s] * len(d[f"y_{s}"]) for s in subs])
    return X, y, groups, subs


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=42, n_jobs=-1)


def pooled_metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    return {"accuracy": round((tp + tn) / len(y), 4), "sensitivity": round(sens, 4),
            "specificity": round(spec, 4), "ppv": round(ppv, 4), "npv": round(npv, 4),
            "f1": round(f1, 4), "mcc": round(float(matthews_corrcoef(y, pred)), 4),
            "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def oof_proba(X, y, groups, protocol):
    """Out-of-fold probabilities + the per-sample subject id."""
    split = (LeaveOneGroupOut().split(X, y, groups) if protocol == "LOSO"
             else StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    proba = np.zeros(len(y)); seen = np.zeros(len(y), bool)
    for tr, te in split:
        sc = StandardScaler().fit(X[tr])
        clf = rf().fit(sc.transform(X[tr]), y[tr])
        proba[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]; seen[te] = True
    assert seen.all()
    return proba


def macro_and_ci(y, proba, groups, subs):
    """Macro per-subject sensitivity/AUC + subject-level bootstrap 95% CI."""
    per = {}
    for s in subs:
        m = groups == s
        if y[m].sum() == 0:
            continue
        pred = (proba[m] >= 0.5).astype(int)
        sens = pred[y[m] == 1].mean() if (y[m] == 1).any() else 0.0
        try:
            auc = roc_auc_score(y[m], proba[m])
        except ValueError:
            auc = np.nan
        per[s] = {"sens": float(sens), "auc": float(auc)}
    sens_arr = np.array([v["sens"] for v in per.values()])
    auc_arr = np.array([v["auc"] for v in per.values()])
    boots = [np.mean(RNG.choice(sens_arr, len(sens_arr), replace=True)) for _ in range(2000)]
    return {"macro_sensitivity": round(float(sens_arr.mean()), 4),
            "macro_auc": round(float(np.nanmean(auc_arr)), 4),
            "sens_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                          round(float(np.percentile(boots, 97.5)), 4)],
            "worst_subjects": sorted(per, key=lambda s: per[s]["sens"])[:4]}


def threshold_sweep(y, proba):
    rows = []
    for thr in np.round(np.arange(0.1, 0.95, 0.1), 2):
        rows.append({"threshold": float(thr), **{k: pooled_metrics(y, proba, thr)[k]
                     for k in ["sensitivity", "specificity", "ppv", "npv", "f1", "mcc"]}})
    # sensitivity at fixed specificity
    order = np.argsort(-proba)
    sens_at = {}
    for target in [0.90, 0.95, 0.99]:
        best = 0.0
        for thr in np.unique(proba):
            pred = (proba >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) else 0
            sens = tp / (tp + fn) if (tp + fn) else 0
            if spec >= target:
                best = max(best, sens)
        sens_at[f"spec_{int(target*100)}"] = round(float(best), 4)
    return rows, sens_at


def main():
    if not CACHE.exists():
        raise SystemExit(f"feature_cache.npz not found at {CACHE}; run chbmit_loso_pipeline.py first.")
    X, y, groups, subs = load()
    print(f"[proto] CHB-MIT same-dataset: {len(y)} epochs, {y.sum()} seizure, {len(subs)} subjects")

    result = {"name": "chbmit_protocol_comparison",
              "dataset": "CHB-MIT (identical 20-D features + balanced RF)",
              "n_epochs": int(len(y)), "n_seizure": int(y.sum()), "n_subjects": len(subs)}

    for proto in ["epoch_5fold", "LOSO"]:
        proba = oof_proba(X, y, groups, proto)
        pooled = pooled_metrics(y, proba)
        macro = macro_and_ci(y, proba, groups, subs)
        block = {"pooled": pooled, "macro_per_subject": macro}
        if proto == "LOSO":
            sweep, sens_at = threshold_sweep(y, proba)
            block["threshold_sweep"] = sweep
            block["sensitivity_at_fixed_specificity"] = sens_at
        result[proto] = block
        print(f"[proto] {proto:11s} pooled acc={pooled['accuracy']} sens={pooled['sensitivity']} "
              f"spec={pooled['specificity']} MCC={pooled['mcc']} PR-AUC={pooled['pr_auc']} | "
              f"macro sens={macro['macro_sensitivity']} CI{macro['sens_ci95']}")

    # the headline same-dataset delta
    e = result["epoch_5fold"]["pooled"]; l = result["LOSO"]["pooled"]
    result["leakage_delta_same_dataset"] = {
        "sensitivity_pooled": [e["sensitivity"], l["sensitivity"]],
        "sensitivity_macro": [result["epoch_5fold"]["macro_per_subject"]["macro_sensitivity"],
                              result["LOSO"]["macro_per_subject"]["macro_sensitivity"]],
        "mcc": [e["mcc"], l["mcc"]], "pr_auc": [e["pr_auc"], l["pr_auc"]]}
    (OUT / "chbmit_protocol_comparison.json").write_text(json.dumps(result, indent=2))
    print(f"[proto] SAME-DATASET leakage delta (macro sens): "
          f"{result['leakage_delta_same_dataset']['sensitivity_macro']}")
    print(f"[proto] LOSO sens@spec: {result['LOSO']['sensitivity_at_fixed_specificity']}")
    print(f"[proto] -> {OUT/'chbmit_protocol_comparison.json'}")


if __name__ == "__main__":
    main()
