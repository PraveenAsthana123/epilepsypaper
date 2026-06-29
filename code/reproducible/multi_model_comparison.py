#!/usr/bin/env python3
"""Multi-model + ensemble LOSO comparison + per-subject + statistical analysis (§162).

Reuses the SAME engineered features and patient-independent Leave-One-Subject-Out
protocol as the published RandomForest result, then for each of
    SVM(RBF) · RandomForest · XGBoost · MLP · Soft-Voting ENSEMBLE
computes:
  - global metrics @0.5 (accuracy/sens/spec/precision/F1/AUC/MCC)
  - SUBJECTIVE analysis: per-subject metrics (24 subjects) -> mean +/- SD + 95% CI + worst-case
  - STATISTICAL analysis: Wilcoxon signed-rank test comparing each model's per-subject
    AUC against RandomForest (paired across subjects)

Features are cached to a .npz on first run so later analyses are instant. No epoch-level
inflation, no fabricated numbers (section 57.7).

Run: python code/reproducible/multi_model_comparison.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chbmit_loso_pipeline import discover, build_case  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ACC = REPO / "accuracy"
CACHE = REPO / "code" / "reproducible" / "feature_cache.npz"


def load_features():
    """Extract per-subject features once; cache to .npz (numeric arrays, no pickle)."""
    if CACHE.exists():
        z = np.load(CACHE)
        subs = [s for s in z["subjects"]]
        CX = {s: z[f"X_{s}"] for s in subs}
        CY = {s: z[f"y_{s}"] for s in subs}
        print(f"loaded cached features for {len(CX)} subjects")
        return CX, CY
    cases = discover()
    CX, CY = {}, {}
    for cn, (summ, edfs) in cases.items():
        X, y = build_case(cn, summ, edfs)
        if len(y) and y.sum() > 0:
            CX[cn], CY[cn] = X, y
    save = {"subjects": np.array(list(CX), dtype="U10")}
    for k in CX:
        save[f"X_{k}"] = CX[k]; save[f"y_{k}"] = CY[k]
    np.savez_compressed(CACHE, **save)
    print(f"extracted + cached features for {len(CX)} subjects")
    return CX, CY


def make_models():
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                        eval_metric="logloss", random_state=42, n_jobs=-1)
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced",
                                              probability=True, random_state=42))
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                                                        max_iter=300, random_state=42))
    ens = VotingClassifier(estimators=[("svm", make_pipeline(StandardScaler(), SVC(
        kernel="rbf", class_weight="balanced", probability=True, random_state=42))),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                              eval_metric="logloss", random_state=42, n_jobs=-1))], voting="soft", n_jobs=-1)
    return {"SVM_RBF": svm, "RandomForest": rf, "XGBoost": xgb, "MLP": mlp, "Ensemble_Voting": ens}


def metrics_at(y_true, y_prob, thr=0.5):
    yp = (y_prob >= thr).astype(int)
    tp = int(((yp == 1) & (y_true == 1)).sum()); tn = int(((yp == 0) & (y_true == 0)).sum())
    fp = int(((yp == 1) & (y_true == 0)).sum()); fn = int(((yp == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / den if den else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return dict(accuracy=acc, sensitivity=sens, specificity=spec, precision=prec, f1=f1, auc=auc, mcc=mcc)


def ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return (float(a.mean()), float(a.mean()))
    se = a.std(ddof=1) / math.sqrt(len(a))
    return (float(a.mean() - 2.069 * se), float(a.mean() + 2.069 * se))


def main():
    CX, CY = load_features()
    names = list(CX)
    models = make_models()

    glob_prob = {m: [] for m in models}
    truths = []
    per_subj = {m: {} for m in models}  # subjective: model -> subject -> metrics

    for test in names:
        Xtr = np.vstack([CX[n] for n in names if n != test])
        ytr = np.concatenate([CY[n] for n in names if n != test])
        Xte, yte = CX[test], CY[test]
        truths.append(yte)
        for mname, mdl in models.items():
            mdl.fit(Xtr, ytr)
            p = mdl.predict_proba(Xte)[:, 1]
            glob_prob[mname].append(p)
            per_subj[mname][test] = metrics_at(yte, p, 0.5)
        print(f"  LOSO test={test} (n={len(yte)}, seizures={int(yte.sum())})")

    y_true = np.concatenate(truths)
    report = {"cv": "Leave-One-Subject-Out", "n_subjects": len(names), "models": {}}
    rf_auc = np.array([per_subj["RandomForest"][s]["auc"] for s in names])

    for mname in models:
        glob = metrics_at(y_true, np.concatenate(glob_prob[mname]), 0.5)
        subj_metrics = {k: np.array([per_subj[mname][s][k] for s in names])
                        for k in ["accuracy", "sensitivity", "specificity", "auc", "mcc"]}
        subjective = {}
        for k, vals in subj_metrics.items():
            lo, hi = ci95(vals)
            subjective[k] = dict(mean=float(np.nanmean(vals)), std=float(np.nanstd(vals, ddof=1)),
                                 median=float(np.nanmedian(vals)), min=float(np.nanmin(vals)),
                                 max=float(np.nanmax(vals)), ci95=[lo, hi],
                                 worst_subject=names[int(np.nanargmin(vals))])
        this_auc = np.array([per_subj[mname][s]["auc"] for s in names])
        if mname == "RandomForest":
            wil = {"note": "reference model"}
        else:
            try:
                w, p = sps.wilcoxon(this_auc, rf_auc)
                wil = {"stat": float(w), "p_value": float(p), "significant_vs_RF": bool(p < 0.05),
                       "mean_auc_delta_vs_RF": float(np.nanmean(this_auc - rf_auc))}
            except Exception as e:
                wil = {"error": str(e)}
        report["models"][mname] = {"global_at_0.5": glob, "subjective": subjective,
                                   "statistical_vs_RF": wil}

    (ACC / "multi_model_comparison.json").write_text(json.dumps(report, indent=2))

    lines = ["# Multi-Model + Ensemble - CHB-MIT LOSO (real)\n",
             f"Patient-independent ({len(names)} subjects), same features.\n",
             "## Global accuracy @ threshold 0.5",
             "| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC | MCC |",
             "|---|---|---|---|---|---|---|---|"]
    for m, r in report["models"].items():
        g = r["global_at_0.5"]
        lines.append(f"| {m} | {g['accuracy']*100:.1f}% | {g['sensitivity']*100:.1f}% | "
                     f"{g['specificity']*100:.1f}% | {g['precision']*100:.1f}% | {g['f1']:.3f} | "
                     f"{g['auc']:.3f} | {g['mcc']:.3f} |")
    lines += [f"\n## Subjective (per-subject, n={len(names)}) - AUC mean +/- SD [95% CI], worst subject",
              "| Model | AUC mean+/-SD | 95% CI | worst subject (AUC) |", "|---|---|---|---|"]
    for m, r in report["models"].items():
        s = r["subjective"]["auc"]
        lines.append(f"| {m} | {s['mean']:.3f} +/- {s['std']:.3f} | [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | "
                     f"{s['worst_subject']} ({s['min']:.3f}) |")
    lines += ["\n## Statistical - Wilcoxon paired AUC vs RandomForest (across subjects)",
              "| Model | mean AUC delta vs RF | p-value | significant? |", "|---|---|---|---|"]
    for m, r in report["models"].items():
        w = r["statistical_vs_RF"]
        if "p_value" in w:
            lines.append(f"| {m} | {w['mean_auc_delta_vs_RF']:+.3f} | {w['p_value']:.3f} | "
                         f"{'yes' if w['significant_vs_RF'] else 'no'} |")
        else:
            lines.append(f"| {m} | - | - | {w.get('note','')} |")
    lines.append("\n> Honest read: ~90% accuracy is imbalance-driven; compare on **AUC + MCC + per-subject "
                 "spread**. The worst-subject column is the deployability check (subject-wise, section 83).")
    (ACC / "MULTI_MODEL_COMPARISON.md").write_text("\n".join(lines))

    print("\n=== per-model (global @0.5) ===")
    for m, r in report["models"].items():
        g = r["global_at_0.5"]; s = r["subjective"]["auc"]
        print(f"  {m:16} acc={g['accuracy']*100:5.1f}%  sens={g['sensitivity']*100:5.1f}%  "
              f"AUC={g['auc']:.3f}  MCC={g['mcc']:.3f}  | subj-AUC {s['mean']:.3f}+/-{s['std']:.3f}")
    print("OK wrote multi_model_comparison.json + MULTI_MODEL_COMPARISON.md")


if __name__ == "__main__":
    main()
