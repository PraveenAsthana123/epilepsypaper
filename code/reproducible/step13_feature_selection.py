#!/usr/bin/env python3
"""Step 13 — Feature Selection (with subject-level CV validation).

Selects feature subsets from the REAL 20-D CHB-MIT vector with four standard methods
and validates each subset under Leave-One-Subject-Out (LOSO) CV — the leakage-free
unit (Sec. 83). Confirms whether dimensionality can be reduced without losing the
honest LOSO sensitivity.

  * LASSO (L1-penalised logistic regression)  -> non-zero-coef features
  * RFE (recursive feature elimination, RF estimator)
  * SelectKBest (ANOVA f_classif, k=10)
  * PCA (variance retained at n components)

No fabricated numbers (Sec. 57.7): all from feature_cache.npz. Run:
    python code/reproducible/step13_feature_selection.py

Output: accuracy/feature_selection.json
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("FEATURE_CACHE", Path(__file__).resolve().parent / "feature_cache.npz"))
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)

BASE = ["delta", "theta", "alpha", "beta", "gamma",
        "hjorth_activity", "hjorth_mobility", "hjorth_complexity", "line_length", "rms"]
NAMES = [f"{b}_mean" for b in BASE] + [f"{b}_std" for b in BASE]


def load_all(cache: Path):
    d = np.load(cache, allow_pickle=True)
    subs = list(d["subjects"])
    X = np.vstack([d[f"X_{s}"] for s in subs])
    y = np.concatenate([d[f"y_{s}"] for s in subs]).astype(int)
    groups = np.concatenate([[s] * len(d[f"y_{s}"]) for s in subs])
    return X, y, groups, subs


def loso_sensitivity(X, y, groups):
    """Mean per-subject sensitivity + AUC under LOSO with a balanced RF."""
    logo = LeaveOneGroupOut()
    sens, aucs = [], []
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or y[te].sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(sc.transform(X[tr]), y[tr])
        proba = clf.predict_proba(sc.transform(X[te]))[:, 1]
        pred = (proba >= 0.5).astype(int)
        sens.append(recall_score(y[te], pred, zero_division=0))
        try:
            aucs.append(roc_auc_score(y[te], proba))
        except ValueError:
            pass
    return round(float(np.mean(sens)), 4), round(float(np.mean(aucs)), 4)


def main():
    if not CACHE.exists():
        raise SystemExit(f"feature_cache.npz not found at {CACHE}; run chbmit_loso_pipeline.py first.")
    X, y, groups, subs = load_all(CACHE)
    Xs = StandardScaler().fit_transform(X)

    # LASSO: L1 logistic, features with non-zero coefficient
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                               class_weight="balanced", max_iter=2000, random_state=42).fit(Xs, y)
    lasso_idx = np.where(np.abs(lasso.coef_[0]) > 1e-8)[0]

    # RFE with RF, keep 10
    rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
              n_features_to_select=10).fit(Xs, y)
    rfe_idx = np.where(rfe.support_)[0]

    # SelectKBest ANOVA, k=10
    skb = SelectKBest(f_classif, k=10).fit(Xs, y)
    skb_idx = np.where(skb.get_support())[0]

    # PCA: components for 95% variance
    pca = PCA(n_components=0.95, random_state=42).fit(Xs)

    methods = {
        "LASSO_L1": sorted(lasso_idx.tolist()),
        "RFE_RF_k10": sorted(rfe_idx.tolist()),
        "SelectKBest_ANOVA_k10": sorted(skb_idx.tolist()),
    }

    # validate each subset under LOSO
    full_sens, full_auc = loso_sensitivity(X, y, groups)
    subsets = {}
    for name, idx in methods.items():
        if not idx:
            subsets[name] = {"features": [], "loso_sensitivity": None, "loso_auc": None}
            continue
        s, a = loso_sensitivity(X[:, idx], y, groups)
        subsets[name] = {
            "n_features": len(idx),
            "features": [NAMES[i] for i in idx],
            "loso_sensitivity": s,
            "loso_auc": a,
        }

    result = {
        "step": 13,
        "name": "feature_selection",
        "dataset": "CHB-MIT (cached 20-D features, all subjects)",
        "n_samples": int(X.shape[0]),
        "validation": "Leave-One-Subject-Out (balanced RandomForest)",
        "full_20D": {"loso_sensitivity": full_sens, "loso_auc": full_auc},
        "pca_components_for_95pct_variance": int(pca.n_components_),
        "methods": subsets,
    }
    (OUT / "feature_selection.json").write_text(json.dumps(result, indent=2))

    print(f"[step13] full 20-D    : LOSO sens={full_sens}  AUC={full_auc}")
    for name, info in subsets.items():
        print(f"[step13] {name:22s}: {info['n_features']:>2}-D  "
              f"sens={info['loso_sensitivity']}  AUC={info['loso_auc']}")
    print(f"[step13] PCA 95% variance needs {pca.n_components_} components")
    print(f"[step13] -> {OUT/'feature_selection.json'}")


if __name__ == "__main__":
    main()
