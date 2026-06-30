#!/usr/bin/env python3
"""Step 12 — Feature Evaluation (statistical model component).

Statistically ranks the REAL 20-D CHB-MIT feature vector against the seizure label
using three independent criteria, on the cached feature matrix (all 24 subjects):

  * ANOVA F-test            (sklearn.feature_selection.f_classif)   -> F, p
  * Mutual information      (mutual_info_classif)                   -> nats
  * Point-biserial / Pearson correlation with the binary label     -> r, p

No fabricated numbers (Sec. 57.7): every value is derived from feature_cache.npz,
the same 20-D features the LOSO pipeline trains on. Run:
    python code/reproducible/step12_feature_evaluation.py

Output: accuracy/feature_evaluation.json (+ stdout table).
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import f_classif, mutual_info_classif

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("FEATURE_CACHE", Path(__file__).resolve().parent / "feature_cache.npz"))
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)

# 20-D layout: per-channel 10-D (5 band powers + 3 Hjorth + line-length + RMS),
# aggregated across channels by mean then std -> 20 features.
BASE = ["delta", "theta", "alpha", "beta", "gamma",
        "hjorth_activity", "hjorth_mobility", "hjorth_complexity", "line_length", "rms"]
NAMES = [f"{b}_mean" for b in BASE] + [f"{b}_std" for b in BASE]


def load_all(cache: Path):
    d = np.load(cache, allow_pickle=True)
    subs = list(d["subjects"])
    X = np.vstack([d[f"X_{s}"] for s in subs])
    y = np.concatenate([d[f"y_{s}"] for s in subs]).astype(int)
    return X, y, subs


def main():
    if not CACHE.exists():
        raise SystemExit(f"feature_cache.npz not found at {CACHE}; run chbmit_loso_pipeline.py first "
                         f"or set FEATURE_CACHE.")
    X, y, subs = load_all(CACHE)
    n, p = X.shape
    assert p == len(NAMES), f"expected {len(NAMES)} features, got {p}"

    F, p_anova = f_classif(X, y)
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    corr = np.array([pearsonr(X[:, j], y)[0] for j in range(p)])
    corr_p = np.array([pearsonr(X[:, j], y)[1] for j in range(p)])

    feats = []
    for j in range(p):
        feats.append({
            "feature": NAMES[j],
            "anova_F": round(float(F[j]), 4),
            "anova_p": float(f"{p_anova[j]:.3e}"),
            "mutual_info": round(float(mi[j]), 5),
            "pearson_r": round(float(corr[j]), 4),
            "pearson_p": float(f"{corr_p[j]:.3e}"),
            "significant_p_lt_0_05": bool(p_anova[j] < 0.05),
        })
    # rank by ANOVA F (descending)
    ranked = sorted(feats, key=lambda d: d["anova_F"], reverse=True)

    result = {
        "step": 12,
        "name": "feature_evaluation",
        "dataset": "CHB-MIT (cached 20-D features, all subjects)",
        "n_samples": int(n),
        "n_features": int(p),
        "n_seizure": int(y.sum()),
        "n_subjects": len(subs),
        "criteria": ["ANOVA F-test", "mutual information", "Pearson/point-biserial correlation"],
        "n_significant_anova_p_lt_0_05": int(sum(f["significant_p_lt_0_05"] for f in feats)),
        "top5_by_anova_F": [f["feature"] for f in ranked[:5]],
        "top5_by_mutual_info": [f["feature"] for f in sorted(feats, key=lambda d: d["mutual_info"], reverse=True)[:5]],
        "features": ranked,
    }
    (OUT / "feature_evaluation.json").write_text(json.dumps(result, indent=2))

    print(f"[step12] {n} samples x {p} features, {y.sum()} seizure, {len(subs)} subjects")
    print(f"[step12] {result['n_significant_anova_p_lt_0_05']}/{p} features ANOVA-significant (p<0.05)")
    print(f"[step12] top-5 by ANOVA F : {result['top5_by_anova_F']}")
    print(f"[step12] top-5 by MI       : {result['top5_by_mutual_info']}")
    print(f"[step12] -> {OUT/'feature_evaluation.json'}")


if __name__ == "__main__":
    main()
