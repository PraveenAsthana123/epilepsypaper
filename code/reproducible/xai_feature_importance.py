#!/usr/bin/env python3
"""Explainable AI for the CHB-MIT RandomForest epilepsy model (§162 step 17).

Drafted by Ollama (qwen2.5-coder:14b) per §158; the model's skeleton (imports, RF,
SHAP TreeExplainer, plots) was correct — Claude fixed two project-specific bugs it
could not know (§156 verify): (1) it hand-built non-existent EDF paths → reuse the
pipeline's real `discover()`; (2) feature names were `features*2` (duplicates) →
the real aggregation is mean+std across channels, so names are mean_<f> + std_<f>.

Computes RandomForest feature_importances_ + SHAP values on real CHB-MIT data and
saves a feature-importance bar plot, a SHAP summary plot, and a JSON of top features.
Run: python code/reproducible/xai_feature_importance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier

# reuse the real pipeline (its FLAT/SUMM_DIR are absolute paths to the real data)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chbmit_loso_pipeline import discover, build_case  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
IMG = REPO / "images"
ACC = REPO / "accuracy"
N_SUBJECTS = 3  # keep the XAI run light


def feature_names():
    """20 features = mean + std (across channels) of the 10 per-channel features."""
    per_channel = ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
                   "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
                   "line_length", "rms"]
    return [f"mean_{f}" for f in per_channel] + [f"std_{f}" for f in per_channel]


def main():
    IMG.mkdir(exist_ok=True)
    ACC.mkdir(exist_ok=True)
    names = feature_names()

    cases = discover()
    if not cases:
        print("No CHB-MIT data discovered — cannot run XAI (real data required, §57.7).")
        return

    # build X,y from the first few subjects that have seizures
    X_parts, y_parts, used = [], [], []
    for cn, (summ, edfs) in cases.items():
        X, y = build_case(cn, summ, edfs)
        if len(y) and y.sum() > 0:
            X_parts.append(X); y_parts.append(y); used.append(cn)
        if len(used) >= N_SUBJECTS:
            break
    if not X_parts:
        print("No seizure-positive epochs found — cannot run XAI.")
        return

    X = np.vstack(X_parts); y = np.concatenate(y_parts)
    print(f"XAI on subjects {used}: X={X.shape}, seizure epochs={int(y.sum())}")

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    rf.fit(X, y)

    # --- 1. RandomForest feature importance ---
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    top = [{"feature": names[i], "importance": float(imp[i])} for i in order]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([names[i] for i in order[::-1]], imp[order[::-1]], color="#3b82f6")
    ax.set_xlabel("RandomForest feature importance")
    ax.set_title(f"Feature importance — CHB-MIT (subjects {','.join(used)})")
    fig.tight_layout(); fig.savefig(IMG / "feature_importance.png", dpi=120); plt.close(fig)

    # --- 2. SHAP (TreeExplainer) ---
    shap_ok = True
    try:
        # sample for speed; SHAP on a few hundred epochs is plenty
        idx = np.random.RandomState(42).choice(len(X), min(300, len(X)), replace=False)
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X[idx])
        sv_pos = sv[1] if isinstance(sv, list) else sv  # seizure-class SHAP
        plt.figure()
        shap.summary_plot(sv_pos, X[idx], feature_names=names, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(IMG / "shap_summary.png", dpi=120); plt.close()
    except Exception as e:
        shap_ok = False
        print(f"SHAP step failed: {e}")

    (ACC / "xai_feature_importance.json").write_text(json.dumps({
        "model": "RandomForest(300,balanced)", "subjects": used,
        "n_epochs": int(len(y)), "n_seizure": int(y.sum()),
        "feature_importance_top": top,
        "shap_summary": "images/shap_summary.png" if shap_ok else "failed",
    }, indent=2))

    print(f"✓ top features: {[t['feature'] for t in top[:5]]}")
    print(f"✓ wrote images/feature_importance.png{', images/shap_summary.png' if shap_ok else ''}"
          f", accuracy/xai_feature_importance.json")


if __name__ == "__main__":
    main()
