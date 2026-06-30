#!/usr/bin/env python3
"""EDA — Exploratory Data Analysis on the REAL epilepsy datasets.

Summarises both datasets before modelling: class balance, per-subject distribution,
per-feature statistics (mean/std/skew/kurtosis), and the most correlated feature
pairs (redundancy check that motivates feature selection in step13).

No fabricated numbers (Sec. 57.7). Run:
    UCI_CSV="/path/Epileptic Seizure Recognition.csv" \
    python code/reproducible/eda_analysis.py

Output: accuracy/eda.json (+ images/eda_class_balance.png)
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("FEATURE_CACHE", Path(__file__).resolve().parent / "feature_cache.npz"))
UCI_CSV = os.environ.get("UCI_CSV", sys.argv[1] if len(sys.argv) > 1 else "data/Epileptic Seizure Recognition.csv")
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)
IMG = ROOT / "images"; IMG.mkdir(parents=True, exist_ok=True)

BASE = ["delta", "theta", "alpha", "beta", "gamma",
        "hjorth_activity", "hjorth_mobility", "hjorth_complexity", "line_length", "rms"]
NAMES = [f"{b}_mean" for b in BASE] + [f"{b}_std" for b in BASE]


def chbmit_eda():
    if not CACHE.exists():
        return None
    d = np.load(CACHE, allow_pickle=True)
    subs = list(d["subjects"])
    per_subject = {s: {"n": int(len(d[f"y_{s}"])), "seizure": int(d[f"y_{s}"].sum())} for s in subs}
    X = np.vstack([d[f"X_{s}"] for s in subs])
    y = np.concatenate([d[f"y_{s}"] for s in subs]).astype(int)
    stats = {NAMES[j]: {"mean": round(float(X[:, j].mean()), 4), "std": round(float(X[:, j].std()), 4),
                        "skew": round(float(skew(X[:, j])), 3), "kurtosis": round(float(kurtosis(X[:, j])), 3)}
             for j in range(X.shape[1])}
    # most-correlated feature pairs (redundancy)
    C = np.corrcoef(X.T); pairs = []
    for i in range(len(NAMES)):
        for k in range(i + 1, len(NAMES)):
            pairs.append((abs(C[i, k]), NAMES[i], NAMES[k], round(float(C[i, k]), 3)))
    pairs.sort(reverse=True)
    return {"n_samples": int(len(y)), "n_features": int(X.shape[1]),
            "class_balance": {"seizure": int(y.sum()), "non_seizure": int((y == 0).sum()),
                              "seizure_rate": round(float(y.mean()), 4)},
            "n_subjects": len(subs), "per_subject": per_subject,
            "feature_stats": stats,
            "top5_correlated_pairs": [{"f1": a, "f2": b, "r": r} for _, a, b, r in pairs[:5]],
            "missing_values": int(np.isnan(X).sum())}


def uci_eda():
    import pandas as pd
    try:
        df = pd.read_csv(UCI_CSV)
    except Exception:
        return None
    X = df.iloc[:, 1:-1].values.astype(float)
    y5 = df.iloc[:, -1].values.astype(int)
    yb = (y5 == 1).astype(int)
    return {"n_samples": int(len(y5)), "n_features": int(X.shape[1]),
            "binary_balance": {"seizure": int(yb.sum()), "non_seizure": int((yb == 0).sum()),
                               "seizure_rate": round(float(yb.mean()), 4)},
            "multiclass_counts": {int(c): int((y5 == c).sum()) for c in sorted(set(y5))},
            "amplitude_range": [round(float(X.min()), 1), round(float(X.max()), 1)],
            "missing_values": int(np.isnan(X).sum())}


def main():
    eda = {"name": "exploratory_data_analysis",
           "CHB-MIT-20D": chbmit_eda(), "UCI-178raw": uci_eda()}
    (OUT / "eda.json").write_text(json.dumps(eda, indent=2))

    # class-balance bar figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    if eda["CHB-MIT-20D"]:
        cb = eda["CHB-MIT-20D"]["class_balance"]
        axes[0].bar(["non-seizure", "seizure"], [cb["non_seizure"], cb["seizure"]], color=["#4c78a8", "#e45756"])
        axes[0].set_title("CHB-MIT epochs")
    if eda["UCI-178raw"]:
        bb = eda["UCI-178raw"]["binary_balance"]
        axes[1].bar(["non-seizure", "seizure"], [bb["non_seizure"], bb["seizure"]], color=["#4c78a8", "#e45756"])
        axes[1].set_title("UCI segments")
    plt.tight_layout(); plt.savefig(IMG / "eda_class_balance.png", dpi=110); plt.close()

    if eda["CHB-MIT-20D"]:
        c = eda["CHB-MIT-20D"]
        print(f"[eda] CHB-MIT: {c['n_samples']} epochs, seizure-rate {c['class_balance']['seizure_rate']}, "
              f"{c['n_subjects']} subjects, missing={c['missing_values']}")
        print(f"[eda] top correlated pair: {c['top5_correlated_pairs'][0]}")
    if eda["UCI-178raw"]:
        u = eda["UCI-178raw"]
        print(f"[eda] UCI: {u['n_samples']} segments, multiclass {u['multiclass_counts']}, "
              f"missing={u['missing_values']}")
    print(f"[eda] -> {OUT/'eda.json'}, {IMG/'eda_class_balance.png'}")


if __name__ == "__main__":
    main()
