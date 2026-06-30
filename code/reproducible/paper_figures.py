#!/usr/bin/env python3
"""Generate ALL paper figures from REAL data + committed results.

Produces (to images/): ROC curves, confusion matrices (UCI + CHB-MIT), feature
histograms, band-power/frequency bars, PCA+KMeans clustering, EDA before/after
(raw vs filtered 1D signal, raw vs standardised features), and the Butterworth
1D filter frequency response.

No fabricated numbers (Sec. 57.7). Run:
    UCI_CSV="/path/Epileptic Seizure Recognition.csv" \
    python code/reproducible/paper_figures.py

Output: images/{roc_curves,confusion_uci,confusion_chbmit,feature_histograms,
        bandpower_frequency,cluster_pca,eda_before_after,filter_response}.png
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, welch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("FEATURE_CACHE", Path(__file__).resolve().parent / "feature_cache.npz"))
UCI_CSV = os.environ.get("UCI_CSV", sys.argv[1] if len(sys.argv) > 1 else "data/Epileptic Seizure Recognition.csv")
IMG = ROOT / "images"; IMG.mkdir(parents=True, exist_ok=True)
ACC = ROOT / "accuracy"
FS = 173.61
BASE = ["delta", "theta", "alpha", "beta", "gamma", "hj_act", "hj_mob", "hj_cmp", "linelen", "rms"]
NAMES = [f"{b}_mean" for b in BASE] + [f"{b}_std" for b in BASE]


def load_uci():
    import pandas as pd
    df = pd.read_csv(UCI_CSV)
    return df.iloc[:, 1:-1].values.astype(float), (df.iloc[:, -1].values == 1).astype(int)


def load_chb():
    d = np.load(CACHE, allow_pickle=True)
    subs = list(d["subjects"])
    X = np.vstack([d[f"X_{s}"] for s in subs]); y = np.concatenate([d[f"y_{s}"] for s in subs]).astype(int)
    return X, y


def fig_roc_and_confusion(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler().fit(Xtr)
    models = {"RandomForest": RandomForestClassifier(300, random_state=42, n_jobs=-1),
              "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced")}
    plt.figure(figsize=(5, 4))
    rf_pred = None
    for name, clf in models.items():
        clf.fit(sc.transform(Xtr), ytr)
        proba = clf.predict_proba(sc.transform(Xte))[:, 1]
        if name == "RandomForest":
            rf_pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(yte, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC — UCI seizure detection"); plt.legend(); plt.tight_layout()
    plt.savefig(IMG / "roc_curves.png", dpi=120); plt.close()

    ConfusionMatrixDisplay(confusion_matrix(yte, rf_pred),
                           display_labels=["non-sz", "seizure"]).plot(cmap="Blues", colorbar=False)
    plt.title("Confusion — RF on UCI"); plt.tight_layout()
    plt.savefig(IMG / "confusion_uci.png", dpi=120); plt.close()


def fig_confusion_chbmit():
    j = json.loads((ACC / "comprehensive_metrics.json").read_text())
    m = j["aggregate_confusion_matrix"]
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    ConfusionMatrixDisplay(cm, display_labels=["non-sz", "seizure"]).plot(cmap="Oranges", colorbar=False)
    plt.title("Confusion — CHB-MIT LOSO (aggregate)"); plt.tight_layout()
    plt.savefig(IMG / "confusion_chbmit.png", dpi=120); plt.close()


def fig_histograms_bandpower_cluster(X, y):
    # histograms of 4 informative features, seizure vs non
    feats = [5, 9, 1, 18]  # hj_act_mean, rms_mean, theta_mean, hj_cmp_std
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    for a, fi in zip(ax.ravel(), feats):
        a.hist(X[y == 0, fi], bins=40, alpha=0.6, label="non-sz", color="#4c78a8", density=True)
        a.hist(X[y == 1, fi], bins=40, alpha=0.6, label="seizure", color="#e45756", density=True)
        a.set_title(NAMES[fi]); a.legend(fontsize=7)
    plt.suptitle("Feature histograms (seizure vs non-seizure)"); plt.tight_layout()
    plt.savefig(IMG / "feature_histograms.png", dpi=120); plt.close()

    # band-power / frequency bars (mean relative band power, first 5 features = bands)
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    nz = X[y == 0, :5].mean(0); sz = X[y == 1, :5].mean(0)
    x = np.arange(5); w = 0.38
    plt.figure(figsize=(6, 3.5))
    plt.bar(x - w / 2, nz, w, label="non-sz", color="#4c78a8")
    plt.bar(x + w / 2, sz, w, label="seizure", color="#e45756")
    plt.xticks(x, bands); plt.ylabel("relative power"); plt.title("Band-power by frequency band")
    plt.legend(); plt.tight_layout(); plt.savefig(IMG / "bandpower_frequency.png", dpi=120); plt.close()

    # PCA + KMeans clustering
    Z = PCA(2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
    km = KMeans(2, random_state=42, n_init=10).fit(Z)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].scatter(Z[y == 0, 0], Z[y == 0, 1], s=3, alpha=0.3, color="#4c78a8", label="non-sz")
    ax[0].scatter(Z[y == 1, 0], Z[y == 1, 1], s=3, alpha=0.3, color="#e45756", label="seizure")
    ax[0].set_title("PCA — true labels"); ax[0].legend(fontsize=7)
    ax[1].scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.3, c=km.labels_, cmap="viridis")
    ax[1].set_title("PCA — KMeans (k=2)")
    plt.tight_layout(); plt.savefig(IMG / "cluster_pca.png", dpi=120); plt.close()


def fig_eda_before_after_and_filter(Xuci):
    # 1D filter: raw vs bandpass-filtered EEG segment (EDA before/after)
    b, a = butter(4, [0.5, 40], btype="band", fs=FS)
    sig = Xuci[np.random.RandomState(0).randint(len(Xuci))]
    filt = filtfilt(b, a, sig)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(sig, lw=0.7, color="#888", label="raw"); ax[0].plot(filt, lw=0.9, color="#e45756", label="0.5-40Hz")
    ax[0].set_title("EDA before/after — 1D bandpass filter"); ax[0].legend(fontsize=7)
    # raw vs standardised feature distribution
    raw = Xuci[:, 89]; std = StandardScaler().fit_transform(raw.reshape(-1, 1)).ravel()
    ax[1].hist(raw, bins=50, alpha=0.6, color="#888", density=True, label="raw")
    ax[1].hist(std, bins=50, alpha=0.6, color="#4c78a8", density=True, label="standardised")
    ax[1].set_title("Before/after standardisation"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(IMG / "eda_before_after.png", dpi=120); plt.close()

    # Butterworth 1D filter frequency response
    w, h = freqz(b, a, fs=FS, worN=2048)
    plt.figure(figsize=(6, 3.2))
    plt.plot(w, np.abs(h), color="#4c78a8")
    plt.axvline(0.5, ls="--", c="g", lw=0.8); plt.axvline(40, ls="--", c="g", lw=0.8)
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Gain"); plt.title("1D Butterworth 0.5-40 Hz response")
    plt.xlim(0, 60); plt.tight_layout(); plt.savefig(IMG / "filter_response.png", dpi=120); plt.close()


def main():
    Xu, yu = load_uci()
    fig_roc_and_confusion(Xu, yu)
    fig_confusion_chbmit()
    fig_eda_before_after_and_filter(Xu)
    if CACHE.exists():
        Xc, yc = load_chb()
        fig_histograms_bandpower_cluster(Xc, yc)
    figs = sorted(p.name for p in IMG.glob("*.png"))
    print(f"[figs] generated {len(figs)} figures in {IMG}:")
    for f in figs:
        print("   ", f)


if __name__ == "__main__":
    main()
