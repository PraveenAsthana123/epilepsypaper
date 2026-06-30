#!/usr/bin/env python3
"""Step 8/9 — Time-Frequency Transformation + 1D->2D image conversion (Fourier).

Takes REAL raw EEG segments (UCI Epileptic Seizure Recognition, 178 samples ~1.024 s
at 173.61 Hz) and produces, for a seizure and a non-seizure example:

  * FFT / Welch PSD            (Fourier transform -> power spectrum)
  * STFT spectrogram           (scipy.signal.stft -> time x frequency 2D map)
  * CWT scalogram              (pywt.cwt, Morlet -> scale x time 2D map)

These are the "1D EEG -> 2D image" inputs a CNN/ViT path would consume. Images are
saved to images/; spectral band-power summaries to accuracy/time_frequency.json.

No fabricated numbers (Sec. 57.7). Run:
    UCI_CSV="/path/Epileptic Seizure Recognition.csv" \
    python code/reproducible/step08_time_frequency.py

Output: images/spectrogram_*.png, scalogram_*.png, psd_*.png + accuracy/time_frequency.json
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, stft
import pywt

ROOT = Path(__file__).resolve().parents[2]
UCI_CSV = os.environ.get("UCI_CSV", sys.argv[1] if len(sys.argv) > 1 else "data/Epileptic Seizure Recognition.csv")
IMG = ROOT / "images"; IMG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)

FS = 173.61  # Bonn/UCI sampling rate (Hz)
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def band_powers(sig):
    f, pxx = welch(sig, fs=FS, nperseg=min(128, len(sig)))
    total = np.trapezoid(pxx, f) + 1e-12
    return {b: round(float(np.trapezoid(pxx[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]) / total), 4)
            for b, (lo, hi) in BANDS.items()}, f, pxx


def save_psd(f, pxx, title, path):
    plt.figure(figsize=(5, 3))
    plt.semilogy(f, pxx); plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD")
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def save_spectrogram(sig, title, path):
    f, t, Z = stft(sig, fs=FS, nperseg=32, noverlap=24)
    plt.figure(figsize=(5, 3))
    plt.pcolormesh(t, f, np.abs(Z), shading="gouraud")
    plt.xlabel("Time (s)"); plt.ylabel("Freq (Hz)"); plt.title(title)
    plt.colorbar(label="|STFT|"); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def save_scalogram(sig, title, path):
    scales = np.arange(1, 64)
    coef, _ = pywt.cwt(sig, scales, "morl", sampling_period=1.0 / FS)
    plt.figure(figsize=(5, 3))
    plt.imshow(np.abs(coef), aspect="auto", extent=[0, len(sig) / FS, scales[-1], scales[0]],
               cmap="viridis")
    plt.xlabel("Time (s)"); plt.ylabel("Scale"); plt.title(title)
    plt.colorbar(label="|CWT|"); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    import pandas as pd
    df = pd.read_csv(UCI_CSV)
    X = df.iloc[:, 1:-1].values.astype(float)
    y = (df.iloc[:, -1].values == 1).astype(int)   # class 1 = seizure

    examples = {"seizure": int(np.where(y == 1)[0][0]), "nonseizure": int(np.where(y == 0)[0][0])}
    summary = {"step": 8, "name": "time_frequency", "dataset": "UCI-178raw",
               "fs_hz": FS, "segment_len": int(X.shape[1]), "examples": {}}

    for label, idx in examples.items():
        sig = X[idx]
        bp, f, pxx = band_powers(sig)
        save_psd(f, pxx, f"PSD — {label}", IMG / f"psd_{label}.png")
        save_spectrogram(sig, f"STFT spectrogram — {label}", IMG / f"spectrogram_{label}.png")
        save_scalogram(sig, f"CWT scalogram — {label}", IMG / f"scalogram_{label}.png")
        summary["examples"][label] = {"row": idx, "relative_band_power": bp,
                                      "dominant_band": max(bp, key=bp.get)}
        print(f"[step08] {label:11s} row={idx}  band-power={bp}  dominant={max(bp, key=bp.get)}")

    (OUT / "time_frequency.json").write_text(json.dumps(summary, indent=2))
    print(f"[step08] images -> {IMG}/(psd|spectrogram|scalogram)_*.png")
    print(f"[step08] -> {OUT/'time_frequency.json'}")


if __name__ == "__main__":
    main()
