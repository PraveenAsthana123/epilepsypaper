# EEG Data — raw scalp EEG, 10-20 system

Only **real raw EEG** datasets. Samples = 10 rows of raw signal (channel × time) for inspection.

| Dataset | Sampling freq | Bandwidth (Nyquist) | Channels | Montage (10-20) |
|---|---|---|---|---|
| **CHB-MIT** | **256 Hz** | 0–128 Hz | 23 | bipolar: FP1-F7, F7-T7, T7-P7, P7-O1, … |
| **Siena** (PhysioNet) | **512 Hz** | 0–256 Hz | 35 | referential: EEG Fp1, F3, C3, P3, O1, … |

## Frequency bands (δθαβγ) used in feature extraction
δ 0.5–4 · θ 4–8 · α 8–13 · β 13–30 · γ 30–45 Hz.
Preprocessing bandpass 0.5–45 Hz + 50/60 Hz notch (§162 step 5); band-power via Welch PSD.

## 10-20 electrodes
Fp1 Fp2 · F7 F3 Fz F4 F8 · T7 C3 Cz C4 T8 · P7 P3 Pz P4 P8 · O1 O2 (+ refs A1/A2).
