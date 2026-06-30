#!/usr/bin/env python3
"""Advanced models — Computer Vision (CNN), RNN (LSTM), Transformer, and
Reinforcement Learning — on REAL epilepsy EEG, compared head-to-head.

All four are trained with PyTorch (CPU) on the UCI Epileptic Seizure Recognition
set (178-sample raw segments, seizure=class 1), on a class-balanced subsample with
a fixed stratified 80/20 split. Honest protocol label: epoch-level hold-out.

  * CNN  (computer vision): STFT spectrogram image -> 2-layer ConvNet
  * RNN  (LSTM)          : raw 1D sequence (178 timesteps) -> LSTM -> linear
  * Transformer          : raw sequence -> linear embed + positional enc -> encoder
  * RL   (REINFORCE)     : contextual-bandit policy net on the spectrogram features;
                           reward +1 correct / -1 wrong; learns a classification policy

No fabricated numbers (Sec. 57.7). Run:
    UCI_CSV="/path/Epileptic Seizure Recognition.csv" \
    python code/reproducible/advanced_models.py

Output: accuracy/advanced_models.json + accuracy/ADVANCED_MODELS.md
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

torch.manual_seed(42); np.random.seed(42)
torch.set_num_threads(max(1, os.cpu_count() // 2))
DEV = "cpu"

ROOT = Path(__file__).resolve().parents[2]
UCI_CSV = os.environ.get("UCI_CSV", sys.argv[1] if len(sys.argv) > 1 else "data/Epileptic Seizure Recognition.csv")
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)
FS = 173.61
N_PER_CLASS = 2300      # balanced subsample per class (all seizures available)
EPOCHS = 14


def load_balanced():
    import pandas as pd
    df = pd.read_csv(UCI_CSV)
    X = df.iloc[:, 1:-1].values.astype(np.float32)
    y = (df.iloc[:, -1].values == 1).astype(np.int64)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    neg = rng.choice(neg, len(pos), replace=False)
    idx = rng.permutation(np.concatenate([pos, neg]))
    return X[idx], y[idx]


def spectrograms(X):
    """1D segments -> STFT magnitude images (computer-vision input)."""
    imgs = []
    for sig in X:
        _, _, Z = stft(sig, fs=FS, nperseg=32, noverlap=16)
        imgs.append(np.abs(Z).astype(np.float32))
    A = np.stack(imgs)                       # (N, F, T)
    A = (A - A.mean()) / (A.std() + 1e-6)
    return A[:, None, :, :]                   # (N, 1, F, T)


def metrics(y, p, proba):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = round(float(roc_auc_score(y, proba)), 4)
    except ValueError:
        auc = None
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "sensitivity": round(float(recall_score(y, p, zero_division=0)), 4),
            "specificity": round(float(spec), 4),
            "f1": round(float(f1_score(y, p, zero_division=0)), 4), "auc": auc}


# ----------------------- model definitions -----------------------
class CNN(nn.Module):
    def __init__(self, fbins, tbins):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1)).squeeze(1)


class TCN(nn.Module):
    """Time-series model: 1D temporal convolution over the raw EEG sequence."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(32, 1)

    def forward(self, x):                     # x: (N, T)
        return self.fc(self.net(x.unsqueeze(1)).flatten(1)).squeeze(1)


class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):                     # x: (N, T)
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1, :]).squeeze(1)


class TransformerNet(nn.Module):
    def __init__(self, seq=178, d=32):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, seq, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=64,
                                         batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Linear(d, 1)

    def forward(self, x):                     # x: (N, T)
        h = self.embed(x.unsqueeze(-1)) + self.pos[:, :x.shape[1], :]
        return self.fc(self.tr(h).mean(1)).squeeze(1)


def train_torch(model, Xtr, ytr, Xte, yte, pos_weight):
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.float32)
    n = len(Xtr_t); bs = 128
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xtr_t[b])
            loss = lossf(out, ytr_t[b]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte))
        proba = torch.sigmoid(logits).numpy()
    return metrics(yte, (proba >= 0.5).astype(int), proba)


# ----------------------- reinforcement learning -----------------------
class Policy(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(1)


def train_rl(Xtr, ytr, Xte, yte):
    """REINFORCE contextual bandit: action = predict seizure(1)/not(0),
    reward = +1 if action==true label else -1. Learns a classification policy."""
    d = Xtr.shape[1]
    pol = Policy(d).to(DEV)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    n = len(Xtr_t); bs = 256; baseline = 0.0
    for ep in range(EPOCHS * 3):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            p1 = pol(Xtr_t[b])                       # P(action=1)
            dist = torch.distributions.Bernoulli(p1)
            action = dist.sample()
            reward = torch.where(action == ytr_t[b].float(),
                                 torch.tensor(1.0), torch.tensor(-1.0))
            baseline = 0.95 * baseline + 0.05 * reward.mean().item()
            loss = -((reward - baseline) * dist.log_prob(action)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        proba = pol(torch.tensor(Xte)).numpy()
    return metrics(yte, (proba >= 0.5).astype(int), proba)


def main():
    t0 = time.time()
    X, y = load_balanced()
    print(f"[adv] balanced UCI subsample: {len(X)} segments ({y.sum()} seizure)")

    # raw-sequence split (for LSTM / Transformer / RL-on-features)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    # spectrogram split (computer vision)
    S = spectrograms(X)
    Sx_tr, Sx_te, sy_tr, sy_te = train_test_split(S, y, test_size=0.2, stratify=y, random_state=42)
    fbins, tbins = S.shape[2], S.shape[3]

    results = {}
    print("[adv] training CNN (computer vision on spectrograms)...")
    results["CNN_computer_vision"] = train_torch(CNN(fbins, tbins), Sx_tr, sy_tr, Sx_te, sy_te, pw)
    print(f"      {results['CNN_computer_vision']}")

    print("[adv] training Time-Series model (1D temporal CNN)...")
    results["TimeSeries_TCN"] = train_torch(TCN(), Xtr, ytr, Xte, yte, pw)
    print(f"      {results['TimeSeries_TCN']}")

    print("[adv] training RNN (LSTM on raw sequence)...")
    results["RNN_LSTM"] = train_torch(LSTMNet(), Xtr, ytr, Xte, yte, pw)
    print(f"      {results['RNN_LSTM']}")

    print("[adv] training Transformer (attention on raw sequence)...")
    results["Transformer"] = train_torch(TransformerNet(seq=X.shape[1]), Xtr, ytr, Xte, yte, pw)
    print(f"      {results['Transformer']}")

    print("[adv] training RL (REINFORCE contextual bandit on spectrogram features)...")
    Xf = S.reshape(len(S), -1)
    Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-6)
    Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(Xf.astype(np.float32), y, test_size=0.2,
                                                  stratify=y, random_state=42)
    results["RL_REINFORCE"] = train_rl(Xf_tr, yf_tr, Xf_te, yf_te)
    print(f"      {results['RL_REINFORCE']}")

    out = {"benchmark": "advanced_models", "framework": f"pytorch {torch.__version__} (cpu)",
           "dataset": "UCI-178raw (balanced subsample)", "n_segments": int(len(X)),
           "validation": "epoch-level stratified 80/20 hold-out",
           "epochs": EPOCHS, "spectrogram_shape": [fbins, tbins],
           "runtime_sec": round(time.time() - t0, 1), "results": results}
    (OUT / "advanced_models.json").write_text(json.dumps(out, indent=2))

    lines = ["# Advanced Models — CNN / RNN / Transformer / RL\n",
             f"PyTorch (CPU), UCI balanced subsample ({len(X)} segments), epoch-level 80/20. "
             "Real data, no fabrication (Sec. 57.7).\n",
             "| Model | Type | Acc | Sens | Spec | F1 | AUC |", "|---|---|---|---|---|---|---|"]
    typ = {"CNN_computer_vision": "Computer Vision (CNN)", "TimeSeries_TCN": "Time-Series (1D temporal CNN)",
           "RNN_LSTM": "RNN (LSTM)", "Transformer": "Transformer (attention)",
           "RL_REINFORCE": "Reinforcement Learning"}
    for k, m in results.items():
        lines.append(f"| {k} | {typ[k]} | {m['accuracy']} | {m['sensitivity']} | "
                     f"{m['specificity']} | {m['f1']} | {m['auc']} |")
    (OUT / "ADVANCED_MODELS.md").write_text("\n".join(lines) + "\n")

    print(f"[adv] done in {out['runtime_sec']}s -> {OUT/'advanced_models.json'}, {OUT/'ADVANCED_MODELS.md'}")


if __name__ == "__main__":
    main()
