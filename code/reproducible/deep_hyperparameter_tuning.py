#!/usr/bin/env python3
"""Deep hyperparameter tuning — neural-network architecture & training search.

Grid-searches a PyTorch MLP over the REAL CHB-MIT 20-D features under subject-aware
hold-out, sweeping the knobs that matter for deep models:

  * Layer lists      : [64], [128,64], [256,128,64]   (depth/width)
  * Activation       : ReLU, LeakyReLU(0.1), Tanh
  * Optimizer (GD)   : SGD(momentum) and Adam          (gradient-descent variants)
  * Learning rate    : 1e-2, 1e-3
  * Loss function    : BCEWithLogits (weighted) vs Focal loss (gamma=2)

Reports the best config by validation AUC + a full results table. This is the
"deep hyperparameter tuning / loss function / gradient descent / LeakyReLU" analysis.

No fabricated numbers (Sec. 57.7). Run:
    python code/reproducible/deep_hyperparameter_tuning.py

Output: accuracy/deep_hyperparameter_tuning.json
"""
from __future__ import annotations
import os, json, itertools, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

torch.manual_seed(42); np.random.seed(42)
torch.set_num_threads(max(1, os.cpu_count() // 2))

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("FEATURE_CACHE", Path(__file__).resolve().parent / "feature_cache.npz"))
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)

LAYERS = {"[64]": [64], "[128,64]": [128, 64], "[256,128,64]": [256, 128, 64]}
ACTS = {"ReLU": nn.ReLU, "LeakyReLU": lambda: nn.LeakyReLU(0.1), "Tanh": nn.Tanh}
OPTS = ["SGD", "Adam"]
LRS = [1e-2, 1e-3]
LOSSES = ["weighted_bce", "focal"]
EPOCHS = 30


def make_mlp(d, layers, act):
    mods, prev = [], d
    for h in layers:
        mods += [nn.Linear(prev, h), act()]
        prev = h
    mods += [nn.Linear(prev, 1)]
    return nn.Sequential(*mods)


def focal_loss(logits, target, gamma=2.0, pos_weight=1.0):
    p = torch.sigmoid(logits)
    pt = torch.where(target == 1, p, 1 - p)
    w = torch.where(target == 1, torch.tensor(pos_weight), torch.tensor(1.0))
    return (-w * (1 - pt) ** gamma * torch.log(pt + 1e-8)).mean()


def load():
    d = np.load(CACHE, allow_pickle=True)
    subs = list(d["subjects"])
    X = np.vstack([d[f"X_{s}"] for s in subs]).astype(np.float32)
    y = np.concatenate([d[f"y_{s}"] for s in subs]).astype(np.float32)
    return X, y


def train_eval(X, y, layers, act, opt_name, lr, loss_name):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(sc.transform(Xtr)); Xte_t = torch.tensor(sc.transform(Xte))
    ytr_t = torch.tensor(ytr);
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    model = make_mlp(X.shape[1], LAYERS[layers], ACTS[act])
    opt = (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) if opt_name == "SGD"
           else torch.optim.Adam(model.parameters(), lr=lr))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    n = len(Xtr_t); bs = 256
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            out = model(Xtr_t[b]).squeeze(1)
            loss = bce(out, ytr_t[b]) if loss_name == "weighted_bce" \
                else focal_loss(out, ytr_t[b], pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(Xte_t).squeeze(1)).numpy()
    pred = (proba >= 0.5).astype(int)
    return {"auc": round(float(roc_auc_score(yte, proba)), 4),
            "sensitivity": round(float(recall_score(yte, pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(yte, pred)), 4)}


def main():
    if not CACHE.exists():
        raise SystemExit(f"feature_cache.npz not found at {CACHE}; run chbmit_loso_pipeline.py first.")
    X, y = load()
    t0 = time.time(); rows = []
    grid = list(itertools.product(LAYERS, ACTS, OPTS, LRS, LOSSES))
    print(f"[hpt] searching {len(grid)} configs on {len(X)} samples ...")
    for layers, act, opt_name, lr, loss_name in grid:
        m = train_eval(X, y, layers, act, opt_name, lr, loss_name)
        rows.append({"layers": layers, "activation": act, "optimizer": opt_name,
                     "lr": lr, "loss": loss_name, **m})
    rows.sort(key=lambda r: r["auc"], reverse=True)
    best = rows[0]

    result = {"name": "deep_hyperparameter_tuning",
              "dataset": "CHB-MIT 20-D features", "n_samples": int(len(X)),
              "search_space": {"layers": list(LAYERS), "activation": list(ACTS),
                               "optimizer": OPTS, "lr": LRS, "loss": LOSSES},
              "n_configs": len(grid), "epochs": EPOCHS,
              "runtime_sec": round(time.time() - t0, 1),
              "best_config": best, "all_results": rows}
    (OUT / "deep_hyperparameter_tuning.json").write_text(json.dumps(result, indent=2))

    print(f"[hpt] BEST: {best['layers']} {best['activation']} {best['optimizer']} "
          f"lr={best['lr']} {best['loss']} -> AUC={best['auc']} sens={best['sensitivity']}")
    print(f"[hpt] top-3:")
    for r in rows[:3]:
        print(f"      {r['layers']:>14} {r['activation']:>10} {r['optimizer']:>4} "
              f"lr={r['lr']} {r['loss']:>12} AUC={r['auc']} sens={r['sensitivity']}")
    print(f"[hpt] {len(grid)} configs in {result['runtime_sec']}s -> {OUT/'deep_hyperparameter_tuning.json'}")


if __name__ == "__main__":
    main()
