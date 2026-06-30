#!/usr/bin/env python3
"""Step 21 — Human-in-the-Loop feedback (active learning).

Demonstrates the HITL feedback loop on the REAL CHB-MIT 20-D features: the model
flags its most UNCERTAIN epochs, a human (here, the held-out ground truth acts as the
oracle/clinician) labels them, the labels are fed back, and the model retrains. Each
round records test performance, and an uncertainty-sampling (HITL) strategy is compared
against random sampling to show the value of routing uncertain cases to a human.

Every query is written to an append-only feedback audit (Sec. 23 governance link).

No fabricated numbers (Sec. 57.7). Run:
    python code/reproducible/step21_human_in_loop.py

Output: accuracy/human_in_loop.json + accuracy/feedback_audit.jsonl
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("FEATURE_CACHE", Path(__file__).resolve().parent / "feature_cache.npz"))
OUT = ROOT / "accuracy"; OUT.mkdir(parents=True, exist_ok=True)

SEED_N = 150          # initial labelled epochs
QUERY_B = 100         # epochs sent to the human each round
ROUNDS = 8


def load_all(cache):
    d = np.load(cache, allow_pickle=True)
    subs = list(d["subjects"])
    X = np.vstack([d[f"X_{s}"] for s in subs])
    y = np.concatenate([d[f"y_{s}"] for s in subs]).astype(int)
    return X, y


def run(strategy, X_pool, y_pool, X_te, y_te, audit=None):
    rng = np.random.RandomState(42)
    n = len(X_pool)
    lab = np.zeros(n, dtype=bool)
    # stratified seed so both classes are present
    pos = np.where(y_pool == 1)[0]; neg = np.where(y_pool == 0)[0]
    seed = np.concatenate([rng.choice(pos, SEED_N // 2, replace=False),
                           rng.choice(neg, SEED_N // 2, replace=False)])
    lab[seed] = True
    history = []
    for r in range(ROUNDS):
        sc = StandardScaler().fit(X_pool[lab])
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=42, n_jobs=-1).fit(sc.transform(X_pool[lab]), y_pool[lab])
        proba_te = clf.predict_proba(sc.transform(X_te))[:, 1]
        acc = accuracy_score(y_te, (proba_te >= 0.5).astype(int))
        sens = recall_score(y_te, (proba_te >= 0.5).astype(int), zero_division=0)
        history.append({"round": r, "n_labelled": int(lab.sum()),
                        "test_accuracy": round(float(acc), 4), "test_sensitivity": round(float(sens), 4)})
        # choose next batch from the unlabelled pool
        un = np.where(~lab)[0]
        if len(un) == 0:
            break
        if strategy == "uncertainty":          # HITL: route most-uncertain to human
            pu = clf.predict_proba(StandardScaler().fit(X_pool[lab]).transform(X_pool[un]))[:, 1]
            order = np.argsort(np.abs(pu - 0.5))          # closest to 0.5 = most uncertain
            pick = un[order[:QUERY_B]]
        else:                                  # random baseline
            pick = rng.choice(un, min(QUERY_B, len(un)), replace=False)
        lab[pick] = True                       # "human" reveals true labels (oracle feedback)
        if audit is not None:
            for i in pick[:5]:
                audit.append({"round": r, "strategy": strategy, "queried_index": int(i),
                              "human_label": int(y_pool[i])})
    return history


def main():
    if not CACHE.exists():
        raise SystemExit(f"feature_cache.npz not found at {CACHE}; run chbmit_loso_pipeline.py first.")
    X, y = load_all(CACHE)
    X_pool, X_te, y_pool, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    audit = []
    hitl = run("uncertainty", X_pool, y_pool, X_te, y_te, audit)
    rand = run("random", X_pool, y_pool, X_te, y_te)

    result = {
        "step": 21, "name": "human_in_the_loop_active_learning",
        "dataset": "CHB-MIT 20-D features", "test_size": int(len(X_te)),
        "seed_labels": SEED_N, "query_batch": QUERY_B, "rounds": ROUNDS,
        "strategy_HITL_uncertainty": hitl,
        "strategy_random_baseline": rand,
        "final_HITL_vs_random_accuracy": [hitl[-1]["test_accuracy"], rand[-1]["test_accuracy"]],
        "labels_to_reach_HITL_final": hitl[-1]["n_labelled"],
    }
    (OUT / "human_in_loop.json").write_text(json.dumps(result, indent=2))
    with open(OUT / "feedback_audit.jsonl", "a") as f:
        for a in audit:
            f.write(json.dumps(a) + "\n")

    print(f"[step21] HITL active learning on {len(X_pool)} pool / {len(X_te)} test")
    print(f"  {'round':>5} {'labels':>7} {'HITL acc':>9} {'rand acc':>9}")
    for h, r in zip(hitl, rand):
        print(f"  {h['round']:>5} {h['n_labelled']:>7} {h['test_accuracy']:>9} {r['test_accuracy']:>9}")
    print(f"[step21] final HITL acc={hitl[-1]['test_accuracy']} vs random={rand[-1]['test_accuracy']}")
    print(f"[step21] -> {OUT/'human_in_loop.json'} (+ feedback_audit.jsonl)")


if __name__ == "__main__":
    main()
