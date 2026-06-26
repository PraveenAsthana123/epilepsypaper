#!/usr/bin/env python3
"""Comprehensive accuracy + statistical analysis of the REAL CHB-MIT LOSO results.

Reads accuracy/chbmit_loso_results.json (which carries tp/tn/fp/fn PER subject),
computes every standard classification metric + statistical tests across the 24
subjects, and renders the confusion matrix + ROC/AUC + per-subject plots.

No fabricated numbers (§57.7): everything is derived from the committed per-fold
true/false positive/negative counts. Run:
    python code/reproducible/comprehensive_accuracy_analysis.py
"""
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "accuracy" / "chbmit_loso_results.json"
IMG = REPO / "images"
OUT_JSON = REPO / "accuracy" / "comprehensive_metrics.json"
OUT_MD = REPO / "accuracy" / "COMPREHENSIVE_METRICS.md"


def metrics_from_confusion(tp, tn, fp, fn):
    """Compute the full metric suite from raw confusion counts."""
    tot = tp + tn + fp + fn
    acc = (tp + tn) / tot if tot else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0          # precision / PPV
    rec = tp / (tp + fn) if (tp + fn) else 0.0           # recall / sensitivity
    spec = tn / (tn + fp) if (tn + fp) else 0.0          # specificity
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    bal_acc = (rec + spec) / 2
    # Matthews correlation coefficient
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    # Cohen's kappa
    po = acc
    pe = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / (tot * tot) if tot else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    return dict(accuracy=acc, precision=prec, recall=rec, sensitivity=rec,
                specificity=spec, npv=npv, f1=f1, balanced_accuracy=bal_acc,
                mcc=mcc, cohens_kappa=kappa)


def ci95(arr):
    """95% confidence interval of the mean (t-based)."""
    a = np.asarray(arr, float)
    n = len(a)
    if n < 2:
        return (float(a.mean()), float(a.mean()))
    se = a.std(ddof=1) / math.sqrt(n)
    t = 2.069  # t_{0.975, df≈23}
    return (float(a.mean() - t * se), float(a.mean() + t * se))


def main():
    IMG.mkdir(exist_ok=True)
    d = json.loads(RESULTS.read_text())
    folds = d["per_fold"]

    # --- aggregate confusion matrix (sum across all 24 LOSO folds) ---
    TP = sum(f["tp"] for f in folds); TN = sum(f["tn"] for f in folds)
    FP = sum(f["fp"] for f in folds); FN = sum(f["fn"] for f in folds)
    agg = metrics_from_confusion(TP, TN, FP, FN)
    agg["auc_mean"] = float(np.mean([f["auc"] for f in folds]))
    confusion = [[TN, FP], [FN, TP]]  # rows=actual[neg,pos], cols=pred[neg,pos]

    # --- per-subject statistics for each metric ---
    stats = {}
    for key in ["accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "auc"]:
        vals = [f[key] for f in folds]
        lo, hi = ci95(vals)
        stats[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)),
                          median=float(np.median(vals)),
                          q1=float(np.percentile(vals, 25)), q3=float(np.percentile(vals, 75)),
                          min=float(np.min(vals)), max=float(np.max(vals)),
                          ci95_low=lo, ci95_high=hi)

    # --- statistical test: is accuracy significantly above chance (0.5)? ---
    accs = np.array([f["accuracy"] for f in folds])
    t_stat = (accs.mean() - 0.5) / (accs.std(ddof=1) / math.sqrt(len(accs)))
    sig_above_chance = bool(t_stat > 2.069)  # one-sided, df≈23

    report = dict(
        dataset=d["dataset"], cv=d["cv"], n_subjects=len(folds), model=d["model"],
        aggregate_confusion_matrix=dict(matrix=confusion, labels=["no-seizure", "seizure"],
                                        TP=TP, TN=TN, FP=FP, FN=FN),
        aggregate_metrics={k: round(v, 4) for k, v in agg.items()},
        per_subject_statistics=stats,
        accuracy_above_chance=dict(t_stat=round(t_stat, 3), significant_p_lt_0_05=sig_above_chance),
    )
    OUT_JSON.write_text(json.dumps(report, indent=2))

    # --- plots: confusion matrix + per-subject AUC ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = np.array(confusion)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no-seizure", "seizure"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["no-seizure", "seizure"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"CHB-MIT LOSO confusion (acc {agg['accuracy']*100:.1f}%)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout(); fig.savefig(IMG / "confusion_matrix.png", dpi=120); plt.close(fig)
        # per-subject AUC bar
        fig, ax = plt.subplots(figsize=(8, 3))
        aucs = [f["auc"] for f in folds]; subs = [f["test_subject"] for f in folds]
        ax.bar(range(len(aucs)), aucs, color="#3b82f6")
        ax.axhline(0.5, color="red", ls="--", lw=1, label="chance")
        ax.set_xticks(range(len(subs))); ax.set_xticklabels(subs, rotation=90, fontsize=7)
        ax.set_ylabel("AUC"); ax.set_title("Per-subject AUC (LOSO, 24 subjects)")
        ax.legend(); fig.tight_layout(); fig.savefig(IMG / "per_subject_auc.png", dpi=120); plt.close(fig)
        plots = "confusion_matrix.png, per_subject_auc.png"
    except Exception as e:
        plots = f"(plotting skipped: {e})"

    # --- markdown report ---
    m = report["aggregate_metrics"]
    OUT_MD.write_text(f"""# Comprehensive Accuracy + Statistical Report — CHB-MIT LOSO (real)

Derived from `chbmit_loso_results.json` (per-subject tp/tn/fp/fn, {len(folds)} subjects).
Model: {d['model']}. No fabricated numbers (§57.7).

## Aggregate confusion matrix (summed over 24 LOSO folds)
|              | Pred no-seizure | Pred seizure |
|--------------|----------------:|-------------:|
| **Actual no-seizure** | {TN:,} (TN) | {FP:,} (FP) |
| **Actual seizure**    | {FN:,} (FN) | {TP:,} (TP) |

## All accuracy metrics (aggregate)
| Metric | Value |
|---|---|
| **Accuracy** | **{m['accuracy']*100:.2f}%** |
| Precision (PPV) | {m['precision']*100:.2f}% |
| Recall (Sensitivity) | {m['recall']*100:.2f}% |
| Specificity | {m['specificity']*100:.2f}% |
| NPV | {m['npv']*100:.2f}% |
| F1-score | {m['f1']:.4f} |
| Balanced accuracy | {m['balanced_accuracy']*100:.2f}% |
| MCC | {m['mcc']:.4f} |
| Cohen's kappa | {m['cohens_kappa']:.4f} |
| AUC (mean) | {m['auc_mean']:.4f} |

## Statistical (per-subject, n={len(folds)})
| Metric | Mean ± SD | 95% CI | Median [IQR] |
|---|---|---|---|
""" + "\n".join(
        f"| {k} | {s['mean']*100:.1f}% ± {s['std']*100:.1f} | "
        f"[{s['ci95_low']*100:.1f}, {s['ci95_high']*100:.1f}] | "
        f"{s['median']*100:.1f}% [{s['q1']*100:.1f}–{s['q3']*100:.1f}] |"
        for k, s in stats.items()
    ) + f"""

**Significance:** accuracy is {'significantly' if sig_above_chance else 'NOT significantly'}
above chance (one-sample t vs 0.5: t={t_stat:.2f}, df={len(folds)-1}, {'p<0.05' if sig_above_chance else 'n.s.'}).

## Honest caveat (§162 invariant)
Accuracy is high ({m['accuracy']*100:.1f}%) because non-seizure epochs dominate. **Sensitivity
is {m['recall']*100:.1f}%** — the deployable signal. High accuracy alone is misleading on
imbalanced seizure data; report sensitivity + AUC + MCC, not accuracy alone.

Plots: {plots} (in `images/`).
""")
    print(f"✓ aggregate accuracy: {m['accuracy']*100:.2f}% · sensitivity: {m['recall']*100:.2f}% · "
          f"MCC: {m['mcc']:.3f} · kappa: {m['cohens_kappa']:.3f}")
    print(f"✓ wrote {OUT_JSON.name}, {OUT_MD.name}, plots: {plots}")


if __name__ == "__main__":
    main()
