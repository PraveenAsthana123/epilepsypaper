"""Siena Scalp EEG — Leave-One-Subject-Out seizure detection (3rd dataset, real).
Same 20-D feature pipeline + RandomForest as the CHB-MIT LOSO, for a fair cross-dataset comparison.
Annotations: Seizures-list-PNxx.txt (clock times → seconds offset from registration start).
Run: python siena_loso_pipeline.py /path/to/siena_epilepsy_physionet
"""
import sys, os, glob, re, json
import numpy as np
from datetime import datetime
try:
    import mne
    from scipy.signal import welch
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
except ImportError as e:
    print("needs mne, scipy, scikit-learn:", e); sys.exit(1)

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
FS_TARGET = 256; EPOCH_S = 8; BANDS = [(0.5,4),(4,8),(8,13),(13,30),(30,45)]

def parse_clock(t):
    t = t.strip().replace(".", ":")
    for fmt in ("%H:%M:%S",):
        try: return datetime.strptime(t, fmt)
        except: pass
    return None

def seizure_offsets(ann_file):
    """Return list of (start_s, end_s) offsets from registration start."""
    txt = open(ann_file, errors="replace").read()
    rs = re.search(r"Registration start time:\s*([0-9.:]+)", txt)
    reg = parse_clock(rs.group(1)) if rs else None
    if not reg: return []
    out = []
    for m in re.finditer(r"Start time:\s*([0-9.:]+).*?End time:\s*([0-9.:]+)", txt, re.DOTALL):
        s, e = parse_clock(m.group(1)), parse_clock(m.group(2))
        if not s or not e: continue
        ss = (s - reg).total_seconds() % 86400
        ee = (e - reg).total_seconds() % 86400
        if ee < ss: ee += 86400
        out.append((ss, ee))
    return out

def features(epoch, fs):
    f = []
    for ch in epoch:
        freqs, psd = welch(ch, fs, nperseg=min(len(ch), fs*2))
        tot = psd.sum() + 1e-9
        for lo, hi in BANDS:
            f.append(psd[(freqs >= lo) & (freqs < hi)].sum() / tot)
        d = np.diff(ch); dd = np.diff(d)
        act = ch.var() + 1e-9; mob = np.sqrt((d.var()+1e-9)/act)
        f += [act, mob, np.sqrt((dd.var()+1e-9)/(d.var()+1e-9))/(mob+1e-9)]
        f.append(np.abs(d).sum()); f.append(np.sqrt((ch**2).mean()))
    f = np.array(f).reshape(-1, 10)
    return np.concatenate([f.mean(0), f.std(0)])  # 20-D

def load_subject(pn_dir, ann):
    edfs = glob.glob(os.path.join(pn_dir, "*.edf")) + glob.glob(os.path.join(os.path.dirname(pn_dir), os.path.basename(pn_dir)+"*.edf"))
    sz = seizure_offsets(ann) if ann else []
    X, y = [], []
    for edf in sorted(set(edfs)):
        try:
            raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
            raw.filter(0.5, 40, verbose="ERROR"); raw.resample(FS_TARGET, verbose="ERROR")
            data = raw.get_data(); fs = FS_TARGET; n = int(EPOCH_S*fs)
            for i in range(0, data.shape[1]-n, n):
                t0 = i/fs; t1 = (i+n)/fs
                lab = int(any(t0 < e and t1 > s for s, e in sz))
                X.append(features(data[:, i:i+n], fs)); y.append(lab)
        except Exception as ex:
            print(f"   skip {os.path.basename(edf)}: {ex}")
    return np.array(X), np.array(y)

def main():
    subs = {}
    for pn in sorted(glob.glob(os.path.join(ROOT, "PN*"))):
        name = os.path.basename(pn)
        if not re.match(r"PN\d+$", name): continue
        ann = glob.glob(os.path.join(ROOT, f"*{name}*.txt")) + glob.glob(os.path.join(pn, "*.txt"))
        ann = [a for a in ann if "eizure" in a.lower()]
        X, y = load_subject(pn, ann[0] if ann else None)
        if len(X) and y.sum() > 0:  # need at least one seizure epoch
            subs[name] = (X, y); print(f"  {name}: {len(X)} epochs, {int(y.sum())} seizure")
    if len(subs) < 3:
        print(f"\nONLY {len(subs)} usable subjects — Siena download incomplete. Re-run when full."); return
    # LOSO
    res = []
    names = list(subs)
    for test in names:
        Xtr = np.vstack([subs[n][0] for n in names if n != test])
        ytr = np.concatenate([subs[n][1] for n in names if n != test])
        Xte, yte = subs[test]
        clf = RandomForestClassifier(300, class_weight="balanced", random_state=42, n_jobs=-1).fit(Xtr, ytr)
        yp = clf.predict(Xte); yprob = clf.predict_proba(Xte)[:, 1]
        sens = recall_score(yte, yp, zero_division=0)
        try: auc = roc_auc_score(yte, yprob)
        except: auc = float("nan")
        tn, fp, fn, tp = confusion_matrix(yte, yp, labels=[0,1]).ravel()
        spec = tn/(tn+fp+1e-9)
        res.append({"subject": test, "sensitivity": round(sens,3), "specificity": round(spec,3), "auc": round(auc,3)})
        print(f"  LOSO {test}: sens={sens:.3f} spec={spec:.3f} auc={auc:.3f}")
    sens = np.mean([r["sensitivity"] for r in res]); spec = np.mean([r["specificity"] for r in res])
    auc = np.nanmean([r["auc"] for r in res])
    summary = {"dataset": "Siena Scalp EEG (LOSO)", "n_subjects": len(subs),
               "mean": {"sensitivity": round(sens,3), "specificity": round(spec,3), "auc": round(auc,3)},
               "per_subject": res}
    out = os.path.join(os.path.dirname(__file__), "..", "results", "siena_loso_results.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n=== Siena LOSO ({len(subs)} subjects): sens={sens:.3f} spec={spec:.3f} auc={auc:.3f} → {out}")

if __name__ == "__main__":
    main()
