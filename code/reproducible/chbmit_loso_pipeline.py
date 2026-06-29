#!/usr/bin/env python3
"""chbmit_pipeline.py — REAL CHB-MIT seizure-detection pipeline with SUBJECT-WISE
(Leave-One-Subject-Out) cross-validation, for the Epilepsy dissertation.

Operator 2026-06-23: downloaded CHB-MIT; "run real pipeline -> dissertation".

Honesty (§57.7 + §83): uses only the fully-annotated cases (those with a
chbNN-summary.txt) and reports SUBJECT-WISE CV (LOSO) — the defensible metric
for EEG, NOT the over-optimistic sample-wise 99% in prior runs. Clinical metrics
per §75: sensitivity, specificity, PPV, NPV, AUC, F1.

Pipeline: parse summary -> load EDF (mne) -> 0.5-40Hz bandpass -> 8s epochs ->
label seizure/non-seizure -> per-channel features (δθαβγ band power + Hjorth +
line-length + RMS), mean across channels -> RandomForest -> LOSO CV.

Output: data/eeg/chbmit/chbmit_loso_results.json + chbmit_loso_summary.md
"""
from __future__ import annotations
import os, re, json, sys, random, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
# Full 24-case cohort: flat EDF copy + summaries fetched from PhysioNet.
# Point CHBMIT_DIR (env) or argv[1] at your local CHB-MIT EDF directory; the
# corpus itself is large patient data and is NOT shipped in this repo.
FLAT = Path(os.environ.get("CHBMIT_DIR", sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "chbmit_edf")))
SUMM_DIR = FLAT / "_summaries"
OUT = ROOT / "data" / "eeg" / "chbmit"; OUT.mkdir(parents=True, exist_ok=True)
SF = 256; EPOCH_S = 8; EPOCH_N = SF*EPOCH_S
NEG_PER_POS = 8                 # cap negative:positive epoch ratio per case
NEG_FILES = 2                   # non-seizure EDF files to add per case (runtime cap)
BANDS = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}
random.seed(42); np.random.seed(42)

def parse_summary(p: Path):
    """-> {edf_filename: [(start_s,end_s),...]}"""
    txt = p.read_text(errors="replace"); res={}; cur=None
    for ln in txt.splitlines():
        m=re.match(r"File Name:\s*(\S+)", ln)
        if m: cur=m.group(1); res.setdefault(cur,[])
        s=re.search(r"Seizure.*Start Time:\s*(\d+)", ln)
        e=re.search(r"Seizure.*End Time:\s*(\d+)", ln)
        if s and cur: res[cur].append([int(s.group(1)),None])
        if e and cur and res[cur] and res[cur][-1][1] is None: res[cur][-1][1]=int(e.group(1))
    return {k:[(a,b) for a,b in v if b is not None] for k,v in res.items()}

def hjorth(x):
    d1=np.diff(x); d2=np.diff(d1)
    v0=np.var(x)+1e-12; v1=np.var(d1)+1e-12; v2=np.var(d2)+1e-12
    mob=np.sqrt(v1/v0); comp=np.sqrt(v2/v1)/mob
    return v0, mob, comp

def feats_channel(x):
    f,psd=welch(x, fs=SF, nperseg=min(EPOCH_N,512))
    tot=np.sum(psd)+1e-12
    bp=[np.sum(psd[(f>=lo)&(f<hi)])/tot for lo,hi in BANDS.values()]
    act,mob,comp=hjorth(x)
    ll=np.sum(np.abs(np.diff(x)))         # line length
    rms=np.sqrt(np.mean(x**2))
    return bp+[np.log(act+1e-12),mob,comp,np.log(ll+1e-12),np.log(rms+1e-12)]

def epoch_features(raw):
    data=raw.get_data()                    # (nch, nsamp)
    # drop near-dead channels
    keep=[i for i in range(data.shape[0]) if np.std(data[i])>1e-9]
    data=data[keep]
    n=data.shape[1]//EPOCH_N
    feats=[]
    for k in range(n):
        seg=data[:, k*EPOCH_N:(k+1)*EPOCH_N]
        per=[feats_channel(seg[c]) for c in range(seg.shape[0])]
        per=np.array(per)
        feats.append(np.concatenate([per.mean(0), per.std(0)]))   # mean+std across channels
    return np.array(feats), n

def build_case(cn: str, summ_path: Path, present: dict):
    """cn=case name; present={edf_filename: Path} for this subject."""
    summ=parse_summary(summ_path)
    X=[]; y=[]
    sz_files=[f for f in summ if summ[f] and f in present]
    neg_files=[f for f in present if f not in sz_files]
    random.shuffle(neg_files)
    use=sz_files + neg_files[:NEG_FILES]              # all seizure files + a few negatives
    for fn in use:
        try:
            raw=mne.io.read_raw_edf(present[fn], preload=True, verbose="ERROR")
            raw.filter(0.5,40.0, verbose="ERROR")
            fe,n=epoch_features(raw)
        except Exception as ex:
            print(f"   skip {fn}: {ex}"); continue
        lab=np.zeros(n,dtype=int)
        for (s,e) in summ.get(fn,[]):
            ks=s//EPOCH_S; ke=min(n, e//EPOCH_S+1)
            lab[ks:ke]=1
        for i in range(n):
            X.append(fe[i]); y.append(lab[i])
    X=np.array(X); y=np.array(y)
    # balance negatives per case
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    if len(pos):
        keepn=np.random.choice(neg, min(len(neg), len(pos)*NEG_PER_POS), replace=False)
        idx=np.concatenate([pos,keepn]); np.random.shuffle(idx)
        X,y=X[idx],y[idx]
    print(f"  {cn}: epochs={len(y)} seizure={int(y.sum())} files={len(use)}")
    return X,y

def metrics(yt,yp,ys):
    tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
    sens=tp/(tp+fn+1e-12); spec=tn/(tn+fp+1e-12)
    ppv=tp/(tp+fp+1e-12); npv=tn/(tn+fn+1e-12)
    f1=2*ppv*sens/(ppv+sens+1e-12); acc=(tp+tn)/(tp+tn+fp+fn)
    try: auc=roc_auc_score(yt,ys)
    except Exception: auc=float("nan")
    return dict(sensitivity=sens,specificity=spec,ppv=ppv,npv=npv,f1=f1,
                accuracy=acc,auc=auc,tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn))

def discover():
    """-> {case: (summary_path, {edf_filename: Path})} from the flat copy.
    chb17a/b/c map to subject chb17. Skips .edf.1 re-download dupes."""
    import re as _re
    cases={}
    for n in range(1,25):
        cn=f"chb{n:02d}"
        summ=SUMM_DIR/f"{cn}-summary.txt"
        if not summ.exists(): continue
        edfs={}
        for p in FLAT.glob(f"{cn}*_*.edf"):          # chb05_*, chb17a_* ...
            if p.name.endswith(".edf.1"): continue
            edfs[p.name]=p
        if edfs: cases[cn]=(summ, edfs)
    return cases

def main():
    cases=discover()
    print(f"Cohort: {len(cases)} cases {list(cases)}")
    CX={}; CY={}
    for cn,(summ,edfs) in cases.items():
        X,y=build_case(cn, summ, edfs)
        if len(y) and y.sum()>0: CX[cn]=X; CY[cn]=y
    names=list(CX)
    print(f"Usable cases (with seizures): {names}")
    folds=[]
    for test in names:
        Xtr=np.vstack([CX[n] for n in names if n!=test]); ytr=np.concatenate([CY[n] for n in names if n!=test])
        Xte=CX[test]; yte=CY[test]
        clf=RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=-1,random_state=42)
        clf.fit(Xtr,ytr)
        ys=clf.predict_proba(Xte)[:,1]; yp=(ys>=0.5).astype(int)
        m=metrics(yte,yp,ys); m["test_subject"]=test; m["n_test"]=len(yte); m["n_seizure"]=int(yte.sum())
        folds.append(m)
        print(f"  LOSO test={test}: sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
              f"PPV={m['ppv']:.3f} NPV={m['npv']:.3f} AUC={m['auc']:.3f} F1={m['f1']:.3f}")
    agg={k:float(np.nanmean([f[k] for f in folds])) for k in ("sensitivity","specificity","ppv","npv","f1","accuracy","auc")}
    out=dict(dataset="CHB-MIT", cv="Leave-One-Subject-Out", cases=names,
             epoch_s=EPOCH_S, model="RandomForest(300,balanced)",
             features="per-channel δθαβγ band-power + Hjorth(act/mob/comp) + line-length + RMS, mean+std across channels",
             per_fold=folds, mean=agg)
    (OUT/"chbmit_loso_results.json").write_text(json.dumps(out,indent=2))
    md=[f"# CHB-MIT Seizure Detection — Subject-Wise (LOSO) Results","",
        f"Dataset: CHB-MIT | Cases: {', '.join(names)} | CV: Leave-One-Subject-Out",
        f"Epoch: {EPOCH_S}s | Model: RandomForest(300, class-balanced)","",
        "| Test subject | Sens | Spec | PPV | NPV | AUC | F1 | n(epochs) | n(seizure) |",
        "|---|---|---|---|---|---|---|---|---|"]
    for f in folds:
        md.append(f"| {f['test_subject']} | {f['sensitivity']:.3f} | {f['specificity']:.3f} | "
                  f"{f['ppv']:.3f} | {f['npv']:.3f} | {f['auc']:.3f} | {f['f1']:.3f} | {f['n_test']} | {f['n_seizure']} |")
    md.append(f"| **MEAN** | **{agg['sensitivity']:.3f}** | **{agg['specificity']:.3f}** | "
              f"**{agg['ppv']:.3f}** | **{agg['npv']:.3f}** | **{agg['auc']:.3f}** | **{agg['f1']:.3f}** | | |")
    (OUT/"chbmit_loso_summary.md").write_text("\n".join(md)+"\n")
    print("\nMEAN LOSO:", {k:round(v,3) for k,v in agg.items()})
    print("Results:", OUT/"chbmit_loso_results.json")

if __name__=="__main__":
    main()
