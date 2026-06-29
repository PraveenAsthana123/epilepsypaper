"""Re-run the epoch-level UCI Epileptic Seizure Recognition benchmark on REAL data — verify the paper's number."""
import os, sys
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
# Point UCI_CSV (env) or argv[1] at the public UCI "Epileptic Seizure Recognition" CSV.
CSV = os.environ.get("UCI_CSV", sys.argv[1] if len(sys.argv) > 1 else "data/Epileptic Seizure Recognition.csv")
df=pd.read_csv(CSV)
print("shape:",df.shape,"| last col (label) values:",sorted(df.iloc[:,-1].unique()))
# UCI: col 'y' 1=seizure, 2-5=non-seizure → binary
X=df.iloc[:,1:-1].values; y=(df.iloc[:,-1].values==1).astype(int)
print(f"X:{X.shape} | seizure:{y.sum()} non:{(y==0).sum()}")
clf=RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42,n_jobs=-1)
skf=StratifiedKFold(5,shuffle=True,random_state=42)
yp=cross_val_predict(clf,X,y,cv=skf,n_jobs=-1)
yprob=cross_val_predict(clf,X,y,cv=skf,method='predict_proba',n_jobs=-1)[:,1]
acc=accuracy_score(y,yp); auc=roc_auc_score(y,yprob); sens=recall_score(y,yp); 
tn,fp,fn,tp=confusion_matrix(y,yp).ravel(); spec=tn/(tn+fp); ppv=precision_score(y,yp); f1=f1_score(y,yp)
print("\n=== REAL UCI epoch-level 5-fold (RandomForest 300) ===")
print(f"  accuracy : {acc*100:.2f}%   (paper claims 97.52%)")
print(f"  AUC      : {auc:.4f}  (paper claims 0.996)")
print(f"  sensitivity: {sens*100:.2f}%  (paper claims 93.2%)")
print(f"  specificity: {spec*100:.2f}%  PPV: {ppv*100:.2f}%  F1: {f1*100:.2f}%")
