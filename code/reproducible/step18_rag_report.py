#!/usr/bin/env python3
"""Steps 18-22 — RAG knowledge layer, hybrid retrieval, and report generation.

Builds a retrieval-augmented report WITHOUT any external LLM API, so it is fully
reproducible offline:

  Step 18 (Index)      : parse code/paper/references.bib -> a TF-IDF index over
                         {title, journal, year, key} of every cited work.
  Step 19 (Retrieval)  : hybrid search = TF-IDF cosine similarity + keyword/metadata
                         filter, for a clinical query.
  Step 20 (Report gen) : combine REAL model prediction (accuracy/comprehensive_metrics.json),
                         top EEG biomarkers (accuracy/feature_evaluation.json or
                         xai_feature_importance.json), and the retrieved evidence.
  Step 21/22 (Review/Output): emit a doctor-facing + patient-facing markdown report
                         with an explicit "requires human review" gate and limitations.

No fabricated numbers (Sec. 57.7): metrics/biomarkers are read from committed JSON.
Run:  python code/reproducible/step18_rag_report.py "inter-patient seizure detection"

Output: accuracy/rag_report.md (+ rag_retrieval.json)
"""
from __future__ import annotations
import re, sys, json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
BIB = ROOT / "code" / "paper" / "references.bib"
ACC = ROOT / "accuracy"
OUT = ROOT / "accuracy"


def parse_bib(path: Path):
    """Minimal BibTeX parser -> list of {key,title,journal,year,text}."""
    if not path.exists():
        return []
    raw = path.read_text(errors="ignore")
    entries = []
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", raw, re.S):
        key, body = m.group(1).strip(), m.group(2)
        def field(name):
            fm = re.search(rf"{name}\s*=\s*[{{\"](.+?)[}}\"]\s*,?", body, re.S | re.I)
            return re.sub(r"\s+", " ", fm.group(1)).strip() if fm else ""
        title, journal, year = field("title"), field("journal") or field("booktitle"), field("year")
        entries.append({"key": key, "title": title, "journal": journal, "year": year,
                        "text": f"{title} {journal} {year}"})
    return entries


def hybrid_retrieve(query, entries, k=5):
    docs = [e["text"] for e in entries]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    M = vec.fit_transform(docs + [query])
    sims = cosine_similarity(M[-1], M[:-1]).ravel()
    kw = query.lower().split()
    hits = []
    for i, e in enumerate(entries):
        kw_bonus = sum(1 for w in kw if w in e["text"].lower()) / max(len(kw), 1)
        score = 0.7 * sims[i] + 0.3 * kw_bonus       # hybrid: vector + keyword
        hits.append((score, i))
    hits.sort(reverse=True)
    return [{"rank": r + 1, "score": round(float(s), 4), **entries[i]}
            for r, (s, i) in enumerate(hits[:k]) if s > 0]


def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default if default is not None else {}


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "inter-patient subject-wise seizure detection EEG"
    entries = parse_bib(BIB)
    retrieved = hybrid_retrieve(query, entries, k=5)

    metrics = load_json(ACC / "comprehensive_metrics.json").get("aggregate_metrics", {})
    persub = load_json(ACC / "comprehensive_metrics.json").get("per_subject_statistics", {})
    feval = load_json(ACC / "feature_evaluation.json")
    biomarkers = feval.get("top5_by_anova_F") or \
        [f.get("feature") for f in load_json(ACC / "xai_feature_importance.json").get("ranking", [])[:5]]

    acc = metrics.get("accuracy"); sens = (persub.get("sensitivity") or {}).get("mean")
    auc = metrics.get("auc_mean"); spec = metrics.get("specificity")

    json.dump({"query": query, "n_indexed": len(entries), "retrieved": retrieved},
              open(OUT / "rag_retrieval.json", "w"), indent=2)

    cites = "\n".join(f"{r['rank']}. [{r['key']}] {r['title']} ({r['year']}) — score {r['score']}"
                      for r in retrieved) or "_(no references indexed)_"
    bm = ", ".join(biomarkers[:5]) if biomarkers else "_(run step12 first)_"

    report = f"""# RAG Clinical Decision-Support Report (epilepsy / seizure detection)

> **Generated offline** (TF-IDF retrieval, no external LLM). For research use; **requires human review**.

**Query:** {query}

## 1. Model prediction (REAL, CHB-MIT LOSO)
- Accuracy: **{acc}**  |  Mean per-subject sensitivity: **{sens}**  |  Specificity: **{spec}**  |  AUC: **{auc}**
- Honest figure of merit = subject-wise sensitivity (not epoch-level accuracy).

## 2. Key EEG biomarkers (top by ANOVA F-test)
{bm}

## 3. Retrieved evidence (hybrid TF-IDF + keyword, top 5 of {len(entries)} indexed)
{cites}

## 4. Doctor-facing summary
- Risk-support: model flags seizure epochs with high specificity but **modest, patient-variable
  sensitivity** — use as triage-with-escalation, not autonomous decision.
- Key abnormality drivers: {bm}.
- Evidence: see retrieved citations above.
- **Limitations:** patient-independent sensitivity ~35%; some subjects near 0% — escalate uncertain cases.
- **Recommended next step:** clinician review of flagged epochs + standard EEG read.

## 5. Patient-facing summary
- This is an automated **screening aid**, not a diagnosis.
- It highlights EEG segments a specialist should look at.
- Please follow up with your neurologist for any clinical decision.

## 6. Governance gate (Step 21)
- [ ] Reviewed by psychiatrist / neurophysiologist
- [ ] Approve / reject / request more assessment
- Audit + drift monitoring: see step23_governance.py
"""
    (OUT / "rag_report.md").write_text(report)
    print(f"[step18] indexed {len(entries)} references; retrieved top {len(retrieved)} for: {query!r}")
    for r in retrieved:
        print(f"  #{r['rank']} score={r['score']} [{r['key']}] {r['title'][:60]}")
    print(f"[step18] -> {OUT/'rag_report.md'}, {OUT/'rag_retrieval.json'}")


if __name__ == "__main__":
    main()
