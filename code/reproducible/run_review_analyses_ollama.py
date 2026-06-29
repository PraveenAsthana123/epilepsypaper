#!/usr/bin/env python3
"""Dispatch the 26-category Scopus-Q1 review analyses (§164) to a local Ollama model.

Reads config/review_analyses_plan.json; for each category NOT already drafted and NOT
status 'done', sends its `ollama_task` to a local Ollama coder model and saves the
draft to .loop/review_analyses/<id>/draft.md. Incremental: processes --batch N per run
so a cron can chip away without overloading. Drafts are NEVER auto-applied — the
operator/Claude reviews + verifies before anything lands (§50.5/§156). Audit row per
dispatch to .loop/review_analyses/audit.jsonl.

Run:  python code/reproducible/run_review_analyses_ollama.py --batch 3
Cron: */30 * * * * cd <repo> && <venv>/python code/reproducible/run_review_analyses_ollama.py --batch 2
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "config" / "review_analyses_plan.json"
OUT = REPO / ".loop" / "review_analyses"
AUDIT = OUT / "audit.jsonl"
OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5-coder:14b"
SKIP_STATUS = {"done"}


def ollama(prompt: str, model: str = MODEL, timeout: int = 600) -> tuple[str, int, float]:
    """Call the local Ollama model; return (text, n_tokens, latency_s)."""
    t0 = time.time()
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "options": {"temperature": 0.2, "num_ctx": 8192}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d.get("response", ""), d.get("eval_count", 0), round(time.time() - t0, 1)


def build_prompt(cat: dict) -> str:
    return (
        "You are drafting ONE analysis for a Scopus Q1 systematic review on AI for "
        "epilepsy EEG. Output ONLY: (1) a short method (what to compute + how), (2) the "
        "Python/markdown to produce it, (3) what real data it needs. Do not fabricate "
        "numbers — describe the procedure that yields real results.\n\n"
        f"## Analysis: {cat['title']} ({cat['id']})\n"
        f"Sub-analyses: {', '.join(cat['items'])}\n"
        f"Data source on disk: {cat['data_source']}\n"
        f"Task: {cat['ollama_task']}\n")


def audit(row: dict):
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2, help="how many to draft this run")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--force", action="store_true", help="re-draft even if a draft exists")
    a = ap.parse_args()

    plan = json.loads(PLAN.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    done = 0
    for cat in plan["categories"]:
        if done >= a.batch:
            break
        if cat["status"] in SKIP_STATUS:
            continue
        draft_path = OUT / cat["id"] / "draft.md"
        if draft_path.exists() and not a.force:
            continue
        print(f"[dispatch] {cat['id']} :: {cat['title']} -> Ollama ({a.model})")
        try:
            text, ntok, lat = ollama(build_prompt(cat), a.model)
        except Exception as e:
            print(f"  ✗ failed: {e}")
            audit({"ts": datetime.now(timezone.utc).isoformat(), "id": cat["id"],
                   "outcome": f"error:{e}", "model": a.model})
            continue
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(
            f"# DRAFT (Ollama, NOT applied — review per §50.5/§156) · {cat['title']}\n\n"
            f"- model: {a.model} · {lat}s · {ntok} tokens · status was: {cat['status']}\n\n"
            f"---\n\n{text}\n")
        audit({"ts": datetime.now(timezone.utc).isoformat(), "id": cat["id"],
               "outcome": "drafted", "model": a.model, "tokens": ntok, "latency_s": lat})
        print(f"  ✓ draft: {draft_path} ({lat}s, {ntok} tok)")
        done += 1

    remaining = sum(1 for c in plan["categories"]
                    if c["status"] not in SKIP_STATUS and not (OUT / c["id"] / "draft.md").exists())
    print(f"drafted {done} this run · {remaining} categories still pending")


if __name__ == "__main__":
    main()
