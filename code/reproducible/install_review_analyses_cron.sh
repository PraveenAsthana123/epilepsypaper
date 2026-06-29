#!/usr/bin/env bash
# Install/remove the cron that drafts the §164 review analyses via local Ollama.
# Idempotent (tagged). Drafts are NOT auto-applied — operator reviews (§50.5).
#   install:  install_review_analyses_cron.sh install
#   remove:   install_review_analyses_cron.sh remove
#   status:   install_review_analyses_cron.sh status
set -uo pipefail

# Override REPO/VENV via env if needed; defaults resolve to this repo + system python3.
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
VENV="${VENV:-python3}"
TAG="# EPILEPSY-REVIEW-ANALYSES (Ollama §164)"
LINE="*/30 * * * * cd $REPO && $VENV code/reproducible/run_review_analyses_ollama.py --batch 2 >> $REPO/.loop/review_analyses/cron.log 2>&1 $TAG"

case "${1:-status}" in
  install)
    crontab -l 2>/dev/null | grep -vF "$TAG" > /tmp/cron.rev || true
    printf '%s\n' "$LINE" >> /tmp/cron.rev
    crontab /tmp/cron.rev
    echo "installed (every 30 min, 2 analyses/run):"; crontab -l | grep -F "$TAG" ;;
  remove)
    crontab -l 2>/dev/null | grep -vF "$TAG" | crontab -
    echo "removed" ;;
  status)
    crontab -l 2>/dev/null | grep -F "$TAG" || echo "not installed" ;;
esac
