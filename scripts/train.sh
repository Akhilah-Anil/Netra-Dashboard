#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train.sh  — trigger training via the REST API and tail the job until done
#
# Usage:
#   ./scripts/train.sh <segments.csv> <dataset.csv> [max_len] [epochs] [latent]
#
# Examples:
#   ./scripts/train.sh data/segments.csv data/dataset.csv
#   ./scripts/train.sh data/segments.csv data/dataset.csv 100 50 64
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
 
API="${ANOMALY_API:-http://localhost:8080}"
SEGMENTS="${1:-}"
DATASET="${2:-}"
MAX_LEN="${3:-100}"
EPOCHS="${4:-50}"
LATENT="${5:-64}"
 
if [[ -z "$SEGMENTS" || -z "$DATASET" ]]; then
  echo "Usage: $0 <segments.csv> <dataset.csv> [max_len] [epochs] [latent]"
  exit 1
fi
 
echo "⏳  Uploading files and starting training..."
RESPONSE=$(curl -s -X POST "$API/train" \
  -F "segments_file=@$SEGMENTS" \
  -F "dataset_file=@$DATASET"   \
  -F "max_len=$MAX_LEN"         \
  -F "epochs=$EPOCHS"           \
  -F "latent=$LATENT")
 
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "✅  Job started: $JOB_ID"
echo "    Training up to $EPOCHS epochs, max_len=$MAX_LEN, latent=$LATENT"
echo ""
 
# Poll until done
DOTS=0
while true; do
  STATUS_JSON=$(curl -s "$API/jobs/$JOB_ID")
  STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
 
  if [[ "$STATUS" == "done" ]]; then
    echo ""
    echo "✅  Training complete!"
    echo "$STATUS_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)['result']
print(f'  Threshold   : {d[\"threshold\"]:.8f}')
print(f'  Train F1    : {d[\"train_f1\"]:.4f}')
print(f'  Precision   : {d[\"train_precision\"]:.4f}')
print(f'  Recall      : {d[\"train_recall\"]:.4f}')
print(f'  Epochs run  : {d[\"epochs_trained\"]}')
print(f'  Sep ratio   : {d[\"sep_ratio\"]:.2f}x')
"
    break
  elif [[ "$STATUS" == "failed" ]]; then
    echo ""
    echo "❌  Training failed!"
    echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['error'])"
    exit 1
  fi
 
  printf "."
  DOTS=$((DOTS+1))
  if [[ $((DOTS % 30)) -eq 0 ]]; then echo " (still running...)"; fi
  sleep 10
done
