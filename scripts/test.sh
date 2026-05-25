#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# test.sh  — trigger inference via the REST API and tail the job until done
#
# Usage:
#   ./scripts/test.sh <segments.csv> <dataset.csv> [threshold_override]
#
# Examples:
#   ./scripts/test.sh data/segments.csv data/dataset.csv
#   ./scripts/test.sh data/new_segments.csv data/new_dataset.csv 0.00042
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
 
API="${ANOMALY_API:-http://localhost:8080}"
SEGMENTS="${1:-}"
DATASET="${2:-}"
THRESHOLD="${3:-}"
 
if [[ -z "$SEGMENTS" || -z "$DATASET" ]]; then
  echo "Usage: $0 <segments.csv> <dataset.csv> [threshold]"
  exit 1
fi
 
echo "⏳  Uploading files and starting inference..."
 
EXTRA=""
if [[ -n "$THRESHOLD" ]]; then
  EXTRA="-F threshold=$THRESHOLD"
  echo "    Using threshold override: $THRESHOLD"
fi
 
RESPONSE=$(curl -s -X POST "$API/test" \
  -F "segments_file=@$SEGMENTS" \
  -F "dataset_file=@$DATASET"   \
  $EXTRA)
 
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "✅  Job started: $JOB_ID"
echo ""
 
DOTS=0
while true; do
  STATUS_JSON=$(curl -s "$API/jobs/$JOB_ID")
  STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
 
  if [[ "$STATUS" == "done" ]]; then
    echo ""
    echo "✅  Inference complete!"
    echo "$STATUS_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)['result']
print(f'  Run ID         : {d[\"run_id\"]}')
print(f'  Total segments : {d[\"total_segments\"]}')
print(f'  Anomalies found: {d[\"flagged_anomalies\"]}')
print(f'  Threshold used : {d[\"threshold_used\"]:.8f}')
if 'metrics' in d:
    m = d['metrics']
    print(f'  F1             : {m[\"f1\"]:.4f}')
    print(f'  Precision      : {m[\"precision\"]:.4f}')
    print(f'  Recall         : {m[\"recall\"]:.4f}')
    print(f'  TP={m[\"tp\"]}  FP={m[\"fp\"]}  TN={m[\"tn\"]}  FN={m[\"fn\"]}')
print()
print(f'  Download results:')
for f in d['files']:
    print(f'    curl \$API/results/{d[\"run_id\"]}/{f} -o {f}')
"
    break
  elif [[ "$STATUS" == "failed" ]]; then
    echo ""
    echo "❌  Inference failed!"
    echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['error'])"
    exit 1
  fi
 
  printf "."
  DOTS=$((DOTS+1))
  if [[ $((DOTS % 30)) -eq 0 ]]; then echo " (still running...)"; fi
  sleep 5
done
