#!/usr/bin/env bash
# Smoke test for bruce-server. Build, launch, hit every endpoint, kill.
# Used by `make test-server`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cargo build -p bruce-server --release

PORT=${PORT:-18080}
LOG=$(mktemp)
./target/release/bruce-server --addr "127.0.0.1:${PORT}" --d-k 4 --d-v 2 \
    > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true; rm -f "$LOG"' EXIT
sleep 1

base="http://127.0.0.1:${PORT}"
fail=0
check() {
    local name=$1 expected=$2 actual=$3
    if [[ "$actual" == *"$expected"* ]]; then
        echo "  ✓ $name"
    else
        echo "  ✗ $name: expected '$expected' got '$actual'"
        fail=1
    fi
}

check "/health"        "ok"            "$(curl -fsS --max-time 3 $base/health)"
check "/info"          '"alive":0'     "$(curl -fsS --max-time 3 $base/info)"

curl -fsS --max-time 3 -X POST $base/facts \
  -H 'Content-Type: application/json' \
  -d '{"fact_id":"a","k":[1,0,0,0],"v":[10,0],"owner":"u"}' >/dev/null
curl -fsS --max-time 3 -X POST $base/facts \
  -H 'Content-Type: application/json' \
  -d '{"fact_id":"b","k":[0,1,0,0],"v":[0,20],"owner":"u"}' >/dev/null

check "GET /facts/a"   '10'            "$(curl -fsS --max-time 3 $base/facts/a)"

# ε=0 indicator picks fact a exactly
out=$(curl -fsS --max-time 3 -X POST $base/query/attention \
  -H 'Content-Type: application/json' \
  -d '{"x":[1,0,0,0],"eps":0,"sim":"indicator"}')
check "/query/attention ε=0"  "[10.0,0.0]"  "$out"

# owner mismatch on delete must 403
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 \
       -X DELETE "$base/facts/a?owner=intruder")
check "DELETE wrong owner 403" "403" "$code"

check "/audit/length=2"  "2"  "$(curl -fsS --max-time 3 $base/audit/length)"

if (( fail )); then
    echo "FAIL"
    cat "$LOG"
    exit 1
fi
echo "all bruce-server endpoints OK"
