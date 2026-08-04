#!/usr/bin/env bash
# Workstream 19 — the nightly gate: one command, PASS/FAIL summary.
#
# Stages (always):
#   1. workspace cargo test            (bruce-core/query/py/cli/server)
#   2. python pytest                   (bruce-py/tests, shipped wheel)
#   3. one_query golden band           (real 460k-row movie workload,
#      f64 path; query median must land in [10, 20] ms — the frozen
#      SIGMOD number is 13.5 ms @ 32 threads)
# Stages (opt-in, expensive):
#   REGRESS_BENCH=1  cargo bench fold + bench_compare.py vs the saved
#                    baseline (adds ~4 min)
#   REGRESS_SOAK=1   scripts/soak.py 10M-row scale soak (adds ~2 min)
#
# run_m2 (paper_sigmod_bruce/experiments/m2_mixed_precision/run_m2.py)
# is NOT run here: it has no quick mode — it sweeps 1/4/8/32 threads in
# fresh subprocesses (minutes) to produce paper numbers, and the same
# engine path + golden number is already gated by stage 3 in one run.
#
# Exit 0 iff every executed stage passed. Designed for cron:
#   0 3 * * *  /home/jding/bruce_tool/scripts/regress.sh >> nightly.log 2>&1

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA=/home/jding/bruce/experiments/cidr_one_query
GOLD_LO=10.0
GOLD_HI=20.0

declare -a NAMES=()
declare -a RESULTS=()
overall=0

stage() { # stage <name> <cmd...>
    local name="$1"; shift
    echo "==== [$name] $*"
    if "$@"; then
        NAMES+=("$name"); RESULTS+=(PASS)
    else
        NAMES+=("$name"); RESULTS+=(FAIL); overall=1
    fi
}

golden_one_query() {
    # build once (quiet), then run and check the printed median
    cargo build --release -p bruce-query --example one_query --quiet || return 1
    local out
    out="$("$ROOT/target/release/examples/one_query" \
        "$DATA/movies.parquet" "$DATA/emb.npy" 0.1 "$DATA/query_emb.npy")" || return 1
    echo "$out"
    local ms
    ms=$(echo "$out" | sed -n 's/^query: \([0-9.]*\) ms.*/\1/p')
    if [ -z "$ms" ]; then
        echo "golden: could not parse 'query: X ms' from output"
        return 1
    fi
    awk -v ms="$ms" -v lo="$GOLD_LO" -v hi="$GOLD_HI" 'BEGIN {
        if (ms >= lo && ms <= hi) {
            printf "golden: %.1f ms within band [%.0f, %.0f]\n", ms, lo, hi; exit 0
        } else {
            printf "golden: %.1f ms OUTSIDE band [%.0f, %.0f]\n", ms, lo, hi; exit 1
        }
    }'
}

cd "$ROOT"

stage "cargo-test-workspace" cargo test --workspace --quiet
stage "pytest-wheel"         python3 -m pytest bruce-py/tests -q
stage "one-query-golden"     golden_one_query

if [ "${REGRESS_BENCH:-0}" = "1" ]; then
    stage "criterion-fold"   cargo bench -p bruce-core --bench fold --quiet
    stage "bench-compare"    python3 "$ROOT/scripts/bench_compare.py"
fi
if [ "${REGRESS_SOAK:-0}" = "1" ]; then
    stage "scale-soak"       python3 "$ROOT/scripts/soak.py"
fi

echo
echo "==== nightly gate summary ===="
for i in "${!NAMES[@]}"; do
    printf "  %-22s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}"
done
if [ "$overall" -eq 0 ]; then echo "OVERALL: PASS"; else echo "OVERALL: FAIL"; fi
exit "$overall"
