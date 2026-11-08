#!/usr/bin/env bash
# Build the extension variants used only for cost attribution / bitwise checks (never shipped):
#   build/diag/softavg.duckdb_extension    engines/duckdb with --features diag
#                                           (adds softavg_diag_noop / _touch / _2arg)
#   build/pre_p1/softavg.duckdb_extension  engines/duckdb against a temporary copy of engines/core
#                                           with core_proposal/p1_one_exp.patch REVERSED, i.e. the
#                                           two-exp absorb the extension shipped with before core
#                                           adopted P1 (2026-11-05)
# engines/core itself is never modified. Everything lands in build/ (deleted by cleanup).
# core_proposal/*.patch are the historical proposals against the pre-P1 core; P1 is now in core,
# P2 was not adopted.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG="$(cd "$HERE/../../engines" && pwd)"
B="$HERE/build"
mkdir -p "$B"
export CARGO_TARGET_DIR="$B/target"
footer() {  # $1 = .so, $2 = output dir
  mkdir -p "$2"
  python3 "$ENG/duckdb/tools/append_extension_metadata.py" -l "$1" -n softavg \
    -o "$2/softavg.duckdb_extension" -p linux_amd64 -dv v1.2.0 -ev 0.1.0 --abi-type C_STRUCT >/dev/null
}

(cd "$ENG/duckdb" && nice -n 10 cargo build --release --offline --features diag)
footer "$CARGO_TARGET_DIR/release/libsoftavg.so" "$B/diag"

S="$B/src_pre_p1"
rm -rf "$S" && mkdir -p "$S/duckdb"
cp -r "$ENG/core" "$S/core" && rm -rf "$S/core/target"
(cd "$S/core" && patch -p1 -R --quiet < "$HERE/core_proposal/p1_one_exp.patch")
cp -r "$ENG/duckdb/Cargo.toml" "$ENG/duckdb/Cargo.lock" "$ENG/duckdb/src" "$S/duckdb/"
(cd "$S/duckdb" && nice -n 10 cargo build --release --offline)
footer "$CARGO_TARGET_DIR/release/libsoftavg.so" "$B/pre_p1"
ls -la "$B"/*/softavg.duckdb_extension
