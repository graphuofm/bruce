#!/usr/bin/env bash
# Build the softavg DuckDB extension: cargo cdylib + DuckDB metadata footer.
# Output: build/softavg.duckdb_extension  (LOAD it with allow_unsigned_extensions=true)
set -euo pipefail
cd "$(dirname "$0")"
EXT_VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
CAPI_VERSION=${CAPI_VERSION:-v1.2.0}          # must match CAPI_VERSION in src/lib.rs
PLATFORM=${DUCKDB_PLATFORM:-linux_amd64}      # `PRAGMA platform` of the target DuckDB
cargo build --release --offline
mkdir -p build
python3 tools/append_extension_metadata.py \
  -l target/release/libsoftavg.so -n softavg -o build/softavg.duckdb_extension \
  -p "$PLATFORM" -dv "$CAPI_VERSION" -ev "$EXT_VERSION" --abi-type C_STRUCT
ls -la build/softavg.duckdb_extension
