# UNSAFE INVENTORY — memory-safety audit (workstream 20)

Date: 2026-08-03 (night six). Auditor: lifecycle-safety track.
Scope: every crate in this tree — bruce-core, bruce-query, bruce-py,
bruce-server, bruce-cli, bruce-pg, bruce-cdc.

## Sweep method

```
rg -n 'unsafe' -g '!target'          # whole tree, all file types
rg -c 'unsafe' <crate>/src           # per-crate cross-check
```

Run twice: at campaign start (04:41) and at end of this track's run
(after the f32 track's AVX2 work landed), because tonight's f32 track
was expected to add one AVX2 `unsafe` block to bruce-core/src/mask.rs.

## Result: ZERO hand-written `unsafe` blocks

| Crate | src files | `unsafe` occurrences | Notes |
|---|---|---|---|
| bruce-core | 20 | 1 (comment only) | mask.rs:543 — doc comment recording that an AVX2+FMA `unsafe` kernel was implemented, measured (-2.2% on the gated operator config, below the >=5% keep gate), and REVERTED on 2026-08-03. No code block. |
| bruce-query | 13 | 0 | |
| bruce-py | 1 | 0 | see macro boundary below |
| bruce-server | 5 | 0 | |
| bruce-cli | 1 | 0 | |
| bruce-pg | 1 | 0 | see macro boundary below |
| bruce-cdc | 7 | 0 | pgoutput protocol parsing is 100% safe Rust over `&[u8]` |

There is no hand-written `unsafe` block anywhere in the workspace, so
there are no per-block safety arguments to carry: the audit obligation
collapses to the two macro boundaries below plus dependency posture.

## Macro-generated unsafe (crate boundaries, not hand-written)

1. **bruce-py — pyo3 0.22 (abi3-py39) + rust-numpy 0.22.**
   `#[pyclass]`/`#[pymethods]`/`wrap_pyfunction!` expand to unsafe
   CPython FFI, and `PyReadonlyArray*` wraps the numpy C API. The
   generated code is upstream-audited; our obligation is the boundary
   contract (GIL held at entry, buffer lifetimes tied to
   `PyReadonly*` guards, no `&mut` aliasing of borrowed arrays — all
   enforced by pyo3's type system in safe code on our side).
   Exercised by: the full pytest suite (285 tests as of tonight)
   drives every exported class/function, including error paths
   (test_error_paths.py) and lifecycle churn (test_lifecycle.py);
   plus the abi3 matrix (same .so under CPython 3.10.12 and 3.13.13,
   docs/qa/abi3_matrix.json).

2. **bruce-pg — pgrx =0.18.1.**
   `#[pg_extern]` (3 sites) and the aggregate registration macros
   expand to unsafe PG FFI (`pg_guard` wrappers, palloc-backed
   datums). Constitution C2 keeps our sfunc/combinefunc/finalfunc
   bodies pure safe Rust over owned values; pgrx owns the
   datum<->Rust marshalling. Exercised by: 21/21 pgrx tests on both
   pg16 and pg17 (last night's campaign).

## Dependency posture

`deny.toml` at the workspace root (cargo-deny) gates the dependency
graph. Heavy unsafe lives in upstream ndarray/rayon/arrow/parquet —
constitution C1 keeps arrow OUT of bruce-core (pure algorithm,
pgrx-portable) and C3 confines Arrow/Parquet to the server/py layer.

## Recommendation (not applied tonight — lib.rs files owned by other tracks)

Add `#![forbid(unsafe_code)]` to bruce-query, bruce-server, bruce-cli,
bruce-cdc, bruce-pg (their zero count is then compiler-enforced).
bruce-core should take `#![deny(unsafe_code)]` rather than forbid: the
AVX2 experiment is documented as revisitable if the gated config ever
becomes cache-resident, and deny permits a scoped
`#[allow(unsafe_code)]` at one block. bruce-py cannot forbid
(pyo3 macro expansion).

## ASAN / LSAN — DEFERRED (needs nightly rustc)

`rustup toolchain list` on this box: `stable-x86_64-unknown-linux-gnu`
only (rustc 1.95; pgrx 0.18.1 is pinned to it). Not installing a
nightly mid-campaign (toolchain/network churn). A future run needs:

```
rustup toolchain install nightly --component rust-src
RUSTFLAGS=-Zsanitizer=address cargo +nightly test -Zbuild-std \
    --target x86_64-unknown-linux-gnu -p bruce-core -p bruce-query
```

(pyo3/pgrx crates need their host runtimes instrumented too; start
with the pure-Rust crates above, which contain all the kernel math.)

## Leak sampling (in lieu of valgrind — not installed on this box)

`which valgrind` is empty. Method used instead: /proc/self/status
VmRSS sampled per cycle, 5-cycle block medians, post-warmup growth
asserted < 32 MiB (bound justification in
bruce-py/tests/test_lifecycle.py docstring — >100x measured clean
noise, a fraction of one-leaked-table-generation-per-cycle).

Measured 2026-08-03 (loadavg ~10 under concurrent track builds,
results in docs/qa/leak_soak.json and docs/qa/lifecycle_rss.json):

- session-churn (build + drop 200 whole QuerySessions): -28 kB
  post-warmup growth over 180 cycles;
- replace-churn (200 register-over/cascade/view-rebuild cycles in one
  session): 0 kB post-warmup growth over 180 cycles;
- long-lived session (10,000 queries + 100 lifecycle cycles,
  pytest test_lifecycle.py): 0 kB post-warmup growth.

## Panic-freedom adjunct (typed-error doctrine)

Registration-time totality gap closed this campaign: DictU32 codes
outside `[0, dict.len())` and NaN key rows no longer panic in stats
collection (bruce-query/src/stats.rs guard + NaN skip in
`KeySketch::sims`; pinned by bruce-query/tests/register_safety.rs,
5 tests).
