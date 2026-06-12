# Contributing to Bruce

Thank you for considering a contribution. Bruce is a Rust workspace
(`bruce-core`, `bruce-py`, `bruce-cli`, `bruce-server`) with a Python
package built from `bruce-py` via maturin.

## Development setup

```bash
git clone <repo> && cd bruce_tool
cargo test --workspace                 # Rust suites
cd bruce-py && maturin build --release # build the wheel
pip install --force-reinstall ../target/wheels/bruce-*.whl
pytest tests -q                        # Python suite
```

`maturin develop` requires an activated virtualenv; `maturin build` +
`pip install` works everywhere (this is also what CI does).

## Quality gates (all enforced by CI)

- `cargo fmt --all --check` — formatting.
- `cargo clippy --workspace --all-targets -- -D warnings` — zero
  warnings, no exceptions; crate-level `allow`s need a comment
  explaining why and when they can be removed.
- `cargo test --workspace` and `pytest bruce-py/tests` — all green.
- MSRV is **1.81** (`rust-version` in Cargo.toml); do not use newer
  std APIs without bumping it deliberately in a dedicated PR.

## Engineering rules

1. **No panics on user input in library code.** Anything reachable
   from `bruce-core`'s public API or the Python surface returns
   `Result` / raises `ValueError`. `expect()` is allowed only for
   genuine internal invariants, with a comment naming the invariant.
2. **Numerical claims need tests.** Any algebraic identity (operator
   equivalences, partition-reduce, order invariance) is tested at
   `1e-12` absolute tolerance against an independent reference
   implementation (usually numpy).
3. **The math is the spec.** Algorithms mirror published statements
   (see the paper trail in the repository this project grew out of);
   if code and theorem disagree, one of them is wrong — stop and find
   out which.
4. **Breaking API changes** are fine pre-1.0 but must be listed under
   a "Breaking" heading in `CHANGELOG.md`.
5. **Every PR updates `CHANGELOG.md`.**

## Commit style

Imperative subject, body explains *why*. Group mechanical changes
(formatting, renames) into separate commits from behavioural ones.
