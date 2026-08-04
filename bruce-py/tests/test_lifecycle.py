"""Workstream 16 — session lifecycle (docs/TESTING_MATRIX.md).

Pins, at the PYTHON level (the wheel users actually load), the
lifecycle semantics the Rust layer defined:

* register over an existing name REPLACES the table, REPLACES stats,
  and DROPS maintained views built on the old contents (PG vocabulary:
  replace = drop + create; dropping a table cascades to dependents) —
  db.rs::register, pinned in Rust by tests/stateful_writes.rs.
* stale references after a replace (attached key columns are part of
  the dropped generation) surface as typed ValueError, never a crash.
* view names are unique per session (PG: duplicate CREATE VIEW
  errors); the cascade frees the name.
* views on OTHER tables survive an unrelated re-register and keep
  answering incrementally.
* a long-lived session (O(10k) queries, O(100) full lifecycle cycles)
  has bounded RSS growth.

RSS bound justification (measured on this box, 2026-08-03, 32-thread
box under concurrent build load ~4):
  - clean growth after a 10-cycle warmup: 256 kB over 290 lifecycle
    cycles (plateau; the pre-warmup delta is allocator arena growth);
  - the failure modes the bound must catch:
      (a) one leaked table generation per re-register cycle: the key
          column alone is 4096 x 32 x 8 B = 1 MiB, so >= 90 MiB over
          90 post-warmup cycles;
      (b) a 4 KiB-per-query leak: 36 MiB over 9000 post-warmup queries;
  - bound: 32 MiB post-warmup growth — >100x the measured clean noise,
    and a fraction of either leak signature. (Last night's smoke bound
    of 50 MB over 200 queries was a ceiling, not a measurement; this
    one is tied to the leak sizes above.)
Measured samples land in docs/qa/lifecycle_rss.json (doctrine: every
measured number in a results.json).
"""

import json
import pathlib

import numpy as np
import pytest

import bruce

D = 32          # key dimension
N = 4096        # rows per table generation
G = 8           # genres
EPS = 0.5

QA_OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "qa" / "lifecycle_rss.json"

Q = ("SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.5) "
     "FROM movies GROUP BY genre")


def rss_kb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not found")  # pragma: no cover


def make_parquet(path, seed, rating_shift=0.0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    pd.DataFrame({
        "genre": [f"g{i % G}" for i in range(N)],
        "rating": rng.uniform(0, 10, N) + rating_shift,
        "year": rng.uniform(1990, 2025, N),
    }).to_parquet(path)
    return str(path)


@pytest.fixture()
def rng():
    return np.random.default_rng(7)


@pytest.fixture()
def pq(tmp_path):
    return make_parquet(tmp_path / "gen1.parquet", seed=1)


@pytest.fixture()
def pq2(tmp_path):
    # rating_shift makes generation-2 answers distinguishable
    return make_parquet(tmp_path / "gen2.parquet", seed=2, rating_shift=100.0)


def fresh_session(pq_path, rng):
    s = bruce.QuerySession()
    s.register_parquet("movies", pq_path)
    emb = rng.standard_normal((N, D))
    s.attach_key("movies", "emb", emb)
    return s, emb


def chosen_plan(explain: str) -> str:
    """The '== chosen plan ==' section only (candidates list every
    considered plan, including unchosen view scans)."""
    return explain.split("== candidates ==")[0]


# ---------------------------------------------------------------------
# Re-register semantics
# ---------------------------------------------------------------------

def test_reregister_replaces_table_and_drops_views(pq, pq2, rng):
    s, _ = fresh_session(pq, rng)
    x = rng.standard_normal(D)
    s.create_view("v1", "movies", "genre", "rating", "emb", x, eps=EPS)
    _, vals1, explain = s.run(Q, {"q": x})
    assert "MaintainedViewScan view=v1" in chosen_plan(explain)

    # replace: new contents (shifted ratings), fresh key column
    s.register_parquet("movies", pq2)
    s.attach_key("movies", "emb", rng.standard_normal((N, D)))
    labels, vals2, explain = s.run(Q, {"q": x})

    # (1) contents replaced: generation-2 ratings are shifted by +100
    assert all(v > 90.0 for v in vals2)
    assert all(v < 11.0 for v in vals1)
    # (2) the view was DROPPED with the old table: not even a candidate
    assert "MaintainedViewScan" not in explain
    # (3) cascade freed the name: recreating v1 now succeeds
    s.create_view("v1", "movies", "genre", "rating", "emb", x, eps=EPS)
    _, _, explain = s.run(Q, {"q": x})
    assert "MaintainedViewScan view=v1" in chosen_plan(explain)


def test_reregister_stale_key_reference_is_typed_error(pq, pq2, rng):
    s, _ = fresh_session(pq, rng)
    x = rng.standard_normal(D)
    assert len(s.run(Q, {"q": x})[0]) == G

    # replacement drops the attached key column with its generation
    s.register_parquet("movies", pq2)

    # stale read: typed error naming the column, no crash
    with pytest.raises(ValueError, match="emb"):
        s.run(Q, {"q": x})
    # stale write: the insert names a key column the new generation
    # does not have — typed error (typo protection), no crash
    with pytest.raises(ValueError, match="unknown key column emb"):
        s.insert_row("movies", {"rating": 1.0, "year": 2000.0},
                     {"genre": "g0"}, {"emb": np.zeros(D)})
    # the session survives both: scalar-only queries still answer
    s.attach_key("movies", "emb", rng.standard_normal((N, D)))
    labels, _, _ = s.run(Q, {"q": x})
    assert len(labels) == G


def test_lifecycle_errors_are_typed(pq, rng):
    s, _ = fresh_session(pq, rng)
    x = rng.standard_normal(D)
    with pytest.raises(ValueError, match="no table"):
        s.delete_where("gone", "year", ">=", 0.0)
    with pytest.raises(ValueError, match="no table"):
        s.create_view("v", "gone", "genre", "rating", "emb", x, eps=EPS)
    with pytest.raises(ValueError, match="rows"):
        s.attach_key("movies", "emb2", np.zeros((5, D)))


# ---------------------------------------------------------------------
# View lifecycle
# ---------------------------------------------------------------------

def test_create_view_duplicate_name_is_typed_error(pq, rng):
    s, _ = fresh_session(pq, rng)
    x = rng.standard_normal(D)
    s.create_view("v1", "movies", "genre", "rating", "emb", x, eps=EPS)
    with pytest.raises(ValueError, match="already exists"):
        s.create_view("v1", "movies", "genre", "rating", "emb", x, eps=EPS)


def test_views_survive_unrelated_reregister(pq, pq2, rng, tmp_path):
    s, _ = fresh_session(pq, rng)
    x = rng.standard_normal(D)
    s.create_view("v1", "movies", "genre", "rating", "emb", x, eps=EPS)
    _, before, explain = s.run(Q, {"q": x})
    assert "MaintainedViewScan view=v1" in chosen_plan(explain)

    # churn an UNRELATED table: register, replace, replace again
    other = make_parquet(tmp_path / "other.parquet", seed=9)
    s.register_parquet("other", other)
    s.register_parquet("other", pq2)
    s.register_parquet("other", other)

    # movies' view still exists, still chosen, and still incrementally
    # correct: an insert routes through it and shifts the answer
    _, after, explain = s.run(Q, {"q": x})
    assert "MaintainedViewScan view=v1" in chosen_plan(explain)
    assert after == pytest.approx(before, rel=1e-12)

    s.insert_row("movies", {"rating": 1000.0, "year": 2020.0},
                 {"genre": "g0"}, {"emb": (x / np.linalg.norm(x)) * 10.0})
    labels, shifted, _ = s.run(Q, {"q": x})
    got = dict(zip(labels, shifted))
    # the inserted row scores far above every random key: its weight
    # dominates g0's fold, dragging the answer toward 1000
    assert got["g0"] > 900.0


# ---------------------------------------------------------------------
# Long-lived session: bounded RSS
# ---------------------------------------------------------------------

def test_long_session_rss_bounded(pq, rng):
    n_cycles = 100          # full lifecycle churns
    queries_per_cycle = 100  # -> 10_000 queries total
    warmup_cycles = 10       # allocator arena growth happens here
    bound_kb = 32 * 1024     # justification in module docstring

    s, emb = fresh_session(pq, rng)
    x_view = rng.standard_normal(D)     # view-hit path (O(groups))
    x_scan = rng.standard_normal(D)     # scan path (O(rows))
    s.create_view("v1", "movies", "genre", "rating", "emb", x_view, eps=EPS)

    samples = []  # (cycle, rss_kb)
    baseline = None
    for c in range(n_cycles):
        for q in range(queries_per_cycle):
            x = x_view if q % 2 == 0 else x_scan
            labels, values, _ = s.run(Q, {"q": x})
            assert len(labels) == G and np.isfinite(values).all()
        # one full lifecycle churn: replace table (drops v1), re-attach
        # keys, recreate the view, write through it, read back
        s.register_parquet("movies", pq)
        s.attach_key("movies", "emb", emb)
        s.create_view("v1", "movies", "genre", "rating", "emb", x_view, eps=EPS)
        s.insert_row("movies", {"rating": 5.0, "year": 2020.0},
                     {"genre": "g0"}, {"emb": np.zeros(D)})
        s.delete_where("movies", "year", ">=", 2024.5)
        s.run(Q, {"q": x_view})

        samples.append((c + 1, rss_kb()))
        if c + 1 == warmup_cycles:
            baseline = samples[-1][1]

    final = int(np.median([r for _, r in samples[-5:]]))
    growth = final - baseline

    QA_OUT.parent.mkdir(parents=True, exist_ok=True)
    QA_OUT.write_text(json.dumps({
        "date": "2026-08-03",
        "workstream": 16,
        "queries": n_cycles * queries_per_cycle,
        "lifecycle_cycles": n_cycles,
        "warmup_cycles": warmup_cycles,
        "baseline_rss_kb": baseline,
        "final_rss_kb_median5": final,
        "post_warmup_growth_kb": growth,
        "bound_kb": bound_kb,
        "samples": samples,
    }, indent=1))

    assert growth < bound_kb, (
        f"RSS grew {growth} kB over {n_cycles - warmup_cycles} post-warmup "
        f"cycles (bound {bound_kb} kB) — see {QA_OUT}"
    )
