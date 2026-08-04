# bruce-cdc — the maintenance plane

The incremental semantic-view materializer beside PostgreSQL: PG
stays the system of record, bruce mirrors committed changes through
logical replication, and maintained SOFTAVG views answer fresh
without re-scanning.

## Three planes

```
            SQL (SOFTAVG/SIM/eps)          INSERT / DELETE
                    |                            |
              +-----v------+              +------v------+
              | bruce-query|              |  PostgreSQL |
   QUERY      | parse/plan |              |  (system of |
   PLANE      | views/exec |              |   record)   |
              +-----+------+              +------+------+
                    |                            | wal_level=logical
              +-----v------+   START_REPLICATION | pgoutput
   COMPUTE    | bruce-core |              +------v------+
   PLANE      | (mu,z,u)   |              |  bruce-cdc  |  MAINTENANCE
              | kernels    |              |  subscriber |  PLANE
              +------------+              +------+------+
                                                 |
                        Database::insert_row / delete_where
                        (same write path as every other caller,
                         so maintained views update incrementally)
```

- `protocol.rs` — raw v3 frontend/backend protocol over the local
  unix socket: `replication=database` startup, replication commands
  via simple query, the CopyBoth stream, standby status updates.
  (Why raw: released rust-postgres — postgres 0.19.14 /
  tokio-postgres 0.7.18 — exposes no replication mode and no
  CopyBoth; the SQL polling interface would consume changes outside
  the walsender ack protocol. The transport is ~400 lines.)
- `pgoutput.rs` — pgoutput proto_version 1 decoding: Begin / Commit /
  Relation (replica identity carried) / Insert / Delete / Update.
  Tuple datums keep the three wire markers distinct:
  `TupleDatum::{Null, Unchanged, Text}` ('n' / 'u' / 't') — the
  unchanged-TOAST marker is NOT NULL.
- `source.rs` — `ChangeSource`, the transport seam: committed
  transactions out, LSN acks in. `TxAssembler` is the connection-free
  pgoutput -> `CommittedTx` state machine (conformance tests drive it
  with synthetic streams); `PgOutputSource` implements the trait with
  real START_REPLICATION.
- `apply.rs` — `Mirror`: snapshot rows -> a `bruce_query::Database`
  table; committed transactions -> `insert_row` / `delete_where`;
  Update = delete(old pk) + insert(resolved new) through the same
  write path, so maintained views absorb it via the (m,num,den) group
  inverse. `Unchanged` datums resolve from the mirror's current row —
  that is why the mirror exists.
- `durable.rs` — the mirror on disk (`Mirror::save` / `Mirror::load`):
  one `BRCDCM01` file with the table, counters and the `last_lsn`
  watermark, FNV-1a integrity trailer, atomic tmp+rename+fsync writes.
  Exactly-once survives SIGKILL: state is saved BEFORE the ack.

## Correctness anchors

- **Consistent start.** The slot is created with `USE_SNAPSHOT`
  inside a read-only repeatable-read transaction on the walsender
  connection; the initial table copy runs under that snapshot, so
  snapshot and stream meet exactly at the slot's consistent point —
  no lost or doubled rows.
- **Native checkpoint.** `ack(end_lsn)` sends a standby status
  update; the slot's `confirmed_flush_lsn` advances. Restarting with
  `START_REPLICATION ... 0/0` resumes from there — PG itself is the
  resume state.
- **Deletes carry the old tuple.** The table has
  `REPLICA IDENTITY FULL`; the apply path maps the old tuple's
  primary key to `Pred::Eq` and demands exactly one row died, so a
  drifted mirror fails loudly instead of silently diverging.
- **Updates under both identities.** REPLICA IDENTITY FULL sends the
  complete old tuple ('O'); DEFAULT sends a key-only 'K' tuple iff an
  identity column changed, else NO old tuple (the row is located by
  the new tuple's pk). REPLICA IDENTITY NOTHING is a typed error
  naming the `ALTER TABLE ... REPLICA IDENTITY` fix (PG itself
  refuses such UPDATEs with SQLSTATE 55000 — pinned live in
  tests/update_toast.rs — so the guard only fires on untrusted
  streams). Untouched TOASTed columns arrive as the 'u' marker and
  resolve from the mirror's current row, byte-exactly
  (tests/update_toast.rs proves it with a 10KB STORAGE EXTERNAL
  column against a real walsender).

## PG setup (once)

```sql
-- postgresql.conf: wal_level = logical   (then restart)
CREATE TABLE cdc_movies(
  movie_id int primary key, genre text, rating float8,
  year float8, e0 float8, e1 float8);
ALTER TABLE cdc_movies REPLICA IDENTITY FULL;
CREATE PUBLICATION bruce_pub FOR TABLE cdc_movies;
```

The local instance for this repo: unix socket `/tmp`, port `54329`,
data dir `~/bruce/experiments/cidr_one_query/pgdata`, binaries
`~/miniforge3/envs/pgv/bin/`.

## Run

```bash
# demo sidecar: snapshot, stream, print the maintained answer per commit
cargo run --release -- --idle-exit 30
# durable demo: saved after every commit, resumes from disk if the
# slot already exists (exactly-once across kill -9)
cargo run --release -- --state /tmp/bruce_cdc_demo.mirror
# then, from psql: INSERT/UPDATE/DELETE on cdc_movies, watch the answer
# afterwards: SELECT pg_drop_replication_slot('bruce_cdc_demo');

# tests: pure unit tests + golden pgoutput byte-corpus conformance
# (tests/conformance.rs over tests/corpus/*.bin, no PG needed) +
# durable-format tests (tests/durable.rs, no PG) + the end-to-end
# proof + live update/TOAST round trip + update differential + chaos:
# kill-resume x5, update-heavy kill-resume x5, pg_ctl restart
# mid-stream, 5000-row single-tx atomicity, kill -9 x5 of a REAL
# subscriber process restarted from disk, recovery/save-throughput
# medians. The live tests need the PG instance up and own their
# tables/publications/slots: cdc_movies+bruce_cdc_e2e,
# cdc_movies_{chaos,updchaos,restart,bigtx,toast,diff,dur,perf}
cargo test
```

The end-to-end test seeds 1000 rows, streams 100 INSERTs + 50
DELETEs while applying, and asserts the maintained answers equal (a)
a from-scratch bruce snapshot of the final PG state and (b) the same
soft average computed by PG in SQL — then disconnects, writes 10 more
rows, reconnects, and proves slot-based resume. Measured numbers land
in `bruce/paper_sigmod_bruce/experiments/m12_cdc/`.

## Consistency contract (pinned by tests/chaos.rs + durable_chaos.rs)

- **Commit-buffered delivery**: `next_tx` yields only whole committed
  transactions; nothing is delivered (or applied) before Commit — a
  5000-row transaction lands in one `apply_tx` call, and readers
  sequenced between `apply_tx` calls never see a partial transaction.
- **Exactly-once across resume**: `Mirror.last_lsn` is the watermark;
  a crash between apply and ack redelivers a prefix of
  already-applied transactions, which `apply_tx` filters (returns 0)
  instead of double-applying. The caller still acks them.
- **Exactly-once across process death**: with the durable mirror,
  save-THEN-ack makes every SIGKILL window safe — killed before save:
  the tx was never acked, redelivered onto on-disk state that never
  saw it; killed after save: redelivery filtered by the DURABLE
  watermark; killed mid-save: atomic rename keeps the previous
  complete file. tests/durable_chaos.rs SIGKILLs a real subscriber
  process 5x under an update-heavy workload and asserts the final
  on-disk state equals PG bit for bit.
- **Reconnect**: `RetryingSource` wraps `PgOutputSource` with bounded
  redial (transient = Io / Protocol / SQLSTATE 57P01, 57P03, 55006 —
  see `is_transient`); semantic errors propagate immediately.
- **Schema change**: a Relation re-sent mid-stream (ADD COLUMN)
  widens tuples; columns unknown to the `TableMap` are ignored until
  re-snapshot, a dropped mapped column fails loudly at apply.

## v1 boundaries

- Truncate/Origin/Type are ignored. One table map per mirror.
- Text tuple format only; trust auth over the unix socket only.
- NULLs in mapped columns are typed errors (the demo schema forbids
  them); an UPDATE that sets a mapped column NULL is rejected with
  the mirror untouched (pinned in tests/update_toast.rs).
- The durable file is a cache of PG's authoritative state: delete it
  and the next start re-snapshots. Views are not serialized —
  recreate them after `Mirror::load` (they recompute from the table).
