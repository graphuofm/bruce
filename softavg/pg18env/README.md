# pg18env — PostgreSQL 18.6 for paper_icde (set up 2026-11-07)

The PostgreSQL used by paper_icde from 2026-11-07 on. It replaces the release
PostgreSQL 17.10 under `~/.pgrx` that was deleted by the 2026-11-06 disk cleanup.

- **Server binaries and headers: the conda env `/home/jding/miniforge3/envs/ggdb-pg`**
  (PostgreSQL 18.6, conda-forge release build, `-O2`, no cassert; `pg_config` there).
  That env belongs to another project and is used **read-only**: nothing is installed
  into it. Its compiled-in `$libdir` / sharedir are the conda `lib/` and `share/`.
- **Extensions live here**, in `ext/`, and the server finds them through two GUCs
  (PostgreSQL 18):
  - `dynamic_library_path = '/home/jding/bruce/paper_icde/pg18env/ext/lib:$libdir'`
  - `extension_control_path = '/home/jding/bruce/paper_icde/pg18env/ext/share:$system'`
- **Cluster**: `data/` (initdb'd with the conda binaries), port **28860**, listening on
  127.0.0.1 and a unix socket in `data/`.
- Do not use `../pg18/` (a stray 164 MB copy of the conda env made by mistake; left for the
  author to delete).

## Start / stop

```bash
/home/jding/bruce/paper_icde/pg18env/start.sh   # pg_ctl start -w, log in log/server.log, prints version()
/home/jding/bruce/paper_icde/pg18env/stop.sh    # pg_ctl stop -m fast -w (only this data directory)
psql -h 127.0.0.1 -p 28860 -U jding -d postgres # psql = /home/jding/miniforge3/envs/ggdb-pg/bin/psql
```

Check that it is down: `ps aux | grep '[p]g18env/data'` prints nothing. The other
postgres on this box (pgserver under `~/dbtaint`) is a different project: leave it alone.

## Server settings (appended to `data/postgresql.conf`)

Same as `experiments/postgres/README.md` "Rerun", plus `jit = off` and the two path GUCs:

```
port = 28860
listen_addresses = '127.0.0.1'
unix_socket_directories = '/home/jding/bruce/paper_icde/pg18env/data'
shared_buffers = 4GB
work_mem = 256MB
maintenance_work_mem = 4GB
effective_cache_size = 16GB
max_worker_processes = 32
max_parallel_workers = 16
max_parallel_workers_per_gather = 0
max_parallel_maintenance_workers = 8
max_wal_size = 8GB
jit = off
shared_preload_libraries = 'pg_ivm'
dynamic_library_path = '/home/jding/bruce/paper_icde/pg18env/ext/lib:$libdir'
extension_control_path = '/home/jding/bruce/paper_icde/pg18env/ext/share:$system'
```

Recreate the cluster from scratch: `initdb -D data -U jding --no-locale -E UTF8` with the
conda `initdb`, then append the block above.

### What the two GUCs do (verified on this server)

- `extension_control_path` entries are **share directories**: PostgreSQL 18 appends
  `/extension` to each entry (`$system` = the conda `share/extension`). So the control
  and script files go in `ext/share/extension/`. The install / update scripts of an
  extension are read from the directory where its control file was found, not from
  the compiled-in sharedir.
- `dynamic_library_path` is searched for module names without a directory
  (`bruce_pg`, `softavg_ablation`: pgrx writes `module_pathname = 'bruce_pg'`) and also
  resolved pgvector's and pg_ivm's `$libdir/vector`, `$libdir/pg_ivm` (the conda
  `lib/` has no such files; the backend maps `ext/lib/*.so`, checked in
  `/proc/<pid>/maps`). `shared_preload_libraries = 'pg_ivm'` is found the same way.
- `CREATE EXTENSION vector; CREATE EXTENSION pg_ivm; CREATE EXTENSION bruce_pg;
  CREATE EXTENSION softavg_ablation;` all succeed (vector 0.8.7, pg_ivm 1.16,
  bruce_pg 0.2.0, softavg_ablation 0.1.0); a smoke query of each works (pgvector `<#>`,
  `pgivm.create_immv` + incremental insert, `softavg`, `softavg_s3`).
  `CREATE EXTENSION bruce_pg VERSION '0.1.2'` + `ALTER EXTENSION bruce_pg UPDATE TO '0.2.0'`
  also works (same value before and after).

## What is in `ext/`

| extension | source | build |
|---|---|---|
| `bruce_pg` 0.2.0 | `../engines/postgres` (no source change for PG 18) | `cargo pgrx package` (release profile: opt-level 3, lto fat, codegen-units 1), `--features pg18 --no-default-features` |
| `softavg_ablation` 0.1.0 | `../experiments/softmax_ladder/pg_ablation` (experiment-only S3 build, not shipped) | same |
| `vector` (pgvector) 0.8.7 | `~/pgext_src/pgvector` (git f37c13f), copied to `build/pgvector` | PGXS, `-O2` (+ pgvector's own `-march=native`) |
| `pg_ivm` 1.16 | `~/pgext_src/pg_ivm` (git 22b4b45), copied to `build/pg_ivm` | PGXS, `-O2`; also installs `ext/bin/pg_ivm_dump_metadata` |

`ext/share/extension/bruce_pg--0.1.2.sql` is copied by hand from
`engines/postgres/sql/` (needed only for the 0.1.2 -> 0.2.0 upgrade check; pgrx does not
package it).

## How the extensions were staged (reproduce)

Build environment (`build/env.sh`, source it first):

```bash
E=/home/jding/bruce/paper_icde/pg18env
. $E/build/env.sh   # LIBCLANG_PATH (pip libclang), BINDGEN_EXTRA_CLANG_ARGS / CFLAGS with -I$E/build/compat,
                    # CARGO_NET_OFFLINE=true, CARGO_TARGET_DIR=$E/build/target, PG_CONFIG=<conda pg_config>
```

Two quirks of building against the conda headers with the system toolchain (gcc 11,
glibc 2.35):
- the conda `pg_config.h` defines `HAVE_XLOCALE_H` (conda sysroot, glibc 2.17), and glibc
  >= 2.26 has no `xlocale.h`: `build/compat/xlocale.h` is a one-line shim
  (`#include <locale.h>`) put on the include path of gcc and of bindgen;
- the conda `CFLAGS` contain `-Wmissing-variable-declarations` (gcc >= 14) and the PGXS
  makefiles say `with_llvm = yes` with a `CLANG` that does not exist here, so PGXS builds
  override `CFLAGS` (the conda flags minus that one, `-O2 -fPIC`) and set `with_llvm=no`.

PGXS extensions, installed into a staging root with `DESTDIR` and then moved:

```bash
PGC=/home/jding/miniforge3/envs/ggdb-pg/bin/pg_config
CF="-Wall -Wmissing-prototypes -Wpointer-arith -Wdeclaration-after-statement -Werror=vla -Wendif-labels \
    -Wmissing-format-attribute -Wimplicit-fallthrough=3 -Wcast-function-type -Wformat-security \
    -fno-strict-aliasing -fwrapv -fexcess-precision=standard -Wno-format-truncation -Wno-stringop-truncation \
    -O2 -fPIC -I$E/build/compat"
for x in pgvector pg_ivm; do
  rsync -a --exclude '*.o' --exclude '*.so' --exclude '*.bc' ~/pgext_src/$x/ $E/build/$x/
  (cd $E/build/$x && make PG_CONFIG=$PGC clean && make PG_CONFIG=$PGC CC=gcc "CFLAGS=$CF" with_llvm=no -j16 \
     && make PG_CONFIG=$PGC CC=gcc "CFLAGS=$CF" with_llvm=no DESTDIR=$E/build/stage install)
done
S=$E/build/stage/home/jding/miniforge3/envs/ggdb-pg
mv $S/lib/*.so $E/ext/lib/; mv $S/share/extension/* $E/ext/share/extension/
mkdir -p $E/ext/bin; mv $S/bin/pg_ivm_dump_metadata $E/ext/bin/; rm -rf $E/build/stage
```

pgrx extensions (`cargo pgrx init --pg18 $PGC` was run once; it only writes
`~/.pgrx/config.toml` and initdb's an unused `~/.pgrx/data-18`, it builds no PostgreSQL):

```bash
cd /home/jding/bruce/paper_icde/engines/postgres
nice -n 10 cargo pgrx package --features pg18 --no-default-features --pg-config $PGC --out-dir $E/build/pkg_bruce_pg
P=$E/build/pkg_bruce_pg/home/jding/miniforge3/envs/ggdb-pg
cp $P/lib/bruce_pg.so $E/ext/lib/; cp $P/share/extension/* $E/ext/share/extension/
cp sql/bruce_pg--0.1.2.sql $E/ext/share/extension/

cd /home/jding/bruce/paper_icde/experiments/softmax_ladder/pg_ablation   # has a pg18 feature + Cargo.lock
nice -n 10 cargo pgrx package --features pg18 --no-default-features --pg-config $PGC --out-dir $E/build/pkg_ablation
P=$E/build/pkg_ablation/home/jding/miniforge3/envs/ggdb-pg
cp $P/lib/softavg_ablation.so $E/ext/lib/; cp $P/share/extension/* $E/ext/share/extension/
```

(`cargo pgrx package` builds the release profile by default; it has no `--release` flag.)
After changing an extension: rebuild, copy, then in each database
`DROP EXTENSION ... CASCADE; CREATE EXTENSION ...` (or restart the server for a new `.so`).

## bruce_pg test suite on PG 18 (`cargo pgrx test`)

`cargo pgrx test` normally installs into `pg_config --pkglibdir/--sharedir`, i.e. the conda
env. It was pointed elsewhere with a separate `PGRX_HOME` whose `pg18` entry is a test-only
`pg_config` wrapper (`build/pgrxtest/bin/pg_config`) that redirects `--pkglibdir`,
`--sharedir` into `build/pgrxtest/ext` and `--bindir` into `build/pgrxtest/bin` (symlinks to
the conda binaries + an `initdb` wrapper that appends `dynamic_library_path` /
`extension_control_path` for `build/pgrxtest/ext` to each fresh test cluster):

```bash
. $E/build/env.sh; export PGRX_HOME=$E/build/pgrxtest/pgrx_home
cd /home/jding/bruce/paper_icde/engines/postgres
cargo pgrx test pg18 --release --no-default-features --features pg18 --pgdata $E/build/pgrxtest/pgdata
```

Result 2026-11-07 (`experiments/postgres/tests_pg18.json`): 55 passed, 1 failed of 56
(8 pure Rust + 48 `#[pg_test]`, incl. both conformance-vector tests). The failure is
`test_i_upgrade_path_alter_extension`: it writes its probe update script into the
compiled-in `SHAREDIR/extension`, but under `extension_control_path` PostgreSQL 18 reads
update scripts from the directory of the control file. **Do not rerun the full suite as is:**
the ERROR aborts the test before its cleanup, so it leaves
`bruce_pg--0.2.0--0.2.1.sql` (88 bytes) in the conda env's `share/extension` (happened once
on 2026-11-07; removed right away, the env's file listing is again identical to the pre-run
snapshot). Run the other tests with a name filter, or fix the test first (TODO in
`notes/INTEGRATION_ISSUES.md`). The same probe done by hand in `ext/share/extension`
passes (extversion 0.2.1, comment set; rolled back, probe removed).

## Files

| path | what | keep? |
|---|---|---|
| `start.sh`, `stop.sh` | start / stop the cluster | yes |
| `data/` | the cluster (stopped when not in use) | yes |
| `ext/` | staged extensions (lib, share/extension, bin) | yes |
| `log/server.log` | server log | yes |
| `build/env.sh`, `build/compat/xlocale.h` | build environment | yes |
| `build/pgvector`, `build/pg_ivm` | copied sources + objects | small |
| `build/pgrxtest/` | test-only wrappers (`pg_config`, `initdb`, symlinks), `PGRX_HOME` | small |
| `build/*.log` | build / package / test logs | small |

Removed after the build: `build/target` (cargo target dir, 1.2 GB) and `build/pkg_*` (the
packaged trees, already copied into `ext/`); the commands above recreate them (~1 min).
`data/pg_wal` holds ~2.2 GB of recycled WAL segments from the ladder runs (`max_wal_size =
8GB`); left as is (do not delete files there by hand). `cargo pgrx init` also left an unused
`~/.pgrx/data-18` (39 MB).
