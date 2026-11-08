#!/bin/bash
# Start the paper_icde PostgreSQL 18.6 cluster (binaries of the conda env ggdb-pg, read-only;
# extensions from pg18env/ext via dynamic_library_path / extension_control_path). Port 28860.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
B=/home/jding/miniforge3/envs/ggdb-pg/bin
mkdir -p "$HERE/log"
"$B/pg_ctl" -D "$HERE/data" -l "$HERE/log/server.log" -w start
"$B/psql" -h 127.0.0.1 -p 28860 -U jding -d postgres -XAtc "SELECT version()"
