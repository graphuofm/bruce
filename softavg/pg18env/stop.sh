#!/bin/bash
# Stop the paper_icde PostgreSQL 18.6 cluster started by start.sh (only this data directory).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
/home/jding/miniforge3/envs/ggdb-pg/bin/pg_ctl -D "$HERE/data" -m fast -w stop
