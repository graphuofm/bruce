# build environment for pgrx against the conda PG 18.6 (sourced by the commands in ../README.md)
export LIBCLANG_PATH=$HOME/.local/lib/python3.10/site-packages/clang/native
export BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/gcc/x86_64-linux-gnu/12/include -I/home/jding/bruce/paper_icde/pg18env/build/compat"
export CFLAGS="-I/home/jding/bruce/paper_icde/pg18env/build/compat"
export PATH="$HOME/miniforge3/bin:$PATH"
export CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR=/home/jding/bruce/paper_icde/pg18env/build/target
export PG_CONFIG=/home/jding/miniforge3/envs/ggdb-pg/bin/pg_config
