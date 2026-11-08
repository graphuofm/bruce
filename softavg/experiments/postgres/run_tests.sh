#!/bin/bash
# Run `cargo pgrx test pg17` for engines/postgres and write a digest to
# tests_pg17.json. NOTE: cargo pgrx test installs a debug pg_test build of
# bruce_pg into ~/.pgrx/17.10/pgrx-install (replacing the bruce_pg.so there).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ENG=$HERE/../../engines/postgres
LOG=$(mktemp)
export LIBCLANG_PATH=$HOME/.local/lib/python3.10/site-packages/clang/native
export BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/gcc/x86_64-linux-gnu/12/include"
export PATH="$HOME/miniforge3/bin:$PATH"
(cd "$ENG" && nice -n 10 cargo pgrx test pg17) > "$LOG" 2>&1
RC=$?
python3 - "$LOG" "$RC" "$HERE/tests_pg17.json" <<'EOF'
import json, re, sys, time
log, rc, out = open(sys.argv[1]).read(), int(sys.argv[2]), sys.argv[3]
tests = re.findall(r"^test (\S+) \.\.\. (ok|FAILED|ignored)", log, re.M)
summ = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored", log)
pure = [t for t, s in tests if not t.startswith("tests::")]
pg = [t for t, s in tests if t.startswith("tests::")]
d = dict(command="cargo pgrx test pg17", exit_code=rc, at=time.strftime("%Y-%m-%dT%H:%M:%S"),
         passed=sum(s == "ok" for _, s in tests), failed=sum(s == "FAILED" for _, s in tests),
         ignored=sum(s == "ignored" for _, s in tests), pure_rust=len(pure), pg_test=len(pg),
         result_lines=[" ".join(x) for x in summ], tests={t: s for t, s in tests},
         server="pgrx-managed PostgreSQL 17.10 (~/.pgrx/17.10/pgrx-install, cassert), pg_test feature build")
json.dump(d, open(out, "w"), indent=1)
print(json.dumps({k: d[k] for k in ("exit_code", "passed", "failed", "pure_rust", "pg_test")}))
EOF
rm -f "$LOG"
exit $RC
