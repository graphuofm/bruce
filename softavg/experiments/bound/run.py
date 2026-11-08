"""Run the bound experiment end to end and write results.json.

  python3 run.py [n_queries=20] [threads=10] [sweep_queries=5]
  python3 run.py bench        (re-time only; replaces results.json -> bench)

1. builds rust/ (release; links ../../engines/core unmodified),
2. per corpus: prep.py -> work/<corpus>, `bound-exp run`, then deletes
   work/<corpus>,
3. `bound-exp bench` on IMDb (before its work dir is deleted),
4. writes results.json (all numbers the README and the figures use).
Pure single-threaded arithmetic per unit; units (query x eps) run on
`threads` threads. Timings are smoke numbers (shared machine).
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ICDE = HERE.parents[1]
CORE = ICDE / "engines" / "core"
RUST = HERE / "rust"
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", RUST / "target"))
EXE = TARGET / "release" / "bound-exp"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import quiet  # noqa: E402

sys.path.insert(0, str(HERE))
import prep  # noqa: E402


def sha(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def machine_conditions():
    try:
        c = quiet.require_quiet(wait_seconds=0, verbose=False)
        c["quiet"] = True
    except SystemExit as e:
        c = {"quiet": False, "require_quiet_message": str(e),
             "cpu_busy_fraction": round(quiet.cpu_busy_fraction(), 3),
             "load_average_1min": round(os.getloadavg()[0], 2)}
    c["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return c


def bench_only():
    """Re-run only the per-row timing on IMDb and replace results.json -> bench."""
    env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET))
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(RUST / "Cargo.toml")], check=True, env=env)
    out = json.load(open(HERE / "results.json"))
    prep.prep("imdb", 1)
    wd = HERE / "work" / "imdb"
    bj = HERE / "work" / "bench.json"
    out["machine_before_bench"] = machine_conditions()
    subprocess.run([str(EXE), "bench", str(wd), str(bj)], check=True)
    out["bench"] = json.load(open(bj))
    out["bench"]["timing_status"] = "smoke, machine shared"
    out["bench"]["experiment_src_sha256_16"] = sha(list((RUST / "src").glob("*.rs")) + [RUST / "Cargo.toml"])
    out["machine_after_bench"] = machine_conditions()
    json.dump(out, open(HERE / "results.json", "w"), indent=1)
    shutil.rmtree(HERE / "work", ignore_errors=True)
    print("bench updated in results.json")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        return bench_only()
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sweep_q = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET))
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(RUST / "Cargo.toml")], check=True, env=env)
    out = {
        "what": "slack of softagg-core InvAcc's running error bound vs the exact value; refold tolerance trade-off; "
                "a compensated (double-double) prototype; mergeable ESS diagnostic",
        "host_engine": "none: softagg-core driven directly from Rust (the arithmetic every host extension links)",
        "softagg_core_version": "0.1.0",
        "softagg_core_src_sha256_16": sha(list((CORE / "src").glob("*.rs")) + [CORE / "Cargo.toml"]),
        "experiment_src_sha256_16": sha(list((RUST / "src").glob("*.rs")) + [RUST / "Cargo.toml"]),
        "build": "release (opt-level 3, lto fat, codegen-units 1)",
        "rustc": subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip(),
        "cpu": platform.processor() or platform.machine(),
        "timing_status": "smoke, machine shared",
        "machine_before": machine_conditions(),
        "queries": nq, "threads": threads, "sweep_queries": sweep_q,
        "corpora": {}, "prep": {},
    }
    for corpus in ["imdb", "amazon", "stackexchange"]:
        t0 = time.time()
        prep.prep(corpus, nq)
        wd = HERE / "work" / corpus
        out["prep"][corpus] = json.load(open(wd / "meta.json"))
        raw = HERE / "work" / f"{corpus}.json"
        subprocess.run([str(EXE), "run", str(wd), str(raw), str(nq), str(threads), str(sweep_q)], check=True)
        out["corpora"][corpus] = json.load(open(raw))
        if corpus == "imdb":
            bj = HERE / "work" / "bench.json"
            out["machine_before_bench"] = machine_conditions()
            subprocess.run([str(EXE), "bench", str(wd), str(bj)], check=True)
            out["bench"] = json.load(open(bj))
            out["bench"]["timing_status"] = "smoke, machine shared"
        shutil.rmtree(wd)
        raw.unlink()
        print(f"{corpus}: {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(HERE / "results.json", "w"), indent=1)
    out["machine_after"] = machine_conditions()
    json.dump(out, open(HERE / "results.json", "w"), indent=1)
    shutil.rmtree(HERE / "work", ignore_errors=True)
    print("wrote results.json")


if __name__ == "__main__":
    main()
