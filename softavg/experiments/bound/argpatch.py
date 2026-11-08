"""Shipped InvAcc bound vs the proposed soundness patch (argument rounding +
underflow terms; local copy with arg_term = true), same tolerance, exact
reference; writes results.json -> argpatch.<corpus>.

  python3 argpatch.py [threads=10]       (IMDb 20 queries, Amazon 5, StackExchange 20)
"""
import json
import os
import shutil
import subprocess
import sys

import prep
import run

PLAN = [("imdb", 20), ("amazon", 5), ("stackexchange", 20)]


def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    env = dict(os.environ, CARGO_TARGET_DIR=str(run.TARGET))
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(run.RUST / "Cargo.toml")], check=True, env=env)
    res = json.load(open(run.HERE / "results.json"))
    res.pop("argpatch_imdb", None)
    out = {"what": "shipped InvAcc (local copy, default tol) vs the proposed patch (arg_term=true: charge 2U|x| per "
                   "weight for the rounding of the exp argument, and 2^-1074 per underflowed weight/product)",
           "experiment_src_sha256_16": run.sha(list((run.RUST / "src").glob("*.rs")) + [run.RUST / "Cargo.toml"]),
           "corpora": {}}
    for corpus, nq in PLAN:
        prep.prep(corpus, nq)
        wd = run.HERE / "work" / corpus
        oj = run.HERE / "work" / f"argpatch_{corpus}.json"
        subprocess.run([str(run.EXE), "argpatch", str(wd), str(oj), str(nq), str(threads)], check=True)
        out["corpora"][corpus] = json.load(open(oj))
        shutil.rmtree(wd)
        oj.unlink()
    res["argpatch"] = out
    json.dump(res, open(run.HERE / "results.json", "w"), indent=1)
    shutil.rmtree(run.HERE / "work", ignore_errors=True)
    print("argpatch written to results.json")


if __name__ == "__main__":
    main()
