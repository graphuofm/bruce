#!/usr/bin/env python3
"""Workstream 16/20 — abi3 import matrix (docs/TESTING_MATRIX.md).

The wheel is built cp39-abi3: ONE binary artifact for every CPython
>= 3.9. This script enumerates the interpreters ACTUALLY PRESENT on
this box (it never fakes absent ones), loads the same wheel file into
each, runs one end-to-end SOFTAVG query, and checks the answer against
a pure-numpy oracle computed inside that interpreter.

Discovery scope (dedup by realpath): /usr/bin/python3*,
/usr/local/bin/python3*, ~/miniforge3/bin/python3*,
~/miniforge3/envs/*/bin/python3*. Conda envs without a python binary
(e.g. the pgv PG-runtime env) are recorded as such, not skipped
silently.

The wheel is unzipped (not pip-installed) into a temp dir and put on
sys.path of the probe subprocess — the identical .so bytes in every
interpreter, which is exactly the abi3 claim under test.

Output: docs/qa/abi3_matrix.json + a table on stdout.
Exit 0 iff every present interpreter >= 3.9 with numpy passes
import + query; interpreters without numpy are recorded (the wheel
requires numpy at runtime) but do not fail the run.

Run:  python3 scripts/abi3_matrix.py
"""

import glob
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "qa" / "abi3_matrix.json"
WHEEL_DIR = ROOT / "target" / "wheels"
ABI3_FLOOR = (3, 9)  # cp39-abi3

PROBE = r"""
import json, sys
wheel_dir, pq = sys.argv[1], sys.argv[2]
sys.path.insert(0, wheel_dir)
out = {"python": sys.version.split()[0], "exe": sys.executable}
try:
    import numpy as np
    out["numpy"] = np.__version__
except Exception as e:
    out["numpy"] = None
    out["error"] = "numpy unavailable: %r" % (e,)
    print(json.dumps(out)); sys.exit(0)
try:
    import bruce
    out["import_ok"] = True
    out["bruce"] = bruce.__version__
except Exception as e:
    out["import_ok"] = False
    out["error"] = repr(e)
    print(json.dumps(out)); sys.exit(0)
try:
    s = bruce.QuerySession()
    s.register_parquet("t", pq)
    emb = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.0, 0.5]])
    s.attach_key("t", "emb", emb)
    x = np.array([1.0, 0.0])
    labels, values, _ = s.run(
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.5) "
        "FROM t GROUP BY genre", {"q": x})
    got = dict(zip(labels, values))
    # oracle: same fixed data the driver wrote into the parquet
    ratings = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    sims = {"a": [1.0, 0.5], "b": [0.0, 0.0]}
    ok = True
    for g in ("a", "b"):
        w = np.exp((np.array(sims[g]) - max(sims[g])) / 0.5)
        want = float(w @ ratings[g] / w.sum())
        ok = ok and abs(got[g] - want) < 1e-12
    out["query_ok"] = bool(ok)
    if not ok:
        out["error"] = "answer mismatch: %r" % (got,)
except Exception as e:
    out["query_ok"] = False
    out["error"] = repr(e)
print(json.dumps(out))
"""


def find_interpreters():
    pats = [
        "/usr/bin/python3*",
        "/usr/local/bin/python3*",
        os.path.expanduser("~/miniforge3/bin/python3*"),
        os.path.expanduser("~/miniforge3/envs/*/bin/python3*"),
    ]
    seen, out = set(), []
    for p in pats:
        for c in sorted(glob.glob(p)):
            base = os.path.basename(c)
            # binaries only: python3, python3.12 — not python3-config
            if not base.replace("python", "").replace(".", "").isdigit() and base != "python3":
                continue
            r = os.path.realpath(c)
            if r in seen or not os.access(r, os.X_OK):
                continue
            seen.add(r)
            out.append(r)
    return out


def python_free_conda_envs():
    envs = glob.glob(os.path.expanduser("~/miniforge3/envs/*"))
    return [e for e in envs if not glob.glob(os.path.join(e, "bin", "python3*"))]


def main() -> int:
    wheels = sorted(WHEEL_DIR.glob("bruce-*-abi3-*.whl"), key=os.path.getmtime)
    if not wheels:
        print(f"no abi3 wheel under {WHEEL_DIR}; run `make python` first")
        return 1
    wheel = wheels[-1]

    try:
        import pandas as pd
    except ImportError:
        print("driver needs pandas (to write the probe parquet)")
        return 1

    results = {"date": "2026-08-03", "wheel": wheel.name,
               "wheel_mtime": int(os.path.getmtime(wheel)),
               "interpreters": [], "conda_envs_without_python": python_free_conda_envs()}
    rc = 0
    with tempfile.TemporaryDirectory() as td:
        site = os.path.join(td, "site")
        zipfile.ZipFile(wheel).extractall(site)
        pq = os.path.join(td, "probe.parquet")
        pd.DataFrame({"genre": ["a", "a", "b", "b"],
                      "rating": [1.0, 2.0, 3.0, 4.0],
                      "year": [2000.0] * 4}).to_parquet(pq)
        probe = os.path.join(td, "probe.py")
        pathlib.Path(probe).write_text(PROBE)

        for exe in find_interpreters():
            r = subprocess.run([exe, probe, site, pq],
                               capture_output=True, text=True, timeout=120)
            try:
                row = json.loads(r.stdout.strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                row = {"exe": exe, "python": "?", "import_ok": False,
                       "error": (r.stderr or r.stdout).strip()[-500:]}
            results["interpreters"].append(row)
            ver = tuple(int(v) for v in row.get("python", "0.0").split(".")[:2])
            eligible = ver >= ABI3_FLOOR and row.get("numpy")
            passed = row.get("import_ok") and row.get("query_ok")
            if eligible and not passed:
                rc = 1
            print(f"  {row.get('python', '?'):<9} {exe:<50} "
                  f"numpy={row.get('numpy') or '-':<8} "
                  f"import={'ok' if row.get('import_ok') else 'FAIL':<5} "
                  f"query={'ok' if row.get('query_ok') else ('FAIL' if eligible else '-')}"
                  + (f"  [{row['error']}]" if row.get("error") else ""))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"{'MATRIX PASS' if rc == 0 else 'MATRIX FAIL'} -> {OUT}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
