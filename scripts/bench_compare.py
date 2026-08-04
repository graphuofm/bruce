#!/usr/bin/env python3
"""Workstream 17 — criterion baseline snapshot + regression compare.

Reads criterion's own estimates (target/criterion/<group>/<bench>/new/
estimates.json) after a `cargo bench -p bruce-core --bench fold` run.

Modes
-----
  --save     snapshot the current run as the baseline (with machine
             metadata) into the baselines dir
  (default)  compare the current run against the saved baseline and
             exit 1 if any bench's MEDIAN regressed by more than
             --threshold (default 15%)

Anti-noise protocol (the contract behind the 15% gate)
------------------------------------------------------
  * the compared statistic is criterion's *median* point estimate —
    robust to descheduling spikes; means are not used;
  * idle-box assumption: the nightly gate runs on the otherwise-quiet
    32-core box. No taskset pinning: the kernels are rayon-wide by
    design, so pinning would benchmark a different engine;
  * fixed deterministic inputs in benches/fold.rs (no RNG state);
  * 15% is far above the observed run-to-run jitter of medians on an
    idle box (single-digit percent); improvements are reported but
    never fail the gate;
  * baselines are machine-tagged — comparing against a baseline from a
    different hostname/core-count is refused unless --force.

Usage
-----
  cargo bench -p bruce-core --bench fold
  python3 scripts/bench_compare.py --save          # once, to freeze
  python3 scripts/bench_compare.py                 # nightly: diff
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CRIT = REPO / "target" / "criterion"
BASELINE_DIR = Path(
    "/home/jding/bruce/paper_sigmod_bruce/experiments/perf_baselines"
)
BASELINE = BASELINE_DIR / "fold_baseline.json"

# The stable bench ids the gate watches (group, bench). Keep in sync
# with bruce-core/benches/fold.rs.
WATCHED_GROUPS = ["grouped_softavg", "masked_attention", "kv_memory"]


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def rustc_version() -> str:
    try:
        return subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def collect_run() -> dict:
    """Parse criterion's estimates.json for every watched bench."""
    out = {}
    if not CRIT.is_dir():
        sys.exit(f"FAIL: no criterion output at {CRIT}; run cargo bench first")
    for group in WATCHED_GROUPS:
        gdir = CRIT / group
        if not gdir.is_dir():
            continue
        for bdir in sorted(gdir.iterdir()):
            est = bdir / "new" / "estimates.json"
            if not est.is_file():
                continue
            with open(est) as f:
                e = json.load(f)
            out[f"{group}/{bdir.name}"] = {
                "median_ns": e["median"]["point_estimate"],
                "median_ci_lo_ns": e["median"]["confidence_interval"]["lower_bound"],
                "median_ci_hi_ns": e["median"]["confidence_interval"]["upper_bound"],
                "mean_ns": e["mean"]["point_estimate"],  # informational only
            }
    if not out:
        sys.exit(f"FAIL: no estimates.json found under {CRIT} for {WATCHED_GROUPS}")
    return out


def machine_meta() -> dict:
    return {
        "hostname": platform.node(),
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "kernel": platform.release(),
        "rustc": rustc_version(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "criterion median point estimate; idle box; no taskset "
            "(rayon-wide kernels); deterministic inputs; gate = +15% on median"
        ),
    }


def do_save() -> None:
    run = collect_run()
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE, "w") as f:
        json.dump({"machine": machine_meta(), "benches": run}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"saved baseline for {len(run)} benches -> {BASELINE}")
    for k, v in sorted(run.items()):
        print(f"  {k:45} median {v['median_ns'] / 1e6:10.3f} ms")


def do_compare(threshold: float, force: bool) -> int:
    if not BASELINE.is_file():
        sys.exit(f"FAIL: no baseline at {BASELINE}; run with --save first")
    with open(BASELINE) as f:
        base = json.load(f)
    if base["machine"]["hostname"] != platform.node() and not force:
        sys.exit(
            f"FAIL: baseline is from {base['machine']['hostname']!r}, this is "
            f"{platform.node()!r}; cross-machine compare needs --force"
        )
    run = collect_run()

    failures, missing = [], []
    print(f"{'bench':45} {'baseline':>12} {'current':>12} {'delta':>8}")
    for name, b in sorted(base["benches"].items()):
        if name not in run:
            missing.append(name)
            continue
        cur = run[name]["median_ns"]
        old = b["median_ns"]
        delta = (cur - old) / old
        flag = ""
        if delta > threshold:
            flag = "  << REGRESSION"
            failures.append((name, delta))
        print(
            f"{name:45} {old / 1e6:10.3f}ms {cur / 1e6:10.3f}ms "
            f"{delta * 100:+7.1f}%{flag}"
        )
    new_benches = sorted(set(run) - set(base["benches"]))
    for name in new_benches:
        print(f"{name:45} {'--':>12} {run[name]['median_ns'] / 1e6:10.3f}ms   (new, not gated)")

    if missing:
        print(f"FAIL: benches in baseline but absent from run: {missing}")
        return 1
    if failures:
        print(
            f"FAIL: {len(failures)} regression(s) beyond "
            f"{threshold * 100:.0f}%: "
            + ", ".join(f"{n} ({d * 100:+.1f}%)" for n, d in failures)
        )
        return 1
    print(f"PASS: {len(base['benches'])} benches within {threshold * 100:.0f}% of baseline")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", action="store_true", help="snapshot current run as baseline")
    ap.add_argument("--threshold", type=float, default=0.15, help="regression gate (fraction)")
    ap.add_argument("--force", action="store_true", help="allow cross-machine compare")
    args = ap.parse_args()
    if args.save:
        do_save()
    else:
        sys.exit(do_compare(args.threshold, args.force))


if __name__ == "__main__":
    main()
