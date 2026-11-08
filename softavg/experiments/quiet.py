"""Refuse to record a latency under machine load.

Timings taken while other jobs run are not measurements of the system
under test. Every harness that reports a latency imports this; it
aborts unless the machine is quiet and records the observed conditions
next to the numbers, so a reader sees the conditions rather than
trusting them.

Discovered the hard way: a set of latencies was collected while a GPU
embedding job was running, and two independent harnesses then
disagreed by 1.7x on the same computation.

Busy fraction is measured from /proc/stat over a short window rather
than from load average, which decays over minutes and therefore stays
high long after the machine is actually free.
"""
import os, subprocess, sys, time


def _cpu_times():
    with open("/proc/stat") as f:
        parts = [float(x) for x in f.readline().split()[1:]]
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)
    return sum(parts), idle


def cpu_busy_fraction(window=1.0):
    """Instantaneous CPU busy fraction over `window` seconds."""
    t0, i0 = _cpu_times()
    time.sleep(window)
    t1, i1 = _cpu_times()
    dt = t1 - t0
    return 0.0 if dt <= 0 else max(0.0, min(1.0, 1.0 - (i1 - i0) / dt))


def gpu_utilisation():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return max(int(x) for x in out.stdout.split()) if out.stdout.strip() else 0
    except Exception:
        return 0


def require_quiet(max_busy=0.15, max_gpu=20, wait_seconds=0, verbose=True):
    """Block until the machine is quiet, or abort. Returns the conditions."""
    deadline = time.time() + wait_seconds
    while True:
        busy, gpu = cpu_busy_fraction(), gpu_utilisation()
        la = os.getloadavg()[0]
        if busy <= max_busy and gpu <= max_gpu:
            if verbose:
                print(f"[quiet] cpu {busy:.1%} busy, gpu {gpu}%, "
                      f"load {la:.2f} (lagging) -- recording timings", flush=True)
            return dict(cpu_busy_fraction=round(busy, 3),
                        gpu_utilisation_pct=gpu, load_average_1min=round(la, 2),
                        thresholds=dict(max_cpu_busy=max_busy, max_gpu=max_gpu),
                        note="load average is recorded for information only; "
                             "the gate is instantaneous CPU and GPU use")
        if time.time() >= deadline:
            sys.exit(f"[quiet] REFUSING to record timings: cpu {busy:.1%} busy "
                     f"(limit {max_busy:.0%}), gpu {gpu}% (limit {max_gpu}%).")
        if verbose:
            print(f"[quiet] cpu {busy:.1%}, gpu {gpu}% -- waiting...", flush=True)
        time.sleep(15)
