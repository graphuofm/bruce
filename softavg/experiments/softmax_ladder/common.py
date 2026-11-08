"""Shared pieces of the softmax-ladder harness: grid, data, numpy float64
reference, error metrics, machine state, provenance hashes."""
import hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
ICDE = HERE.parents[1]
WORK = Path(os.environ.get("LADDER_WORK", "/tmp/ladder_work"))
TARGET = HERE / "target"
ABL_SQLITE = TARGET / "release/libsoftavg_ablation_sqlite.so"
SHIPPED_SQLITE = TARGET / "shipped_sqlite/release/libsoftagg_sqlite.so"
REPLAY_BIN = TARGET / "release/ladder_replay"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import quiet  # noqa: E402

DATASETS = ["imdb_all", "imdb_ge", "amazon"]
KS = [10, 100, 1000]
EPS = [1e-4, 1e-3, 0.02, 0.1, 1.0]
NQ = 3
WRONG_TOL = 1e-9  # PROTOCOL window rule: |err| <= 1e-9 * max|v| of the frame
TIMING_STATUS = "smoke, machine shared"


def load(name):
    z = np.load(WORK / f"{name}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def frame_views(x, k, pad):
    """(n, k+1) read-only view: row i = x[i-k .. i], left-padded with `pad`."""
    xp = np.concatenate([np.full(k, pad, dtype=np.float64), x.astype(np.float64)])
    return sliding_window_view(xp, k + 1)


def vmax_frames(v, k):
    return frame_views(np.abs(v), k, 0.0).max(1)


def reference(v, s, eps, k, chunk=4096):
    """numpy float64 two-pass per frame: m = max(s), sum e^{(s-m)/eps} v / sum e^{(s-m)/eps}."""
    S = frame_views(s, k, -np.inf); V = frame_views(v, k, 0.0)
    out = np.empty(len(v))
    for a in range(0, len(v), chunk):
        b = min(len(v), a + chunk)
        Sc = S[a:b]
        m = Sc.max(1, keepdims=True)
        W = np.exp((Sc - m) / eps)
        out[a:b] = np.einsum("ij,ij->i", W, V[a:b]) / W.sum(1)
    return out


def errors(got, ref, vmax):
    """got may hold NaN for SQL NULL."""
    null = np.isnan(got); inf = np.isinf(got); fin = ~(null | inf)
    ab = np.abs(got - ref)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(fin, ab / np.abs(ref), np.nan)
        over = np.where(fin, ab / vmax, np.nan)
    wrong_fin = fin & (ab > WRONG_TOL * vmax)
    worst = int(np.nanargmax(over)) if fin.any() else None
    return dict(n_rows=int(len(got)), n_null=int(null.sum()), n_inf=int(inf.sum()),
                n_wrong_finite=int(wrong_fin.sum()), n_wrong=int(wrong_fin.sum() + null.sum() + inf.sum()),
                max_abs_err_over_vmax=(float(np.nanmax(over)) if fin.any() else None),
                max_rel_err=(float(np.nanmax(rel)) if fin.any() else None),
                worst_row=worst)


def machine_state():
    try:
        st = quiet.require_quiet(wait_seconds=0, verbose=False); st["quiet"] = True
    except SystemExit as e:
        st = dict(quiet=False, refusal=str(e))
    st["load_average"] = [round(x, 2) for x in os.getloadavg()]
    st["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return st


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_tree(paths):
    h = hashlib.sha256()
    for f in sorted(Path(p) for p in paths):
        h.update(str(f.relative_to(ICDE)).encode()); h.update(f.read_bytes())
    return h.hexdigest()


def host():
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    return dict(cpu=sh("lscpu | sed -n 's/Model name: *//p'"), nproc=os.cpu_count(), kernel=platform.release(),
                rustc=sh("rustc --version"), python=platform.python_version(), numpy=np.__version__)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t0) * 1e3


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))
