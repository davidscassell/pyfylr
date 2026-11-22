# Not collected by pytest (the name does not start with test_).
# Not executed with the within-fit E-step edit.
#
# Record the serial fits on f1cf2bd7cb41e302da7db2194642f8c004bf074d, which
# has no estep_threads argument. On this PR, estep_threads=1 must match that
# record bitwise. estep_threads 2, 4, and 8 must match each other bitwise
# and lie within 1e-8 of the serial fit. They are not claimed bitwise-equal
# to serial. Do not set LTA_ETHREAD_RECORD and LTA_ETHREAD_BASE together.
#
#   git worktree add /tmp/ethread-base f1cf2bd7cb41e302da7db2194642f8c004bf074d
#   cp pyfylr/tests/unrun_estep_threads_equivalence.py /tmp/ethread-base/pyfylr/tests/
#   PYTHONPATH=/tmp/ethread-base/pyfylr/src \
#     LTA_ETHREAD_RECORD=/tmp/ethread-base.rec \
#     python3 /tmp/ethread-base/pyfylr/tests/unrun_estep_threads_equivalence.py
#   LTA_ETHREAD_BASE=/tmp/ethread-base.rec \
#     python3 pyfylr/tests/unrun_estep_threads_equivalence.py

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pyfylr as pf

RECORD_COMMIT = "f1cf2bd7cb41e302da7db2194642f8c004bf074d"
RECORD_HASHES = {
    "lta.cpp": "a7a9d7796f2451977cb23cdda70744e5ee45882843514a35d237781a73ce5d36",
    "lta.hpp": "dc9d9fe767b0c61adecb838059c4ff1d47cc10c2c3b5adbbd5080dc766433e5c",
    "_core.cpp": "ea95b126a980ccf64b89e65e9a1f98b035661f05a354c079310965250ea9e84c",
    "__init__.py": "82230229b1ca8a16c37e90ca10b67530d2dc84eeb3401938d760ff3ddebd6872",
}
HEAD_HASHES = {
    "lta.cpp": "c9cdff1bdbd41be39af2d0a6e6826d4c3a14c2de4520b2aae4f436c68a949a56",
    "lta.hpp": "1ebe8680c1e1949a9dc0a828fcb571764d453256858cefeec65d7ad21e5034ca",
    "_core.cpp": "2e6d9abf3e7360aa7c14d287c3c9084ceb15cf114f57db097b546d1ed607fefd",
    "__init__.py": "14e9ee0f5a407d0121a1a962014dcd101bcd3b36a6f52c2182c9f2cdb5743c85",
}
SOURCES = {
    "lta.cpp": ("core", "src", "lta.cpp"),
    "lta.hpp": ("core", "include", "lta", "lta.hpp"),
    "_core.cpp": ("pyfylr", "src", "pyfylr", "_core.cpp"),
    "__init__.py": ("pyfylr", "src", "pyfylr", "__init__.py"),
}
ABS = 1e-8


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _inside(path, root):
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _git(root, *args):
    proc = subprocess.run(["git", "-C", str(root), *args], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("FAIL git " + " ".join(args))
    return proc.stdout.strip()


def _repo_root():
    pkg = Path(pf.__file__).resolve()
    extension = Path(pf._core.__file__).resolve()
    root = None
    for parent in pkg.parents:
        if (parent / "core" / "src" / "lta.cpp").is_file() and (parent / ".git").exists():
            root = parent
            break
    if root is None or not _inside(pkg, root) or not _inside(extension, root):
        raise SystemExit("FAIL loaded pyfylr is not inside the checked tree")
    init = root.joinpath(*SOURCES["__init__.py"]).resolve()
    if pkg != init or Path(inspect.getfile(pf.fit_lpa)).resolve() != init:
        raise SystemExit("FAIL loaded fit_lpa is not this tree's __init__.py")
    return root


def _fingerprint(root):
    return {name: _sha256(root.joinpath(*parts)) for name, parts in SOURCES.items()}


def _require_record_identity():
    root = _repo_root()
    if _git(root, "rev-parse", "HEAD") != RECORD_COMMIT or _fingerprint(root) != RECORD_HASHES:
        raise SystemExit("FAIL record tree is not f1cf2bd with its sources")


def _require_compare_identity():
    root = _repo_root()
    found = _fingerprint(root)
    if _git(root, "rev-parse", "HEAD") == RECORD_COMMIT or found == RECORD_HASHES or found != HEAD_HASHES:
        raise SystemExit("FAIL compare sources are not this PR's copies")
    if _git(root, "cat-file", "-e", "HEAD:pyfylr/tests/unrun_estep_threads_equivalence.py") != "":
        raise SystemExit("FAIL this harness is not in the compare commit")


def _bytes(value):
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return array.tobytes().hex()


def _pack(result):
    packed = {"loglik": _bytes([result["loglik"]]), "n_iter": int(result["n_iter"])}
    for key in ("delta", "tau", "rho", "means", "cov"):
        packed["params." + key] = _bytes(result["params"][key])
    for key in ("gamma", "filtered", "xi"):
        if key in result:
            packed[key] = _bytes(result[key])
    return packed


def _close_array(left, right, name):
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise SystemExit("FAIL shape " + name)
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    pos_a = np.isposinf(a)
    pos_b = np.isposinf(b)
    neg_a = np.isneginf(a)
    neg_b = np.isneginf(b)
    if not (np.array_equal(nan_a, nan_b) and np.array_equal(pos_a, pos_b) and np.array_equal(neg_a, neg_b)):
        raise SystemExit("FAIL non-finite pattern " + name)
    finite = ~(nan_a | pos_a | neg_a)
    if not np.any(finite):
        return
    diff = np.abs(a[finite] - b[finite])
    if np.any(np.isnan(diff)):
        raise SystemExit("FAIL NaN in finite diff " + name)
    if float(np.max(diff)) > ABS:
        raise SystemExit("FAIL " + name + " outside 1e-8")


def _close(serial, parallel):
    if int(serial["n_iter"]) != int(parallel["n_iter"]):
        raise SystemExit("FAIL iteration count changed")
    _close_array([serial["loglik"]], [parallel["loglik"]], "loglik")
    for key in ("delta", "tau", "rho", "means", "cov", "gamma", "filtered", "xi"):
        left = serial["params"][key] if key in ("delta", "tau", "rho", "means", "cov") else serial[key]
        right = parallel["params"][key] if key in ("delta", "tau", "rho", "means", "cov") else parallel[key]
        _close_array(left, right, key)


def _frames():
    # 16 full blocks of 32 plus a one-subject remainder. 2, 4, and 8 threads
    # therefore receive different block assignments.
    n = 513
    rng = np.random.default_rng(11)
    cat = np.array([0] * 256 + [1] * 257)
    u = cat.astype(float)
    v = np.where(rng.random(n) < 0.75, cat, 1 - cat).astype(float)
    u[0] = np.nan
    lca = pd.DataFrame({"u": u, "v": v})
    ids = np.repeat(np.arange(n), 2)
    waves = np.tile([1, 2], n)
    group = np.repeat(cat, 2)
    uu = group.astype(float)
    vv = np.where(rng.random(n * 2) < 0.75, group, 1 - group).astype(float)
    uu[1] = np.nan
    lta = pd.DataFrame({"id": ids, "wave": waves, "u": uu, "v": vv})
    g = np.where(cat == 0, -1.0, 1.0)
    gauss = pd.DataFrame({
        "y1": g + rng.normal(0, 0.3, size=n),
        "y2": g + rng.normal(0, 0.3, size=n),
    })
    return lca, lta, gauss


def _fit_all(extra=None):
    extra = {} if extra is None else dict(extra)
    controls = dict(n_starts=2, screen_iter=0, n_full=2, max_iter=15, seed=11, n_threads=1)
    controls.update(extra)
    lca, lta, gauss = _frames()
    out = {
        "lca": pf.fit_lca(lca, ["u", "v"], n_states=2, **controls).result,
        "lta": pf.fit_lta(lta, ["u", "v"], "id", "wave", n_states=2, **controls).result,
        "lta-stream": pf.fit_lta(lta, ["u", "v"], "id", "wave", n_states=2, estep="streaming", **controls).result,
        "lpa": pf.fit_lpa(gauss, ["y1", "y2"], n_states=2, covariance="EEI", **controls).result,
    }
    analyzed = int(out["lta"]["n_analyzed"])
    if analyzed < 512 or analyzed % 32 == 0:
        raise SystemExit("FAIL fixture is not 512 or more subjects with a partial last block")
    nblocks = (analyzed + 31) // 32

    def assignment(nthreads):
        spans = []
        cursor = 0
        left_blocks = nblocks
        left_threads = nthreads
        for _ in range(nthreads):
            take = left_blocks // left_threads
            spans.append((cursor, cursor + take))
            cursor += take
            left_blocks -= take
            left_threads -= 1
        return tuple(spans)

    if len({assignment(2), assignment(4), assignment(8)}) != 3:
        raise SystemExit("FAIL 2/4/8 threads share a block assignment")

    def waves(nthreads):
        grouped = []
        for wave in range(0, nblocks, nthreads):
            grouped.append(tuple(range(wave, min(wave + nthreads, nblocks))))
        return tuple(grouped)

    if len({waves(2), waves(4), waves(8)}) != 3:
        raise SystemExit("FAIL 2/4/8 threads share a streaming wave")
    return out


def main():
    record = os.environ.get("LTA_ETHREAD_RECORD")
    base = os.environ.get("LTA_ETHREAD_BASE")
    if bool(record) == bool(base):
        raise SystemExit("FAIL set exactly one of LTA_ETHREAD_RECORD and LTA_ETHREAD_BASE")
    if record:
        _require_record_identity()
        payload = {name: _pack(result) for name, result in _fit_all().items()}
        with open(record, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
        return
    _require_compare_identity()
    with open(base, encoding="utf-8") as handle:
        recorded = json.load(handle)
    serial = _fit_all({"estep_threads": 1})
    for name, result in serial.items():
        if _pack(result) != recorded[name]:
            raise SystemExit("FAIL estep_threads=1 does not match the serial record " + name)
    parallel = {threads: _fit_all({"estep_threads": threads}) for threads in (2, 4, 8)}
    for name in serial:
        for threads in (4, 8):
            if _pack(parallel[threads][name]) != _pack(parallel[2][name]):
                raise SystemExit("FAIL thread counts disagree " + name + " " + str(threads))
        _close(serial[name], parallel[2][name])


if __name__ == "__main__":
    raise SystemExit(main())
