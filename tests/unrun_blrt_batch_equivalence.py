# Not collected by pytest (the name does not start with test_).
# Not executed with the bounded-BLRT edit.
#
# This file is not in a861687943823dcaa39e4366421816536e072a6a. Copy it onto a
# worktree of that revision. The record run there calls blrt_check with
# n_threads=2 and no batch_size. On that revision that is one blrt_fit_draws
# of every simulated panel. The head run compares that record with this
# head's pool: the default batch (n_threads) and sizes 1, 7, and B, at
# worker counts 1, 2, and 4. Comparing head batches only with each other is
# not this check.
#
# Do not set LTA_BLRT_RECORD and LTA_BLRT_BASE together, and do not point
# them at the same path.
#
#   git worktree add /tmp/blrt-base a861687943823dcaa39e4366421816536e072a6a
#   cp pyfylr/tests/unrun_blrt_batch_equivalence.py /tmp/blrt-base/pyfylr/tests/
#   PYTHONPATH=/tmp/blrt-base/pyfylr/src \
#     LTA_BLRT_RECORD=/tmp/blrt-base.rec \
#     python3 /tmp/blrt-base/pyfylr/tests/unrun_blrt_batch_equivalence.py
#   LTA_BLRT_BASE=/tmp/blrt-base.rec python3 pyfylr/tests/unrun_blrt_batch_equivalence.py

import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pyfylr as pf
import pyfylr.selection as pysel


# Public counts from blrt_check, including b, p, and p_plus.
FIELDS = (
    "lrs",
    "lr_draws",
    "b",
    "p",
    "p_plus",
    "n_ok",
    "n_failed",
    "n_failed_not_converged",
    "n_failed_error",
    "n_failed_inadmissible",
    "n_not_replicated",
    "n_retried",
    "failures",
    "attempts",
)

# Pool rows that blrt_check copies into those public fields.
POOL_FIELDS = (
    "lrs",
    "lr_draws",
    "n_failed",
    "n_failed_not_converged",
    "n_failed_error",
    "n_failed_inadmissible",
    "n_not_replicated",
    "n_retried",
    "failures",
    "attempts",
)


BASE_COMMIT = "a861687943823dcaa39e4366421816536e072a6a"
# selection.py and core/src/lta.cpp at the record commit, then at af953ef.
# A later edit of either file fails closed until these pins move with it.
BASE_SELECTION = "12b5c6beb887063826019138675f8629b4d868443bc005b754dedca2e6a6d8ed"
BASE_CORE = "8da03e20806891d3749643d6d79fc51216687c7509a4e02710634d0a8edf4bb5"
HEAD_SELECTION = "05d162e586b5d9b577ea4842f1e87360eea3b864d2ea659c8fc52c0cc9850ee6"
HEAD_CORE = "c9cdff1bdbd41be39af2d0a6e6826d4c3a14c2de4520b2aae4f436c68a949a56"
HEAD_ANCHOR = "bdd9e76bba3a3a6dde0872a7c30dc63b06f4b178"
COMPARE_COMMIT = "af953ef27053f3b0a7aae2cf117a9eb4434c5345"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _inside(path, root):
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _repo_root():
    pkg = Path(pf.__file__).resolve()
    extension = Path(pf._core.__file__).resolve()
    selection = Path(pysel.__file__).resolve()
    pool = Path(inspect.getfile(pysel._blrt_draw_pool)).resolve()
    check = Path(inspect.getfile(pysel.blrt_check)).resolve()
    root = None
    for parent in pkg.parents:
        selection_path = parent / "pyfylr" / "src" / "pyfylr" / "selection.py"
        core = parent / "core" / "src" / "lta.cpp"
        git_dir = parent / ".git"
        if selection_path.is_file() and core.is_file() and git_dir.exists():
            root = parent
            break
    if root is None:
        raise SystemExit("FAIL loaded pyfylr.__file__ is not inside a source tree")
    hashed = (root / "pyfylr" / "src" / "pyfylr" / "selection.py").resolve()
    for label, path in (
        ("pyfylr.__file__", pkg),
        ("extension", extension),
        ("selection.py", selection),
        ("_blrt_draw_pool", pool),
        ("blrt_check", check),
    ):
        if not _inside(path, root):
            raise SystemExit("FAIL loaded " + label + " is outside the checked tree")
    if selection != hashed or pool != hashed or check != hashed:
        raise SystemExit("FAIL loaded BLRT functions are not this tree's selection.py")
    return root


def _git(root, *args):
    proc = subprocess.run(["git", "-C", str(root), *args], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("FAIL git " + " ".join(args))
    return proc.stdout.strip()


def _identity():
    root = _repo_root()
    commit = _git(root, "rev-parse", "HEAD")
    fingerprint = {
        "selection.py": _sha256(root / "pyfylr" / "src" / "pyfylr" / "selection.py"),
        "lta.cpp": _sha256(root / "core" / "src" / "lta.cpp"),
    }
    return commit, fingerprint


def _require_record_identity():
    commit, fingerprint = _identity()
    if commit != BASE_COMMIT:
        raise SystemExit("FAIL record worktree is not a861687: " + commit)
    if fingerprint["selection.py"] != BASE_SELECTION or fingerprint["lta.cpp"] != BASE_CORE:
        raise SystemExit("FAIL record sources are not the a861687 selection.py and lta.cpp")
    return commit


def _require_compare_identity():
    root = _repo_root()
    commit, fingerprint = _identity()
    if commit == BASE_COMMIT:
        raise SystemExit("FAIL compare worktree is still a861687")
    if _git(root, "merge-base", "--is-ancestor", HEAD_ANCHOR, "HEAD") != "":
        raise SystemExit("FAIL compare commit does not contain " + HEAD_ANCHOR)
    if _git(root, "merge-base", "--is-ancestor", COMPARE_COMMIT, "HEAD") != "":
        raise SystemExit("FAIL compare commit does not contain " + COMPARE_COMMIT)
    if fingerprint["selection.py"] != HEAD_SELECTION or fingerprint["lta.cpp"] != HEAD_CORE:
        raise SystemExit("FAIL compare sources are not the af953ef selection.py and lta.cpp")
    harness = "pyfylr/tests/unrun_blrt_batch_equivalence.py"
    if _git(root, "cat-file", "-e", "HEAD:" + harness) != "":
        raise SystemExit("FAIL this harness is not in the compare commit")
    return commit


def _bits(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        # Native IEEE-754 bytes, then little-endian. This keeps signed zero
        # and the NaN payload; it does not replace NaN with a token.
        raw = np.float64(value).tobytes()
        if sys.byteorder != "little":
            raw = raw[::-1]
        return raw.hex()
    if isinstance(value, (list, tuple)):
        return [_bits(item) for item in value]
    if isinstance(value, dict):
        return {key: _bits(value[key]) for key in sorted(value)}
    return value


def statistical(result):
    return _bits({key: result[key] for key in FIELDS})


def _gaussian_fits():
    rng = np.random.default_rng(19)
    group = np.repeat([-1.5, 1.5], 12)
    data = pd.DataFrame({
        "y1": group + rng.normal(0, 0.4, size=24),
        "y2": group + rng.normal(0, 0.4, size=24),
    })
    controls = dict(covariance="EEI", n_starts=4, screen_iter=0, n_full=2, max_iter=20, seed=19, n_threads=1)
    h0 = pf.fit_lpa(data, ["y1", "y2"], n_states=1, **controls)
    h1 = pf.fit_lpa(data, ["y1", "y2"], n_states=2, **controls)
    return h0, h1


def _categorical_fits():
    rng = np.random.default_rng(23)
    group = np.repeat([0, 1], 12)
    u = group.copy().astype(float)
    v = np.where(rng.random(24) < 0.8, group, 1 - group).astype(float)
    u[0] = np.nan
    u[7] = np.nan
    v[3] = np.nan
    data = pd.DataFrame({"u": u, "v": v})
    controls = dict(n_starts=4, screen_iter=0, n_full=2, max_iter=20, seed=23, n_threads=1)
    h0 = pf.fit_lca(data, ["u", "v"], n_states=1, **controls)
    h1 = pf.fit_lca(data, ["u", "v"], n_states=2, **controls)
    return h0, h1


def run_check(h0, h1, threads, batch_size=None, use_batch=False):
    kwargs = dict(B=8, max_retries=1, seed=19, max_iter=20, n_threads=threads, experimental=True)
    if use_batch:
        kwargs["batch_size"] = batch_size
    result = pysel.blrt_check(h0, h1, **kwargs)
    if not math.isfinite(result["lr_observed"]):
        raise SystemExit("observed likelihood ratio is not finite")
    if result["B"] != 8 or result.get("validated") is not False:
        raise SystemExit("B or validated changed")
    return statistical(result)


def _pool_key(h0, h1, threads, batch):
    pooled = pysel._blrt_draw_pool(
        h0, h1, 8, 1, 19, 20,
        max(int(h0.spec.get("n_starts", 2)), 2),
        max(int(h0.spec.get("n_full", 2)), 2),
        max(int(h1.spec.get("n_starts", 2)), 2),
        max(int(h1.spec.get("n_full", 2)), 2),
        threads,
        batch_size=batch,
    )
    return _bits({key: pooled[key] for key in POOL_FIELDS})


def _expect_pool_matches(h0, h1, reference):
    # reference is a blrt_check record. Compare the overlapping pool fields.
    expected = {key: reference[key] for key in POOL_FIELDS}
    for threads in (1, 2, 4):
        for batch in (1, 7, 8):
            got = _pool_key(h0, h1, threads, batch)
            if got != expected:
                print("FAIL historical pool mismatch", threads, batch, file=sys.stderr)
                return False
        # Default batch size is n_threads, passed explicitly here.
        got = _pool_key(h0, h1, threads, threads)
        if got != expected:
            print("FAIL default batch mismatch", threads, file=sys.stderr)
            return False
    return True


def _expect_check_matches(h0, h1, reference):
    # n_threads > 1 uses the pool. Default omits batch_size.
    for threads in (2, 4):
        if run_check(h0, h1, threads) != reference:
            print("FAIL default-batch blrt_check", threads, file=sys.stderr)
            return False
        for batch in (1, 7, 8):
            if run_check(h0, h1, threads, batch, use_batch=True) != reference:
                print("FAIL batched blrt_check", threads, batch, file=sys.stderr)
                return False
    return True


def _write_record(path, text):
    if path == "":
        return False
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        if handle.closed:
            print("FAIL record closed before flush", file=sys.stderr)
            return False
    with open(path, "r", encoding="utf-8") as handle:
        if handle.read() != text:
            print("FAIL record write", file=sys.stderr)
            return False
    return True


def main():
    record = os.environ.get("LTA_BLRT_RECORD")
    base = os.environ.get("LTA_BLRT_BASE")
    if record is not None and base is not None:
        print("FAIL set only one of LTA_BLRT_RECORD and LTA_BLRT_BASE", file=sys.stderr)
        if record == base:
            print("FAIL LTA_BLRT_RECORD and LTA_BLRT_BASE are the same path", file=sys.stderr)
        return 1
    if record is None and base is None:
        print("FAIL set LTA_BLRT_RECORD on a861687 or LTA_BLRT_BASE on this head", file=sys.stderr)
        return 1
    workloads = {
        "gaussian_threaded_pool": _gaussian_fits(),
        "categorical_missing_threaded_pool": _categorical_fits(),
    }
    commit = _require_record_identity() if record is not None else _require_compare_identity()
    # n_threads=2 and no batch_size: the historical full-vector pool on a861687.
    payload = {"commit": commit}
    for name, (h0, h1) in workloads.items():
        payload[name] = run_check(h0, h1, 2)
    if record is not None:
        text = json.dumps(payload, sort_keys=True)
        if not _write_record(record, text):
            return 1
        print("ok blrt base threaded-pool record")
        return 0
    with open(base, "r", encoding="utf-8") as handle:
        saved = json.loads(handle.read())
    if saved.get("commit") != "a861687943823dcaa39e4366421816536e072a6a":
        print("FAIL base record commit", file=sys.stderr)
        return 1
    for name, (h0, h1) in workloads.items():
        reference = saved.get(name)
        if reference is None:
            print("FAIL base record missing", name, file=sys.stderr)
            return 1
        if not _expect_check_matches(h0, h1, reference):
            return 1
        if not _expect_pool_matches(h0, h1, reference):
            return 1
    print("ok blrt batch equivalence source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
