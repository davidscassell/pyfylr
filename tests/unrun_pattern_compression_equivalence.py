# Not collected by pytest (the name does not start with test_).
# Not executed with the pattern-compression edit.
#
# Record uncompressed fits on c8feb1bd7bb3e6710933bc8a4b58e9c9ee7c2635.
# patterns="none" must match that record bitwise. patterns="compress"
# must lie within the non-finite-then-1e-8 rule. Threads 2, 4, and 8 in
# compressed mode must match each other bitwise.
#
#   git worktree add /tmp/pat-base c8feb1bd7bb3e6710933bc8a4b58e9c9ee7c2635
#   cp pyfylr/tests/unrun_pattern_compression_equivalence.py /tmp/pat-base/pyfylr/tests/
#   PYTHONPATH=/tmp/pat-base/pyfylr/src \
#     LTA_PATTERN_RECORD=/tmp/pat-base.rec \
#     python3 /tmp/pat-base/pyfylr/tests/unrun_pattern_compression_equivalence.py
#   LTA_PATTERN_BASE=/tmp/pat-base.rec \
#     python3 pyfylr/tests/unrun_pattern_compression_equivalence.py

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

RECORD_COMMIT = "c8feb1bd7bb3e6710933bc8a4b58e9c9ee7c2635"
RECORD_HASHES = {
    "lta.cpp": "70b3f8d9c5cb1299184194d3d64b2dd70d72ca163b63e772530ad01d8f363320",
    "lta.hpp": "fa6565abdbc992afb0a71da65da4ee1399c9d2e1809277c4b5aeaa8694dcc9f3",
    "_core.cpp": "2f66181151fc57169745db437553bafdb31779d0b3e369f7e11e7aa65b7eb91b",
    "__init__.py": "9e7a9f20169e84c8471dca990475e70459945cc082f94ca7e8d3c898e4291569",
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
    if pkg != init or Path(inspect.getfile(pf.fit_lca)).resolve() != init:
        raise SystemExit("FAIL loaded fit_lca is not this tree's __init__.py")
    return root


def _fingerprint(root):
    return {name: _sha256(root.joinpath(*parts)) for name, parts in SOURCES.items()}


def _require_record_identity():
    root = _repo_root()
    if _git(root, "rev-parse", "HEAD") != RECORD_COMMIT or _fingerprint(root) != RECORD_HASHES:
        raise SystemExit("FAIL record tree is not c8feb1b with its sources")


def _require_compare_identity():
    root = _repo_root()
    found = _fingerprint(root)
    if _git(root, "rev-parse", "HEAD") == RECORD_COMMIT or found == RECORD_HASHES or found != HEAD_HASHES:
        raise SystemExit("FAIL compare sources are not this PR's copies")
    if _git(root, "cat-file", "-e", "HEAD:pyfylr/tests/unrun_pattern_compression_equivalence.py") != "":
        raise SystemExit("FAIL this harness is not in the compare commit")


def _bytes(value):
    return np.ascontiguousarray(np.asarray(value, dtype="<f8")).tobytes().hex()


def _pack(result, posteriors=True):
    packed = {"loglik": _bytes([result["loglik"]]), "n_iter": int(result["n_iter"])}
    for key in ("delta", "tau", "rho", "means", "cov"):
        packed["params." + key] = _bytes(result["params"][key])
    if posteriors:
        for key in ("gamma", "filtered", "xi"):
            packed[key] = _bytes(result[key])
    return packed


def _close_array(left, right, name):
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise SystemExit("FAIL shape " + name)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    pos_a, pos_b = np.isposinf(a), np.isposinf(b)
    neg_a, neg_b = np.isneginf(a), np.isneginf(b)
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


def _close(serial, parallel, posteriors=True):
    if int(serial["n_iter"]) != int(parallel["n_iter"]):
        raise SystemExit("FAIL iteration count changed")
    _close_array([serial["loglik"]], [parallel["loglik"]], "loglik")
    for key in ("delta", "tau", "rho", "means", "cov"):
        _close_array(serial["params"][key], parallel["params"][key], key)
    if posteriors:
        for key in ("gamma", "filtered", "xi"):
            _close_array(serial[key], parallel[key], key)


def _frames():
    rng = np.random.default_rng(11)
    base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    pick = np.resize(np.arange(4), 80)
    dup = base[pick].copy()
    dup[0, 0] = np.nan
    dup[3, 1] = np.nan
    dup = np.vstack([dup, [np.nan, np.nan]])
    ids = np.repeat(np.arange(40), 2)
    waves = np.tile([1, 2], 40)
    group = np.resize(np.arange(4), 40)
    uu = base[group, 0].repeat(2).astype(float)
    vv = base[group, 1].repeat(2).astype(float)
    uu[1] = np.nan
    lta = pd.DataFrame({"id": np.append(ids, [40, 40]), "wave": np.append(waves, [1, 2]),
                        "u": np.append(uu, [np.nan, np.nan]), "v": np.append(vv, [np.nan, np.nan])})
    wide_n = 513
    bits = ((np.arange(wide_n)[:, None] >> np.arange(10)) & 1).astype(float)
    bits[0, 0] = np.nan
    gauss = pd.DataFrame({"y": np.linspace(-1.0, 1.0, 12)})
    mixed = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "wave": [1, 2, 1, 2],
        "u": [0.0, 1.0, 0.0, 1.0],
        "y": [0.2, -0.4, 1.5, 0.7],
    })
    return dup, bits, gauss, lta, mixed


def _fit_dup(extra=None, posteriors=True, estep="stored"):
    extra = {} if extra is None else dict(extra)
    dup, _, _, _, _ = _frames()
    frame = pd.DataFrame(dup, columns=["u", "v"])
    controls = dict(n_starts=2, screen_iter=0, n_full=2, max_iter=12, seed=11, n_threads=1, posteriors=posteriors)
    controls.update(extra)
    return pf.fit_lca(frame, ["u", "v"], n_states=2, estep=estep, **controls).result


def _fit_wide(extra=None, estep="stored"):
    extra = {} if extra is None else dict(extra)
    _, bits, _, _, _ = _frames()
    frame = pd.DataFrame(bits, columns=["b" + str(j) for j in range(10)])
    controls = dict(n_starts=1, screen_iter=0, n_full=1, max_iter=4, seed=3, n_threads=1)
    controls.update(extra)
    return pf.fit_lca(frame, list(frame.columns), n_states=2, estep=estep, **controls).result


def _codes(left, right, name):
    if len(left) != len(right):
        raise SystemExit("FAIL " + name + " length")
    for a, b in zip(left, right):
        if int(a) != int(b):
            raise SystemExit("FAIL " + name + " code")


def _blrt_frame():
    # Two repeated patterns, far apart, plus two crossovers and one missing cell.
    # The observed 1-class and 2-class fits have to converge and stay admissible,
    # so the bootstrap runs. Counts above 1 exercise compression on that input.
    # B stays at 2, with one retry so a single draw can recover.
    rows = [(0.0, 0.0)] * 16 + [(1.0, 1.0)] * 16 + [(0.0, 1.0), (1.0, 0.0), (0.0, np.nan)]
    return pd.DataFrame(rows, columns=["u", "v"])


def _blrt(mode):
    frame = _blrt_frame()
    kw = dict(n_starts=4, n_full=2, screen_iter=0, max_iter=40, seed=23, n_threads=1, patterns=mode)
    h0 = pf.fit_lca(frame, ["u", "v"], n_states=1, **kw)
    h1 = pf.fit_lca(frame, ["u", "v"], n_states=2, **kw)
    return pf.blrt_check(h0, h1, B=2, max_retries=1, n_threads=2, experimental=True, max_iter=40, seed=23)


def _fit_lta(extra=None):
    extra = {} if extra is None else dict(extra)
    _, _, _, lta, _ = _frames()
    controls = dict(n_starts=2, screen_iter=0, n_full=2, max_iter=8, seed=5, n_threads=1)
    controls.update(extra)
    return pf.fit_lta(lta, ["u", "v"], "id", "wave", n_states=2, **controls).result


def _repeated_frame(waves):
    """257 unique patterns, each repeated 2, 3, or 4 times, with some missingness.
    257 is 8*32+1, so the pattern list has 9 blocks of 32, which is more than 8 threads.
    """
    rows = []
    subject = 0
    for pid in range(257):
        bits = [(pid >> bit) & 1 for bit in range(9)]
        missing = pid % 19 == 0
        reps = 2 + (pid % 3)
        for _rep in range(reps):
            if waves == 1:
                vals = [float(bit) for bit in bits]
                if missing:
                    vals[0] = np.nan
                rows.append(vals)
            else:
                wave1 = [float(bit) for bit in bits[:5]]
                wave2 = [float(bit) for bit in bits[5:]] + [0.0]
                if missing:
                    wave1[0] = np.nan
                    wave2[-1] = np.nan
                rows.append([subject, 1] + wave1)
                rows.append([subject, 2] + wave2)
            subject += 1
    if waves == 1:
        return pd.DataFrame(rows, columns=["c" + str(i) for i in range(9)])
    return pd.DataFrame(rows, columns=["id", "wave"] + ["c" + str(i) for i in range(5)])


def _fit_repeated(kind, extra=None, estep="stored", posteriors=True):
    extra = {} if extra is None else dict(extra)
    controls = dict(n_starts=1, screen_iter=0, n_full=1, max_iter=3, seed=7, n_threads=1, posteriors=posteriors)
    controls.update(extra)
    if kind == "lca":
        frame = _repeated_frame(1)
        return pf.fit_lca(frame, list(frame.columns), n_states=2, estep=estep, **controls).result
    frame = _repeated_frame(2)
    indicators = ["c" + str(i) for i in range(5)]
    if kind == "lca-wave":
        return pf.fit_lca(frame, indicators, "id", "wave", n_states=2, estep=estep, **controls).result
    return pf.fit_lta(frame, indicators, "id", "wave", n_states=2, estep=estep, **controls).result


def _require_repeated(result, threads):
    unique = int(result["n_patterns"])
    analyzed = int(result["n_analyzed"])
    if unique <= 32 or analyzed <= unique:
        raise SystemExit("FAIL repeated fixture does not have U>32 and counts>1")
    blocks = (unique + 31) // 32
    if blocks <= threads:
        raise SystemExit("FAIL pattern blocks are not greater than estep_threads")


def _cover_blocked():
    # Stored recount is one pattern-order fold, so every thread count matches bitwise.
    plain = _fit_repeated("lca")
    serial = _fit_repeated("lca", {"patterns": "compress", "estep_threads": 1})
    _require_repeated(serial, 8)
    _close(plain, serial)
    stored = {threads: _fit_repeated("lca", {"patterns": "compress", "estep_threads": threads}) for threads in (2, 4, 8)}
    for threads, result in stored.items():
        _require_repeated(result, threads)
        if _pack(result) != _pack(serial):
            raise SystemExit("FAIL stored compress threads are not bitwise-equal " + str(threads))
        _close(plain, result)
    # Omitted streaming: threads above 1 share the block fold; serial is only a tolerance match.
    plain_omit = _fit_repeated("lca", estep="streaming", posteriors=False)
    serial_omit = _fit_repeated("lca", {"patterns": "compress", "estep_threads": 1}, estep="streaming", posteriors=False)
    _require_repeated(serial_omit, 8)
    _close(plain_omit, serial_omit, posteriors=False)
    omitted = {
        threads: _fit_repeated("lca", {"patterns": "compress", "estep_threads": threads}, estep="streaming", posteriors=False)
        for threads in (2, 4, 8)
    }
    for threads, result in omitted.items():
        _require_repeated(result, threads)
        for key in ("gamma", "filtered", "xi"):
            if len(result[key]) != 0:
                raise SystemExit("FAIL repeated omitted posterior is not empty")
        _close(serial_omit, result, posteriors=False)
        _close(plain_omit, result, posteriors=False)
    for threads in (4, 8):
        if _pack(omitted[threads], posteriors=False) != _pack(omitted[2], posteriors=False):
            raise SystemExit("FAIL omitted streaming threads disagree " + str(threads))
    # Multi-wave streaming, LCA and LTA, with more blocks than 8 threads.
    for kind in ("lca-wave", "lta"):
        plain_wave = _fit_repeated(kind, estep="streaming")
        serial_wave = _fit_repeated(kind, {"patterns": "compress", "estep_threads": 1}, estep="streaming")
        if int(serial_wave["n_waves"]) < 2:
            raise SystemExit("FAIL repeated multi-wave fixture")
        _require_repeated(serial_wave, 8)
        _close(plain_wave, serial_wave)
        threaded = {
            threads: _fit_repeated(kind, {"patterns": "compress", "estep_threads": threads}, estep="streaming")
            for threads in (2, 4, 8)
        }
        for threads, result in threaded.items():
            _require_repeated(result, threads)
            _close(serial_wave, result)
            _close(plain_wave, result)
        for threads in (4, 8):
            if _pack(threaded[threads]) != _pack(threaded[2]):
                raise SystemExit("FAIL multi-wave streaming threads disagree " + kind + " " + str(threads))


def _compare_blrt(left, right):
    for key in ("b", "f", "B", "n_ok", "ran", "lr_observed", "p"):
        if key not in left or key not in right:
            raise SystemExit("FAIL BLRT missing " + key)
    for name, result in (("compress", left), ("none", right)):
        if bool(result["ran"]) is not True:
            raise SystemExit("FAIL BLRT did not run " + name)
        if int(result["n_ok"]) < 1:
            raise SystemExit("FAIL BLRT kept no draws " + name)
        if not np.isfinite(float(result["lr_observed"])) or not np.isfinite(float(result["p"])):
            raise SystemExit("FAIL BLRT skip payload " + name)
    for key in ("b", "f", "B", "n_ok", "ran"):
        if left[key] != right[key]:
            raise SystemExit("FAIL BLRT field " + key)
    _close_array([left["lr_observed"]], [right["lr_observed"]], "lr_observed")
    _close_array([left["p"]], [right["p"]], "blrt p")


def main():
    record = os.environ.get("LTA_PATTERN_RECORD")
    base = os.environ.get("LTA_PATTERN_BASE")
    if bool(record) == bool(base):
        raise SystemExit("FAIL set exactly one of LTA_PATTERN_RECORD and LTA_PATTERN_BASE")
    if record:
        _require_record_identity()
        payload = {
            "dup": _pack(_fit_dup()),
            "dup-stream": _pack(_fit_dup(estep="streaming")),
            "wide": _pack(_fit_wide()),
            "lta": _pack(_fit_lta()),
        }
        with open(record, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
        return
    _require_compare_identity()
    with open(base, encoding="utf-8") as handle:
        recorded = json.load(handle)
    plain = {
        "dup": _fit_dup({"patterns": "none"}),
        "dup-stream": _fit_dup({"patterns": "none"}, estep="streaming"),
        "wide": _fit_wide({"patterns": "none"}),
        "lta": _fit_lta({"patterns": "none"}),
    }
    bare = _fit_dup()
    if "patterns" in bare or "n_patterns" in bare or "pattern_id" in bare:
        raise SystemExit("FAIL default fit exported a patterns key")
    for name, result in plain.items():
        if _pack(result) != recorded[name]:
            raise SystemExit("FAIL patterns=none does not match the record " + name)
        if "patterns" in result or "n_patterns" in result or "pattern_id" in result:
            raise SystemExit("FAIL default fit exported pattern metadata")
    compressed = _fit_dup({"patterns": "compress"})
    if int(compressed["n_patterns"]) >= int(compressed["n_analyzed"]):
        raise SystemExit("FAIL duplicate fixture did not collapse")
    if len(compressed["pattern_id"]) != int(compressed["n_subjects"]):
        raise SystemExit("FAIL pattern_id length")
    lta_compressed = _fit_lta({"patterns": "compress"})
    _close(plain["lta"], lta_compressed)
    _codes(plain["lta"]["viterbi"], lta_compressed["viterbi"], "lta viterbi")
    _codes(plain["lta"]["modal"], lta_compressed["modal"], "lta modal")
    _codes(plain["dup"]["viterbi"], compressed["viterbi"], "viterbi")
    _codes(plain["dup"]["modal"], compressed["modal"], "modal")
    _close(plain["dup"], compressed)
    if int(plain["lta"]["n_waves"]) < 2:
        raise SystemExit("FAIL multi-wave fixture")
    pid = list(compressed["pattern_id"])
    if not any(int(p) < 0 for p in pid):
        raise SystemExit("FAIL excluded subject is not pattern -1")
    width = int(compressed["n_waves"])
    vit = list(compressed["viterbi"])
    modal = list(compressed["modal"])
    if len(vit) != int(compressed["n_subjects"]) * width or len(modal) != len(vit):
        raise SystemExit("FAIL expanded state length")
    seen = {}
    for i, pattern in enumerate(pid):
        if int(pattern) < 0:
            continue
        sl = tuple(vit[i * width:(i + 1) * width])
        md = tuple(modal[i * width:(i + 1) * width])
        if pattern in seen and seen[pattern] != (sl, md):
            raise SystemExit("FAIL pattern did not expand to the same modal and Viterbi")
        seen[pattern] = (sl, md)
    _close(plain["dup"], compressed)
    streamed = _fit_dup({"patterns": "compress"}, estep="streaming")
    _close(plain["dup"], streamed)
    omitted = _fit_dup({"patterns": "compress"}, posteriors=False, estep="streaming")
    if omitted.get("posteriors_computed") is not False:
        raise SystemExit("FAIL omitted streaming posterior flag")
    for key in ("gamma", "filtered", "xi"):
        if len(omitted[key]) != 0:
            raise SystemExit("FAIL omitted posterior is not empty: " + key)
    if omitted.get("patterns") != "compress":
        raise SystemExit("FAIL compressed fit omitted the patterns key")
    pooled = _fit_dup({"patterns": "compress", "n_threads": 2, "estep_threads": 8})
    if _pack(pooled) != _pack(compressed):
        raise SystemExit("FAIL start pool changed the compressed fit")
    dup_threads = {threads: _fit_dup({"patterns": "compress", "estep_threads": threads}) for threads in (2, 4, 8)}
    _close(plain["dup"], dup_threads[2])
    for threads in (4, 8):
        if _pack(dup_threads[threads]) != _pack(dup_threads[2]):
            raise SystemExit("FAIL threaded compress with repeated patterns disagrees " + str(threads))
    lta_threads = {threads: _fit_lta({"patterns": "compress", "estep_threads": threads}) for threads in (2, 4, 8)}
    _close(plain["lta"], lta_threads[2])
    for threads in (4, 8):
        if _pack(lta_threads[threads]) != _pack(lta_threads[2]):
            raise SystemExit("FAIL multi-wave threaded compress disagrees " + str(threads))
    omit_threads = {
        threads: _fit_dup({"patterns": "compress", "estep_threads": threads}, posteriors=False, estep="streaming")
        for threads in (2, 4, 8)
    }
    for result in omit_threads.values():
        for key in ("gamma", "filtered", "xi"):
            if len(result[key]) != 0:
                raise SystemExit("FAIL threaded omitted posterior is not empty")
    _close(plain["dup"], omit_threads[2], posteriors=False)
    for threads in (4, 8):
        if _pack(omit_threads[threads], posteriors=False) != _pack(omit_threads[2], posteriors=False):
            raise SystemExit("FAIL threaded omitted streaming disagrees " + str(threads))
    blrt_compress = _blrt("compress")
    blrt_plain = _blrt("none")
    _compare_blrt(blrt_compress, blrt_plain)
    _cover_blocked()
    stream_threads = {
        threads: _fit_wide({"patterns": "compress", "estep_threads": threads}, estep="streaming")
        for threads in (2, 4, 8)
    }
    _close(plain["wide"], stream_threads[2])
    for threads in (4, 8):
        if _pack(stream_threads[threads]) != _pack(stream_threads[2]):
            raise SystemExit("FAIL threaded streaming compress disagrees " + str(threads))
    _close(plain["dup"], omitted, posteriors=False)
    wide = {threads: _fit_wide({"patterns": "compress", "estep_threads": threads}) for threads in (1, 2, 4, 8)}
    if int(wide[1]["n_patterns"]) < 513 or int(wide[1]["n_patterns"]) % 32 == 0:
        raise SystemExit("FAIL wide fixture does not cross distinct pattern blocks")
    nblocks = (int(wide[1]["n_patterns"]) + 31) // 32

    def assignment(nthreads):
        spans = []
        cursor = 0
        left_blocks = nblocks
        left_threads = min(nthreads, nblocks)
        for _ in range(left_threads):
            take = left_blocks // left_threads
            spans.append((cursor, cursor + take))
            cursor += take
            left_blocks -= take
            left_threads -= 1
        return tuple(spans)

    if len({assignment(2), assignment(4), assignment(8)}) != 3:
        raise SystemExit("FAIL 2/4/8 threads share a pattern-block assignment")
    _close(plain["wide"], wide[1])
    for threads in (2, 4, 8):
        if _pack(wide[threads]) != _pack(wide[2]):
            raise SystemExit("FAIL compressed thread counts disagree " + str(threads))
    _, _, gauss, _, mixed = _frames()
    try:
        pf.fit_lpa(gauss, ["y"], n_states=2, covariance="EEI", patterns="compress", n_starts=1, max_iter=2, screen_iter=0)
    except ValueError as exc:
        if "categorical models only" not in str(exc):
            raise SystemExit("FAIL gaussian refusal text") from exc
    else:
        raise SystemExit("FAIL gaussian compress was accepted")
    try:
        pf.fit_lta(mixed, ["u", "y"], "id", "wave", n_states=2, patterns="compress",
                   n_starts=1, max_iter=2, screen_iter=0)
    except ValueError as exc:
        if "categorical models only" not in str(exc):
            raise SystemExit("FAIL mixed refusal text") from exc
    else:
        raise SystemExit("FAIL mixed compress was accepted")


if __name__ == "__main__":
    raise SystemExit(main())
