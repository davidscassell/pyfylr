# Not collected by pytest (the name does not start with test_).
# Not executed with the streaming-E-step edit.
#
# Record the default stored fit on 2c53f102351ca57bcbbb3d58c75f3327ba3cf7be.
# Compare estep="streaming" on this PR, with posteriors requested and omitted.
# Identity follows the loaded package. The record file does not carry a SHA.
# Do not set LTA_STREAM_RECORD and LTA_STREAM_BASE together.
#
#   git worktree add /tmp/stream-base 2c53f102351ca57bcbbb3d58c75f3327ba3cf7be
#   cp pyfylr/tests/unrun_streaming_estep_equivalence.py /tmp/stream-base/pyfylr/tests/
#   PYTHONPATH=/tmp/stream-base/pyfylr/src \
#     LTA_STREAM_RECORD=/tmp/stream-base.rec \
#     python3 /tmp/stream-base/pyfylr/tests/unrun_streaming_estep_equivalence.py
#   LTA_STREAM_BASE=/tmp/stream-base.rec \
#     python3 pyfylr/tests/unrun_streaming_estep_equivalence.py

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

RECORD_COMMIT = "2c53f102351ca57bcbbb3d58c75f3327ba3cf7be"
RECORD_HASHES = {
    "lta.cpp": "068e877b99bf3ef24465eb00c0c9f2ecc3f525227f276dd6aa522797e2821e4f",
    "lta.hpp": "1ba2eada0e2f75d3e731a99b1ab3d07ffa6ac06874d05737a9637b6b284f4f73",
    "_core.cpp": "bf9ab9fda32cdd323cc927f99cceb0bd07410b7b567cc009bb24a1aa4120946a",
    "__init__.py": "4f45bfb46f51b9766d68b96d2580d503cdf45a2fb7a2155e704c7a0df5e79157",
    "selection.py": "384c23df31dc039a5d32844caa913c87f6cc568a5371e2ee0e5c6e9c528b67ff",
}
HEAD_HASHES = {
    "lta.cpp": "c9cdff1bdbd41be39af2d0a6e6826d4c3a14c2de4520b2aae4f436c68a949a56",
    "lta.hpp": "1ebe8680c1e1949a9dc0a828fcb571764d453256858cefeec65d7ad21e5034ca",
    "_core.cpp": "2e6d9abf3e7360aa7c14d287c3c9084ceb15cf114f57db097b546d1ed607fefd",
    "__init__.py": "14e9ee0f5a407d0121a1a962014dcd101bcd3b36a6f52c2182c9f2cdb5743c85",
    "selection.py": "05d162e586b5d9b577ea4842f1e87360eea3b864d2ea659c8fc52c0cc9850ee6",
}
SOURCES = {
    "lta.cpp": ("core", "src", "lta.cpp"),
    "lta.hpp": ("core", "include", "lta", "lta.hpp"),
    "_core.cpp": ("pyfylr", "src", "pyfylr", "_core.cpp"),
    "__init__.py": ("pyfylr", "src", "pyfylr", "__init__.py"),
    "selection.py": ("pyfylr", "src", "pyfylr", "selection.py"),
}
PARAM_KEYS = ("delta", "tau", "rho", "means", "cov")
POSTERIOR_KEYS = ("gamma", "filtered", "xi")


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
    loaded = {
        "fit_lca": Path(inspect.getfile(pf.fit_lca)).resolve(),
        "fit_lta": Path(inspect.getfile(pf.fit_lta)).resolve(),
    }
    root = None
    for parent in pkg.parents:
        init = parent.joinpath(*SOURCES["__init__.py"])
        core = parent.joinpath(*SOURCES["lta.cpp"])
        if init.is_file() and core.is_file() and (parent / ".git").exists():
            root = parent
            break
    if root is None:
        raise SystemExit("FAIL loaded pyfylr.__file__ is not inside a source tree")
    hashed_init = root.joinpath(*SOURCES["__init__.py"]).resolve()
    for label, path in (("pyfylr.__file__", pkg), ("extension", extension), ("__init__.py", hashed_init)):
        if not _inside(path, root):
            raise SystemExit("FAIL loaded " + label + " is outside the checked tree")
    if pkg != hashed_init or loaded["fit_lca"] != hashed_init or loaded["fit_lta"] != hashed_init:
        raise SystemExit("FAIL loaded fit functions are not this tree's __init__.py")
    return root


def _fingerprint(root):
    return {name: _sha256(root.joinpath(*parts)) for name, parts in SOURCES.items()}


def _require_record_identity():
    root = _repo_root()
    commit = _git(root, "rev-parse", "HEAD")
    if commit != RECORD_COMMIT or _fingerprint(root) != RECORD_HASHES:
        raise SystemExit("FAIL record tree is not 2c53f102351ca57bcbbb3d58c75f3327ba3cf7be with its sources")
    return commit


def _require_compare_identity():
    root = _repo_root()
    commit = _git(root, "rev-parse", "HEAD")
    found = _fingerprint(root)
    if commit == RECORD_COMMIT or found == RECORD_HASHES:
        raise SystemExit("FAIL compare tree is still the record revision")
    if found != HEAD_HASHES:
        raise SystemExit("FAIL compare sources are not this PR's copies")
    if _git(root, "cat-file", "-e", "HEAD:pyfylr/tests/unrun_streaming_estep_equivalence.py") != "":
        raise SystemExit("FAIL this harness is not in the compare commit")
    return commit


def _list_shape(value):
    if not isinstance(value, list):
        raise SystemExit("FAIL nested field is not a list")
    if not value or not isinstance(value[0], list):
        return [len(value)]
    inner = _list_shape(value[0])
    for item in value[1:]:
        if _list_shape(item) != inner:
            raise SystemExit("FAIL ragged list")
    return [len(value)] + inner


def _bytes(value):
    # A list records its full nested shape, not only the outer length.
    # An array keeps its shape and must already be float64. Bytes are
    # little-endian either way.
    if isinstance(value, list):
        kind = "list"
        shape = _list_shape(value)
        array = np.asarray(value, dtype="<f8")
    elif isinstance(value, np.ndarray):
        if value.dtype != np.float64:
            raise SystemExit("FAIL array is not float64")
        kind = "ndarray"
        shape = list(value.shape)
        array = np.asarray(value, dtype="<f8")
    elif isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(value, bool):
        kind = "scalar"
        shape = [1]
        array = np.asarray([value], dtype="<f8")
    else:
        raise SystemExit("FAIL field is not a list, float64 array, or scalar")
    if array.dtype != np.dtype("<f8"):
        raise SystemExit("FAIL value is not little-endian float64")
    return {"kind": kind, "shape": shape, "bytes": np.ascontiguousarray(array).tobytes().hex()}


def _pack(result, posteriors=True):
    packed = {"loglik": _bytes(result["loglik"]), "n_iter": int(result["n_iter"])}
    params = result["params"]
    for key in PARAM_KEYS:
        packed["params." + key] = _bytes(params[key])
    if posteriors:
        for key in POSTERIOR_KEYS:
            packed[key] = _bytes(result[key])
    return packed


def _categorical_lca():
    rng = np.random.default_rng(23)
    group = np.repeat([0, 1], 12)
    u = group.astype(float)
    v = np.where(rng.random(24) < 0.8, group, 1 - group).astype(float)
    u[0] = np.nan
    return pd.DataFrame({"u": u, "v": v})


def _categorical_lta():
    rng = np.random.default_rng(31)
    n, t = 16, 2
    group = np.repeat(np.repeat([0, 1], n // 2), t)
    u = group.astype(float)
    v = np.where(rng.random(n * t) < 0.8, group, 1 - group).astype(float)
    u[1] = np.nan
    v[4] = np.nan
    return pd.DataFrame({
        "id": np.repeat(np.arange(n), t),
        "wave": np.tile([1, 2], n),
        "u": u,
        "v": v,
    })


def _fits(**extra):
    controls = dict(n_starts=4, screen_iter=0, n_full=2, max_iter=20, seed=23, n_threads=1)
    controls.update(extra)
    lca = pf.fit_lca(_categorical_lca(), ["u", "v"], n_states=2, **controls)
    lta = pf.fit_lta(_categorical_lta(), ["u", "v"], "id", "wave", n_states=2, **controls)
    if int(lta.result["n_waves"]) < 2:
        raise SystemExit("FAIL categorical LTA fixture is not two waves")
    if extra.get("posteriors", True) and len(lta.result["xi"]) == 0:
        raise SystemExit("FAIL two-wave categorical fixture did not produce xi")
    if not np.isnan(_categorical_lta()[["u", "v"]].to_numpy(dtype=float)).any():
        raise SystemExit("FAIL categorical LTA fixture has no missing code")
    return {"lca": lca, "lta": lta}


def _require_omitted(fit):
    result = fit.result
    if result.get("posteriors_computed") is not False:
        raise SystemExit("FAIL posteriors_computed is not false")
    for key in POSTERIOR_KEYS:
        value = result[key]
        if not isinstance(value, list) or len(value) != 0:
            raise SystemExit("FAIL omitted posterior is not an empty list: " + key)
    message = "posteriors were not computed; refit with posteriors=True"
    calls = (
        fit.augment,
        lambda: pf.posterior_states(fit),
        lambda: pf.posterior_transitions(fit),
        lambda: pf.state_prevalence(fit, "posterior"),
        lambda: pf.state_prevalence(fit, "hard"),
        lambda: pf.expected_counts(fit),
        lambda: pf.plot_prevalence(fit, kind="posterior"),
        lambda: pf.plot_prevalence(fit, kind="hard"),
        lambda: pf.plot_avepp(fit),
        lambda: pf.plot_max_posterior(fit),
    )
    for call in calls:
        try:
            call()
        except ValueError as exc:
            if message not in str(exc):
                raise SystemExit("FAIL accessor error text") from exc
            continue
        raise SystemExit("FAIL accessor did not fail closed")
    pf.state_prevalence(fit, "model")
    pf.plot_prevalence(fit, kind="model")


def _missing_state(column):
    """Historical frames store an excluded code as None, which pandas may keep
    as None or as NaN. Either is missing; a literal None identity is not."""
    if column.dtype != object and not pd.api.types.is_float_dtype(column.dtype):
        raise SystemExit("FAIL excluded viterbi dtype is not a missing-value dtype")
    if not pd.isna(column.iloc[0]):
        raise SystemExit("FAIL excluded viterbi code is not missing")


def _check_viterbi(fit):
    from dataclasses import replace

    marked = dict(fit.result)
    codes = list(marked["viterbi"])
    codes[0] = -1
    marked["viterbi"] = codes
    marked_fit = replace(fit, result=marked)
    aug = marked_fit.augment()
    if "smoothed_1" not in list(aug.columns) or "filtered_1" not in list(aug.columns):
        raise SystemExit("FAIL augment did not produce posterior columns")
    present = pf.decode_states(marked_fit, "viterbi")
    expected = aug[["id", "wave", "viterbi_state"]]
    if list(present.columns) != ["id", "wave", "viterbi_state"]:
        raise SystemExit("FAIL viterbi columns")
    if not present.equals(expected):
        raise SystemExit("FAIL viterbi decode is not the augment column subset")
    _missing_state(present["viterbi_state"])
    smoothed = pf.decode_states(marked_fit, "smoothed")
    if not smoothed.equals(aug):
        raise SystemExit("FAIL smoothed decode is not augment()")
    bare = dict(marked)
    bare.pop("posteriors_computed", None)
    routed = pf.decode_states(replace(fit, result=bare), "viterbi")
    if not present.equals(routed):
        raise SystemExit("FAIL absent posteriors_computed did not use the stored decode")
    omitted = dict(marked)
    omitted["posteriors_computed"] = False
    decoded = pf.decode_states(replace(fit, result=omitted), "viterbi")
    if list(decoded.columns) != ["id", "wave", "viterbi_state"]:
        raise SystemExit("FAIL omitted viterbi columns")
    _missing_state(decoded["viterbi_state"])


def main():
    record = os.environ.get("LTA_STREAM_RECORD")
    base = os.environ.get("LTA_STREAM_BASE")
    if bool(record) == bool(base):
        raise SystemExit("FAIL set exactly one of LTA_STREAM_RECORD and LTA_STREAM_BASE")
    if record:
        _require_record_identity()
        fits = _fits()
        payload = {name: _pack(fit.result) for name, fit in fits.items()}
        with open(record, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return
    _require_compare_identity()
    with open(base, encoding="utf-8") as handle:
        recorded = json.load(handle)
    if not isinstance(recorded, dict) or set(recorded) != {"lca", "lta"}:
        raise SystemExit("FAIL record does not contain the categorical fixtures")
    streamed = _fits(estep="streaming", posteriors=True)
    omitted = _fits(estep="streaming", posteriors=False)
    for name in recorded:
        if streamed[name].result.get("posteriors_computed") is not True:
            raise SystemExit("FAIL streaming posterior fit did not set posteriors_computed")
        _check_viterbi(streamed[name])
        if _pack(streamed[name].result) != recorded[name]:
            raise SystemExit("FAIL streaming posterior fit does not match stored " + name)
        kept = {key: recorded[name][key] for key in recorded[name] if key not in POSTERIOR_KEYS}
        if _pack(omitted[name].result, posteriors=False) != kept:
            raise SystemExit("FAIL streaming fit without posteriors does not match stored parameters " + name)
        _require_omitted(omitted[name])


if __name__ == "__main__":
    sys.exit(main())
