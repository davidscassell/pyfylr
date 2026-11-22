# Not collected by pytest (the name does not start with test_).
# Not executed with the hot-loop and compact-output edit.
#
# Identity follows the loaded package. The record tree must be
# 5832c47550fc401a6b9a926982d2d1d987a99708 and its source hashes must be
# that commit's. The compare tree must contain
# af953ef27053f3b0a7aae2cf117a9eb4434c5345 and its source hashes must be
# that commit's bytes. A later edit of those files fails closed until the
# pins move with it. The record file does not carry a SHA.
# Record mode refuses unless every public fit field and a direct _core.fit
# are historical Python lists, so a stale array-returning extension cannot
# write the record. Compare mode requires those recorded types to be lists
# and the loaded _core.fit to return NumPy arrays.
# Matching paths do not prove a stale in-tree extension matches lta.cpp.
#
# Do not set LTA_COMPACT_RECORD and LTA_COMPACT_BASE together.
#
#   git worktree add /tmp/compact-base 5832c47550fc401a6b9a926982d2d1d987a99708
#   cp pyfylr/tests/unrun_compact_output_equivalence.py /tmp/compact-base/pyfylr/tests/
#   PYTHONPATH=/tmp/compact-base/pyfylr/src \
#     LTA_COMPACT_RECORD=/tmp/compact-base.rec \
#     python3 /tmp/compact-base/pyfylr/tests/unrun_compact_output_equivalence.py
#   LTA_COMPACT_BASE=/tmp/compact-base.rec \
#     python3 pyfylr/tests/unrun_compact_output_equivalence.py

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
import pyfylr.selection as pysel
import pyfylr.sklearn_api as pysk

FLOAT_KEYS = (
    "gamma", "filtered", "xi",
    "prevalence_model", "prevalence_posterior", "prevalence_hard",
    "expected_membership", "expected_transitions", "hard_transitions",
    "start_loglik", "rejected_loglik",
)
PARAM_KEYS = ("delta", "tau", "rho", "means", "cov")
RECORD_COMMIT = "5832c47550fc401a6b9a926982d2d1d987a99708"
# File bytes at RECORD_COMMIT, then the bytes at COMPARE_COMMIT. Not written into the record.
# COMPARE_COMMIT pins stay fail-closed: the compare tree must contain that commit
# and these files must still be those bytes.
RECORD_HASHES = {
    "__init__.py": "157ce56159df84b2bc5fcb9515ca2e4c18873f90ee965018b93ddb681ce63eaf",
    "selection.py": "74f891234831b10b6b2d1face8deac1f89666695a78f18b361e476228201b3e3",
    "sklearn_api.py": "3301ef12e38d80a468864583dabb21390e2b5e79912b1645242f5abd97793d63",
    "_core.cpp": "5cc5e6b47eeb1b67c5118eb0f48ca7b15ff5f827f89d0293eeadee1b006f8f30",
    "lta.cpp": "d7f6a1fb022f97583b95d839373f83519ce7d7d8ee00e8ba28f2da87d399534f",
}
COMPARE_COMMIT = "af953ef27053f3b0a7aae2cf117a9eb4434c5345"
HEAD_HASHES = {
    "__init__.py": "14e9ee0f5a407d0121a1a962014dcd101bcd3b36a6f52c2182c9f2cdb5743c85",
    "selection.py": "05d162e586b5d9b577ea4842f1e87360eea3b864d2ea659c8fc52c0cc9850ee6",
    "sklearn_api.py": "3cbf2e237492b53c3fee48d0e41c57fa88345d260798d3b077cf626c88d4118a",
    "_core.cpp": "2e6d9abf3e7360aa7c14d287c3c9084ceb15cf114f57db097b546d1ed607fefd",
    "lta.cpp": "c9cdff1bdbd41be39af2d0a6e6826d4c3a14c2de4520b2aae4f436c68a949a56",
}
SOURCES = {
    "__init__.py": ("pyfylr", "src", "pyfylr", "__init__.py"),
    "selection.py": ("pyfylr", "src", "pyfylr", "selection.py"),
    "sklearn_api.py": ("pyfylr", "src", "pyfylr", "sklearn_api.py"),
    "_core.cpp": ("pyfylr", "src", "pyfylr", "_core.cpp"),
    "lta.cpp": ("core", "src", "lta.cpp"),
}


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
    selection = Path(pysel.__file__).resolve()
    sklearn = Path(pysk.__file__).resolve()
    loaded = {
        "fit_lpa": Path(inspect.getfile(pf.fit_lpa)).resolve(),
        "fit_lca": Path(inspect.getfile(pf.fit_lca)).resolve(),
        "fit_lta": Path(inspect.getfile(pf.fit_lta)).resolve(),
        "blrt_check": Path(inspect.getfile(pysel.blrt_check)).resolve(),
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
    hashed_selection = root.joinpath(*SOURCES["selection.py"]).resolve()
    hashed_sklearn = root.joinpath(*SOURCES["sklearn_api.py"]).resolve()
    for label, path in (
        ("pyfylr.__file__", pkg),
        ("extension", extension),
        ("selection.py", selection),
        ("sklearn_api.py", sklearn),
        ("__init__.py", hashed_init),
    ):
        if not _inside(path, root):
            raise SystemExit("FAIL loaded " + label + " is outside the checked tree")
    if pkg != hashed_init or loaded["fit_lpa"] != hashed_init or loaded["fit_lca"] != hashed_init or loaded["fit_lta"] != hashed_init:
        raise SystemExit("FAIL loaded fit functions are not this tree's __init__.py")
    if selection != hashed_selection or loaded["blrt_check"] != hashed_selection:
        raise SystemExit("FAIL loaded selection functions are not this tree's selection.py")
    if sklearn != hashed_sklearn:
        raise SystemExit("FAIL loaded sklearn_api is not this tree's sklearn_api.py")
    return root


def _fingerprint(root):
    return {name: _sha256(root.joinpath(*parts)) for name, parts in SOURCES.items()}


def _require_record_identity():
    root = _repo_root()
    commit = _git(root, "rev-parse", "HEAD")
    fingerprint = _fingerprint(root)
    if commit != RECORD_COMMIT:
        raise SystemExit("FAIL record worktree HEAD is not 5832c47550fc401a6b9a926982d2d1d987a99708: " + commit)
    if fingerprint != RECORD_HASHES:
        raise SystemExit("FAIL record sources are not the 5832c47 files")
    return commit


def _require_compare_identity():
    root = _repo_root()
    commit = _git(root, "rev-parse", "HEAD")
    fingerprint = _fingerprint(root)
    if commit == RECORD_COMMIT:
        raise SystemExit("FAIL compare worktree is still 5832c47550fc401a6b9a926982d2d1d987a99708")
    if fingerprint == RECORD_HASHES:
        raise SystemExit("FAIL compare sources still match 5832c47")
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", COMPARE_COMMIT, "HEAD"],
        check=False, capture_output=True, text=True,
    )
    if ancestor.returncode != 0:
        raise SystemExit("FAIL compare commit does not contain " + COMPARE_COMMIT)
    if fingerprint != HEAD_HASHES:
        raise SystemExit("FAIL compare sources are not the af953ef bytes")
    harness = "pyfylr/tests/unrun_compact_output_equivalence.py"
    if _git(root, "cat-file", "-e", "HEAD:" + harness) != "":
        raise SystemExit("FAIL this harness is not in the compare commit")
    init = root.joinpath(*SOURCES["__init__.py"]).resolve()
    for name in ("as_lists", "match_result_form", "uses_compact_result"):
        fn = getattr(pf, name, None)
        if fn is None or Path(inspect.getfile(fn)).resolve() != init:
            raise SystemExit("FAIL loaded " + name + " is not this tree's __init__.py")
    return commit


def _bytes(value):
    # dtype "<f8" forces little-endian IEEE-754 bytes, including the sign of
    # zero and the NaN payload. It does not replace NaN with a token.
    array = np.asarray(value, dtype="<f8")
    if array.dtype != np.dtype("<f8"):
        raise SystemExit("value is not little-endian float64")
    return np.ascontiguousarray(array).tobytes()


def _pack(result):
    packed = {}
    for key in FLOAT_KEYS:
        packed[key] = _bytes(result[key]).hex()
    params = result["params"]
    for key in PARAM_KEYS:
        packed["params." + key] = _bytes(params[key]).hex()
    packed["loglik"] = _bytes([result["loglik"]]).hex()
    return packed


def _gaussian_frame():
    rng = np.random.default_rng(19)
    group = np.repeat([-1.5, 1.5], 12)
    return pd.DataFrame({
        "y1": group + rng.normal(0, 0.4, size=24),
        "y2": group + rng.normal(0, 0.4, size=24),
    })


def _categorical_frame():
    rng = np.random.default_rng(23)
    group = np.repeat([0, 1], 12)
    u = group.astype(float)
    v = np.where(rng.random(24) < 0.8, group, 1 - group).astype(float)
    u[0] = np.nan
    return pd.DataFrame({"u": u, "v": v})


def _gaussian_lta():
    rng = np.random.default_rng(29)
    n, t = 16, 2
    group = np.repeat(np.repeat([-1.5, 1.5], n // 2), t)
    y1 = group + rng.normal(0, 0.4, size=n * t)
    y2 = group + rng.normal(0, 0.4, size=n * t)
    y1[0] = np.nan
    y2[3] = np.nan
    y1[5] = np.nan
    return pd.DataFrame({
        "id": np.repeat(np.arange(n), t),
        "wave": np.tile([1, 2], n),
        "y1": y1,
        "y2": y2,
    })


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


def _fits(compact=None):
    # Omit compact on the record revision. That revision's fit functions
    # have no compact argument and would otherwise store it in the spec.
    extra = {} if compact is None else {"compact": compact}
    gaussian = dict(covariance="EEI", n_starts=4, screen_iter=0, n_full=2, max_iter=20, seed=19, n_threads=1)
    categorical = dict(n_starts=4, screen_iter=0, n_full=2, max_iter=20, seed=23, n_threads=1)
    return {
        "lpa-gaussian": pf.fit_lpa(
            _gaussian_frame(), ["y1", "y2"], n_states=2, **gaussian, **extra,
        ).result,
        "lca-categorical": pf.fit_lca(
            _categorical_frame(), ["u", "v"], n_states=2, **categorical, **extra,
        ).result,
        "lta-gaussian": pf.fit_lta(
            _gaussian_lta(), ["y1", "y2"], "id", "wave", n_states=2, **gaussian, **extra,
        ).result,
        "lta-categorical": pf.fit_lta(
            _categorical_lta(), ["u", "v"], "id", "wave", n_states=2, **categorical, **extra,
        ).result,
    }


def _require_exercise(name, result):
    if not name.startswith("lta-"):
        return
    if int(result["n_waves"]) < 2:
        raise SystemExit("FAIL " + name + " did not fit T >= 2")
    if len(result["xi"]) == 0:
        raise SystemExit("FAIL " + name + " produced an empty xi")
    if name == "lta-gaussian" and not np.isnan(_gaussian_lta()[["y1", "y2"]].to_numpy(dtype=float)).any():
        raise SystemExit("FAIL gaussian LTA fixture has no missing value")


def _require_arrays(result):
    for key in FLOAT_KEYS:
        value = result[key]
        if not isinstance(value, np.ndarray) or value.dtype != np.float64 or not value.flags["C_CONTIGUOUS"]:
            raise SystemExit("FAIL compact field is not a contiguous float64 array: " + key)
    for key in PARAM_KEYS:
        value = result["params"][key]
        if not isinstance(value, np.ndarray) or value.dtype != np.float64 or not value.flags["C_CONTIGUOUS"]:
            raise SystemExit("FAIL compact parameter is not a contiguous float64 array: " + key)


def _require_lists(result):
    for key in FLOAT_KEYS:
        if not isinstance(result[key], list):
            raise SystemExit("FAIL field is not a list: " + key)
    for key in PARAM_KEYS:
        if not isinstance(result["params"][key], list):
            raise SystemExit("FAIL parameter is not a list: " + key)


def _form_name(value):
    if type(value) is list:
        return "list"
    if isinstance(value, np.ndarray):
        return "ndarray"
    raise SystemExit("FAIL core field is neither a list nor an ndarray")


def _core_forms(result):
    forms = {key: _form_name(result[key]) for key in FLOAT_KEYS}
    params = result["params"]
    for key in PARAM_KEYS:
        forms["params." + key] = _form_name(params[key])
    return forms


def _historical_core_forms():
    forms = {key: "list" for key in FLOAT_KEYS}
    for key in PARAM_KEYS:
        forms["params." + key] = "list"
    return forms


def _raw_core_result():
    """Call the loaded extension directly, not the list-converting wrapper."""
    frame = _gaussian_frame().copy()
    frame["_id"] = np.arange(len(frame))
    spec = pf.lpa_spec(
        2, covariance="EEI", n_starts=4, screen_iter=0, n_full=2, max_iter=20, seed=19, n_threads=1,
    )
    panel = pf.prepare_panel(frame, ["y1", "y2"], "_id", None)
    return dict(pf._core.fit(panel, spec))


def main():
    record = os.environ.get("LTA_COMPACT_RECORD")
    base = os.environ.get("LTA_COMPACT_BASE")
    if record and base:
        raise SystemExit("FAIL set only one of LTA_COMPACT_RECORD and LTA_COMPACT_BASE")
    if not record and not base:
        raise SystemExit("FAIL set LTA_COMPACT_RECORD or LTA_COMPACT_BASE")
    if record:
        _require_record_identity()
        fits = _fits()
        for name, result in fits.items():
            _require_exercise(name, result)
            _require_lists(result)
        raw = _raw_core_result()
        _require_lists(raw)
        core_types = _core_forms(raw)
        if core_types != _historical_core_forms():
            raise SystemExit("FAIL raw _core.fit did not return historical lists")
        payload = {"fits": {name: _pack(result) for name, result in fits.items()}, "core_types": core_types}
        with open(record, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return
    _require_compare_identity()
    from pyfylr import as_lists

    with open(base, encoding="utf-8") as handle:
        recorded = json.load(handle)
    fits_recorded = recorded.get("fits") if isinstance(recorded, dict) else None
    if not isinstance(fits_recorded, dict) or set(fits_recorded) != {"lpa-gaussian", "lca-categorical", "lta-gaussian", "lta-categorical"}:
        raise SystemExit("FAIL record does not contain the four fixtures")
    if recorded.get("core_types") != _historical_core_forms():
        raise SystemExit("FAIL record core fields were not historical lists")
    raw = _raw_core_result()
    _require_arrays(raw)
    default = _fits()
    compact = _fits(compact=True)
    for name in fits_recorded:
        _require_exercise(name, default[name])
        _require_exercise(name, compact[name])
        _require_lists(default[name])
        _require_arrays(compact[name])
        listed = as_lists(compact[name])
        _require_lists(listed)
        packed_default = _pack(default[name])
        packed_compact = _pack(compact[name])
        packed_lists = _pack(listed)
        if packed_default != fits_recorded[name] or packed_compact != fits_recorded[name] or packed_lists != fits_recorded[name]:
            raise SystemExit("FAIL byte mismatch for " + name)


if __name__ == "__main__":
    sys.exit(main())
