import math
import os
import shutil
import subprocess
import textwrap

import numpy as np
import pandas as pd
import pytest

import pyfylr as pf


def test_fewer_analyzed_subjects_than_states_is_refused():
    one = pd.DataFrame({"y": [0.2]})
    with pytest.raises(ValueError, match="too few analyzed subjects to identify 2 states"):
        pf.fit_lpa(one, ["y"], n_states=2, covariance="EEI", n_starts=1,
                   screen_iter=0, n_full=1, max_iter=5, seed=1)
    missing = pd.DataFrame({"id": [1, 2], "y": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="too few analyzed subjects to identify 1 states: 0 subject"):
        pf.fit_lpa(missing, ["y"], id="id", n_states=1, covariance="EEI", n_starts=1,
                   screen_iter=0, n_full=1, max_iter=5, seed=1)
    longitudinal = pd.DataFrame({"id": [1, 1], "wave": [1, 2], "y": [-1.0, 1.0]})
    with pytest.raises(ValueError, match="too few analyzed subjects to identify 2 states: 1 subject"):
        pf.fit_lta(longitudinal, ["y"], id="id", wave="wave", n_states=2,
                   covariance="EEI", n_starts=1, screen_iter=0, n_full=1,
                   max_iter=5, seed=1)
    two = pd.DataFrame({"y": [-1.0, 1.0]})
    fit = pf.fit_lpa(two, ["y"], n_states=2, covariance="EEI", n_starts=1,
                     screen_iter=0, n_full=1, max_iter=5, seed=1)
    assert fit.result["n_analyzed"] == 2


def test_declared_version_matches_pyproject_and_build_info():
    import tomllib
    from importlib.metadata import version
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert declared == "0.1.1"
    assert pf.__version__ == declared
    assert pf.build_info()["version"] == declared
    assert version("pyfylr") == declared


def test_build_pins_eigen():
    info = pf.build_info()
    assert info["eigen_version"] == "3.4.0"
    assert info["eigen_dont_parallelize"] is True
    assert info["external_blas"] is False
    assert "ffast-math" not in info["flags"]
    assert "fast-math=off" in info["flags"]
    assert info["engine"] == "cpp-em"


def test_binding_recovers_separated_gaussian_means():
    """One separated dataset through the installed extension.

    The eight-replicate C0 rate stays in the core suite. This is the binding check.
    """
    rng = np.random.default_rng(4101)
    y = np.concatenate([rng.normal(-3, 0.4, 40), rng.normal(3, 0.4, 40)])
    data = pd.DataFrame({"id": np.arange(len(y)), "y": y})
    fit = pf.fit_lpa(
        data, ["y"], n_states=2, covariance="EEI", n_starts=10, screen_iter=5,
        n_full=4, max_iter=50, seed=7001, align_labels=False, tol=1e-7,
    )
    means = sorted(fit.result["params"]["means"])
    assert abs(means[0] - (-3)) < 0.50
    assert abs(means[1] - 3) < 0.50
    assert fit.result["admissible"]


def test_hand_likelihood_through_the_panel():
    data = pd.DataFrame({"id": [1, 1], "wave": [1, 2], "u": [1, 1]})
    panel = pf.prepare_panel(data, ["u"], "id", "wave", family={"u": "bernoulli"})
    spec = pf.lca_spec(2, n_starts=1, align_labels=False)
    ll = pf._core.observed_loglik(panel, spec, {
        "delta": [0.6, 0.4],
        "tau": [0.7, 0.3, 0.2, 0.8],
        "rho": [0.8, 0.2, 0.3, 0.7],
    })
    assert ll == pytest.approx(math.log(0.2100), abs=1e-10)


def test_two_state_gaussian_likelihood_matches_the_hand_sum():
    data = pd.DataFrame({"id": [1, 1], "wave": [1, 2], "y": [0.0, 0.5]})
    panel = pf.prepare_panel(data, ["y"], "id", "wave", family={"y": "gaussian"})
    spec = pf.lpa_spec(2, covariance="EEI", align_labels=False)
    ll = pf._core.observed_loglik(panel, spec, {
        "delta": [0.7, 0.3],
        "tau": [0.8, 0.2, 0.4, 0.6],
        "means": [-1.0, 1.0],
        "cov": [1.0],
    })
    assert ll == pytest.approx(-3.024718794919309, abs=1e-10)


def test_free_transitions_with_a_missing_wave_match_the_hand_sum():
    data = pd.DataFrame({"id": [1, 1, 1], "wave": [1, 2, 3], "u": [1, np.nan, 0]})
    panel = pf.prepare_panel(data, ["u"], "id", "wave", family={"u": "bernoulli"})
    spec = pf.lca_spec(2, transitions="free", align_labels=False)
    ll = pf._core.observed_loglik(panel, spec, {
        "delta": [0.6, 0.4],
        "tau": [0.9, 0.1, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        "rho": [0.8, 0.2, 0.3, 0.7],
    })
    assert ll == pytest.approx(-1.6513021165327992, abs=1e-10)


def test_missing_wave_is_not_collapsed():
    data = pd.DataFrame({"id": [1, 1], "wave": [1, 3], "u": [1, 1]})
    panel = pf.prepare_panel(data, ["u"], "id", "wave")
    assert panel["T"] == 3
    spec = pf.lca_spec(2, align_labels=False)
    params = {"delta": [0.6, 0.4], "tau": [0.7, 0.3, 0.2, 0.8], "rho": [0.8, 0.2, 0.3, 0.7]}
    gapped = pf._core.observed_loglik(panel, spec, params)
    adjacent = data.copy()
    adjacent["wave"] = [1, 2]
    collapsed = pf.prepare_panel(adjacent, ["u"], "id", "wave")
    other = pf._core.observed_loglik(collapsed, spec, params)
    assert abs(gapped - other) > 1e-6


def test_mixed_product_and_fit():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(80):
        state = i < 40
        rows.append({
            "id": i,
            "y": rng.normal(-1.5 if state else 1.5, 0.4),
            "u": int(rng.random() < (0.2 if state else 0.8)),
        })
    data = pd.DataFrame(rows)
    fit = pf.fit_lpa(data, ["y", "u"], n_states=2, covariance="EEI", family={"y": "gaussian", "u": "bernoulli"},
                     n_starts=3, screen_iter=5, n_full=2, max_iter=40, seed=1)
    assert fit.engine == "cpp-em"
    assert fit.draws is None
    assert fit.interval_type == "none"
    assert fit.result["parameter_uncertainty"] == "none"
    assert fit.result["state_uncertainty"] == "plugin_posterior"
    assert fit.result["n_gaussian"] == 1 and fit.result["n_categorical"] == 1
    assert fit.glance().shape[0] == 1
    assert "mean" in set(fit.tidy()["parameter"])
    assert "response" in set(fit.tidy()["parameter"])
    assert fit.augment().shape[0] == 80


def test_ordinal_count_covariate_and_distal_are_refused():
    with pytest.raises(Exception, match="Release 2"):
        pf._core.parse_family("ordinal")
    with pytest.raises(Exception, match="not available"):
        pf._core.parse_family("poisson")
    data = pd.DataFrame({"id": [1], "y": [0.2]})
    with pytest.raises(Exception, match="Release 1"):
        pf.fit_lpa(data, ["y"], n_states=1, covariates=["x"])
    with pytest.raises(Exception, match="Release 3"):
        pf.estimate_distal("outcome", "binary", timing="final")
    with pytest.raises(Exception, match="Modal-assignment"):
        pf.estimate_distal("outcome", "continuous", timing="initial", method="modal")
    with pytest.raises(Exception, match="wave"):
        pf.estimate_distal("outcome", "ordinal", timing="wave")
    with pytest.raises(Exception, match="from_state"):
        pf.estimate_distal("outcome", "nominal", timing="transition")


def test_profile_plot_draws():
    data = pd.DataFrame({"y": [-1.0, -1.2, 1.1, 0.9]})
    fit = pf.fit_lpa(data, ["y"], n_states=1, covariance="EEI", n_starts=1, max_iter=5, seed=1)
    ax = pf.plot_profiles(fit)
    assert ax.get_ylabel() == "State mean"


def test_one_wave_has_no_transitions_and_labels_are_ordered():
    rng = np.random.default_rng(2)
    data = pd.DataFrame({
        "y1": np.concatenate([rng.normal(-2, 0.3, 30), rng.normal(2, 0.3, 30)]),
        "y2": np.concatenate([rng.normal(-1, 0.3, 30), rng.normal(1, 0.3, 30)]),
    })
    fit = data.pyfylr.fit_lpa(["y1", "y2"], n_states=2, covariance="VVI", n_starts=2, max_iter=50, seed=2)
    tidy = fit.tidy()
    assert tidy[tidy["parameter"] == "transition"].empty
    initials = tidy[tidy["parameter"] == "initial"]["estimate"].tolist()
    assert initials[0] >= initials[1] - 1e-8
    profiles = pf.describe_profiles(fit)
    assert (profiles["source"] == "model").any()
    assert profiles["reference_sd_source"].isin(["pooled_sample", "not_applicable"]).all()


# Fields that are not part of the shared numeric result. n_threads is compared
# and both calls set it to 1. A caller that asked for a different thread count
# would be a different experiment, not a field this test ignores.
def _is_number(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


# R exports the same result fields as Python, including n_threads.
_PYTHON_ONLY_RESULT_FIELDS = set()


# Measurements. Counts, flags, codes, and labels are compared exactly.
_FLOAT_ABS_TOL = 1e-8
_FLOAT_FIELDS = {
    "loglik", "aic", "bic", "sabic", "tol", "replicate_tol",
    "entropy_wave", "entropy_sequence", "entropy_wave_raw", "entropy_sequence_raw",
    "min_occupancy", "min_eigenvalue", "condition_number", "min_relative_variance",
    "viterbi_logprob", "gamma", "filtered", "xi",
    "prevalence_model", "prevalence_posterior", "prevalence_hard",
    "expected_membership", "expected_transitions",
    "start_loglik", "rejected_loglik", "rejected_delta", "rejected_means",
    "rejected_cov", "rejected_class_eigen", "rejected_min_class_n", "trace_ll",
}


def test_r_and_python_match_on_one_fixed_start():
    """Compare shared fields from the two packages built from one checkout.

    Discrete fields match exactly. Floating-point fields use absolute tolerance
    1e-8. A run against a library installed earlier is preliminary.
    """
    # Only the ubuntu python job builds profylr and sets this variable. Other
    # jobs skip even when a system Rscript exists, because that Rscript does
    # not load the profylr built from this commit.
    required = os.environ.get("PYFYLR_CROSS_LANGUAGE") == "required"
    if not required:
        pytest.skip("this job does not build both packages; the ubuntu python job sets PYFYLR_CROSS_LANGUAGE=required")
    if shutil.which("Rscript") is None:
        raise AssertionError("this job builds both packages and requires Rscript")
    data = pd.DataFrame({"id": [1, 2, 3, 4], "y": [-1.2, -0.8, 0.9, 1.4]})
    start = {
        "delta": [0.55, 0.45],
        "tau": [1.0, 0.0, 0.0, 1.0],
        "means": [-1.0, 1.0],
        "cov": [0.5],
    }
    controls = dict(
        n_states=2, covariance="EEI", n_starts=1, screen_iter=0, n_full=1,
        max_iter=6, seed=1, align_labels=False, tol=1e-7, n_threads=1,
    )
    py = pf.fit_lpa(data, ["y"], start=start, **controls)
    script = textwrap.dedent(r"""
        suppressPackageStartupMessages(library(profylr))
        cell <- Sys.getenv("R_LIBS", unset = "")
        if (!nzchar(cell)) stop("cross-language R requires R_LIBS", call. = FALSE)
        if (!identical(Sys.getenv("R_LIBS_SITE"), "NULL") || !identical(Sys.getenv("R_LIBS_USER"), "NULL")) {
          stop("cross-language R requires R_LIBS_SITE=NULL and R_LIBS_USER=NULL", call. = FALSE)
        }
        if (!"--vanilla" %in% commandArgs(trailingOnly = FALSE)) {
          stop("cross-language R requires Rscript --vanilla", call. = FALSE)
        }
        cell <- normalizePath(cell, mustWork = TRUE)
        lib_root <- normalizePath(.Library, mustWork = TRUE)
        for (entry in normalizePath(.libPaths(), mustWork = TRUE)) {
          if (!identical(entry, cell) && !identical(entry, lib_root)) {
            stop("library path outside the cell library: ", entry, call. = FALSE)
          }
          cat(paste("libpath", entry, sep = "\t"), "\n", sep = "")
        }
        base <- rownames(installed.packages(lib.loc = .Library, priority = "base"))
        for (name in sort(loadedNamespaces())) {
          if (name %in% base) next
          loaded <- normalizePath(getNamespaceInfo(name, "path"), mustWork = TRUE)
          prefix <- paste0(cell, .Platform$file.sep)
          if (!identical(loaded, cell) && !startsWith(loaded, prefix)) {
            stop(name, " namespace resolved outside the cell library: ", loaded, call. = FALSE)
          }
          ver <- as.character(getNamespaceVersion(name))
          if (!nzchar(ver)) stop(name, " has no version", call. = FALSE)
          cat(paste("namespace", name, loaded, ver, sep = "\t"), "\n", sep = "")
        }
        cat("startup\tvanilla\n")
        data <- data.frame(id = c(1, 2, 3, 4), y = c(-1.2, -0.8, 0.9, 1.4))
        start <- list(delta = c(0.55, 0.45), tau = c(1, 0, 0, 1), means = c(-1, 1), cov = 0.5)
        fit <- fit_lpa(data, "y", id = "id", n_states = 2, covariance = "EEI",
                       n_starts = 1, screen_iter = 0, n_full = 1, max_iter = 6,
                       seed = 1, align_labels = FALSE, tol = 1e-7, n_threads = 1, start = start)
        r <- fit$result
        line <- function(name, x) {
          if (is.null(x)) x <- numeric()
          if (is.logical(x)) x <- as.integer(x)
          if (is.character(x)) {
            cat(name, paste(x, collapse = " "), "\n", sep = "\t")
          } else {
            cat(name, paste(format(as.vector(x), digits = 17, scientific = TRUE), collapse = " "), "\n", sep = "\t")
          }
        }
        for (nm in sort(names(r))) {
          if (nm == "params") next
          line(nm, r[[nm]])
        }
        for (nm in sort(names(r$params))) line(paste0("params.", nm), r$params[[nm]])
    """)
    r_libs = os.environ.get("R_LIBS", "")
    if not r_libs:
        raise AssertionError("cross-language comparison requires R_LIBS")
    env = os.environ.copy()
    env["R_LIBS"] = r_libs
    env["R_LIBS_SITE"] = "NULL"
    env["R_LIBS_USER"] = "NULL"
    proc = subprocess.run(
        ["Rscript", "--vanilla", "-e", script],
        capture_output=True, text=True, env=env, check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)
    cell = os.path.realpath(r_libs)
    got = {}
    namespaces = []
    libpaths = []
    vanilla = False
    for ln in proc.stdout.splitlines():
        if ln.startswith("namespace\t"):
            parts = ln.split("\t")
            if len(parts) != 4 or not parts[3]:
                raise AssertionError(ln)
            namespaces.append((parts[1], os.path.realpath(parts[2]), parts[3]))
            continue
        if ln.startswith("libpath\t"):
            libpaths.append(os.path.realpath(ln.split("\t", 1)[1]))
            continue
        if ln == "startup\tvanilla":
            vanilla = True
            continue
        if "\t" not in ln:
            continue
        name, rest = ln.split("\t", 1)
        got[name] = rest.split() if rest.strip() else []
    if not vanilla:
        raise AssertionError("R child did not confirm --vanilla")
    if cell not in libpaths:
        raise AssertionError(f"cell library {cell} not in {libpaths}")
    if not any(name == "profylr" for name, _, _ in namespaces):
        raise AssertionError(f"profylr namespace was not reported: {namespaces}")
    for name, path, _version in namespaces:
        if os.path.commonpath([cell, path]) != cell:
            raise AssertionError(f"{name} resolved outside {cell}: {path}")
    result = py.result
    params = result["params"]

    def py_vals(name):
        if name.startswith("params."):
            value = params[name.split(".", 1)[1]]
        else:
            value = result[name]
        if isinstance(value, bool):
            return [str(int(value))]
        if isinstance(value, int) and not isinstance(value, bool):
            return [str(value)]
        if isinstance(value, float):
            return [format(value, ".16e")]
        if isinstance(value, str):
            return [value]
        out = []
        for v in value:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, bool):
                out.append(str(int(v)))
            elif isinstance(v, int):
                out.append(str(v))
            else:
                out.append(format(float(v), ".16e"))
        return out

    r_names = {name for name in got if not name.startswith("params.")}
    py_names = set(result) - {"params"}
    assert py_names - r_names == _PYTHON_ONLY_RESULT_FIELDS
    assert r_names - py_names == set()
    # R stores family names. Python stores the core codes: gaussian=0, bernoulli=1, nominal=2.
    family_code = {"gaussian": "0", "bernoulli": "1", "nominal": "2"}
    mismatches = []
    for name, rvals in got.items():
        if name == "family":
            rvals = [family_code.get(v, v) for v in rvals]
        left = py_vals(name)
        floating = name in _FLOAT_FIELDS or name.startswith("params.")
        if floating and left and rvals and all(_is_number(v) for v in left) and all(_is_number(v) for v in rvals):
            if [float(v) for v in left] != pytest.approx([float(v) for v in rvals], abs=_FLOAT_ABS_TOL):
                mismatches.append(name)
        elif left != rvals:
            # Integer-valued discrete fields may be printed in scientific form.
            if left and rvals and all(_is_number(v) for v in left) and all(_is_number(v) for v in rvals):
                if [float(v) for v in left] != [float(v) for v in rvals]:
                    mismatches.append(name)
            else:
                mismatches.append(f"{name}:{left}!={rvals}")
    assert not mismatches, mismatches
