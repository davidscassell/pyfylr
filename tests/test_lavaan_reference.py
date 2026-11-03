"""Compare the one-state Gaussian model with lavaan numbers generated at test time.

lavaan is GPL. This test calls it and reads the numbers it prints. It does not
vendor lavaan source.
"""

import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyfylr as pf

R_SCRIPT = textwrap.dedent(
    r"""
    suppressPackageStartupMessages(library(lavaan))
    suppressPackageStartupMessages(library(semTools))
    mask <- function(Y, every, patterns) {
      for (i in seq_len(nrow(Y))) {
        if (i %% every != 0) next
        code <- patterns[((i / every - 1) %% length(patterns)) + 1]
        digits <- if (code == 0) integer(0) else as.integer(strsplit(as.character(code), "")[[1]])
        if (length(digits)) Y[i, digits] <- NA_real_
      }
      Y
    }
    one <- function(seed, every, patterns, tag, dir) {
      set.seed(seed)
      n <- 160
      sigma <- matrix(c(1, .35, .15, .05, .35, 1, .25, .1, .15, .25, 1, .2, .05, .1, .2, 1), 4, 4)
      Y <- matrix(rnorm(n * 4), n, 4) %*% chol(sigma)
      Y <- sweep(Y, 2, c(0.2, -0.4, 0.8, 0.1), "+")
      Y <- mask(Y, every, patterns)
      colnames(Y) <- paste0("y", 1:4)
      df <- as.data.frame(Y)
      vars <- names(df)
      pairs <- combn(vars, 2)
      free <- apply(pairs, 2, function(ab) paste(ab[[1]], "~~", ab[[2]]))
      zero <- apply(pairs, 2, function(ab) paste(ab[[1]], "~~ 0*", ab[[2]]))
      saturated <- paste(c(paste(vars, "~~", vars), free, paste(vars, "~ 1")), collapse = "\n")
      diagonal <- paste(c(paste(vars, "~~", vars), zero, paste(vars, "~ 1")), collapse = "\n")
      full <- sem(saturated, data = df, missing = "ml", meanstructure = TRUE)
      diagm <- sem(diagonal, data = df, missing = "ml", meanstructure = TRUE)
      desc <- vapply(df, function(x) {
        observed <- na.omit(x)
        c(mean(observed), sd(observed),
          as.numeric(semTools::skew(observed, population = TRUE)),
          as.numeric(semTools::kurtosis(observed, population = TRUE)))
      }, numeric(4))
      write.csv(df, file.path(dir, paste0(tag, ".csv")), row.names = FALSE)
      out <- file.path(dir, paste0(tag, ".txt"))
      writeLines(c(
        paste(fitMeasures(full, "logl"), fitMeasures(full, "npar")),
        paste(fitMeasures(diagm, "logl"), fitMeasures(diagm, "npar")),
        paste(as.numeric(lavInspect(full, "mean.ov")), collapse = " "),
        paste(as.numeric(t(lavInspect(full, "cov.ov"))), collapse = " "),
        paste(as.numeric(t(lavInspect(diagm, "cov.ov"))), collapse = " "),
        paste(as.numeric(t(lavCor(df, missing = "ml"))), collapse = " "),
        paste(as.numeric(t(lavCor(df, missing = "pairwise"))), collapse = " "),
        paste(as.numeric(desc), collapse = " ")
      ), out)
    }
    args <- commandArgs(trailingOnly = TRUE)
    one(11, 5, c(0, 1, 2, 3, 12, 13, 23, 123), "scattered", args[[1]])
    one(19, 4, c(4, 14, 23, 1), "grouped", args[[1]])
    """
)


def _reference(tmp_path: Path):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not available; lavaan fixtures are generated at test time")
    script = tmp_path / "reference.R"
    script.write_text(R_SCRIPT)
    run = subprocess.run([rscript, str(script), str(tmp_path)], capture_output=True, text=True)
    if run.returncode != 0:
        message = run.stderr + run.stdout
        if "there is no package called" in message:
            pytest.skip("lavaan or semTools is not installed")
        raise AssertionError(message)
    return tmp_path


def _check(tmp_path: Path, tag: str):
    df = pd.read_csv(tmp_path / f"{tag}.csv")
    lines = (tmp_path / f"{tag}.txt").read_text().splitlines()
    logl_full, npar_full = map(float, lines[0].split())
    logl_diag, _npar_diag = map(float, lines[1].split())
    mean = np.fromstring(lines[2], sep=" ")
    cov = np.fromstring(lines[3], sep=" ").reshape(4, 4)
    diag = np.fromstring(lines[4], sep=" ").reshape(4, 4)
    cor_ml = np.fromstring(lines[5], sep=" ").reshape(4, 4)
    cor_pair = np.fromstring(lines[6], sep=" ").reshape(4, 4)
    desc = np.fromstring(lines[7], sep=" ").reshape(-1, 4)
    full = pf.fiml_moments(df, list(df.columns), covariance="VVV")
    diagonal = pf.fiml_moments(df, list(df.columns), covariance="EEI")
    assert full["converged"]
    assert full["loglik"] == pytest.approx(logl_full, abs=1e-6)
    assert full["n_parameters"] == int(npar_full)
    assert np.allclose(full["mean"].to_numpy(), mean, atol=1e-5)
    assert np.allclose(full["cov"].to_numpy(), cov, atol=1e-5)
    assert np.allclose(full["cor"].to_numpy(), cor_ml, atol=1e-5)
    assert diagonal["loglik"] == pytest.approx(logl_diag, abs=1e-6)
    assert np.allclose(diagonal["cov"].to_numpy(), diag, atol=1e-5)
    pairs = pf.correlate_indicators(df, list(df.columns))
    names = list(df.columns)
    for _, row in pairs.iterrows():
        i = names.index(row["variable_a"])
        j = names.index(row["variable_b"])
        if i == j:
            continue
        assert row["correlation"] == pytest.approx(cor_pair[i, j], abs=1e-8)
    described = pf.describe_indicators(df.assign(id=np.arange(len(df))), list(df.columns), id="id")
    for j, name in enumerate(names):
        row = described.loc[described["variable"] == name].iloc[0]
        assert row["mean"] == pytest.approx(desc[j, 0], abs=1e-10)
        assert row["sd"] == pytest.approx(desc[j, 1], abs=1e-10)
        assert row["skew"] == pytest.approx(desc[j, 2], abs=1e-6)
        assert row["kurtosis"] == pytest.approx(desc[j, 3], abs=1e-6)
    direct = pf.fit_lpa(df, list(df.columns), n_states=1, covariance="VVV",
                        n_starts=1, max_iter=500, tol=1e-12, screen_iter=0, seed=1)
    assert direct.result["loglik"] == pytest.approx(logl_full, abs=1e-6)


def test_python_matches_lavaan_fixtures(tmp_path):
    folder = _reference(tmp_path)
    _check(folder, "scattered")
    _check(folder, "grouped")
