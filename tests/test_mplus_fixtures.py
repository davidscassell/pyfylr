"""The five Mplus inputs agree with pyfylr at the precision Mplus printed."""

import numpy as np
import pandas as pd

import pyfylr as pf

from fixture_paths import fixtures_dir

FIX = fixtures_dir() / "mplus"
CTRL = dict(
    n_starts=40, n_full=4, screen_iter=10, max_iter=200, tol=1e-8,
    replicate_tol=1e-3, seed=1, n_threads=1, align_labels=False,
)


def read_table(name):
    rows = []
    for line in (FIX / name).read_text().splitlines():
        if not line.strip():
            continue
        rows.append([np.nan if tok in (".", "nan") else float(tok) for tok in line.split()])
    return pd.DataFrame(rows)


def test_default_start_budget_is_40():
    assert pf.lta_spec(2)["n_starts"] == 40
    assert pf.LatentProfileModel().n_starts == 40


def test_vvv_missing_reaches_and_replicates_the_mplus_mode():
    frame = read_table("lpa_vvv_k2_missing.dat")
    frame.columns = ["y1", "y2"]
    frame.insert(0, "id", np.arange(1, len(frame) + 1))
    fit = pf.fit_lpa(frame, ["y1", "y2"], id="id", n_states=2, covariance="VVV", **CTRL)
    result = fit.result
    assert abs(result["loglik"] - (-132.902976)) < 1e-4
    assert result["n_best"] >= 2
    assert result["status"] == "ok"
    assert result["n_parameters"] == 11
    assert abs(result["bic"] - 308.838) < 0.001
    assert result["cov_positive_definite"]
    assert not result["cov_ill_conditioned"]
    assert not result["cov_collapsed_component"]
    assert not result["regularized"]
    assert result["n_boundary"] == 0
    assert result["condition_number"] < 1e6
    delta = np.asarray(result["params"]["delta"], dtype=float)
    small = int(np.argmin(delta))
    assert abs(delta[small] - 0.257) < 2e-3
    assert abs(result["params"]["cov"][small * 3 + 1] - (-0.25)) < 2e-3


def test_lta_hungarian_alignment_is_within_mplus_rounding():
    wide = read_table("lta_bernoulli_k2_t2.dat")
    rows = []
    for i, rec in wide.iterrows():
        rows.append({"id": int(i) + 1, "wave": 1, "u1": float(rec.iloc[0]), "u2": float(rec.iloc[1])})
        rows.append({"id": int(i) + 1, "wave": 2, "u1": float(rec.iloc[2]), "u2": float(rec.iloc[3])})
    fit = pf.fit_lta(pd.DataFrame(rows), ["u1", "u2"], id="id", wave="wave", n_states=2, **CTRL)
    assert abs(fit.result["loglik"] - (-360.432800)) < 1e-4
    assert fit.result["n_parameters"] == 7
    rho = []
    for item in ((0.838, 0.054), (0.804, 0.140)):
        for p1 in item:
            rho.extend([1 - p1, p1])
    # rho above is item-major (j, k, r) which is the packed layout ((j)*K+k)*R+r.
    reference = {
        "delta": [0.507, 0.493],
        "tau": [0.860, 0.140, 0.041, 0.959],
        "rho": rho,
        "means": [],
        "cov": [],
    }
    aligned, alignment = pf.align_states(fit, reference)
    params = aligned.result["params"]
    assert list(alignment["permutation"]) == [0, 1]
    assert abs(params["delta"][0] - 0.507) < 5e-4
    assert abs(params["delta"][1] - 0.493) < 5e-4
    tau = np.asarray(params["tau"], dtype=float).reshape(2, 2)
    assert np.max(np.abs(tau - np.array([[0.860, 0.140], [0.041, 0.959]]))) < 5e-4
    got = []
    for j in range(2):
        for k in range(2):
            got.append(params["rho"][((j * 2 + k) * 2) + 1])
    assert np.max(np.abs(np.asarray(got) - np.array([0.838, 0.054, 0.804, 0.140]))) < 5e-4


def test_lca_eei_and_eee_match_the_printed_mplus_likelihood():
    lca = read_table("lca_bernoulli_k2.dat")
    lca.columns = ["u1", "u2", "u3", "u4"]
    lca.insert(0, "id", np.arange(1, len(lca) + 1))
    lca_fit = pf.fit_lca(lca, ["u1", "u2", "u3", "u4"], id="id", n_states=2, **CTRL)
    assert lca_fit.result["n_parameters"] == 9
    assert abs(lca_fit.result["loglik"] - (-183.948578)) < 1e-4
    assert abs(lca_fit.result["bic"] - 407.335) < 0.001

    eei = read_table("lpa_eei_k2.dat")
    eei.columns = ["y1", "y2", "y3"]
    eei.insert(0, "id", np.arange(1, len(eei) + 1))
    eei_fit = pf.fit_lpa(eei, ["y1", "y2", "y3"], id="id", n_states=2, covariance="EEI", **CTRL)
    assert eei_fit.result["n_parameters"] == 10
    assert abs(eei_fit.result["loglik"] - (-318.559608)) < 1e-4
    assert abs(eei_fit.result["bic"] - 680.939) < 0.001

    eee = read_table("lpa_eee_k2.dat")
    eee.columns = ["y1", "y2"]
    eee.insert(0, "id", np.arange(1, len(eee) + 1))
    eee_fit = pf.fit_lpa(eee, ["y1", "y2"], id="id", n_states=2, covariance="EEE", **CTRL)
    assert eee_fit.result["n_parameters"] == 8
    assert abs(eee_fit.result["loglik"] - (-185.678407)) < 1e-4
    assert abs(eee_fit.result["bic"] - 404.112) < 0.001


def test_iteration_cap_is_exact_including_sklearn():
    rng = np.random.default_rng(1)
    y = np.column_stack([
        np.concatenate([rng.normal(-3, 0.6, 20), rng.normal(3, 0.6, 20)]),
        np.concatenate([rng.normal(-3, 0.6, 20), rng.normal(3, 0.6, 20)]),
    ])
    frame = pd.DataFrame(y, columns=["y1", "y2"])
    fit = pf.fit_lpa(
        frame, ["y1", "y2"], n_states=2, covariance="VVI", n_starts=1, screen_iter=0,
        n_full=1, max_iter=2, tol=0.0, seed=1, align_labels=False,
    )
    assert fit.result["converged"] is False
    assert fit.result["n_iter"] == 2
    assert list(fit.result["trace_iter"]) == [1, 2]
    screened = pf.fit_lpa(
        frame, ["y1", "y2"], n_states=2, covariance="VVI", n_starts=4, screen_iter=2,
        n_full=3, max_iter=5, tol=0.0, seed=1, align_labels=False,
    )
    assert screened.result["n_iter"] <= 5
    assert len(screened.result["trace_ll"]) == screened.result["n_iter"]
    assert max(screened.result["trace_iter"]) <= 5
    model = pf.LatentProfileModel(
        n_states=2, covariance="VVI", n_starts=1, screen_iter=0, n_full=1,
        max_iter=2, tol=0.0, random_state=1, align_labels=False,
    ).fit(frame)
    assert model.n_iter_ == 2
    assert model.n_iter_ == fit.result["n_iter"]
    assert model.converged_ is False


def test_selection_returns_the_usable_vvv_solution():
    frame = read_table("lpa_vvv_k2_missing.dat")
    frame.columns = ["y1", "y2"]
    fit = pf.fit_lpa(
        frame, ["y1", "y2"], n_states=2, covariance="VVV", n_starts=500, screen_iter=0,
        n_full=500, max_iter=200, tol=1e-8, replicate_tol=1e-3, seed=1,
        n_threads=1, align_labels=False,
    )
    result = fit.result
    assert abs(result["loglik"] - (-132.902976)) < 1e-3
    assert result["admissible"] is True
    assert result["regularized"] is False
    assert result["status"] != "unusable"
    assert result["n_best"] >= 2
    rejected = list(result["rejected_loglik"])
    assert rejected
    singular = [
        i for i, ll in enumerate(rejected) if abs(ll - (-129.595734)) < 5e-3
    ]
    assert singular, "the regularized point near -129.596 must be reported, not selected"
    i = singular[0]
    assert result["rejected_level"][i] == "blocking"
    assert "variance_floor" in result["rejected_reason"][i]
    assert result["rejected_n"][i] == 1
    assert rejected[i] > result["loglik"]
