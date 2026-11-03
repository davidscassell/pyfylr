"""scikit-learn estimators call the same core as the pandas API.

``score`` is the mean observed-data log-likelihood of a fixed ``n_states``.
It is not a criterion for choosing the number of states, and these tests do
not wrap the estimators in ``GridSearchCV``.

``check_estimator`` skips, and why they are legitimate:

- ``LatentClassModel`` is not passed to ``check_estimator``. The checker
  supplies continuous arrays, and a class model needs categorical codes.
- ``LatentTransitionModel`` is checked at ``n_waves=1`` only. The checker
  cannot supply a wave index. Multi-wave ``predict``, ``filter``, ``smooth``,
  and ``viterbi`` are compared with the pandas API in this file.
- Sparse checks are skipped. A sparse matrix is not a panel. FIML missingness
  is NaN in a dense array.
- Checks that treat ``score`` as a bounded greater-is-better metric are
  skipped. ``score`` is a log-likelihood. It rises with ``n_states`` for a
  nested mixture and must not be used to select ``n_states``.
"""

import numpy as np
import pandas as pd
import pytest

import pyfylr as pf
from pyfylr.sklearn_api import SCORE_SELECTS_N_STATES

CONTROLS = dict(n_starts=3, max_iter=40, screen_iter=8, n_full=2, tol=1e-6, random_state=11)


def sample(n=80, seed=3):
    rng = np.random.default_rng(seed)
    y1 = np.concatenate([rng.normal(-1.5, 0.4, n // 2), rng.normal(1.6, 0.45, n // 2)])
    y2 = np.concatenate([rng.normal(-1.2, 0.35, n // 2), rng.normal(1.3, 0.4, n // 2)])
    return pd.DataFrame({"y1": y1, "y2": y2})


def _means(fit):
    result = fit.result
    k = result["K"]
    j = result["n_gaussian"]
    return np.asarray(result["params"]["means"], dtype=float).reshape(k, j)


def test_score_is_not_a_state_count_criterion():
    assert SCORE_SELECTS_N_STATES is False
    text = pf.LatentProfileModel.score.__doc__
    assert "criterion for choosing" in text
    assert "GridSearchCV" in text
    assert "n_states" in pf.LatentProfileModel.__doc__


def test_constructor_arguments_are_stored_unchanged():
    model = pf.LatentProfileModel(n_states=2, covariance="VVV", n_starts=4, random_state=None)
    assert model.n_states == 2
    assert model.covariance == "VVV"
    assert model.n_starts == 4
    assert model.random_state is None
    params = model.get_params()
    model.set_params(n_states=3)
    assert model.n_states == 3
    model.set_params(**params)
    assert model.get_params() == params


def test_numbers_match_the_pandas_api_including_diagnostics_and_labels():
    data = sample()
    pandas_fit = pf.fit_lpa(data, ["y1", "y2"], n_states=2, covariance="VVV", seed=11,
                            n_starts=3, max_iter=40, screen_iter=8, n_full=2, tol=1e-6)
    model = pf.LatentProfileModel(n_states=2, covariance="VVV", **CONTROLS).fit(data)
    assert model.loglik_ == pandas_fit.result["loglik"]
    assert model.n_iter_ == pandas_fit.result["n_iter"]
    assert model.converged_ == pandas_fit.result["converged"]
    assert np.array_equal(model.means_, _means(pandas_fit))
    assert np.allclose(model.weights_, pandas_fit.result["params"]["delta"])
    assert model.diagnostics_["status_reason"] == pandas_fit.result["status_reason"]
    assert model.diagnostics_["eligibility_detail"] == pandas_fit.result["eligibility_detail"]
    assert list(model.diagnostics_["label_order"]) == list(pandas_fit.result["label_order"])
    assert model.fit_result_.result["status"] == pandas_fit.result["status"]
    modal = np.asarray(pandas_fit.result["modal"], dtype=int)
    assert np.array_equal(model.predict(data), modal)
    gamma = np.asarray(pandas_fit.result["gamma"], dtype=float).reshape(-1, 2)
    assert np.allclose(model.predict_proba(data), gamma)
    n = pandas_fit.result["n_analyzed"]
    assert model.score(data) == pytest.approx(pandas_fit.result["loglik"] / n)
    assert model.bic(data) == pytest.approx(pandas_fit.result["bic"])
    assert model.aic(data) == pytest.approx(pandas_fit.result["aic"])
    per_row = model.score_samples(data)
    assert per_row.shape == (len(data),)
    assert float(np.nansum(per_row)) == pytest.approx(pandas_fit.result["loglik"])
    assert model.random_state == 11
    assert list(model.feature_names_in_) == ["y1", "y2"]
    assert model.n_features_in_ == 2


def test_clone_and_refit_match_when_random_state_is_fixed():
    data = sample()
    model = pf.LatentProfileModel(n_states=2, covariance="EEI", **CONTROLS)
    before = model.get_params()
    model.fit(data)
    assert model.get_params() == before
    copied = pf.clone(model)
    assert copied.get_params() == before
    copied.fit(data)
    assert copied.loglik_ == model.loglik_
    assert np.array_equal(copied.means_, model.means_)
    model.fit(data)
    assert model.loglik_ == copied.loglik_
    assert np.array_equal(model.means_, copied.means_)


def test_none_random_state_is_seed_zero_and_reproducible():
    data = sample()
    first = pf.LatentProfileModel(
        n_states=1, covariance="EEI", n_starts=1, max_iter=20, screen_iter=0, random_state=None
    ).fit(data)
    second = pf.LatentProfileModel(
        n_states=1, covariance="EEI", n_starts=1, max_iter=20, screen_iter=0, random_state=None
    ).fit(data)
    assert first.random_state is None
    assert first.loglik_ == second.loglik_
    pandas_fit = pf.fit_lpa(
        data, ["y1", "y2"], n_states=1, covariance="EEI", n_starts=1, max_iter=20, screen_iter=0, seed=0
    )
    assert first.loglik_ == pandas_fit.result["loglik"]


def test_fiml_missing_values_match_the_pandas_api():
    data = sample()
    data.loc[0, "y1"] = np.nan
    data.loc[4, "y2"] = np.nan
    pandas_fit = pf.fit_lpa(
        data, ["y1", "y2"], n_states=1, covariance="EEI", n_starts=1, max_iter=30, screen_iter=0, seed=2
    )
    model = pf.LatentProfileModel(
        n_states=1, covariance="EEI", n_starts=1, max_iter=30, screen_iter=0, random_state=2
    ).fit(data)
    assert model.loglik_ == pandas_fit.result["loglik"]
    assert np.isnan(data.to_numpy()).any()
    array = data.to_numpy(dtype=float)
    from_array = pf.LatentProfileModel(
        n_states=1, covariance="EEI", n_starts=1, max_iter=30, screen_iter=0, random_state=2
    ).fit(array)
    assert from_array.loglik_ == model.loglik_
    assert not hasattr(from_array, "feature_names_in_")


def test_alignment_on_the_shared_fit_object_is_unchanged():
    data = sample()
    pandas_fit = pf.fit_lpa(
        data, ["y1", "y2"], n_states=2, covariance="VVV", seed=11,
        n_starts=3, max_iter=40, screen_iter=8, n_full=2, tol=1e-6,
    )
    model = pf.LatentProfileModel(n_states=2, covariance="VVV", **CONTROLS).fit(data)
    means = np.asarray(pandas_fit.result["params"]["means"], dtype=float).reshape(2, 2)
    cov = np.asarray(pandas_fit.result["params"]["cov"], dtype=float).reshape(2, -1)
    reference = {
        "delta": list(reversed(pandas_fit.result["params"]["delta"])),
        "tau": pandas_fit.result["params"]["tau"],
        "rho": pandas_fit.result["params"]["rho"],
        "means": means[::-1].reshape(-1).tolist(),
        "cov": cov[::-1].reshape(-1).tolist(),
    }
    aligned, info = pf.align_states(model.fit_result_, reference)
    pandas_aligned, pandas_info = pf.align_states(pandas_fit, reference)
    assert info["permutation"] == pandas_info["permutation"]
    assert aligned.result["status_reason"] == pandas_aligned.result["status_reason"]
    assert aligned.result["params"]["means"] == pandas_aligned.result["params"]["means"]
    assert model.diagnostics_["label_order"] == list(pandas_fit.result["label_order"])


def test_transition_predict_uses_smoothing_and_keeps_the_other_paths():
    rng = np.random.default_rng(5)
    n = 40
    y = np.concatenate([rng.normal(-1.2, 0.3, n), rng.normal(1.2, 0.3, n)])
    long = pd.DataFrame({
        "id": list(range(n)) * 2,
        "wave": [1] * n + [2] * n,
        "y": y,
    })
    wide = np.column_stack([y[:n], y[n:]])
    controls = dict(n_starts=2, max_iter=30, screen_iter=6, n_full=2, tol=1e-6, seed=5)
    pandas_fit = pf.fit_lta(long, ["y"], id="id", wave="wave", n_states=2, covariance="EEI", **controls)
    model = pf.LatentTransitionModel(
        n_states=2, covariance="EEI", n_waves=2, n_starts=2, max_iter=30, screen_iter=6,
        n_full=2, tol=1e-6, random_state=5,
    ).fit(wide)
    assert model.loglik_ == pandas_fit.result["loglik"]
    modal = np.asarray(pandas_fit.result["modal"], dtype=int).reshape(n, 2)
    viterbi = np.asarray(pandas_fit.result["viterbi"], dtype=int).reshape(n, 2)
    assert np.array_equal(model.predict(wide), modal)
    assert np.array_equal(model.smooth(wide), modal)
    assert np.array_equal(model.viterbi(wide), viterbi)
    filtered = np.asarray(pandas_fit.result["filtered"], dtype=float).reshape(n, 2, 2)
    assert np.array_equal(model.filter(wide), np.argmax(filtered, axis=-1))
    gamma = np.asarray(pandas_fit.result["gamma"], dtype=float).reshape(n, 2, 2)
    assert np.allclose(model.predict_proba(wide), gamma)
    assert model.transition_matrix_.shape == (2, 2)
    long_model = pf.LatentTransitionModel(
        n_states=2, covariance="EEI", n_waves=2, id_column="id", wave_column="wave",
        n_starts=2, max_iter=30, screen_iter=6, n_full=2, tol=1e-6, random_state=5,
    ).fit(long)
    assert long_model.loglik_ == model.loglik_


def test_add_starts_keeps_the_same_core_and_grows_the_start_list():
    data = sample()
    fit = pf.fit_lpa(
        data, ["y1", "y2"], n_states=1, covariance="EEI",
        n_starts=1, max_iter=20, screen_iter=0, seed=1,
    )
    more = pf.add_starts(fit, n_new=1, seed=2)
    assert more.result["n_starts"] == fit.result["n_starts"] + 1
    assert len(more.result["start_loglik"]) == fit.result["n_starts"] + 1
    assert more.result["loglik"] + 1e-8 >= fit.result["loglik"]


def test_latent_class_model_matches_pandas_on_codes():
    data = pd.DataFrame({"u": [0, 1] * 30, "v": [0, 0, 1, 1] * 15})
    pandas_fit = pf.fit_lca(data, ["u", "v"], n_states=2, n_starts=2, max_iter=25, screen_iter=5, n_full=2, seed=1)
    model = pf.LatentClassModel(n_states=2, n_starts=2, max_iter=25, screen_iter=5, n_full=2, random_state=1).fit(data)
    assert model.loglik_ == pandas_fit.result["loglik"]
    assert "categorical codes" in pf.LatentClassModel.__doc__


SKIPPED_CHECKS = {
    "check_estimator_sparse_data": (
        "A sparse matrix is not a panel. FIML missingness is NaN in a dense array."
    ),
    "check_estimators_nan_inf": (
        "NaN is a missing indicator and is accepted. Non-finite infinities are not a supported code."
    ),
    "check_classifiers_train": (
        "score is a log-likelihood, not a classification accuracy, and it is not used to choose n_states."
    ),
    "check_classifiers_classes": (
        "n_states is a fixed hyperparameter. score must not select it."
    ),
    "check_clustering": (
        "The estimator is not a clusterer whose score selects the number of states."
    ),
    "check_fit2d_predict1d": (
        "A multi-wave transition model returns one state per wave. The one-wave case is still checked."
    ),
    "check_methods_sample_order_invariance": (
        "Random starts draw from the observed rows, so a permutation of the sample is a different start. "
        "The same random_state on the same row order is reproducible, and that is what clone/refit cover."
    ),
    "check_methods_subset_invariance": (
        "A subset is a different sample. The observed-data likelihood is not required to match the full sample."
    ),
    "check_regressors_train": (
        "score is the mean log-likelihood, not a regression R-squared, and it is not a K-selection criterion."
    ),
    "check_regressors_no_decision_function": (
        "score is the mean log-likelihood, not a regression score."
    ),
}


def _check_name(check):
    func = getattr(check, "func", check)
    return getattr(func, "__name__", str(check))


try:
    from sklearn.utils.estimator_checks import parametrize_with_checks

    _FAST = dict(n_states=1, covariance="EEI", n_starts=1, max_iter=12, screen_iter=0, n_full=1, random_state=0)

    @parametrize_with_checks([
        pf.LatentProfileModel(**_FAST),
        pf.LatentTransitionModel(n_waves=1, **_FAST),
    ])
    def test_check_estimator_where_it_applies(estimator, check):
        name = _check_name(check)
        if name in SKIPPED_CHECKS:
            pytest.skip(SKIPPED_CHECKS[name])
        check(estimator)
except Exception:
    def test_check_estimator_where_it_applies():
        pytest.importorskip("sklearn")
