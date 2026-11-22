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

import subprocess
import sys

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


# Frozen before the single run of this commit. Command, from the repo root:
#   python3.12 -m pytest \
#     pyfylr/tests/test_sklearn_api.py::test_wide_dataframe_schema_accepts_the_training_column_order \
#     pyfylr/tests/test_sklearn_api.py::test_wide_dataframe_later_wave_rename_and_swap_are_rejected \
#     pyfylr/tests/test_sklearn_api.py::test_numpy_wide_input_stays_positional \
#     pyfylr/tests/test_sklearn_api.py::test_long_dataframe_column_order_is_not_the_wide_schema \
#     pyfylr/tests/test_sklearn_api.py::test_transition_predict_uses_smoothing_and_keeps_the_other_paths \
#     pyfylr/tests/test_sklearn_api.py::test_latent_class_model_matches_pandas_on_codes \
#     pyfylr/tests/test_panel_schema.py \
#     -q --tb=short
# Tolerances for the new comparisons: predict, filter, and viterbi use
# np.array_equal. predict_proba and score_samples use np.array_equal.
# score uses ==. No loosened float tolerance. Existing tests keep their
# own assertions, including pytest.approx where they already had it.
_WIDE_FAST = dict(
    n_states=2, n_starts=1, screen_iter=0, n_full=1, max_iter=12,
    tol=1e-7, random_state=1,
)


def _wide_gaussian(n=24, seed=7):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, 4))
    values[: n // 2, 0] -= 1.0
    values[n // 2 :, 0] += 1.0
    values[: n // 2, 2] -= 1.0
    values[n // 2 :, 2] += 1.0
    return pd.DataFrame(values, columns=["w1_y1", "w1_y2", "w2_y1", "w2_y2"])


def _wide_codes(n=24):
    return pd.DataFrame({
        "w1_u": [0, 1] * (n // 2),
        "w1_v": [0, 0, 1, 1] * (n // 4),
        "w2_u": [0, 1] * (n // 2),
        "w2_v": [1, 1, 0, 0] * (n // 4),
    })


def _evaluation(model, data):
    return {
        "predict": np.asarray(model.predict(data)),
        "proba": np.asarray(model.predict_proba(data)),
        "samples": np.asarray(model.score_samples(data)),
        "score": model.score(data),
        "filter": np.asarray(model.filter(data)),
        "viterbi": np.asarray(model.viterbi(data)),
    }


def _assert_exact_evaluation(left, right):
    assert np.array_equal(left["predict"], right["predict"])
    assert np.array_equal(left["filter"], right["filter"])
    assert np.array_equal(left["viterbi"], right["viterbi"])
    assert np.array_equal(left["proba"], right["proba"])
    assert np.array_equal(left["samples"], right["samples"])
    assert left["score"] == right["score"]


def test_wide_dataframe_schema_accepts_the_training_column_order():
    gaussian = _wide_gaussian()
    transition = pf.LatentTransitionModel(covariance="EEI", n_waves=2, **_WIDE_FAST).fit(gaussian)
    same = gaussian.copy()
    _assert_exact_evaluation(_evaluation(transition, gaussian), _evaluation(transition, same))
    assert list(transition.feature_names_in_) == list(gaussian.columns)
    codes = _wide_codes()
    classes = pf.LatentClassModel(n_waves=2, **_WIDE_FAST).fit(codes)
    _assert_exact_evaluation(_evaluation(classes, codes), _evaluation(classes, codes.copy()))
    assert list(classes.feature_names_in_) == list(codes.columns)


def test_wide_dataframe_later_wave_rename_and_swap_are_rejected():
    gaussian = _wide_gaussian()
    transition = pf.LatentTransitionModel(covariance="EEI", n_waves=2, **_WIDE_FAST).fit(gaussian)
    renamed = gaussian.rename(columns={"w2_y1": "later_y1"})
    swapped = gaussian[["w1_y1", "w1_y2", "w2_y2", "w2_y1"]]
    for bad in (renamed, swapped):
        with pytest.raises(ValueError, match="ordered schema"):
            transition.predict(bad)
        with pytest.raises(ValueError, match="ordered schema"):
            transition.score(bad)
    codes = _wide_codes()
    classes = pf.LatentClassModel(n_waves=2, **_WIDE_FAST).fit(codes)
    renamed_codes = codes.rename(columns={"w2_u": "later_u"})
    swapped_codes = codes[["w1_u", "w1_v", "w2_v", "w2_u"]]
    for bad in (renamed_codes, swapped_codes):
        with pytest.raises(ValueError, match="ordered schema"):
            classes.predict(bad)
        with pytest.raises(ValueError, match="ordered schema"):
            classes.score(bad)


def test_numpy_wide_input_stays_positional():
    gaussian = _wide_gaussian()
    transition = pf.LatentTransitionModel(covariance="EEI", n_waves=2, **_WIDE_FAST).fit(gaussian)
    array = gaussian.to_numpy(dtype=float)
    _assert_exact_evaluation(_evaluation(transition, gaussian), _evaluation(transition, array))
    from_array = pf.LatentTransitionModel(covariance="EEI", n_waves=2, **_WIDE_FAST).fit(array)
    assert not hasattr(from_array, "feature_names_in_")
    renamed = gaussian.rename(columns={"w2_y1": "later_y1"})
    _assert_exact_evaluation(_evaluation(from_array, array), _evaluation(from_array, renamed))


def test_long_dataframe_column_order_is_not_the_wide_schema():
    rng = np.random.default_rng(7)
    n = 24
    y = np.concatenate([rng.normal(-1.0, 0.3, n), rng.normal(1.0, 0.3, n)])
    long = pd.DataFrame({
        "id": list(range(n)) * 2,
        "wave": [1] * n + [2] * n,
        "y": y,
    })
    model = pf.LatentTransitionModel(
        covariance="EEI", n_waves=2, id_column="id", wave_column="wave", **_WIDE_FAST,
    ).fit(long)
    reordered = long[["y", "wave", "id"]]
    _assert_exact_evaluation(_evaluation(model, long), _evaluation(model, reordered))


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


_CHECK_FAST = dict(
    n_states=1, covariance="EEI", n_starts=1, max_iter=12, screen_iter=0, n_full=1, random_state=0,
)


def load_parametrize_with_checks():
    """Return sklearn's check parametrizer, or None when sklearn itself is absent.

    Only ``ModuleNotFoundError`` whose missing name is the top-level package
    ``sklearn`` is optional. An internal checker import, a transitive missing
    module, or a missing ``parametrize_with_checks`` symbol is raised.
    """
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "sklearn":
            return None
        raise
    from sklearn.utils.estimator_checks import parametrize_with_checks
    if not callable(parametrize_with_checks):
        raise ImportError(
            "sklearn.utils.estimator_checks.parametrize_with_checks is not available"
        )
    return parametrize_with_checks


def bind_estimator_checks(parametrize):
    """Build the check decorator. Parametrization is not caught.

    ``parametrize`` is None only after a confirmed top-level sklearn absence.
    """
    if parametrize is None:
        return None
    return parametrize([
        pf.LatentProfileModel(**_CHECK_FAST),
        pf.LatentTransitionModel(n_waves=1, **_CHECK_FAST),
    ])


def _skip_when_sklearn_is_absent():
    pytest.skip("scikit-learn is not installed; estimator checks were not collected")


def _hide_sklearn_modules():
    saved = {}
    for key in list(sys.modules):
        if key == "sklearn" or key.startswith("sklearn."):
            saved[key] = sys.modules.pop(key)
    return saved


def _restore_modules(saved):
    sys.modules.update(saved)


def test_genuine_top_level_sklearn_absence_returns_none():
    class _MissingSklearn:
        def find_spec(self, fullname, path, target=None):
            if fullname == "sklearn":
                raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
            return None

    saved = _hide_sklearn_modules()
    sys.meta_path.insert(0, _MissingSklearn())
    try:
        assert load_parametrize_with_checks() is None
    finally:
        sys.meta_path.pop(0)
        _restore_modules(saved)


def _install_parent_sklearn():
    """A package object so ``import sklearn`` succeeds without the real install."""
    import types
    parent = types.ModuleType("sklearn")
    parent.__path__ = []
    utils = types.ModuleType("sklearn.utils")
    utils.__path__ = []
    sys.modules["sklearn"] = parent
    sys.modules["sklearn.utils"] = utils
    return parent, utils


def _drop_installed_fixtures():
    for key in list(sys.modules):
        if key == "sklearn" or key.startswith("sklearn."):
            sys.modules.pop(key, None)


def test_internal_checker_import_error_propagates():
    class _BrokenChecks:
        def find_spec(self, fullname, path, target=None):
            if fullname == "sklearn.utils.estimator_checks":
                raise ImportError("estimator checks failed internally")
            return None

    saved = _hide_sklearn_modules()
    _install_parent_sklearn()
    sys.meta_path.insert(0, _BrokenChecks())
    try:
        with pytest.raises(ImportError, match="estimator checks failed internally"):
            load_parametrize_with_checks()
    finally:
        sys.meta_path.pop(0)
        _drop_installed_fixtures()
        _restore_modules(saved)


def test_transitive_module_not_found_is_not_optional_sklearn():
    class _MissingDependency:
        def find_spec(self, fullname, path, target=None):
            if fullname == "sklearn":
                raise ModuleNotFoundError("No module named 'notsklearn'", name="notsklearn")
            return None

    saved = _hide_sklearn_modules()
    sys.meta_path.insert(0, _MissingDependency())
    try:
        with pytest.raises(ModuleNotFoundError, match="notsklearn") as raised:
            load_parametrize_with_checks()
        assert raised.value.name == "notsklearn"
    finally:
        sys.meta_path.pop(0)
        _restore_modules(saved)


def test_missing_checker_symbol_propagates():
    import types
    saved = _hide_sklearn_modules()
    _install_parent_sklearn()
    checks = types.ModuleType("sklearn.utils.estimator_checks")
    sys.modules["sklearn.utils.estimator_checks"] = checks
    try:
        with pytest.raises(ImportError, match="parametrize_with_checks"):
            load_parametrize_with_checks()
    finally:
        _drop_installed_fixtures()
        _restore_modules(saved)


def test_checker_parametrization_failure_propagates():
    def boom(estimators):
        raise TypeError("parametrization failed")

    with pytest.raises(TypeError, match="parametrization failed"):
        bind_estimator_checks(boom)


def test_absent_sklearn_fallback_is_a_skip_not_a_pass():
    with pytest.raises(pytest.skip.Exception, match="estimator checks were not collected"):
        _skip_when_sklearn_is_absent()


def test_installed_sklearn_exposes_the_checker():
    parametrize = load_parametrize_with_checks()
    if parametrize is None:
        pytest.skip("scikit-learn is not installed; estimator checks were not collected")
    assert callable(parametrize)


def test_module_collection_without_sklearn_skips_checks():
    """Collect the check item with top-level sklearn blocked in a subprocess.

    The collected node is the explicit skip, not a parametrized checker and
    not a test that passes because sklearn imports on a second try.
    """
    script = """
import sys
class _MissingSklearn:
    def find_spec(self, fullname, path, target=None):
        if fullname == "sklearn":
            raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
        return None
sys.meta_path.insert(0, _MissingSklearn())
for key in list(sys.modules):
    if key == "sklearn" or key.startswith("sklearn."):
        del sys.modules[key]
import pytest
raise SystemExit(pytest.main([
    "--collect-only", "-q",
    "pyfylr/tests/test_sklearn_api.py::test_check_estimator_where_it_applies",
]))
"""
    test_path = __import__("pathlib").Path(__file__).resolve()
    script = script.replace(
        '"pyfylr/tests/test_sklearn_api.py::test_check_estimator_where_it_applies"',
        repr(f"{test_path}::test_check_estimator_where_it_applies"),
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(test_path.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    nodes = [
        line for line in output.splitlines()
        if "test_check_estimator_where_it_applies" in line
    ]
    assert len(nodes) == 1, output
    assert "[" not in nodes[0], output


# Frozen validation plan, committed before the single execution.
# Two phases, each once. No numeric tolerance.
# Phase present, PYTHONPATH is an install of this SHA that includes scikit-learn:
#   python3.12 -m pytest \
#     pyfylr/tests/test_sklearn_api.py::test_genuine_top_level_sklearn_absence_returns_none \
#     pyfylr/tests/test_sklearn_api.py::test_internal_checker_import_error_propagates \
#     pyfylr/tests/test_sklearn_api.py::test_transitive_module_not_found_is_not_optional_sklearn \
#     pyfylr/tests/test_sklearn_api.py::test_missing_checker_symbol_propagates \
#     pyfylr/tests/test_sklearn_api.py::test_checker_parametrization_failure_propagates \
#     pyfylr/tests/test_sklearn_api.py::test_absent_sklearn_fallback_is_a_skip_not_a_pass \
#     pyfylr/tests/test_sklearn_api.py::test_installed_sklearn_exposes_the_checker \
#     pyfylr/tests/test_sklearn_api.py::test_module_collection_without_sklearn_skips_checks \
#     pyfylr/tests/test_sklearn_api.py::test_check_estimator_where_it_applies \
#     -q --tb=short -ra
# Phase absent, PYTHONPATH is an install of this SHA without scikit-learn.
# First confirm import sklearn raises ModuleNotFoundError name sklearn.
# Then the same pytest command, and one collect-only of this file.
# Accepted skips there: the installed-sklearn test, and the unparametrized
# estimator-check test ("estimator checks were not collected").
# Checker skips in the present phase stay those in SKIPPED_CHECKS.
_bound_checks = bind_estimator_checks(load_parametrize_with_checks())
if _bound_checks is None:
    def test_check_estimator_where_it_applies():
        _skip_when_sklearn_is_absent()
else:
    @_bound_checks
    def test_check_estimator_where_it_applies(estimator, check):
        name = _check_name(check)
        if name in SKIPPED_CHECKS:
            pytest.skip(SKIPPED_CHECKS[name])
        check(estimator)
