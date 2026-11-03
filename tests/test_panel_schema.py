"""Training schema, duplicate keys, wave schedules, and impossible scores.

These checks are not thread-identity checks. A one-thread and a multi-thread
fit can agree and still rebuild categories from the prediction rows.
"""

import math
import struct

import numpy as np
import pandas as pd
import pytest

import pyfylr as pf


def _train():
    rows = []
    labels = ["a", "b", "c"]
    for i, label in enumerate(labels * 4):
        rows.append({"id": i + 1, "wave": 1, "u": label})
    return pd.DataFrame(rows)


def test_prediction_reuses_training_category_order():
    data = _train()
    trained = pf.prepare_panel(data, ["u"], "id", "wave", family={"u": "nominal"})
    assert list(trained["level_order"][0]) == ["a", "b", "c"]
    subset = data[data["u"] != "b"].iloc[::-1].reset_index(drop=True)
    predicted = pf.prepare_panel(subset, ["u"], "id", "wave", family={"u": "nominal"}, schema=trained)
    codes = [code for code in predicted["cat"] if code >= 0]
    assert set(codes) == {0, 2}
    assert 1 not in codes
    one = data[data["u"] == "b"]
    single = pf.prepare_panel(one, ["u"], "id", "wave", family={"u": "nominal"}, schema=trained)
    assert set(code for code in single["cat"] if code >= 0) == {1}
    unseen = data.copy()
    unseen.loc[0, "u"] = "z"
    with pytest.raises(ValueError, match="unseen category"):
        pf.prepare_panel(unseen, ["u"], "id", "wave", family={"u": "nominal"}, schema=trained)


def test_fitted_predict_rejects_an_unseen_category_and_accepts_a_subset():
    data = _train()
    model = pf.LatentClassModel(
        n_states=2, n_starts=2, max_iter=15, screen_iter=0, n_full=2, random_state=1, family={"u": "nominal"}
    ).fit(data[["u"]])
    subset = data.loc[data["u"] != "c", ["u"]]
    labels = model.predict(subset)
    assert len(labels) == len(subset)
    reordered = data[["u"]].iloc[::-1]
    assert len(model.predict(reordered)) == len(reordered)
    one = data.loc[data["u"] == "a", ["u"]]
    assert len(model.predict(one)) == len(one)
    bad = data[["u"]].copy()
    bad.loc[0, "u"] = "z"
    with pytest.raises(ValueError, match="unseen category"):
        model.predict(bad)


def test_duplicate_and_missing_keys_are_rejected():
    data = _train()
    dup = pd.concat([data.iloc[[0]], data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate subject-wave key"):
        pf.prepare_panel(dup, ["u"], "id", "wave")
    missing = data.copy()
    missing.loc[0, "id"] = np.nan
    with pytest.raises(ValueError, match="missing key"):
        pf.prepare_panel(missing, ["u"], "id", "wave")
    with pytest.raises(ValueError, match="missing key column"):
        pf.prepare_panel(data, ["u"], "person", "wave")


def test_fractional_and_duplicate_schedules_are_rejected():
    data = pd.DataFrame({"id": [1, 1], "wave": [1, 3], "u": [0, 1]})
    with pytest.raises(ValueError, match="duplicate schedule entry"):
        pf.prepare_panel(data, ["u"], "id", "wave", schedule=[1, 2, 2])
    with pytest.raises(ValueError, match="fractional schedules are rejected"):
        pf.prepare_panel(data, ["u"], "id", "wave", schedule=[1, 1.5])
    fractional = data.copy()
    fractional["wave"] = [1.0, 1.5]
    with pytest.raises(ValueError, match="fractional schedules are rejected"):
        pf.prepare_panel(fractional, ["u"], "id", "wave")
    outside = data.copy()
    outside["wave"] = [1, 4]
    with pytest.raises(ValueError, match="not in the schedule"):
        pf.prepare_panel(outside, ["u"], "id", "wave", schedule=[1, 2, 3])


def test_numpy_buffer_matches_the_list_panel():
    data = pd.DataFrame({"id": [1, 1, 2, 2], "wave": [1, 2, 1, 2], "y": [0.2, -0.4, 1.1, 0.7]})
    panel = pf.prepare_panel(data, ["y"], "id", "wave", family={"y": "gaussian"})
    spec = pf.lpa_spec(1, covariance="EEI", n_starts=1, align_labels=False)
    params = {"delta": [1.0], "tau": [1.0], "means": [0.0], "cov": [1.0]}
    listed = pf._core.observed_loglik(panel, spec, params)
    base = np.asarray(panel["gauss"], dtype=float)
    copied = dict(panel)
    copied["gauss"] = base
    assert pf._core.observed_loglik(copied, spec, params) == listed
    ordered = np.array([-2.0, 2.0], dtype=float)
    order_panel = {
        "N": 1, "T": 2, "J": 1, "measurement": "gaussian",
        "family": ["gaussian"], "ncat": [0], "Rmax": 1, "cat": None, "gauss": ordered,
    }
    order_spec = pf.lta_spec(2, covariance="EEI", n_starts=1, align_labels=False)
    order_params = {"delta": [0.9, 0.1], "tau": [0.95, 0.05, 0.2, 0.8], "means": [-2.0, 2.0], "cov": [0.25]}
    forward = pf._core.observed_loglik(order_panel, order_spec, order_params)
    flipped = ordered[::-1]
    assert not flipped.flags.c_contiguous
    order_panel["gauss"] = flipped
    backward = pf._core.observed_loglik(order_panel, order_spec, order_params)
    order_panel["gauss"] = np.ascontiguousarray(flipped)
    assert backward == pf._core.observed_loglik(order_panel, order_spec, order_params)
    assert backward != forward
    copied["gauss"] = np.repeat(base, 2)[::2]
    assert not copied["gauss"].flags.c_contiguous
    assert pf._core.observed_loglik(copied, spec, params) == listed
    copied["gauss"] = base.astype(np.float32)
    float32_ll = pf._core.observed_loglik(copied, spec, params)
    copied["gauss"] = base.astype(np.float32).astype(np.float64)
    assert float32_ll == pf._core.observed_loglik(copied, spec, params)
    owned = base.copy()
    copied["gauss"] = owned
    before = pf._core.observed_loglik(copied, spec, params)
    owned[:] = 99.0
    copied["gauss"] = base
    assert pf._core.observed_loglik(copied, spec, params) == before


def _bits(value):
    return struct.pack("<d", float(value))


def test_numpy_views_match_logical_order_for_a_sticky_chain():
    """Two-state sticky transitions: order changes the likelihood."""
    spec = pf.lta_spec(2, covariance="EEI", n_starts=1, align_labels=False)
    params = {
        "delta": [0.8, 0.2],
        "tau": [0.95, 0.05, 0.15, 0.85],
        "means": [-2.0, 2.0],
        "cov": [0.25],
    }
    values = np.array([-2.0, -1.5, 1.6, 2.1], dtype=np.float64)

    def likelihood(gauss):
        panel = {
            "N": 1, "T": 4, "J": 1, "measurement": "gaussian",
            "family": ["gaussian"], "ncat": [0], "Rmax": 1, "cat": None,
            "gauss": gauss,
        }
        return pf._core.observed_loglik(panel, spec, params)

    forward = likelihood(values)
    flipped = values[::-1]
    assert not flipped.flags.c_contiguous
    backward = likelihood(flipped)
    assert _bits(backward) == _bits(likelihood(np.ascontiguousarray(flipped)))
    assert backward != forward
    stepped_source = np.array([-2.0, 9.0, -1.5, 9.0, 1.6, 9.0, 2.1, 9.0])
    stepped = stepped_source[::2]
    assert not stepped.flags.c_contiguous
    assert _bits(likelihood(stepped)) == _bits(likelihood(np.ascontiguousarray(stepped)))
    square = np.array(values.reshape(2, 2), order="F")
    assert square.flags.f_contiguous and not square.flags.c_contiguous
    logical = np.ascontiguousarray(square).reshape(-1)
    assert _bits(likelihood(square)) == _bits(likelihood(logical))
    assert _bits(likelihood(np.ascontiguousarray(values))) == _bits(forward)


def test_integer_waves_expand_the_closed_interval():
    data = pd.DataFrame({"id": [1, 1], "wave": [1.0, 3.0], "u": [0, 1]})
    panel = pf.prepare_panel(data, ["u"], "id", "wave", family={"u": "bernoulli"})
    assert list(panel["waves"]) == [1, 2, 3]
    assert panel["T"] == 3


def test_impossible_likelihood_stays_in_the_score():
    panel = {
        "N": 3, "T": 1, "J": 1, "measurement": "categorical",
        "family": ["bernoulli"], "ncat": [2], "Rmax": 2,
        "cat": [0, -1, 1], "gauss": None,
        "ids": [1, 2, 3], "waves": [1], "indicators": ["u"],
    }
    spec = pf.lca_spec(1, n_starts=1, align_labels=False)
    params = {"delta": [1.0], "tau": [], "rho": [1.0, 0.0]}
    samples = np.asarray(pf._core.subject_loglik(panel, spec, params), dtype=float)
    assert np.isfinite(samples[0])
    assert math.isnan(samples[1])
    assert samples[2] == -np.inf
    total = pf._core.observed_loglik(panel, spec, params)
    assert total == -np.inf

    data = _train()[["u"]]
    model = pf.LatentClassModel(
        n_states=1, n_starts=1, max_iter=5, screen_iter=0, n_full=1, random_state=1, family={"u": "nominal"}
    ).fit(data)
    model.score_samples = lambda X: np.array([samples[0], np.nan, -np.inf])
    assert model.score(data) == -np.inf
    # -2 * (-inf) is +inf. A finite AIC would mean the impossible subject was dropped.
    assert model.aic(data) == np.inf
    assert model.bic(data) == np.inf
