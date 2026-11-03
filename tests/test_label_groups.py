"""Fixed-parameter label order through the Python binding.

``_core.evaluate`` applies the connected near-tie rule and does not run an
M-step, so the supplied weights stay the weights that are ordered. These
tests do not use the fit thread pool. The numbers are the core fixture in
``test_connected_near_tie_groups``. A separate test fits with more starts
than threads, which is what gives the worker pool more than one worker.
"""

import itertools

import numpy as np
import pandas as pd

import pyfylr as pf


def _gaussian_panel():
    data = pd.DataFrame({"id": [1, 1], "wave": [1, 2], "y": [-1.2, 0.7]})
    return pf.prepare_panel(data, ["y"], "id", "wave", family={"y": "gaussian"})


def _categorical_panel():
    rows = [{"id": i, "wave": wave, "u": code} for i, code in enumerate((0, 1, 2)) for wave in (1, 2)]
    data = pd.DataFrame(rows)
    return pf.prepare_panel(data, ["u"], "id", "wave", family={"u": "nominal"})


def _evaluate(panel, spec, params):
    spec = dict(spec)
    spec["align_labels"] = True
    return pf._core.evaluate(panel, spec, params)["params"]


def _chain_delta():
    offsets = np.array([1.35e-8, 0.45e-8, -0.45e-8, -1.35e-8])
    delta = np.empty(4)
    delta[:3] = 0.25 + offsets[:3]
    delta[3] = 1.0 - delta[:3].sum()
    return delta


def _chain_tau():
    raw = np.array([[(i + 1) * 10 + (j + 1) for j in range(4)] for i in range(4)], dtype=float)
    raw /= raw.sum(axis=1, keepdims=True)
    return raw


def _apply(values, perm):
    return np.asarray(values, dtype=float)[list(perm)]


def _apply_matrix(matrix, perm):
    order = list(perm)
    return np.asarray(matrix, dtype=float)[np.ix_(order, order)]


def _flat_rho(rows, perm):
    return np.asarray(rows, dtype=float)[list(perm)].reshape(-1)


def test_boundary_join_and_split_keep_the_expected_order():
    panel = _gaussian_panel()
    spec = pf.lta_spec(2, covariance="VVI", align_labels=True)
    join_gap = 1e-8
    split_gap = 1e-8 + 1e-12
    join_lo, join_hi = 0.5 - join_gap / 2.0, 0.5 + join_gap / 2.0
    split_lo, split_hi = 0.5 - split_gap / 2.0, 0.5 + split_gap / 2.0
    assert abs(join_hi - join_lo) <= 1e-8
    assert abs(split_hi - split_lo) > 1e-8
    assert join_lo + join_hi == 1.0 and split_lo + split_hi == 1.0
    base_tau = np.array([[0.8, 0.2], [0.3, 0.7]])
    means = np.array([1.0, -1.0])
    cov = np.array([0.2, 0.4])

    def params(low, high, perm):
        return {
            "delta": _apply([low, high], perm).tolist(),
            "means": _apply(means, perm).tolist(),
            "cov": _apply(cov, perm).tolist(),
            "tau": _apply_matrix(base_tau, perm).reshape(-1).tolist(),
        }

    joined = _evaluate(panel, spec, params(join_lo, join_hi, (0, 1)))
    assert np.array_equal(joined["means"], [1.0, -1.0])
    assert np.array_equal(joined["delta"], [join_lo, join_hi])
    assert np.array_equal(joined["cov"], [0.2, 0.4])
    assert np.array_equal(joined["tau"], [0.8, 0.2, 0.3, 0.7])
    assert joined["delta"][0] < joined["delta"][1]
    for perm in ((0, 1), (1, 0)):
        got = _evaluate(panel, spec, params(join_lo, join_hi, perm))
        assert np.array_equal(got["delta"], joined["delta"])
        assert np.array_equal(got["means"], joined["means"])
        assert np.array_equal(got["cov"], joined["cov"])
        assert np.array_equal(got["tau"], joined["tau"])

    split = _evaluate(panel, spec, params(split_lo, split_hi, (0, 1)))
    assert np.array_equal(split["means"], [-1.0, 1.0])
    assert np.array_equal(split["delta"], [split_hi, split_lo])
    assert np.array_equal(split["cov"], [0.4, 0.2])
    assert np.array_equal(split["tau"], [0.7, 0.3, 0.2, 0.8])
    for perm in ((0, 1), (1, 0)):
        got = _evaluate(panel, spec, params(split_lo, split_hi, perm))
        assert np.array_equal(got["delta"], split["delta"])
        assert np.array_equal(got["tau"], split["tau"])


def test_long_chain_uses_the_core_fixture_and_profile_order():
    panel = _gaussian_panel()
    spec = pf.lta_spec(4, covariance="VVI", align_labels=True)
    delta = _chain_delta()
    means = np.array([0.0, 1.0, 2.0, 3.0])
    cov = np.array([0.1, 0.2, 0.3, 0.4])
    tau = _chain_tau()
    order = [3, 2, 1, 0]
    expected_delta = delta[order]
    expected_tau = tau[np.ix_(order, order)].reshape(-1)
    assert expected_delta[0] < expected_delta[3]
    assert delta.max() - delta.min() > 1e-8
    assert delta.max() - delta.min() <= 3 * 1e-8
    reference = None
    for perm in itertools.permutations(range(4)):
        got = _evaluate(panel, spec, {
            "delta": _apply(delta, perm).tolist(),
            "means": _apply(means, perm).tolist(),
            "cov": _apply(cov, perm).tolist(),
            "tau": _apply_matrix(tau, perm).reshape(-1).tolist(),
        })
        assert np.array_equal(got["means"], [3.0, 2.0, 1.0, 0.0])
        assert np.array_equal(got["cov"], [0.4, 0.3, 0.2, 0.1])
        assert np.array_equal(got["delta"], expected_delta)
        assert np.array_equal(got["tau"], expected_tau)
        if reference is None:
            reference = got
        else:
            assert np.array_equal(got["tau"], reference["tau"])


def test_equal_profiles_keep_the_incoming_index():
    panel = _gaussian_panel()
    spec = pf.lta_spec(3, covariance="EEI", align_labels=True)
    hi, mid, lo = 1.0 / 3.0 + 0.75e-8, 1.0 / 3.0, 1.0 / 3.0 - 0.75e-8
    delta = np.array([hi, mid, lo])
    tau = np.array([
        [0.50, 0.30, 0.20],
        [0.20, 0.50, 0.30],
        [0.10, 0.20, 0.70],
    ])
    same = _evaluate(panel, spec, {
        "delta": delta.tolist(),
        "means": [0.0, 0.0, 0.0],
        "cov": [1.0],
        "tau": tau.reshape(-1).tolist(),
    })
    assert np.array_equal(same["delta"], delta)
    assert np.array_equal(same["tau"], tau.reshape(-1))
    flipped = _evaluate(panel, spec, {
        "delta": delta[::-1].tolist(),
        "means": [0.0, 0.0, 0.0],
        "cov": [1.0],
        "tau": _apply_matrix(tau, (2, 1, 0)).reshape(-1).tolist(),
    })
    assert np.array_equal(flipped["delta"], delta[::-1])
    assert not np.array_equal(flipped["delta"], same["delta"])
    mixed = _evaluate(panel, spec, {
        "delta": delta.tolist(),
        "means": [0.0, 0.0, 5.0],
        "cov": [1.0],
        "tau": tau.reshape(-1).tolist(),
    })
    assert np.array_equal(mixed["means"], [5.0, 0.0, 0.0])
    assert np.array_equal(mixed["delta"], [lo, hi, mid])


def test_categorical_permutations_follow_the_highest_category():
    panel = _categorical_panel()
    spec = pf.lca_spec(3, align_labels=True)
    hi, mid, lo = 1.0 / 3.0 + 0.75e-8, 1.0 / 3.0, 1.0 / 3.0 - 0.75e-8
    delta = np.array([hi, mid, lo])
    rho = np.array([
        [0.20, 0.70, 0.10],
        [0.20, 0.30, 0.50],
        [0.05, 0.05, 0.90],
    ])
    tau = np.array([
        [0.50, 0.30, 0.20],
        [0.20, 0.50, 0.30],
        [0.10, 0.20, 0.70],
    ])
    order = [2, 1, 0]
    expected_rho = rho[order].reshape(-1)
    expected_tau = tau[np.ix_(order, order)].reshape(-1)
    assert expected_rho[2] == 0.90 and expected_rho[5] == 0.50 and expected_rho[8] == 0.10
    for perm in itertools.permutations(range(3)):
        got = _evaluate(panel, spec, {
            "delta": _apply(delta, perm).tolist(),
            "rho": _flat_rho(rho, perm).tolist(),
            "tau": _apply_matrix(tau, perm).reshape(-1).tolist(),
        })
        assert np.array_equal(got["delta"], delta[order])
        assert np.array_equal(got["rho"], expected_rho)
        assert np.array_equal(got["tau"], expected_tau)


def test_multiple_fit_workers_match_when_starts_exceed_threads():
    rng = np.random.default_rng(3)
    group = np.repeat([-1.5, 1.5], 16)
    data = pd.DataFrame({"y": group + rng.normal(0, 0.35, size=32)})
    controls = dict(
        covariance="EEI", n_starts=6, screen_iter=0, n_full=4, max_iter=4, seed=3, align_labels=True
    )
    fits = {
        threads: pf.fit_lpa(data, ["y"], n_states=2, n_threads=threads, **controls)
        for threads in (1, 2, 4)
    }
    for threads, fit in fits.items():
        assert threads < 6
        assert fit.result["n_threads"] == threads
        assert fit.result["n_starts"] == 6
        assert np.array_equal(fit.result["params"]["means"], fits[1].result["params"]["means"])
        assert np.array_equal(fit.result["params"]["delta"], fits[1].result["params"]["delta"])
        assert np.array_equal(fit.result["params"]["cov"], fits[1].result["params"]["cov"])
        assert fit.result["loglik"] == fits[1].result["loglik"]
