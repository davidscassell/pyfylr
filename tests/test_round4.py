"""Round 4 binding checks: identity, labels, bootstrap schema, and threads."""

import math

import numpy as np
import pandas as pd
import pytest

import pyfylr as pf
from pyfylr import selection as sel


def _same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
        return left == right
    return left == right


def _without(result, *keys):
    return {key: value for key, value in result.items() if key not in keys}


def _stored_attributes(result):
    try:
        return dict(vars(result))
    except TypeError:
        return {}


def _canonical_state(result):
    assert type(result) is dict
    assert "telemetry" not in result
    attributes = _stored_attributes(result)
    assert "telemetry" not in attributes
    return _without(result, "n_threads"), attributes


def test_fractional_subject_ids_do_not_collide():
    data = pd.DataFrame({"id": [1.1, 1.2], "y": [10.0, 20.0]})
    panel = pf.prepare_panel(data, ["y"], "id")
    assert panel["N"] == 2
    assert len(panel["ids"]) == 2
    assert panel["gauss"][0] == pytest.approx(10.0)
    assert panel["gauss"][1] == pytest.approx(20.0)


def test_text_wave_labels_follow_first_appearance():
    data = pd.DataFrame({"id": [1, 1], "wave": ["z", "a"], "y": [1.0, 2.0]})
    panel = pf.prepare_panel(data, ["y"], "id", "wave")
    assert list(panel["waves"]) == ["z", "a"]
    assert panel["gauss"][0] == pytest.approx(1.0)
    assert panel["gauss"][1] == pytest.approx(2.0)


def _separated():
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        "y": np.concatenate([rng.normal(-2.0, 0.2, 24), rng.normal(2.0, 0.2, 24)]),
    })


def test_identity_and_inverse_alignment_preserve_the_result():
    fit = pf.fit_lpa(
        _separated(), ["y"], n_states=2, covariance="EEI", n_starts=2, screen_iter=0,
        n_full=2, max_iter=25, seed=4, align_labels=False,
    )
    aligned, info = pf.align_states(fit, fit)
    assert list(info["permutation"]) == [0, 1]
    for key, value in fit.result.items():
        assert _same(aligned.result[key], value), key
    assert aligned.result["entropy_wave_raw"] == fit.result["entropy_wave_raw"]
    ic = pf.information_criteria(fit)
    ic_aligned = pf.information_criteria(aligned)
    if math.isfinite(ic["icl_bic"]) or math.isfinite(ic_aligned["icl_bic"]):
        assert ic_aligned["icl_bic"] == pytest.approx(ic["icl_bic"])

    params = dict(fit.result["params"])
    swapped = {
        "delta": list(reversed(params["delta"])),
        "means": list(reversed(params["means"])),
        "cov": list(params["cov"]),
        "tau": list(params.get("tau") or []),
        "rho": list(params.get("rho") or []),
    }
    moved, moved_info = pf.align_states(fit, swapped)
    assert list(moved_info["permutation"]) == [1, 0]
    restored, restored_info = pf.align_states(moved, fit)
    assert list(restored_info["permutation"]) == [1, 0]
    assert _same(restored.result["params"], fit.result["params"])
    for key in ("seed", "n_threads", "entropy_wave_raw", "entropy_sequence_raw", "viterbi_logprob",
                "eligible_model_comparison", "eligible_assignment", "excluded_subjects"):
        assert _same(moved.result[key], fit.result[key]), key
        assert _same(restored.result[key], fit.result[key]), key


def test_h1_below_h0_refits_the_panel():
    data = pd.DataFrame({"y": np.random.default_rng(0).normal(0, 1, 40)})
    controls = dict(covariance="EEI", n_starts=1, screen_iter=0, n_full=1, seed=1, align_labels=False)
    h0 = pf.fit_lpa(data, ["y"], n_states=1, max_iter=25, tol=1e-8, **controls)
    bad = {"delta": [0.5, 0.5], "means": [-20.0, 20.0], "cov": [1.0]}
    h1 = pf.fit_lpa(data, ["y"], n_states=2, max_iter=2, tol=0.1, start=bad, **controls)
    assert h1.result["loglik"] < h0.result["loglik"] - 1e-4
    out = pf.blrt_check(h0, h1, B=1, max_retries=0, seed=1, max_iter=6, n_threads=1, experimental=True)
    assert "ran" in out
    assert isinstance(out["reason"], str)
    assert "not run: observed alternative is materially below the null" in out["reason"] or out["ran"] is True


def _nominal_pair():
    data = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "y": ["a", "b", "c", "a", "b", "c"]})
    controls = dict(n_starts=1, screen_iter=0, n_full=1, max_iter=4, seed=1, align_labels=False)
    h0 = pf.fit_lca(data, ["y"], id="id", n_states=1, **controls)
    h1 = pf.fit_lca(data, ["y"], id="id", n_states=2, **controls)
    return h0, h1


def test_bootstrap_preparation_keeps_generating_categories():
    h0, _h1 = _nominal_pair()
    order = list(h0.panel["level_order"][0])
    assert len(order) == 3
    only = pd.DataFrame({"id": [1, 2], "y": [order[0], order[0]]})
    panel = sel._prepare_bootstrap_panel(h0, only)
    assert panel["ncat"] == [3]
    ends = pd.DataFrame({"id": [1, 2], "y": [order[0], order[2]]})
    kept = sel._prepare_bootstrap_panel(h0, ends)
    assert kept["ncat"] == [3]
    assert kept["cat"][1] == 2


def test_string_categories_run_through_serial_and_pooled_blrt():
    data = pd.DataFrame({
        "id": list(range(24)),
        "y": (["a"] * 8) + (["b"] * 8) + (["c"] * 8),
    })
    controls = dict(n_starts=2, screen_iter=0, n_full=2, max_iter=20, seed=5, align_labels=False)
    h0 = pf.fit_lca(data, ["y"], id="id", n_states=1, **controls)
    h1 = pf.fit_lca(data, ["y"], id="id", n_states=2, **controls)
    assert all(isinstance(label, str) for label in h0.panel["level_order"][0])
    frame = sel._simulate(h0, np.random.default_rng(5), np.arange(h0.panel["N"]))
    assert set(frame["y"].dropna().astype(str)) <= set(h0.panel["level_order"][0])
    for threads in (1, 2):
        out = pf.blrt_check(
            h0, h1, B=2, max_retries=0, seed=5, max_iter=20, n_threads=threads, experimental=True,
        )
        assert out["n_ok"] >= 1
        assert out["n_ok"] + out["n_failed"] == 2


def test_bindings_reject_malformed_inputs():
    panel = {
        "N": 1, "T": 1, "J": 1, "measurement": "categorical", "ncat": [2], "Rmax": 2,
        "cat": [2], "family": ["bernoulli"],
    }
    spec = {"K": 1, "n_starts": 1, "max_iter": 1, "screen_iter": 0, "n_full": 1,
            "tol": 1e-7, "has_start": True, "start": {"delta": [1.0], "rho": [0.5, 0.5]}}
    with pytest.raises(ValueError, match="ncat"):
        pf._core.fit(panel, spec)
    panel["cat"] = [0]
    spec["start"] = {"delta": [], "rho": [0.5, 0.5]}
    with pytest.raises(ValueError, match="delta"):
        pf._core.fit(panel, spec)
    spec["start"] = {"delta": [1.0], "rho": [0.5, 0.5]}
    spec["max_iter"] = 0
    with pytest.raises(ValueError, match="max_iter"):
        pf._core.fit(panel, spec)
    spec["max_iter"] = 1
    spec["tol"] = float("nan")
    with pytest.raises(ValueError, match="tol"):
        pf._core.fit(panel, spec)


def test_fit_and_blrt_canonical_results_match_across_threads():
    data = _separated()
    controls = dict(covariance="EEI", n_starts=2, screen_iter=0, n_full=2, max_iter=15, seed=8, align_labels=False)
    fits = {
        threads: pf.fit_lpa(data, ["y"], n_states=2, n_threads=threads, **controls)
        for threads in (1, 2, 4)
    }
    for threads, fit in fits.items():
        assert fit.result["n_threads"] == threads
        body, attributes = _canonical_state(fit.result)
        ref_body, ref_attributes = _canonical_state(fits[1].result)
        assert _same(body, ref_body)
        assert attributes == ref_attributes
    h0 = pf.fit_lpa(data, ["y"], n_states=1, n_threads=1, **controls)
    h1 = fits[1]
    checks = {}
    telemetry = {}
    for threads in (1, 2, 4):
        checks[threads] = pf.blrt_check(
            h0, h1, B=2, max_retries=0, seed=8, max_iter=12, n_threads=threads, experimental=True
        )
        telemetry[threads] = pf.blrt_last_telemetry()
    for threads, out in checks.items():
        assert out["n_threads"] == threads
        assert "seconds" not in out
        assert "n_threads_used" not in out
        assert "Elapsed" not in out["reason"]
        assert telemetry[threads]["n_threads_used"] == min(threads, pf.blrt_hardware_threads(), 2)
        body, attributes = _canonical_state(out)
        ref_body, ref_attributes = _canonical_state(checks[1])
        assert _same(body, ref_body)
        assert attributes == ref_attributes
