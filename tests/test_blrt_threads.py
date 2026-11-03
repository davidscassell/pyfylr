"""Draw-pool identity. Workers are real threads; the counts must match."""

import math

import numpy as np
import pandas as pd

import pyfylr as pf


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


def _stored_attributes(result):
    try:
        return dict(vars(result))
    except TypeError:
        return {}


def _canonical(result):
    assert type(result) is dict
    assert "telemetry" not in result
    assert "seconds" not in result
    assert "n_threads_used" not in result
    attributes = _stored_attributes(result)
    assert "telemetry" not in attributes
    body = {key: value for key, value in result.items() if key != "n_threads"}
    return body, attributes


def test_bootstrap_draws_agree_for_one_two_four_and_hardware_threads():
    rng = np.random.default_rng(19)
    group = np.repeat([-1.5, 1.5], 20)
    data = pd.DataFrame({
        "y1": group + rng.normal(0, 0.4, size=40),
        "y2": group + rng.normal(0, 0.4, size=40),
    })
    controls = dict(covariance="EEI", n_starts=4, screen_iter=0, n_full=2, max_iter=40, seed=19, n_threads=1)
    h0 = pf.fit_lpa(data, ["y1", "y2"], n_states=1, **controls)
    h1 = pf.fit_lpa(data, ["y1", "y2"], n_states=2, **controls)

    def run(threads):
        out = pf.blrt_check(h0, h1, B=4, max_retries=1, seed=19, max_iter=40, n_threads=threads, experimental=True)
        return out, pf.blrt_last_telemetry()

    one, _ = run(1)
    two, two_tel = run(2)
    four, _ = run(4)
    hw = pf.blrt_hardware_threads()
    cap, _ = run(hw)
    assert hw >= 1
    assert one["n_threads"] == 1
    assert two["n_threads"] == 2
    assert four["n_threads"] == 4
    assert cap["n_threads"] == hw
    assert two_tel["n_threads_used"] == min(2, hw, 4)
    if hw >= 2:
        assert two_tel["n_threads_used"] > 1
    one_body, one_attrs = _canonical(one)
    for other in (two, four, cap):
        body, attributes = _canonical(other)
        assert _same(body, one_body)
        assert attributes == one_attrs
        assert other["p"] == one["p"]
        assert other["b"] == one["b"]
        assert other["lrs"] == one["lrs"]
        assert other["lr_draws"] == one["lr_draws"]
        assert other["n_ok"] == one["n_ok"]
        assert other["n_failed"] == one["n_failed"]
        assert other["n_failed_not_converged"] == one["n_failed_not_converged"]
        assert other["n_failed_inadmissible"] == one["n_failed_inadmissible"]
        assert other["n_failed_error"] == one["n_failed_error"]
        assert other["n_not_replicated"] == one["n_not_replicated"]
        assert other["observed_h0_replicated"] == one["observed_h0_replicated"]
        assert other["observed_h1_replicated"] == one["observed_h1_replicated"]
    assert one["n_ok"] + one["n_failed"] == 4
