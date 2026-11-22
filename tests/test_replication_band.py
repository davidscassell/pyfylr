import numpy as np
import pandas as pd

import pyfylr as pf
from pyfylr.selection import _blrt_solution

# Recorded best and second log-likelihoods from the e8107d5 diagnostic.
# The absolute gaps are about 0.007.
RECORDED = (
    (-19133.760102125754, -19133.767117735992),
    (-191327.57629785853, -191327.58358255043),
)


def test_recorded_pairs_replicate_and_a_half_unit_gap_does_not():
    spec = pf.lta_spec(2)
    assert spec["replicate_tol"] == 0.1
    for best, second in RECORDED:
        assert abs(best - second) > 1e-3
        assert pf.logliks_replicate(best, second) is True
    assert pf.logliks_replicate(-10.0, -10.5) is False
    assert pf.logliks_replicate(float("nan"), -10.0) is False
    assert pf.logliks_replicate(-10.0, -10.05, replicate_tol=0.01) is False


def test_api_and_sklearn_default_is_user_settable():
    assert pf.lca_spec(2)["replicate_tol"] == 0.1
    assert pf.lpa_spec(2)["replicate_tol"] == 0.1
    assert pf.lta_spec(2, replicate_tol=0.01)["replicate_tol"] == 0.01
    for cls in (pf.LatentClassModel, pf.LatentProfileModel, pf.LatentTransitionModel):
        model = cls()
        assert model.replicate_tol == 0.1
        assert model.get_params()["replicate_tol"] == 0.1
    pinned = pf.LatentClassModel(replicate_tol=0.01)
    assert pinned.get_params()["replicate_tol"] == 0.01


def _stream(n):
    rng = np.random.Generator(np.random.PCG64(20261112))
    klass = rng.random(1_000_000) >= 0.5
    prob = np.where(klass, 0.8, 0.2)
    codes = (rng.random((1_000_000, 3)) < prob[:, None]).astype(np.int64)
    return pd.DataFrame({
        "id": np.arange(n),
        "y1": codes[:n, 0],
        "y2": codes[:n, 1],
        "y3": codes[:n, 2],
    })


def _fit(frame, replicate_tol=None):
    kwargs = {"n_states": 2, "seed": 1}
    if replicate_tol is not None:
        kwargs["replicate_tol"] = replicate_tol
    return pf.fit_lca(frame, ["y1", "y2", "y3"], id="id", **kwargs)


def _arrays(fit):
    result = fit.result
    params = result["params"]
    return {
        "loglik": np.float64(result["loglik"]),
        "gamma": np.asarray(result["gamma"], dtype=np.float64),
        "delta": np.asarray(params["delta"], dtype=np.float64),
        "rho": np.asarray(params["rho"], dtype=np.float64),
        "start_loglik": np.asarray(result["start_loglik"], dtype=np.float64),
    }


def _same_arrays(left, right):
    for key, value in _arrays(left).items():
        other = _arrays(right)[key]
        assert value.shape == other.shape
        assert np.array_equal(value, other, equal_nan=True), key


def test_n1000_arrays_match_the_tighter_tolerance():
    frame = _stream(1000)
    default = _fit(frame)
    tight = _fit(frame, replicate_tol=1e-3)
    _same_arrays(default, tight)
    assert int(default.result["n_best"]) >= int(tight.result["n_best"])
    assert default.result["loglik"] == tight.result["loglik"]


def test_recorded_n10000_changes_status_only():
    frame = _stream(10000)
    default = _fit(frame)
    tight = _fit(frame, replicate_tol=1e-3)
    _same_arrays(default, tight)
    assert default.result["loglik"] == -19133.760102125754
    assert default.result["status"] == "ok"
    assert default.result["status_reason"] == "ok"
    assert int(default.result["n_best"]) >= 2
    assert default.result["ic_supported"] is True
    assert default.result["eligible_model_comparison"] is True
    assert tight.result["status"] == "unusable"
    assert "best_loglik_not_replicated" in tight.result["status_reason"]
    assert int(tight.result["n_best"]) == 1
    assert tight.result["ic_supported"] is False
    assert tight.result["eligible_model_comparison"] is False


class _Stub:
    def __init__(self, n_best, status):
        self.result = {
            "loglik": -10.0,
            "converged": True,
            "admissible": True,
            "n_best": n_best,
            "status": status,
        }


def test_blrt_acceptance_ignores_status_and_flags_follow_n_best():
    unreplicated = _blrt_solution(_Stub(1, "unusable"))
    replicated = _blrt_solution(_Stub(2, "ok"))
    assert unreplicated["usable"] is True
    assert replicated["usable"] is True
    assert unreplicated["replicated"] is False
    assert replicated["replicated"] is True
    assert unreplicated["loglik"] == replicated["loglik"]


def test_small_blrt_acceptance_matches_across_the_tolerance():
    # Four starts on this prefix leave the two-class fit unreplicated even at
    # 0.1, so its status is unusable. The bootstrap still runs: acceptance
    # uses admissibility, not status. The replication flag follows n_best.
    frame = _stream(400)
    draws = []
    fits = []
    for tol in (None, 1e-3):
        kwargs = {"n_starts": 4, "n_full": 2, "screen_iter": 5, "max_iter": 100, "seed": 1}
        if tol is not None:
            kwargs["replicate_tol"] = tol
        h1 = pf.fit_lca(frame, ["y1", "y2", "y3"], id="id", n_states=2, **kwargs)
        h0 = pf.fit_lca(frame, ["y1", "y2", "y3"], id="id", n_states=1, **kwargs)
        fits.append(h1)
        draws.append(pf.blrt_check(h0, h1, B=1, max_retries=0, max_iter=100, experimental=True, seed=1))
    wide, narrow = draws
    assert fits[0].result["status"] == "unusable"
    assert fits[0].result["admissible"] is True
    assert wide["ran"] is True and narrow["ran"] is True
    assert wide["n_ok"] == narrow["n_ok"]
    assert wide["b"] == narrow["b"]
    assert wide["lr_observed"] == narrow["lr_observed"]
    assert wide["observed_h1_replicated"] is False
    assert narrow["observed_h1_replicated"] is False
