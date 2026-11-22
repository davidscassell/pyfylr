"""Automatic selection: information criteria, the rule, and the diagnostic plots."""

import copy
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

import pyfylr as pf
from pyfylr.selection import BIAS_NOTE, CANDIDATE_COLUMNS, _h1_materially_below
from pyfylr.sklearn_api import SCORE_SELECTS_N_STATES, StateCountSearch

PUBLISHED = dict(
    n_starts=30, n_full=8, screen_iter=12, max_iter=250, tol=1e-8,
    seed=20261025, replicate_tol=1e-4, n_threads=1,
)


def ll_for_bic(bic, p, n):
    return (p * np.log(n) - bic) / 2


def stub_fit(K=2, ll=-100.0, p=8, n=200, covariance="EEI", model="lpa", n_waves=1,
             converged=True, regularized=False, pd=True, collapsed=False, n_starts=4, n_best=2,
             entropy_wave=0.9, entropy_sequence=0.9, entropy_wave_raw=0.0, entropy_sequence_raw=0.0,
             high=0.95, n_subjects=8, ic_supported=True, loglik_finite=True, membership=None):
    if membership is None:
        membership = [n / K] * (n_waves * K)
    gamma = []
    for i in range(n_subjects):
        for _t in range(n_waves):
            modal = i % K
            rest = 0.0 if K == 1 else (1.0 - high) / (K - 1)
            for state in range(K):
                if K == 1:
                    gamma.append(1.0)
                else:
                    gamma.append(high if state == modal else rest)
    result = {
        "K": int(K),
        "loglik": float(ll) if loglik_finite else float("nan"),
        "n_parameters": int(p),
        "n_analyzed": int(n),
        "n_subjects": int(n_subjects),
        "n_waves": int(n_waves),
        "covariance": covariance or "",
        "ic_supported": bool(ic_supported),
        "aic": float("nan"), "bic": float("nan"), "sabic": float("nan"),
        "converged": bool(converged),
        "regularized": bool(regularized),
        "cov_positive_definite": pd,
        "cov_collapsed_component": bool(collapsed),
        "n_starts": int(n_starts),
        "n_best": int(n_best),
        "n_converged_starts": int(n_best),
        "status": "ok" if converged else "unusable",
        "status_reason": "",
        "entropy_wave": entropy_wave,
        "entropy_sequence": entropy_sequence,
        "expected_membership": list(membership),
        "expected_transitions": [],
        "gamma": gamma,
        "n_boundary": 0,
        "min_eigenvalue": 1.0,
        "measurement_invariant": True,
        "means_invariant": True,
        "covariance_invariant": True,
        "transition_equal": True,
    }
    if entropy_wave_raw is not None:
        result["entropy_wave_raw"] = entropy_wave_raw
    if entropy_sequence_raw is not None:
        result["entropy_sequence_raw"] = entropy_sequence_raw
    return pf.Fit(result=result, panel=None, spec={"replicate_tol": 1e-3, "seed": 1, "n_starts": int(n_starts)}, model=model)


def choose(fits, **kwargs):
    if "second_seed_loglik" not in kwargs:
        seeds = []
        for fit in fits:
            ll = fit.result.get("loglik")
            seeds.append(0.0 if ll is None or not np.isfinite(ll) else float(ll))
        kwargs["second_seed_loglik"] = seeds
        kwargs.setdefault("second_seed_n_best", [2] * len(fits))
    return pf.select_model(fits, **kwargs)


def picked(sel):
    return sel.candidates.loc[sel.candidates["selected"]].iloc[0]


def test_avepp_occ_and_raw_entropy_match_a_brute_force_posterior():
    fit = stub_fit(K=2, n=4, n_subjects=4, p=4, ll=-10, membership=[2, 2],
                   entropy_wave_raw=None, entropy_sequence_raw=None)
    fit.result["gamma"] = [0.9, 0.1, 0.8, 0.2, 0.3, 0.7, 0.1, 0.9]
    diag = pf.classification_diagnostics(fit)
    assert diag["avepp"] == pytest.approx([0.85, 0.80])
    assert diag["avepp_matrix"][0].tolist() == pytest.approx([0.85, 0.15])
    assert diag["avepp_matrix"][1].tolist() == pytest.approx([0.20, 0.80])
    assert diag["min_avepp"] == pytest.approx(0.80)
    assert diag["occ"][0] == pytest.approx((0.85 / 0.15) / (0.5 / 0.5))
    ic = pf.information_criteria(fit)
    assert ic["entropy_raw"] == pytest.approx(1.761432672375978)
    assert ic["icl_bic"] == pytest.approx(ic["bic"] + 2 * ic["entropy_raw"])

    zeros = stub_fit(K=2, n=2, n_subjects=2, membership=[1, 1],
                     entropy_wave_raw=None, entropy_sequence_raw=None)
    zeros.result["gamma"] = [1.0, 0.0, 0.0, 1.0]
    assert pf.information_criteria(zeros)["entropy_raw"] == pytest.approx(0.0)
    assert pf.classification_diagnostics(zeros)["occ"][0] == np.inf

    wave = stub_fit(K=2, n=2, n_subjects=2, n_waves=2, model="lta", membership=[1, 1, 1, 1],
                    entropy_wave_raw=None, entropy_sequence_raw=None)
    wave.result["gamma"] = [0.9, 0.1, 0.6, 0.4, 0.2, 0.8, 0.3, 0.7]
    by = pf.classification_diagnostics(wave)["by_wave"]
    assert by.loc[by["wave"] == 1, "avepp"].tolist() == pytest.approx([0.9, 0.8])
    assert by.loc[by["wave"] == 2, "avepp"].tolist() == pytest.approx([0.6, 0.7])
    assert pf.classification_diagnostics(wave)["min_avepp"] == pytest.approx(0.75)

    lta = stub_fit(K=2, n=200, n_waves=2, model="lta", p=5, ll=-100,
                   entropy_wave_raw=1, entropy_sequence_raw=5)
    ic_lta = pf.information_criteria(lta)
    assert ic_lta["entropy_raw"] == pytest.approx(5)
    assert ic_lta["icl_bic"] == pytest.approx(ic_lta["bic"] + 10)
    assert abs(ic_lta["icl_bic"] - (ic_lta["bic"] + 2)) > 1


def test_new_information_criteria_match_mplus_printed_values():
    def check(ll, p, n, aic, bic, sabic):
        fit = stub_fit(K=2, ll=ll, p=p, n=n, covariance="", model="lca",
                       entropy_wave_raw=0, entropy_sequence_raw=0)
        ic = pf.information_criteria(fit)
        dev = -2 * ll
        assert ic["aic"] == pytest.approx(aic, abs=1e-6)
        assert ic["bic"] == pytest.approx(bic, abs=1e-3)
        assert ic["sabic"] == pytest.approx(sabic, abs=1e-3)
        assert ic["aic3"] == pytest.approx(dev + 3 * p, abs=1e-8)
        assert ic["caic"] == pytest.approx(dev + p * (np.log(n) + 1), abs=1e-8)
        assert ic["awe"] == pytest.approx(dev + 2 * p * (np.log(n) + 1.5), abs=1e-8)
        assert ic["icl_bic"] == pytest.approx(ic["bic"], abs=1e-8)
        blank = stub_fit(K=2, ll=ll, p=p, n=n, ic_supported=False)
        out = pf.information_criteria(blank)
        assert not np.isfinite(out["aic"])
        assert not np.isfinite(out["bic"])
        assert not np.isfinite(out["icl_bic"])

    check(-965.244, 9, 500, 1948.488, 1986.420, 1957.853)
    check(-3177.162, 13, 500, 6380.324, 6435.114, 6393.851)
    check(-3054.963, 24, 500, 6157.926, 6259.077, 6182.899)


def test_rule_never_selects_an_inadmissible_or_unreplicated_cell():
    fits = [
        stub_fit(ll=-50, p=4),
        stub_fit(ll=-30, p=8, converged=False),
        stub_fit(ll=-30, p=8, collapsed=True),
        stub_fit(ll=-30, p=8, n_best=1),
        stub_fit(ll=-30, p=8, regularized=True),
        stub_fit(ll=-30, p=8, pd=False),
        stub_fit(ll=-30, p=8, loglik_finite=False),
        stub_fit(ll=-30, p=8),
    ]
    sel = choose(fits, second_seed_loglik=[-50, -30, -30, -30, -30, -30, -30, -10],
                 second_seed_n_best=[2] * 8)
    assert list(sel.candidates.columns) == CANDIDATE_COLUMNS
    hit = picked(sel)
    assert hit["loglik"] == pytest.approx(-50)
    assert bool(hit["hard_pass"])
    assert hit["rejection_reason"] == ""
    reasons = sel.candidates["rejection_reason"].tolist()
    assert reasons[1:] == [
        "hard:not_converged",
        "hard:collapsed",
        "hard:best_loglik_not_replicated",
        "hard:regularized",
        "hard:not_positive_definite",
        "hard:non_finite,hard:information_criterion_undefined",
        "hard:second_seed_not_reproduced",
    ]
    assert all("second_seed" not in reason for reason in reasons[1:6])
    assert not sel.comparison["selected"].any()
    finite = np.isfinite(sel.candidates["bic_cmp"].to_numpy(dtype=float))
    assert sel.candidates.loc[finite, "bic_cmp"].sum() == pytest.approx(1)
    best = np.abs(sel.candidates["delta_bic"].to_numpy(dtype=float)) < 1e-8
    assert np.allclose(sel.candidates.loc[best, "bic_bf_vs_best"], 1)
    assert sel.candidates["delta_bic"].iloc[0] > 1
    assert "lowest BIC" in sel.reason
    assert "positive definite" in sel.reason
    assert "second seed" in sel.reason
    assert "is not a filter" in sel.reason
    assert "chosen jointly" in sel.reason
    assert "not an analyst record" in pf.report_model(sel)


def test_tie_zone_primary_and_soft_filters_follow_the_rule():
    small = stub_fit(K=1, p=5, ll=ll_for_bic(101.5, 5, 200), membership=[200])
    large = stub_fit(K=2, p=10, ll=ll_for_bic(100, 10, 200))
    tied = choose([small, large], rule=pf.selection_rule(tie_zone=2))
    assert int(picked(tied)["K"]) == 2
    assert tied.ambiguous is False
    assert tied.blrt["ambiguous_kind"] == ""
    assert "lowest BIC" in tied.reason
    exact = choose([small, large], rule=pf.selection_rule(tie_zone=0))
    assert int(picked(exact)["K"]) == 2
    assert int(picked(exact)["model_id"]) == int(picked(tied)["model_id"])

    k2 = stub_fit(K=2, p=10, n=500, ll=-100, membership=[250, 250])
    k3 = stub_fit(K=3, p=20, n=500, ll=-80, membership=[200, 150, 150])
    vote = choose([k2, k3])
    assert vote.primary == "bic"
    assert int(picked(vote)["K"]) == 2
    by = dict(zip(vote.disagreement["index"], vote.disagreement["K"]))
    assert [int(by[name]) for name in ("aic", "aic3", "sabic")] == [3, 3, 3]
    assert [int(by[name]) for name in ("bic", "caic", "awe", "icl_bic")] == [2, 2, 2, 2]
    assert not vote.disagreement["selects"].any()
    assert not any("n_criteria" in name or "favoring" in name for name in vote.disagreement.columns)
    for _, row in vote.disagreement.iterrows():
        assert row["bias_note"] == BIAS_NOTE[row["index"]]
    aic = choose([k2, k3], rule=pf.selection_rule(primary="aic"))
    assert int(picked(aic)["K"]) == 3
    assert "lowest AIC" in aic.reason

    eei = stub_fit(K=2, p=5, covariance="EEI", ll=ll_for_bic(101, 5, 200))
    vvv = stub_fit(K=2, p=11, covariance="VVV", ll=ll_for_bic(100, 11, 200))
    joint = choose([vvv, eei], rule=pf.selection_rule(tie_zone=2))
    assert picked(joint)["covariance"] == "VVV"
    assert joint.ambiguous is False
    assert "lowest BIC" in joint.reason
    assert "chosen jointly" in joint.reason

    low = stub_fit(ll=ll_for_bic(100, 8, 200), entropy_wave=0.4)
    high = stub_fit(ll=ll_for_bic(110, 8, 200), entropy_wave=0.9)
    kept = choose([low, high])
    assert picked(kept)["entropy_wave"] == pytest.approx(0.4)
    filtered = choose([low, high], rule=pf.selection_rule(entropy_filter=True, min_entropy=0.8))
    assert picked(filtered)["entropy_wave"] == pytest.approx(0.9)
    assert "soft:entropy" in filtered.candidates["rejection_reason"].iloc[0]

    sharp = stub_fit(ll=ll_for_bic(90, 8, 200), high=0.95)
    flat = stub_fit(ll=ll_for_bic(80, 8, 200), high=0.5)
    avepp = choose([flat, sharp])
    assert picked(avepp)["loglik"] == pytest.approx(flat.result["loglik"])
    assert "soft:min_avepp" not in avepp.candidates["rejection_reason"].iloc[0]
    held = choose([flat, sharp], rule=pf.selection_rule(min_avepp=0.70))
    assert picked(held)["min_avepp"] == pytest.approx(pf.classification_diagnostics(sharp)["min_avepp"])
    assert picked(held)["min_avepp"] >= 0.70
    flat_row = held.candidates[np.isclose(held.candidates["loglik"], flat.result["loglik"])]
    assert "soft:min_avepp" in flat_row["rejection_reason"].iloc[0]
    assert "analyst threshold" in held.reason

    tiny = stub_fit(ll=ll_for_bic(80, 8, 200), membership=[2, 198])
    big = stub_fit(ll=ll_for_bic(100, 8, 200), membership=[100, 100])
    sized = choose([tiny, big])
    assert picked(sized)["loglik"] == pytest.approx(tiny.result["loglik"])
    assert "soft:" not in sized.candidates["rejection_reason"].iloc[0]
    free = choose([tiny, big], rule=pf.selection_rule(min_class_prop=0.05, min_class_n=30))
    assert "soft:min_class_prop" in free.candidates["rejection_reason"].iloc[0]
    assert "soft:min_class_n" in free.candidates["rejection_reason"].iloc[0]
    assert picked(free)["loglik"] == pytest.approx(big.result["loglik"])

    gap = choose([stub_fit(ll=-40, p=4)], rule=pf.selection_rule(require_second_seed=False),
                 second_seed_loglik=[-1])
    assert bool(picked(gap)["hard_pass"])
    assert not np.isfinite(picked(gap)["second_seed_loglik"])

    quiet = choose([small, large], auto=False)
    assert quiet.decision == "auto_disabled"
    assert not quiet.candidates["selected"].any()
    assert not quiet.comparison["selected"].any()
    assert "auto is FALSE" in quiet.reason
    assert not np.isfinite(quiet.candidates["second_seed_loglik"]).any()

    none = choose([stub_fit(converged=False)])
    assert none.decision == "no_selection"
    assert "No selection" in none.reason
    assert "no fallback" in none.reason
    assert none.selected_model_id is None

    one = stub_fit(K=1, p=5, ll=-100, n_waves=2, model="lta", membership=[200, 200])
    two = stub_fit(K=2, p=12, ll=-90, n_waves=2, model="lta")
    joint_lta = choose([one, two])
    assert joint_lta.primary == "bic"
    assert int(picked(joint_lta)["K"]) == 1
    bic = choose([one, two], rule=pf.selection_rule(primary="bic"))
    assert int(picked(bic)["K"]) == 1
    with pytest.raises(ValueError, match="one model family"):
        choose([stub_fit(model="lpa"), two])
    with pytest.raises(ValueError, match="primary"):
        pf.selection_rule(primary="waic")
    with pytest.raises(ValueError, match="tie_zone"):
        pf.selection_rule(tie_zone=-1)
    with pytest.raises(ValueError, match="does not choose the joint"):
        pf.select_model(pf.WaveEnumeration(tables={}, fits=[], note=""))


def test_plots_mark_the_selected_cell_and_are_not_a_vote():
    fit = stub_fit()
    sel = choose([fit, stub_fit(K=1, p=4, ll=-80, membership=[200])])
    elbow = pf.plot_ic_elbow(sel)
    assert "ringed" in elbow.get_ylabel()
    dots = pf.plot_index_disagreement(sel)
    assert "not a vote" in dots.get_title()
    assert pf.plot_avepp(fit).get_title()
    assert pf.plot_max_posterior(fit).get_title()
    assert pf.plot_prevalence(fit).get_title()


def test_selection_recovers_a_known_two_class_profile_and_max_iter_stays_200():
    assert pf.lpa_spec(2)["max_iter"] == 200
    rng = np.random.default_rng(11)
    n = 240
    group = np.repeat([-2.0, 2.0], n // 2)
    data = pd.DataFrame({
        "y1": group + rng.normal(0, 0.4, n),
        "y2": group + rng.normal(0, 0.4, n),
    })
    fits = [
        pf.fit_lpa(data, ["y1", "y2"], n_states=k, covariance="EEI",
                   n_starts=4, screen_iter=0, n_full=4, max_iter=80, seed=11)
        for k in (1, 2, 3)
    ]
    sel = pf.select_model(fits)
    hit = picked(sel)
    assert int(hit["K"]) == 2
    assert bool(hit["hard_pass"])
    assert hit["rejection_reason"] == ""
    assert float(hit["second_seed_abs_gap"]) <= float(hit["replicate_tol"])
    assert sel.blrt["ran"] is False
    assert sel.blrt["validated"] is False
    assert "Mplus TECH14" in sel.blrt["reason"]
    assert "off by default" in sel.blrt["reason"]
    assert "digit match" not in sel.blrt["reason"]
    assert "bootstrap draws differ" not in sel.blrt["reason"]
    assert "Independent bootstrap p-values are not required to match" in sel.blrt["reason"]
    assert "diagnostic" in sel.blrt["reason"]
    assert pf.plot_profiles(fits[1]).get_title() == "Gaussian profiles"
    elbow = pf.plot_ic_elbow(sel)
    assert elbow.figure._suptitle.get_text() == "Information criteria by covariance"
    assert "ringed" in elbow.get_ylabel()
    assert pf.plot_avepp(fits[1]).get_title() == "Average posterior by modal class"
    assert pf.plot_max_posterior(fits[1]).get_title() == "Assignment certainty"
    assert "not a vote" in pf.plot_index_disagreement(sel).get_title()
    with pytest.raises(ValueError, match="one-wave"):
        pf.plot_transitions(fits[1])


def test_joint_transition_model_warns_when_per_wave_state_counts_differ():
    rng = np.random.default_rng(3)
    n = 120
    ids = np.repeat(np.arange(1, n + 1), 2)
    wave = np.tile([1, 2], n)
    y1 = np.empty(2 * n)
    y2 = np.empty(2 * n)
    y1[wave == 1] = rng.normal(0, 0.4, n)
    y2[wave == 1] = rng.normal(0, 0.4, n)
    shift = np.where(ids[wave == 2] <= n / 2, -2.5, 2.5)
    y1[wave == 2] = shift + rng.normal(0, 0.4, n)
    y2[wave == 2] = shift + rng.normal(0, 0.4, n)
    data = pd.DataFrame({"id": ids, "wave": wave, "y1": y1, "y2": y2})
    fits = [
        pf.fit_lta(data, ["y1", "y2"], id="id", wave="wave", n_states=k, covariance="EEI",
                   n_starts=4, screen_iter=0, n_full=4, max_iter=60, seed=3)
        for k in (1, 2)
    ]
    sel = pf.select_model(fits)
    assert "configural_noninvariance_possible" in sel.warnings
    assert "configural noninvariance" in sel.reason
    assert pf.plot_transitions(fits[1]).get_title() == "Transition probabilities"
    assert pf.plot_profiles(fits[1]).get_title() == "Gaussian profiles"
    elbow = pf.plot_ic_elbow(sel)
    assert elbow.figure._suptitle.get_text() == "Information criteria by covariance"
    assert pf.plot_avepp(fits[1]).get_title() == "Average posterior by modal class"
    assert pf.plot_max_posterior(fits[1]).get_title() == "Assignment certainty"
    assert "not a vote" in pf.plot_index_disagreement(sel).get_title()


def test_wilson_interval_carries_the_uncertainty_of_a_small_bootstrap_count():
    from pyfylr.selection import _wilson_ci
    assert _wilson_ci(0, 100) == pytest.approx((0.0, 0.03699349820698568))
    assert _wilson_ci(1, 100) == pytest.approx((0.00176743206414065, 0.05448619617870532))
    assert _wilson_ci(2, 100) == pytest.approx((0.00550196755016235, 0.07001179072854388))


def test_blrt_prints_its_cost_and_stays_unvalidated(capsys):
    rng = np.random.default_rng(4)
    n = 40
    group = np.repeat([-2.0, 2.0], n // 2)
    data = pd.DataFrame({
        "y1": group + rng.normal(0, 0.3, n),
        "y2": group + rng.normal(0, 0.3, n),
    })
    controls = dict(covariance="EEI", n_starts=4, screen_iter=0, n_full=4, max_iter=30, seed=4)
    h0 = pf.fit_lpa(data, ["y1", "y2"], n_states=1, **controls)
    h1 = pf.fit_lpa(data, ["y1", "y2"], n_states=2, **controls)
    out = pf.blrt_check(h0, h1, B=1, max_retries=1, seed=4, experimental=True)
    captured = capsys.readouterr()
    assert "estimated cost" in captured.out
    assert out["validated"] is False
    assert pf.selection_rule()["blrt"] is False
    assert pf.lpa_spec(2)["max_iter"] == 200
    assert "validated stays false until Monte Carlo agreement" in out["cost_note"]
    assert "digit match" not in out["reason"]
    assert "digit match" not in out["cost_note"]
    assert "Independent bootstrap p-values are not required to match" in out["reason"]
    assert "Unreplicated kept solutions are a diagnostic" in out["reason"]
    if out["n_ok"] > 0:
        assert out["mc_se"] == pytest.approx((out["p"] * (1 - out["p"]) / out["n_ok"]) ** 0.5)
        assert out["p_ci_low"] <= out["p"] <= out["p_ci_high"]
        assert out["p_ci_method"] == "wilson_95"
    assert "Dropped" in out["reason"]
    assert "Mplus TECH14 p=b/n_ok" in out["reason"]
    assert "(b+1)/(n_ok+1)" in out["reason"]
    assert "p_plus" in out["reason"]
    assert "secondary" in out["reason"]
    assert "p_mplus" not in out
    assert out["n_ok"] + out["n_failed"] == 1
    assert out["n_failed"] == out["n_failed_not_converged"] + out["n_failed_error"] + out["n_failed_inadmissible"]
    assert out["n_not_replicated"] <= out["n_ok"]
    if (
        out["n_not_replicated"] > 0
        or not out["observed_h0_replicated"]
        or not out["observed_h1_replicated"]
    ):
        assert (
            "THE BEST LOGLIKELIHOOD VALUE WAS NOT REPLICATED. "
            "THE SOLUTION MAY NOT BE TRUSTWORTHY DUE TO LOCAL MAXIMA. "
            "INCREASE THE NUMBER OF RANDOM STARTS."
        ) in out["reason"]
    assert len(out["lrs"]) == out["n_ok"]
    assert len(out["lr_draws"]) == out["n_ok"]
    if out["n_ok"]:
        assert out["p"] == out["b"] / out["n_ok"]
        assert out["p_plus"] == (1 + out["b"]) / (1 + out["n_ok"])
        assert out["b"] == sum(value >= out["lr_observed"] - 1e-8 for value in out["lrs"])
    ok_p = np.isfinite(out["p"]) and 0 <= out["p"] <= 1
    assert ok_p or out["n_ok"] == 0
    sel = pf.select_model([h0, h1], rule=pf.selection_rule(blrt=True))
    assert sel.blrt["ran"] is False
    assert "only in the tie zone" in sel.blrt["reason"]


def _bootstrap_panel():
    return {
        "N": 2, "T": 1, "J": 1, "family": ["gaussian"], "gauss": [0.0, 1.0],
        "indicators": ["y"], "ids": [1, 2], "waves": [1], "ncat": [1], "cat": [],
    }


def test_default_bootstrap_size_is_1000_and_null_rejection_probability_matches_the_floor_formula():
    import inspect

    assert inspect.signature(pf.blrt_check).parameters["B"].default == 1000
    rule = pf.selection_rule()
    assert rule["blrt_B"] == 1000
    assert rule["blrt"] is False
    assert rule["blrt_max_iter"] == 1000
    spec = pf.lpa_spec(2)
    assert spec["n_starts"] == 40
    assert spec["screen_iter"] == 10
    assert spec["n_full"] == 4
    assert spec["max_iter"] == 200
    assert pf.blrt_null_size(0.01, 100) == 2 / 101
    assert pf.blrt_null_size(0.01, 1000) == 11 / 1001
    assert pf.blrt_null_size(0.05, 100) == 6 / 101
    assert pf.blrt_null_size(0.05, 1000) == 51 / 1001
    for alpha in (0.01, 0.05):
        for draws in (100, 1000):
            crit = int(np.floor(alpha * draws))
            assert pf.blrt_null_size(alpha, draws) == (crit + 1) / (draws + 1)
            reject = [(b / draws) <= alpha for b in range(draws + 1)]
            assert sum(reject) == crit + 1
            assert reject == [b <= crit for b in range(draws + 1)]


def test_selection_defaults_rank_by_bic_and_leave_substantive_filters_off():
    rule = pf.selection_rule()
    assert rule["min_class_prop"] is None
    assert rule["min_class_n"] is None
    assert rule["min_avepp"] is None
    assert rule["entropy_filter"] is False
    assert rule["min_entropy"] == 0.80
    assert rule["primary"] is None
    assert rule["blrt"] is False
    assert pf.lpa_spec(2)["max_iter"] == 200
    sel = choose([stub_fit()])
    assert sel.primary == "bic"
    assert sel.decision == "selected"
    assert "lowest BIC among eligible candidates tested" in sel.reason
    assert "Search grid:" in sel.reason
    assert "Class size" in sel.reason
    assert "AvePP" in sel.reason
    assert "Entropy" in sel.reason
    assert "min_class_prop=NULL" in sel.reason
    assert "min_class_n=NULL" in sel.reason
    assert "min_avepp=NULL" in sel.reason
    assert "entropy_filter=FALSE" in sel.reason
    assert "no analyst threshold" in sel.reason
    assert sel.accepted is None
    assert sel.accepted_model_id is None
    assert sel.automatic_model_id == sel.selected_model_id
    assert sel.automatic_decision == "selected"
    report = pf.report_model(sel)
    assert "not an analyst record" in report
    assert "accepted starts unset" in report


def test_a_higher_unreplicated_admissible_solution_blocks_the_candidate():
    hidden = stub_fit(ll=-100, n_best=2)
    hidden.result["rejected_loglik"] = [-90.0]
    hidden.result["rejected_reason"] = ["admissible"]
    blocked = choose([hidden])
    assert not bool(blocked.candidates["eligible"].iloc[0])
    assert "hard:higher_admissible_not_replicated" in blocked.candidates["rejection_reason"].iloc[0]
    assert blocked.decision == "no_selection"
    assert blocked.selected_model_id is None
    assert "No selection" in blocked.reason
    assert "no fallback" in blocked.reason

    visible = stub_fit(ll=-100, n_best=2)
    visible.result["rejected_loglik"] = [-90.0]
    visible.result["rejected_reason"] = ["not_converged"]
    kept = choose([visible])
    assert bool(kept.candidates["eligible"].iloc[0])
    assert "higher_admissible" not in kept.candidates["rejection_reason"].iloc[0]
    assert kept.decision == "selected"


def test_no_eligible_candidate_returns_no_selection_and_no_fallback():
    none = choose([stub_fit(converged=False), stub_fit(n_best=1)])
    assert none.decision == "no_selection"
    assert none.automatic_decision == "no_selection"
    assert none.selected_model_id is None
    assert none.automatic_model_id is None
    assert not none.candidates["selected"].any()
    assert "No selection" in none.reason
    assert "no fallback" in none.reason
    assert "Search grid:" in none.reason


def test_analyst_thresholds_are_applied_and_recorded():
    tiny = stub_fit(ll=ll_for_bic(80, 8, 200), membership=[2, 198], entropy_wave=0.4)
    big = stub_fit(ll=ll_for_bic(100, 8, 200), membership=[100, 100], entropy_wave=0.9)
    rule = pf.selection_rule(
        min_class_prop=0.05, min_class_n=30, min_avepp=0.70,
        entropy_filter=True, min_entropy=0.80,
    )
    sel = choose([tiny, big], rule=rule)
    assert picked(sel)["loglik"] == pytest.approx(big.result["loglik"])
    reason = sel.candidates["rejection_reason"].iloc[0]
    assert "soft:min_class_prop" in reason
    assert "soft:min_class_n" in reason
    assert "soft:entropy" in reason
    assert "analyst threshold 0.05 applied" in sel.reason
    assert "analyst threshold 30 applied" in sel.reason
    assert "analyst threshold 0.7 applied" in sel.reason
    assert "analyst entropy threshold 0.8 applied" in sel.reason
    assert "min_class_prop=0.05" in sel.reason
    assert "entropy_filter=TRUE" in sel.reason


def test_accept_records_the_analyst_without_changing_the_automatic_selection():
    low = stub_fit(K=1, p=4, ll=ll_for_bic(110, 4, 200), membership=[200])
    high = stub_fit(K=2, p=8, ll=ll_for_bic(100, 8, 200))
    sel = choose([low, high])
    assert sel.automatic_model_id == 2
    assert sel.accepted is None
    assert sel.accepted_model_id is None
    taken = pf.accept(sel, 1)
    assert taken.automatic_model_id == 2
    assert taken.selected_model_id == 2
    assert taken.automatic_decision == "selected"
    assert taken.accepted is True
    assert taken.accepted_model_id == 1
    with pytest.raises(ValueError, match="not one of the tested"):
        pf.accept(sel, 9)


def test_icl_bic_is_reported_and_does_not_rank_the_automatic_selection():
    bic_best = stub_fit(
        K=2, p=8, ll=ll_for_bic(100, 8, 200),
        entropy_wave_raw=25, entropy_sequence_raw=25,
    )
    icl_best = stub_fit(
        K=1, p=4, ll=ll_for_bic(120, 4, 200), membership=[200],
        entropy_wave_raw=0, entropy_sequence_raw=0,
    )
    sel = choose([bic_best, icl_best])
    assert int(picked(sel)["K"]) == 2
    icl = sel.disagreement[sel.disagreement["index"] == "icl_bic"].iloc[0]
    assert int(icl["K"]) == 1
    assert bool(icl["selects"]) is False
    assert "not used for automatic ranking" in sel.reason
    with pytest.raises(ValueError, match="icl_bic is a diagnostic and is not used for automatic ranking"):
        pf.selection_rule(primary="icl_bic")


def test_blrt_uses_the_admissible_likelihood_and_ignores_a_higher_rejected_start():
    h0 = stub_fit(K=1, ll=-100, p=4, membership=[200])
    h0.result["rejected_loglik"] = [-50.0]
    h0.result["rejected_reason"] = ["not_converged"]
    h0.panel = _bootstrap_panel()
    h1 = stub_fit(K=2, ll=-80, p=8)
    h1.result["rejected_loglik"] = [-10.0]
    h1.result["rejected_reason"] = ["not_positive_definite"]
    h1.panel = _bootstrap_panel()
    out = pf.blrt_check(h0, h1, B=0, seed=1, experimental=True)
    assert out["observed_h0"]["loglik"] == pytest.approx(-100)
    assert out["observed_h1"]["loglik"] == pytest.approx(-80)
    assert out["lr_observed"] == pytest.approx(40)
    assert out["validated"] is False
    assert out["report_only"] is True


def test_a_blrt_retry_keeps_one_dataset_and_adds_starts(monkeypatch):
    h0 = stub_fit(K=1, ll=-100, p=4, n_starts=4, membership=[200])
    h1 = stub_fit(K=2, ll=-80, p=8, n_starts=4)
    h0.panel = _bootstrap_panel()
    h1.panel = _bootstrap_panel()
    frame = pd.DataFrame({"id": [1, 2], "y": [0.0, 1.0]})
    calls = {"simulate": 0, "starts": [], "frames": []}

    def simulate(fit, rng, donors):
        calls["simulate"] += 1
        return frame.copy()

    def fit_like(template, data, seed, n_starts=None, n_full=None, max_iter=None):
        calls["starts"].append(int(n_starts))
        calls["frames"].append(data.copy())
        ok = len(calls["starts"]) > 2
        made = stub_fit(
            K=int(template.result["K"]), ll=-100 if int(template.result["K"]) == 1 else -80,
            converged=ok, n_best=2 if ok else 0,
        )
        return made

    monkeypatch.setattr(pf.selection, "_simulate", simulate)
    monkeypatch.setattr(pf.selection, "_fit_like", fit_like)
    out = pf.blrt_check(h0, h1, B=1, max_retries=1, seed=1, experimental=True)
    assert calls["simulate"] == 1
    assert calls["starts"] == [4, 4, 8, 8]
    assert all(item.equals(frame) for item in calls["frames"])
    assert [item["n_starts"] for item in out["attempts"]] == [4, 8]
    assert len({item["dataset_hash"] for item in out["attempts"]}) == 1
    assert out["n_ok"] == 1
    assert out["failures"] == []
    assert "Unresolved failures: none." in out["reason"]


def test_an_unresolved_blrt_failure_is_reported_by_draw_index(monkeypatch):
    h0 = stub_fit(K=1, ll=-100, p=4, n_starts=4, membership=[200])
    h1 = stub_fit(K=2, ll=-80, p=8, n_starts=4)
    h0.panel = _bootstrap_panel()
    h1.panel = _bootstrap_panel()
    calls = {"simulate": 0, "starts": []}

    def simulate(fit, rng, donors):
        calls["simulate"] += 1
        return pd.DataFrame({"id": [1, 2], "y": [0.0, 1.0]})

    def fit_like(template, data, seed, n_starts=None, n_full=None, max_iter=None):
        calls["starts"].append(int(n_starts))
        return stub_fit(K=int(template.result["K"]), converged=False, n_best=0)

    monkeypatch.setattr(pf.selection, "_simulate", simulate)
    monkeypatch.setattr(pf.selection, "_fit_like", fit_like)
    failed = pf.blrt_check(h0, h1, B=1, max_retries=1, seed=1, experimental=True)
    assert calls["simulate"] == 1
    assert calls["starts"] == [4, 4, 8, 8]
    assert failed["failures"] == [{"draw": 1, "reason": "not_converged"}]
    assert "draw 1 not_converged" in failed["reason"]
    assert failed["n_ok"] == 0


def test_an_unvalidated_blrt_does_not_change_the_selected_model(monkeypatch):
    small = stub_fit(K=1, p=5, ll=ll_for_bic(101.5, 5, 200), membership=[200])
    large = stub_fit(K=2, p=10, ll=ll_for_bic(100, 10, 200))
    small.panel = _bootstrap_panel()
    large.panel = _bootstrap_panel()

    def fake_blrt(*args, **kwargs):
        return {
            "ran": True, "validated": False, "switched": True, "report_only": False,
            "p": 0.0, "p_plus": 0.5, "B": 1, "b": 0, "reason": "mocked p=0",
            "n_ok": 1, "seconds": 0.0,
        }

    monkeypatch.setattr(pf.selection, "blrt_check", fake_blrt)
    off = choose([small, large], rule=pf.selection_rule(blrt=False))
    blocked = choose([small, large], rule=pf.selection_rule(blrt=True, blrt_check=True))
    assert blocked.selected_model_id == off.selected_model_id
    assert blocked.blrt["ran"] is False
    assert blocked.blrt["decision"] == "not_run"
    assert blocked.blrt["validated"] is False
    assert "excluded from automatic selection" in blocked.blrt["reason"]
    assert "blrt validated=false" in blocked.blrt["reason"]
    on = choose(
        [small, large],
        rule=pf.selection_rule(blrt=True, blrt_check=True, experimental_blrt=True),
    )
    assert on.selected_model_id == off.selected_model_id
    assert on.automatic_model_id == 2
    assert on.blrt["switched"] is False
    assert on.blrt["validated"] is False
    assert "does not change the selected model" in on.reason
    assert "does not change the selected model" in on.blrt["reason"]


def test_blrt_max_iter_changes_only_the_bootstrap_fits(monkeypatch):
    assert pf.lpa_spec(2)["max_iter"] == 200
    rule = pf.selection_rule()
    assert rule["blrt_max_iter"] == 1000
    assert rule["blrt"] is False
    sel = choose([stub_fit()])
    assert "blrt_max_iter=1000" in sel.reason

    rng = np.random.default_rng(4)
    n = 40
    group = np.repeat([-2.0, 2.0], n // 2)
    data = pd.DataFrame({
        "y1": group + rng.normal(0, 0.3, n),
        "y2": group + rng.normal(0, 0.3, n),
    })
    controls = dict(covariance="EEI", n_starts=4, screen_iter=0, n_full=4, max_iter=200, seed=4)
    h0 = pf.fit_lpa(data, ["y1", "y2"], n_states=1, **controls)
    h1 = pf.fit_lpa(data, ["y1", "y2"], n_states=2, **controls)
    observed = (h0.result["n_iter"], h1.result["n_iter"])
    caps = []

    def simulate(fit, rng, donors):
        return data.assign(id=np.arange(1, len(data) + 1))

    def fit_like(template, frame, seed, n_starts=None, n_full=None, max_iter=None):
        caps.append(int(max_iter))
        return template

    monkeypatch.setattr(pf.selection, "_simulate", simulate)
    monkeypatch.setattr(pf.selection, "_fit_like", fit_like)
    out = pf.blrt_check(h0, h1, B=1, max_retries=0, max_iter=750, seed=4, experimental=True)
    assert caps == [750, 750]
    assert out["max_iter"] == 750
    assert h0.spec["max_iter"] == 200
    assert h1.spec["max_iter"] == 200
    assert (h0.result["n_iter"], h1.result["n_iter"]) == observed
    assert pf.lpa_spec(2)["max_iter"] == 200
    assert "Bootstrap fits use max_iter=750" in out["cost_note"]
    assert "Observed H0 and H1 fits keep their own max_iter" in out["cost_note"]
    caps.clear()
    plain = pf.blrt_check(h0, h1, B=1, max_retries=0, seed=4, experimental=True)
    assert caps == [1000, 1000]
    assert plain["max_iter"] == 1000
    assert h0.spec["max_iter"] == 200
    assert h1.spec["max_iter"] == 200
    assert (h0.result["n_iter"], h1.result["n_iter"]) == observed


def test_state_count_search_does_not_use_score_or_grid_search(monkeypatch):
    rng = np.random.default_rng(1)
    n = 80
    group = np.repeat([-2.0, 2.0], n // 2)
    data = pd.DataFrame({
        "y1": group + rng.normal(0, 0.35, n),
        "y2": group + rng.normal(0, 0.35, n),
    })

    def boom(self, X, y=None):
        raise AssertionError("score")

    monkeypatch.setattr(pf.LatentProfileModel, "score", boom)
    model = pf.LatentProfileModel(covariance="EEI", n_starts=4, screen_iter=0, n_full=4, max_iter=50, random_state=1)
    search = StateCountSearch(model, n_states=(1, 2), covariance="EEI")
    search.fit(data)
    assert search.best_estimator_.n_states == 2
    assert not any(cls.__name__ == "GridSearchCV" for cls in type(search).__mro__)
    assert not hasattr(StateCountSearch, "score")
    assert SCORE_SELECTS_N_STATES is False
    assert len(search.predict(data)) == n
    auto_id = search.selection_.automatic_model_id
    auto_k = search.best_estimator_.n_states
    other = 1 if auto_id != 1 else 2
    search.accept(other)
    assert search.selection_.accepted is True
    assert search.selection_.accepted_model_id == other
    assert search.selection_.automatic_model_id == auto_id
    assert search.best_estimator_.n_states == auto_k
    assert len(search.predict(data)) == n
    search.best_estimator_ = None
    with pytest.raises(ValueError, match="no_selection"):
        search.predict(data)
    with pytest.raises(ValueError, match="no_selection"):
        search.predict_proba(data)


def _published_root():
    from fixture_paths import fixtures_dir
    root = fixtures_dir() / "mplus-ug"
    names = ("ex7.3.dat", "ex7.9.dat", "ex7.22.dat")
    if all((root / name).is_file() for name in names):
        return root
    return None


def _load_published(name, n_use, cols):
    from pathlib import Path
    root = _published_root()
    rows = []
    for line in Path(root, name).read_text().splitlines():
        if not line.strip():
            continue
        rows.append([float(tok) for tok in line.split()])
    frame = pd.DataFrame(rows).iloc[:, :n_use].copy()
    frame.columns = cols
    frame.insert(0, "id", np.arange(1, len(frame) + 1))
    return frame


def _check_published_fit(fit, printed_ll, printed_aic, printed_bic, printed_sabic, printed_entropy):
    ic = pf.information_criteria(fit)
    ll = fit.result["loglik"]
    p = fit.result["n_parameters"]
    n = fit.result["n_analyzed"]
    dev = -2 * ll
    assert ic["aic"] == pytest.approx(dev + 2 * p, abs=1e-8)
    assert ic["aic3"] == pytest.approx(dev + 3 * p, abs=1e-8)
    assert ic["caic"] == pytest.approx(dev + p * (np.log(n) + 1), abs=1e-8)
    assert ic["awe"] == pytest.approx(dev + 2 * p * (np.log(n) + 1.5), abs=1e-8)
    assert ic["icl_bic"] == pytest.approx(ic["bic"] + 2 * ic["entropy_raw"], abs=1e-6)
    assert ic["aic"] == pytest.approx(printed_aic, abs=0.01)
    assert ic["bic"] == pytest.approx(printed_bic, abs=0.01)
    assert ic["sabic"] == pytest.approx(printed_sabic, abs=0.01)
    assert fit.result["entropy_wave"] == pytest.approx(printed_entropy, abs=0.03)
    assert ic["entropy_raw"] == pytest.approx(fit.result["entropy_wave_raw"], abs=1e-8)
    assert ll == pytest.approx(printed_ll, abs=0.05)
    clone = copy.deepcopy(fit)
    stored = clone.result["entropy_wave_raw"]
    clone.result.pop("entropy_wave_raw", None)
    clone.result.pop("entropy_sequence_raw", None)
    assert pf.information_criteria(clone)["entropy_raw"] == pytest.approx(stored, abs=1e-6)


def test_fitted_mplus_examples_keep_the_new_information_criteria():
    if _published_root() is None:
        pytest.skip(
            "Optional external reproduction. Mplus User's Guide files are not "
            "redistributed with this package."
        )
    ex73 = _load_published("ex7.3.dat", 4, ["u1", "u2", "u3", "u4"])
    fit73 = pf.fit_lca(ex73, ["u1", "u2", "u3", "u4"], id="id", n_states=2, **PUBLISHED)
    _check_published_fit(fit73, -965.244, 1948.488, 1986.420, 1957.853, 0.904)
    ex79 = _load_published("ex7.9.dat", 4, ["y1", "y2", "y3", "y4"])
    fit79 = pf.fit_lpa(ex79, ["y1", "y2", "y3", "y4"], id="id", n_states=2, covariance="EEI", **PUBLISHED)
    _check_published_fit(fit79, -3177.162, 6380.324, 6435.114, 6393.851, 0.909)
    ex722 = _load_published("ex7.22.dat", 4, ["y1", "y2", "y3", "y4"])
    fit722 = pf.fit_lpa(ex722, ["y1", "y2", "y3", "y4"], id="id", n_states=3, covariance="EEE", **PUBLISHED)
    _check_published_fit(fit722, -3054.963, 6157.926, 6259.077, 6182.899, 0.812)


def _read_tech14(name):
    import csv
    from fixture_paths import fixtures_dir
    path = fixtures_dir() / "mplus" / name
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        "h0": float(row["h0"]),
        "h1": float(row["h1"]),
        "lr": float(row["lr"]),
        "p": float(row["p"]),
        "successful": int(row["successful"]),
    }


def test_tech14_fixtures_match_the_likelihood_ratio_and_the_p_value_within_monte_carlo_error():
    assert pf.lpa_spec(2)["max_iter"] == 200
    assert pf.selection_rule()["blrt"] is False
    # Guard that p is b/n_ok. Independent runs are not required to match. 0.06 is wider
    # than one Monte Carlo SE at B=100 and rejects a different statistic.
    tolerance = 0.06
    controls = dict(
        n_starts=20, screen_iter=10, n_full=4, max_iter=200,
        tol=1e-7, replicate_tol=1e-3, seed=20261026,
    )
    cases = [
        ("lca_bernoulli_k2", ["u1", "u2", "u3", "u4"], None),
        ("lpa_eei_k2", ["y1", "y2", "y3"], "EEI"),
        ("lpa_eee_k2", ["y1", "y2"], "EEE"),
    ]
    from fixture_paths import fixtures_dir
    root = fixtures_dir() / "mplus"
    for name, indicators, covariance in cases:
        printed = _read_tech14(name + "_tech14.csv")
        frame = pd.read_csv(root / (name + ".dat"), sep=r"\s+", header=None, names=indicators)
        kwargs = dict(controls)
        if covariance is None:
            h0 = pf.fit_lca(frame, indicators, n_states=1, **kwargs)
            h1 = pf.fit_lca(frame, indicators, n_states=2, **kwargs)
        else:
            h0 = pf.fit_lpa(frame, indicators, n_states=1, covariance=covariance, **kwargs)
            h1 = pf.fit_lpa(frame, indicators, n_states=2, covariance=covariance, **kwargs)
        lr = 2.0 * (h1.result["loglik"] - h0.result["loglik"])
        assert h0.result["loglik"] == pytest.approx(printed["h0"], abs=5e-4)
        assert h1.result["loglik"] == pytest.approx(printed["h1"], abs=5e-4)
        assert lr == pytest.approx(printed["lr"], abs=5e-4)
        assert printed["successful"] == 100
        out = pf.blrt_check(h0, h1, B=100, max_retries=3, seed=20261026, experimental=True)
        assert h0.spec["max_iter"] == 200
        assert h1.spec["max_iter"] == 200
        assert h0.spec["tol"] == pytest.approx(1e-7)
        assert out["max_iter"] == 1000
        assert out["max_iter"] == pf.selection_rule()["blrt_max_iter"]
        assert out["lr_observed"] == pytest.approx(lr, abs=1e-12)
        assert out["n_ok"] + out["n_failed"] == 100
        assert out["n_failed"] == out["n_failed_not_converged"] + out["n_failed_error"] + out["n_failed_inadmissible"]
        assert out["n_not_replicated"] <= out["n_ok"]
        assert "p_mplus" not in out
        assert f"Dropped {out['n_failed']} draws" in out["reason"]
        assert f"{out['n_not_replicated']} kept draws did not replicate" in out["reason"]
        assert "Mplus TECH14 p=b/n_ok" in out["reason"]
        assert "p_plus=(b+1)/(n_ok+1)" in out["reason"]
        assert out["p"] == pytest.approx(out["b"] / out["n_ok"])
        assert out["p_plus"] == pytest.approx((out["b"] + 1) / (out["n_ok"] + 1))
        assert abs(out["p"] - printed["p"]) <= tolerance
        warning = (
            "THE BEST LOGLIKELIHOOD VALUE WAS NOT REPLICATED. "
            "THE SOLUTION MAY NOT BE TRUSTWORTHY DUE TO LOCAL MAXIMA. "
            "INCREASE THE NUMBER OF RANDOM STARTS."
        )
        if (
            out["n_not_replicated"] > 0
            or not out["observed_h0_replicated"]
            or (out.get("observed_h1_replication_assessed", True) and not out["observed_h1_replicated"])
        ):
            assert warning in out["reason"]
        else:
            assert "THE BEST LOGLIKELIHOOD VALUE WAS NOT REPLICATED" not in out["reason"]
        assert out["validated"] is False
        assert pf.selection_rule()["blrt"] is False


def _same_panel(value=0.0):
    return {
        "N": 2, "T": 1, "J": 1, "Rmax": 1, "indicators": ["y"], "family": ["gaussian"],
        "ids": [1, 2], "waves": [1], "gauss": [value, value + 0.1], "cat": None,
        "ncat": [0], "level_order": [None],
    }


def test_split_null_state_copies_state_zero():
    src = {
        "delta": [0.25, 0.75],
        "tau": [0.8, 0.2, 0.3, 0.7],
        "means": [1.0, -1.0],
        "cov": [0.4, 0.5],
        "rho": [0.2, 0.8, 0.6, 0.4],
    }
    out = pf._core.split_null_state(src, 2, 3, "VVI", 1, 1, 1, 1, 1, 1, 2)
    assert out["delta"][0] == pytest.approx(0.125)
    assert out["delta"][1] == pytest.approx(0.75)
    assert out["delta"][2] == pytest.approx(0.125)
    assert out["tau"][2] == pytest.approx(0.4)
    assert out["tau"][6] == pytest.approx(0.4)
    assert out["means"][2] == pytest.approx(1.0)
    assert out["cov"] == pytest.approx([0.4, 0.5, 0.4])
    assert out["rho"][4] == pytest.approx(0.2)
    assert out["rho"][5] == pytest.approx(0.8)
    assert pf._core.split_null_state(src, 2, 4, "VVI", 1, 1, 1, 1, 1, 1, 2) is None


def test_unusable_observed_fit_does_not_report_p_zero(monkeypatch):
    h0 = stub_fit(K=1, ll=-100, p=4, loglik_finite=False, converged=False)
    h1 = stub_fit(K=2, ll=-80, p=8)
    h0.panel = _same_panel()
    h1.panel = _same_panel()

    def explode(*args, **kwargs):
        raise AssertionError("bootstrap must not run")

    monkeypatch.setattr(pf.selection, "_simulate", explode)
    out = pf.blrt_check(h0, h1, B=2, seed=1, experimental=True)
    assert out["ran"] is False
    assert out["reason"] == "not run: observed fit is not usable blrt validated=false."
    assert out["decision"] == "not_run"
    assert out["validated"] is False
    assert math.isnan(out["p_lo"]) and math.isnan(out["p_hi"])
    assert math.isnan(out["p"])
    assert math.isnan(out["p_plus"])
    assert math.isnan(out["lr_observed"])
    assert out["b"] == 0
    assert out["p_provisional"] is False


def test_blrt_rejects_a_mismatched_comparison(monkeypatch):
    h0 = stub_fit(K=1, ll=-100, p=4)
    h1 = stub_fit(K=2, ll=-80, p=8)
    h0.panel = _same_panel(0.0)
    h1.panel = _same_panel(1.0)

    def explode(*args, **kwargs):
        raise AssertionError("bootstrap must not run")

    monkeypatch.setattr(pf.selection, "_simulate", explode)
    with pytest.raises(ValueError, match="same observed data"):
        pf.blrt_check(h0, h1, B=1, seed=1, experimental=True)
    h1.panel = _same_panel(0.0)
    h1.result["covariance"] = "VVV"
    with pytest.raises(ValueError, match="covariance"):
        pf.blrt_check(h0, h1, B=1, seed=1, experimental=True)
    h1.result["covariance"] = "EEI"
    h1.spec = dict(h1.spec)
    h1.spec["measurement_invariant"] = False
    with pytest.raises(ValueError, match="measurement_invariant"):
        pf.blrt_check(h0, h1, B=1, seed=1, experimental=True)
    h1.spec["measurement_invariant"] = True
    h1.result["K"] = 1
    with pytest.raises(ValueError, match="nested"):
        pf.blrt_check(h0, h1, B=1, seed=1, experimental=True)


def test_observed_and_bootstrap_share_the_negative_lr_allowance(monkeypatch):
    assert _h1_materially_below(0.0, -1e-5) is False
    assert _h1_materially_below(0.0, -1.0) is True
    assert _h1_materially_below(0.0, 1.0) is False
    h0 = stub_fit(K=1, ll=-10, p=4, n_starts=2)
    h1 = stub_fit(K=2, ll=-30, p=8, n_starts=2)
    panel = _same_panel()
    panel["J"] = 1
    panel["Rmax"] = 1
    h0.panel = panel
    h1.panel = panel
    h0.result["params"] = {"delta": [1.0], "tau": [1.0], "means": [0.0], "cov": [1.0], "rho": []}
    starts = []

    def simulate(fit, rng, donors, masks=None, as_panel=False):
        return pd.DataFrame({"id": [1, 2], "y": [0.0, 1.0]})

    def fit_like(template, frame, seed, n_starts=None, n_full=None, max_iter=None, start=None, append_start=False):
        starts.append({"start": start, "n_starts": n_starts, "append_start": append_start})
        out = copy.deepcopy(template)
        out.result = dict(template.result)
        out.result["converged"] = True
        out.result["admissible"] = True
        out.result["n_starts"] = int(n_starts) + (1 if append_start else 0)
        out.result["n_best"] = 2 if out.result["n_starts"] >= 2 else 1
        out.result["loglik"] = -10.0
        return out

    monkeypatch.setattr(pf.selection, "_simulate", simulate)
    monkeypatch.setattr(pf.selection, "_fit_like", fit_like)
    out = pf.blrt_check(h0, h1, B=1, max_retries=0, seed=1, experimental=True)
    assert starts[0]["start"] is not None
    assert starts[0]["append_start"] is True
    assert starts[0]["n_starts"] == 2
    assert out["observed_h1_replication_assessed"] is True
    assert out["lr_observed"] == pytest.approx(0.0)
    assert out["ran"] is True


def test_material_negative_lr_is_a_distinct_failure_and_is_not_clamped(monkeypatch):
    h0 = stub_fit(K=1, ll=-10, p=4, n_starts=4)
    h1 = stub_fit(K=2, ll=-9, p=8, n_starts=4)
    panel = _same_panel()
    panel["J"] = 1
    panel["Rmax"] = 1
    h0.panel = panel
    h1.panel = panel
    h0.result["params"] = {"delta": [1.0], "tau": [1.0], "means": [0.0], "cov": [1.0], "rho": []}
    calls = []

    def simulate(fit, rng, donors, masks=None, as_panel=False):
        return pd.DataFrame({"id": [1, 2], "y": [0.0, 1.0]})

    def fit_like(template, frame, seed, n_starts=None, n_full=None, max_iter=None, start=None, append_start=False):
        calls.append(start)
        out = copy.deepcopy(template)
        out.result = dict(template.result)
        out.result["converged"] = True
        out.result["n_best"] = 2
        if int(out.result["K"]) == 1:
            out.result["loglik"] = -10.0
        else:
            out.result["loglik"] = -100.0
        return out

    monkeypatch.setattr(pf.selection, "_simulate", simulate)
    monkeypatch.setattr(pf.selection, "_fit_like", fit_like)
    failed = pf.blrt_check(h0, h1, B=1, max_retries=0, seed=1, experimental=True)
    assert failed["n_ok"] == 0
    assert failed["lrs"] == []
    assert failed["failures"][0]["reason"] == "h1_below_h0"
    assert math.isnan(failed["p"])
    assert failed["p_provisional"] is True
    assert "provisional" in failed["reason"]
    assert "n_failed > 0" in failed["reason"]

    calls.clear()

    def near(template, frame, seed, n_starts=None, n_full=None, max_iter=None, start=None, append_start=False):
        out = copy.deepcopy(template)
        out.result = dict(template.result)
        out.result["converged"] = True
        out.result["n_best"] = 2
        out.result["loglik"] = -10.0 if int(out.result["K"]) == 1 else -10.0 - 1e-8
        return out

    monkeypatch.setattr(pf.selection, "_fit_like", near)
    kept = pf.blrt_check(h0, h1, B=1, max_retries=0, seed=1, experimental=True)
    assert kept["n_ok"] == 1
    assert kept["lrs"][0] < 0
    assert kept["lrs"][0] == pytest.approx(-2e-8)
    assert kept["p_provisional"] is False

    def recover(template, frame, seed, n_starts=None, n_full=None, max_iter=None, start=None, append_start=False):
        calls.append(start)
        out = copy.deepcopy(template)
        out.result = dict(template.result)
        out.result["converged"] = True
        out.result["n_best"] = 2
        if int(out.result["K"]) == 1:
            out.result["loglik"] = -10.0
        elif start is None:
            out.result["loglik"] = -100.0
        else:
            out.result["loglik"] = -9.0
        return out

    calls.clear()
    monkeypatch.setattr(pf.selection, "_fit_like", recover)
    recovered = pf.blrt_check(h0, h1, B=1, max_retries=1, seed=1, experimental=True)
    assert recovered["n_ok"] == 1
    assert recovered["lrs"][0] == pytest.approx(2.0)
    assert calls[1] is None
    assert calls[3] is not None
    assert calls[3]["delta"][1] == pytest.approx(0.5)


def test_select_model_does_not_rank_incomparable_fits():
    left = stub_fit(K=1, ll=-100, p=4)
    right = stub_fit(K=2, ll=-80, p=8)
    left.panel = _same_panel(0.0)
    right.panel = _same_panel(3.0)
    with pytest.raises(ValueError, match="not comparable"):
        pf.select_model([left, right], second_seed_loglik=[-100, -80], second_seed_n_best=[2, 2])
    table = pf.compare_models([left, right])
    assert len(table) == 2
    assert bool(table["incomparable"].iloc[0])
    assert not bool(table["comparable"].iloc[0])
    quiet = pf.select_model([left, right], auto=False)
    assert len(quiet.candidates) == 2


def test_second_seed_without_n_best_stays_not_assessed():
    fit = stub_fit()
    sel = pf.select_model([fit], second_seed_loglik=[-100.0])
    row = sel.candidates.iloc[0]
    assert row["second_seed_status"] == "not assessed"
    assert "hard:second_seed_not_assessed" in row["rejection_reason"]
    assert not np.isfinite(row["second_seed_n_best"])
    assert row["eligible"] == False


def test_experimental_blrt_bounds_opt_in_and_not_run():
    assert pf.selection_rule()["experimental_blrt"] is False
    assert pf.selection_rule()["blrt"] is False
    off = pf.blrt_failure_bounds(1, 2, 10, alpha=0.05, ran=False)
    assert off["decision"] == "not_run"
    assert off["validated"] is False
    assert math.isnan(off["p_lo"]) and math.isnan(off["p_hi"])
    reject = pf.blrt_failure_bounds(0, 0, 100, alpha=0.05)
    assert reject["p_lo"] == pytest.approx((0 + 1) / (100 + 1))
    assert reject["p_hi"] == pytest.approx((0 + 0 + 1) / (100 + 1))
    assert reject["decision"] == "reject"
    inconclusive = pf.blrt_failure_bounds(0, 10, 100, alpha=0.05)
    assert inconclusive["p_lo"] == pytest.approx(1 / 101)
    assert inconclusive["p_hi"] == pytest.approx((0 + 10 + 1) / 101)
    assert inconclusive["decision"] == "inconclusive"
    retain = pf.blrt_failure_bounds(20, 0, 20, alpha=0.05)
    assert retain["p_lo"] == pytest.approx(1.0)
    assert retain["decision"] == "retain"
    skipped = pf.blrt_failure_bounds(0, 0, 10, alpha=0.05, n_not_run=2)
    assert skipped["n_unresolved"] == 2
    assert skipped["p_lo"] == pytest.approx(1 / 11)
    assert skipped["p_hi"] == pytest.approx((0 + 2 + 1) / 11)
    assert skipped["bounds_decision"] == "retain"
    assert skipped["decision"] == "inconclusive"
    forced = pf.blrt_failure_bounds(0, 0, 100, alpha=0.05, n_not_run=1)
    assert forced["p_hi"] == pytest.approx((0 + 1 + 1) / 101)
    assert forced["bounds_decision"] == "reject"
    assert forced["decision"] == "inconclusive"
    assert pf.selection._blrt_exceeds(1.0, 1.0)
    assert pf.selection._blrt_exceeds(1.0 - 1e-8, 1.0)
    assert pf.selection._blrt_exceeds(2.0, 1.0)
    assert not pf.selection._blrt_exceeds(1.0 - 1e-7, 1.0)
    h0 = stub_fit(K=1, ll=-100, p=4, membership=[200])
    h1 = stub_fit(K=2, ll=-80, p=8)
    h0.panel = _bootstrap_panel()
    h1.panel = _bootstrap_panel()
    not_run = pf.blrt_check(h0, h1, B=1, seed=1)
    assert not_run["ran"] is False
    assert not_run["decision"] == "not_run"
    assert not_run["validated"] is False
    assert math.isnan(not_run["p"]) and math.isnan(not_run["p_lo"]) and math.isnan(not_run["p_hi"])
    assert "not run" in not_run["reason"]
    assert "blrt validated=false" in not_run["reason"]
    assert "explicit opt-in" in not_run["reason"]


def test_mixed_failed_and_unattempted_draws_count_once_in_the_upper_bound():
    # B=100 scheduled draws: 3 exceedances, 2 failed attempts, 1 not attempted,
    # and 94 successful non-exceedances. A study scores decision, not bounds_decision.
    out = pf.blrt_failure_bounds(3, 2, 100, alpha=0.10, n_not_run=1)
    assert out["b"] + out["f"] + out["n_not_run"] + 94 == out["B"]
    assert out["n_unresolved"] == out["f"] + out["n_not_run"]
    assert out["p_lo"] == pytest.approx(4 / 101)
    assert out["p_hi"] == pytest.approx((3 + 2 + 1 + 1) / 101)
    assert out["bounds_decision"] == "reject"
    assert out["decision"] == "inconclusive"

